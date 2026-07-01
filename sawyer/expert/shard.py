"""Sawyer Expert Shard — a per-expert weight shard ready for distributed loading.

An ExpertShard contains the weight tensors for a single MoE expert,
serialized to disk in a format that can be memory-mapped and loaded
directly into a PyTorch model without the full model context.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Shard format version — bump when the on-disk format changes
SHARD_FORMAT_VERSION = 1

# MoE layer patterns for known architectures
# Each pattern maps to (expert_prefix_format, num_experts_in_layer)
EXPERT_PATTERNS = {
    "mixtral": {
        # Mixtral 8x7B: block_sparse_moe.experts.{0-7}.{w1,w2,w3}.weight
        # in layers model.layers.{0-31}.block_sparse_moe
        "expert_prefix": "model.layers.{layer}.block_sparse_moe.experts.{expert}",
        "gate_prefix": "model.layers.{layer}.block_sparse_moe.gate.weight",
        "shared_prefix": None,  # No shared expert in Mixtral
        "num_layers": 32,
        "num_experts": 8,
        "active_experts": 2,
        "expert_tensors": ["w1.weight", "w2.weight", "w3.weight"],
        "dtype": "float16",
    },
    "deepseek-v2": {
        # DeepSeek-V2: uses shared + routed experts
        "expert_prefix": "model.layers.{layer}.mlp.experts.{expert}",
        "gate_prefix": "model.layers.{layer}.mlp.gate.weight",
        "shared_prefix": "model.layers.{layer}.mlp.shared_expert",
        "num_layers": 60,
        "num_experts": 64,
        "active_experts": 6,
        "expert_tensors": ["gate_proj.weight", "up_proj.weight", "down_proj.weight"],
        "dtype": "float16",
    },
    "qwen1.5-moe": {
        "expert_prefix": "model.layers.{layer}.mlp.experts.{expert}",
        "gate_prefix": "model.layers.{layer}.mlp.gate.weight",
        "shared_prefix": None,
        "num_layers": 24,
        "num_experts": 60,
        "active_experts": 4,
        "expert_tensors": ["gate_proj.weight", "up_proj.weight", "down_proj.weight"],
        "dtype": "float16",
    },
    "dbrx": {
        "expert_prefix": "transformer.block.{layer}.ffn.experts.{expert}",
        "gate_prefix": "transformer.block.{layer}.ffn.router.layer.weight",
        "shared_prefix": None,
        "num_layers": 12,
        "num_experts": 16,
        "active_experts": 4,
        "expert_tensors": ["w1.weight", "w2.weight", "v1.weight", "v2.weight"],
        "dtype": "float16",
    },
}


@dataclass
class ExpertShard:
    """A single expert's weight shard, ready for loading on a node.

    Contains the weight tensors for one expert across all MoE layers,
    plus metadata about the expert's position in the model.
    """

    model_name: str
    expert_id: int
    architecture: str  # "mixtral", "deepseek-v2", etc.
    tensors: dict[str, torch.Tensor] = field(default_factory=dict)
    size_bytes: int = 0
    checksum: str = ""

    def total_params(self) -> int:
        """Total parameter count across all tensors."""
        return sum(t.numel() for t in self.tensors.values())

    def total_bytes(self) -> int:
        """Total memory footprint in bytes."""
        return sum(t.nelement() * t.element_size() for t in self.tensors.values())

    def vram_gb(self) -> float:
        """Estimated VRAM usage in GB."""
        return self.total_bytes() / (1024 ** 3)

    def save(self, path: Path) -> None:
        """Save this shard to disk as a .sawyer-expert file.

        Format: JSON header + numpy tensors, concatenated.
        The header contains metadata and offset table.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Build tensor offset table
        header = {
            "format_version": SHARD_FORMAT_VERSION,
            "model_name": self.model_name,
            "expert_id": self.expert_id,
            "architecture": self.architecture,
            "num_tensors": len(self.tensors),
            "tensor_names": list(self.tensors.keys()),
            "tensor_shapes": {k: list(t.shape) for k, t in self.tensors.items()},
            "tensor_dtypes": {k: str(t.dtype) for k, t in self.tensors.items()},
            "total_params": self.total_params(),
            "total_bytes": self.total_bytes(),
        }

        # Serialize header as JSON with fixed-size padding
        header_json = json.dumps(header, indent=2)
        header_bytes = header_json.encode("utf-8")
        header_size = len(header_bytes)

        # Write header size (4 bytes) + header + tensors
        with open(path, "wb") as f:
            # Header length prefix (uint32, big-endian)
            f.write(header_size.to_bytes(4, "big"))
            f.write(header_bytes)

            # Each tensor: name_len(2) + name + shape_len(2) + shape + dtype_len(2) + dtype + data
            for name, tensor in self.tensors.items():
                np_array = tensor.cpu().numpy()
                data = np_array.tobytes()
                self.size_bytes += len(data)

                name_bytes = name.encode("utf-8")
                f.write(len(name_bytes).to_bytes(2, "big"))
                f.write(name_bytes)
                f.write(len(data).to_bytes(8, "big"))
                f.write(data)

        logger.info(
            "Saved expert shard %s:%d to %s (%.1f MB)",
            self.model_name,
            self.expert_id,
            path,
            self.total_bytes() / (1024 ** 2),
        )

    @classmethod
    def load(cls, path: Path) -> "ExpertShard":
        """Load an expert shard from disk.

        Args:
            path: Path to .sawyer-expert file

        Returns:
            ExpertShard with loaded tensors
        """
        path = Path(path)

        with open(path, "rb") as f:
            # Read header
            header_size = int.from_bytes(f.read(4), "big")
            header_json = f.read(header_size).decode("utf-8")
            header = json.loads(header_json)

            # Read tensors
            tensors = {}
            for _ in range(header["num_tensors"]):
                name_len = int.from_bytes(f.read(2), "big")
                name = f.read(name_len).decode("utf-8")

                data_len = int.from_bytes(f.read(8), "big")
                data = f.read(data_len)

                shape = tuple(header["tensor_shapes"][name])
                dtype_str = header["tensor_dtypes"][name]

                # Map dtype string back to numpy dtype
                dtype_map = {
                    "torch.float16": np.float16,
                    "torch.float32": np.float32,
                    "torch.bfloat16": np.float16,  # stored as float16
                    "torch.int8": np.int8,
                    "torch.int32": np.int32,
                }
                np_dtype = dtype_map.get(dtype_str, np.float16)

                np_array = np.frombuffer(data, dtype=np_dtype).reshape(shape)
                tensors[name] = torch.from_numpy(np_array.copy())

        return cls(
            model_name=header["model_name"],
            expert_id=header["expert_id"],
            architecture=header["architecture"],
            tensors=tensors,
            size_bytes=path.stat().st_size,
        )