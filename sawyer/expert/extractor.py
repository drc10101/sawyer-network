"""Sawyer Expert Extractor — split MoE GGUF models into per-expert weight shards.

Reads a GGUF file, identifies expert-specific weight tensors, and extracts
each expert into a standalone .sawyer-expert shard that can be loaded by a
node without needing the full model.

Architecture support:
- Mixtral 8x7B (8 experts, top-2 gating)
- DeepSeek-V2 Lite (64 experts + shared, top-6 gating)
- Qwen1.5-MoE (60 experts, top-4 gating)
- DBRX (16 experts, top-4 gating)

The extractor also captures the gating network weights (router weights) so
the local router can make routing decisions.
"""

import hashlib
import logging
from pathlib import Path
from typing import Iterator

import torch

from sawyer.expert.shard import EXPERT_PATTERNS, ExpertShard
from sawyer.model.registry import MoEModel, get_model

logger = logging.getLogger(__name__)

# GGUF tensor name patterns that identify expert weights
# Format: (prefix_template, is_expert_specific)
# is_expert_specific=True means the tensor contains "{expert}" placeholder
GGUF_EXPERT_PATTERNS = {
    "mixtral": {
        "expert": "blk.{layer}.ffn_exp.{expert}.{gate}.weight",
        "gate": "blk.{layer}.ffn_gate_inp.weight",
        "gates": ["w1", "w2", "w3"],
    },
    "deepseek-v2": {
        "expert": "blk.{layer}.ffn_exp.{expert}.{gate}.weight",
        "gate": "blk.{layer}.ffn_gate_inp.weight",
        "gates": ["w1", "w2", "w3"],
    },
    "qwen1.5-moe": {
        "expert": "blk.{layer}.ffn_exp.{expert}.{gate}.weight",
        "gate": "blk.{layer}.ffn_gate_inp.weight",
        "gates": ["w1", "w2", "w3"],
    },
    "dbrx": {
        "expert": "blk.{layer}.ffn_exp.{expert}.{gate}.weight",
        "gate": "blk.{layer}.ffn_gate_inp.weight",
        "gates": ["w1", "w2", "v1", "v2"],
    },
}


class ExpertExtractor:
    """Extract per-expert weight shards from a GGUF model file.

    Usage:
        extractor = ExpertExtractor()
        shards = extractor.extract("mixtral-8x7b", gguf_path)
        for shard in shards:
            shard.save(output_dir / f"expert-{shard.expert_id}.sawyer-expert")
    """

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        """Initialize the extractor.

        Args:
            cache_dir: Directory for output shards. Defaults to ~/.sawyer/experts/
        """
        self.cache_dir = Path(cache_dir or "~/.sawyer/experts").expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def extract(
        self,
        model_name: str,
        gguf_path: str | Path | None = None,
        expert_ids: list[int] | None = None,
        layers: list[int] | None = None,
    ) -> list[ExpertShard]:
        """Extract expert shards from a GGUF model file.

        Args:
            model_name: Model identifier (e.g., "mixtral-8x7b")
            gguf_path: Path to GGUF file. If None, looks in Sawyer cache.
            expert_ids: Which experts to extract. None = all experts.
            layers: Which layers to extract from. None = all layers.

        Returns:
            List of ExpertShard objects, one per expert
        """
        model = get_model(model_name)
        arch_key = self._arch_key(model_name)
        pattern = EXPERT_PATTERNS[arch_key]
        gguf_pattern = GGUF_EXPERT_PATTERNS[arch_key]

        if expert_ids is None:
            expert_ids = list(range(model.num_experts))
        if layers is None:
            layers = list(range(pattern["num_layers"]))

        # Load GGUF tensors
        gguf_path = self._resolve_gguf(model_name, gguf_path)
        logger.info("Loading GGUF tensors from %s", gguf_path)
        all_tensors = self._load_gguf_tensors(gguf_path)

        shards = []
        for expert_id in expert_ids:
            logger.info("Extracting expert %d from %s", expert_id, model_name)
            shard = self._extract_expert(
                model=model,
                expert_id=expert_id,
                layers=layers,
                all_tensors=all_tensors,
                arch_key=arch_key,
                gguf_pattern=gguf_pattern,
            )
            shards.append(shard)
            logger.info(
                "Expert %d: %d tensors, %.1f MB",
                expert_id,
                len(shard.tensors),
                shard.total_bytes() / (1024 ** 2),
            )

        return shards

    def extract_and_save(
        self,
        model_name: str,
        gguf_path: str | Path | None = None,
        expert_ids: list[int] | None = None,
        layers: list[int] | None = None,
        output_dir: str | Path | None = None,
    ) -> list[Path]:
        """Extract expert shards and save them to disk.

        Args:
            model_name: Model identifier
            gguf_path: Path to GGUF file (None = use Sawyer cache)
            expert_ids: Which experts to extract (None = all)
            layers: Which layers (None = all)
            output_dir: Output directory (None = ~/.sawyer/experts/{model}/)

        Returns:
            List of paths to saved shard files
        """
        shards = self.extract(model_name, gguf_path, expert_ids, layers)

        if output_dir is None:
            output_dir = self.cache_dir / model_name
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        paths = []
        for shard in shards:
            path = output_dir / f"expert-{shard.expert_id:03d}.sawyer-expert"
            shard.save(path)
            paths.append(path)

        # Save manifest
        manifest_path = output_dir / "manifest.json"
        self._save_manifest(manifest_path, model_name, shards)

        logger.info("Saved %d expert shards to %s", len(paths), output_dir)
        return paths

    def load_shard(
        self,
        model_name: str,
        expert_id: int,
        cache_dir: str | Path | None = None,
    ) -> ExpertShard:
        """Load a previously extracted expert shard from disk.

        Args:
            model_name: Model identifier
            expert_id: Expert number
            cache_dir: Shard directory (None = default cache)

        Returns:
            Loaded ExpertShard
        """
        base = Path(cache_dir) if cache_dir else self.cache_dir / model_name
        path = base / f"expert-{expert_id:03d}.sawyer-expert"

        if not path.exists():
            raise FileNotFoundError(
                f"Expert shard not found: {path}\n"
                f"Run 'sawyer extract {model_name}' first."
            )

        return ExpertShard.load(path)

    def list_shards(self, model_name: str) -> list[dict]:
        """List available expert shards for a model.

        Returns:
            List of dicts with expert_id, path, size, and checksum info
        """
        model_dir = self.cache_dir / model_name
        if not model_dir.exists():
            return []

        import json

        manifest_path = model_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
            return manifest.get("shards", [])

        # Fallback: scan for .sawyer-expert files
        shards = []
        for path in sorted(model_dir.glob("expert-*.sawyer-expert")):
            try:
                expert_id = int(path.stem.split("-")[1])
                shards.append({
                    "expert_id": expert_id,
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                })
            except (ValueError, IndexError):
                continue
        return shards

    def _extract_expert(
        self,
        model: MoEModel,
        expert_id: int,
        layers: list[int],
        all_tensors: dict[str, torch.Tensor],
        arch_key: str,
        gguf_pattern: dict,
    ) -> ExpertShard:
        """Extract one expert's weights across all layers.

        For each MoE layer, pulls the expert-specific tensors (w1, w2, w3)
        and the gating network weight. This gives each shard everything it
        needs to run forward passes for its assigned expert.
        """
        pattern = EXPERT_PATTERNS[arch_key]
        expert_tensors = {}

        for layer_idx in layers:
            # Extract expert FFN weights (w1, w2, w3 or gate/up/down projections)
            for gate_name in gguf_pattern["gates"]:
                # Try GGUF tensor naming: blk.{layer}.ffn_exp.{expert}.{gate}.weight
                tensor_name = gguf_pattern["expert"].format(
                    layer=layer_idx,
                    expert=expert_id,
                    gate=gate_name,
                )

                # Search for the tensor (GGUF names may vary slightly)
                found_name = self._find_tensor(tensor_name, all_tensors)
                if found_name:
                    expert_tensors[f"layer{layer_idx}.{gate_name}.weight"] = all_tensors[found_name]
                else:
                    # Try alternate naming: model.layers.{layer}.block_sparse_moe.experts.{expert}.{gate}.weight
                    alt_name = pattern["expert_prefix"].format(
                        layer=layer_idx,
                        expert=expert_id,
                    ) + f".{gate_name}.weight"
                    found_alt = self._find_tensor(alt_name, all_tensors)
                    if found_alt:
                        expert_tensors[f"layer{layer_idx}.{gate_name}.weight"] = all_tensors[found_alt]

            # Extract gating network weight (shared across all experts in this layer)
            gate_name = gguf_pattern["gate"].format(layer=layer_idx)
            found_gate = self._find_tensor(gate_name, all_tensors)
            if found_gate:
                expert_tensors[f"layer{layer_idx}.gate.weight"] = all_tensors[found_gate]

        return ExpertShard(
            model_name=model.name,
            expert_id=expert_id,
            architecture=arch_key,
            tensors=expert_tensors,
        )

    def _find_tensor(self, name: str, tensors: dict[str, torch.Tensor]) -> str | None:
        """Find a tensor by name, with fuzzy matching for GGUF naming conventions.

        GGUF files may prefix tensor names differently (blk. vs model.layers. etc.)
        """
        # Exact match
        if name in tensors:
            return name

        # Try common GGUF prefixes
        for prefix in ["", "model."]:
            candidate = prefix + name
            if candidate in tensors:
                return candidate

        # Try stripping common prefixes from the search name
        clean = name
        for prefix in ["model.layers.", "blk.", "transformer.block."]:
            if clean.startswith(prefix):
                clean = clean[len(prefix):]

        for tensor_name in tensors:
            tensor_clean = tensor_name
            for prefix in ["model.layers.", "blk.", "transformer.block."]:
                if tensor_clean.startswith(prefix):
                    tensor_clean = tensor_clean[len(prefix):]
            if tensor_clean == clean or tensor_clean.endswith("." + clean):
                return tensor_name

        return None

    def _resolve_gguf(self, model_name: str, gguf_path: str | Path | None) -> Path:
        """Resolve the GGUF file path, searching the Sawyer cache if needed."""
        if gguf_path is not None:
            path = Path(gguf_path)
            if not path.exists():
                raise FileNotFoundError(f"GGUF file not found: {path}")
            return path

        # Search Sawyer cache
        from sawyer.node.weights import WeightLoader

        loader = WeightLoader()
        cached = loader.get_cached_path(model_name)
        if cached and cached.exists():
            return cached

        raise FileNotFoundError(
            f"No GGUF file found for {model_name} in cache.\n"
            f"Run 'sawyer download {model_name}' first."
        )

    def _load_gguf_tensors(self, gguf_path: Path) -> dict[str, torch.Tensor]:
        """Load all tensors from a GGUF file using the gguf library.

        Returns a dict mapping tensor name -> torch.Tensor.
        """
        from gguf import GGUFReader

        reader = GGUFReader(str(gguf_path))
        tensors = {}

        for tensor in reader.tensors:
            # GGUFReader gives us numpy arrays
            name = tensor.name
            # Remove any GGUF name encoding prefix (e.g., "_model_model_")
            if name.startswith("_model_model_"):
                name = name.replace("_model_model_", "model.", 1)
            if name.startswith("_model_"):
                name = "model." + name[7:]

            array = tensor.data
            # Convert to torch tensor
            dtype = self._numpy_to_torch_dtype(array.dtype)
            tensors[name] = torch.from_numpy(array.copy()).to(dtype)

        logger.info("Loaded %d tensors from %s", len(tensors), gguf_path.name)
        return tensors

    @staticmethod
    def _numpy_to_torch_dtype(np_dtype) -> torch.dtype:
        """Map numpy dtype to torch dtype."""
        import numpy as np

        mapping = {
            np.float16: torch.float16,
            np.float32: torch.float32,
            np.float64: torch.float64,
            np.int8: torch.int8,
            np.int16: torch.int16,
            np.int32: torch.int32,
            np.int64: torch.int64,
            np.uint8: torch.uint8,
        }
        return mapping.get(np_dtype, torch.float16)

    @staticmethod
    def _arch_key(model_name: str) -> str:
        """Map model name to architecture key for tensor pattern matching."""
        mapping = {
            "mixtral-8x7b": "mixtral",
            "deepseek-v2-lite": "deepseek-v2",
            "qwen1.5-moe-a2.7b": "qwen1.5-moe",
            "dbrx": "dbrx",
        }
        if model_name not in mapping:
            raise ValueError(
                f"Unsupported model: {model_name}. "
                f"Supported: {', '.join(mapping.keys())}"
            )
        return mapping[model_name]

    @staticmethod
    def _save_manifest(
        manifest_path: Path,
        model_name: str,
        shards: list[ExpertShard],
    ) -> None:
        """Save a manifest.json describing all extracted shards."""
        import json

        manifest = {
            "model_name": model_name,
            "num_shards": len(shards),
            "shards": [
                {
                    "expert_id": s.expert_id,
                    "architecture": s.architecture,
                    "num_tensors": len(s.tensors),
                    "total_params": s.total_params(),
                    "total_bytes": s.total_bytes(),
                    "vram_gb": round(s.vram_gb(), 2),
                    "path": f"expert-{s.expert_id:03d}.sawyer-expert",
                }
                for s in shards
            ],
        }
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)