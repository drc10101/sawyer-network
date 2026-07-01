"""Tests for Sawyer expert extraction and sharding."""

import json
import tempfile
from pathlib import Path

import pytest
import torch

from sawyer.expert.extractor import ExpertExtractor
from sawyer.expert.shard import EXPERT_PATTERNS, ExpertShard, SHARD_FORMAT_VERSION


class TestExpertShard:
    """Test ExpertShard save/load round-trip."""

    def test_shard_save_load_roundtrip(self, tmp_path):
        """Shard saves to disk and loads back with identical tensors."""
        tensors = {
            "layer0.w1.weight": torch.randn(128, 256, dtype=torch.float16),
            "layer0.w2.weight": torch.randn(256, 128, dtype=torch.float16),
            "layer0.w3.weight": torch.randn(128, 256, dtype=torch.float16),
            "layer0.gate.weight": torch.randn(8, 256, dtype=torch.float16),
        }
        shard = ExpertShard(
            model_name="mixtral-8x7b",
            expert_id=3,
            architecture="mixtral",
            tensors=tensors,
        )

        path = tmp_path / "expert-003.sawyer-expert"
        shard.save(path)

        # Verify file exists
        assert path.exists()

        # Load back
        loaded = ExpertShard.load(path)
        assert loaded.model_name == "mixtral-8x7b"
        assert loaded.expert_id == 3
        assert loaded.architecture == "mixtral"
        assert set(loaded.tensors.keys()) == set(tensors.keys())

        # Verify tensor values match
        for name in tensors:
            assert torch.allclose(tensors[name], loaded.tensors[name], atol=1e-3)

    def test_shard_total_params(self):
        """total_params counts all elements across tensors."""
        tensors = {
            "a": torch.randn(10, 20),
            "b": torch.randn(5, 30),
        }
        shard = ExpertShard(
            model_name="test",
            expert_id=0,
            architecture="test",
            tensors=tensors,
        )
        assert shard.total_params() == 10 * 20 + 5 * 30

    def test_shard_vram_gb(self):
        """vram_gb estimates memory usage."""
        tensors = {
            "a": torch.randn(1000, 1000, dtype=torch.float32),  # 4 MB
        }
        shard = ExpertShard(
            model_name="test",
            expert_id=0,
            architecture="test",
            tensors=tensors,
        )
        # float32: 4 bytes per element, 1M elements = 4MB
        expected_gb = 4 * 1000 * 1000 / (1024 ** 3)
        assert abs(shard.vram_gb() - expected_gb) < 0.01

    def test_shard_float16_tensors(self, tmp_path):
        """Float16 tensors save and load correctly."""
        tensors = {
            "w1": torch.randn(512, 512, dtype=torch.float16),
            "w2": torch.randn(512, 512, dtype=torch.float16),
        }
        shard = ExpertShard(
            model_name="mixtral-8x7b",
            expert_id=7,
            architecture="mixtral",
            tensors=tensors,
        )
        path = tmp_path / "expert-007.sawyer-expert"
        shard.save(path)

        loaded = ExpertShard.load(path)
        assert loaded.tensors["w1"].dtype == torch.float16
        assert loaded.tensors["w2"].dtype == torch.float16
        assert torch.allclose(tensors["w1"], loaded.tensors["w1"], atol=1e-3)

    def test_shard_empty_tensors(self, tmp_path):
        """Shard with no tensors saves and loads."""
        shard = ExpertShard(
            model_name="test",
            expert_id=0,
            architecture="test",
            tensors={},
        )
        path = tmp_path / "empty.sawyer-expert"
        shard.save(path)

        loaded = ExpertShard.load(path)
        assert loaded.model_name == "test"
        assert loaded.expert_id == 0
        assert len(loaded.tensors) == 0
        assert loaded.total_params() == 0


class TestExpertPatterns:
    """Test expert pattern definitions for known architectures."""

    def test_mixtral_pattern(self):
        """Mixtral has 8 experts, top-2 gating."""
        p = EXPERT_PATTERNS["mixtral"]
        assert p["num_experts"] == 8
        assert p["active_experts"] == 2
        assert p["num_layers"] == 32
        assert "w1.weight" in p["expert_tensors"]
        assert "w2.weight" in p["expert_tensors"]
        assert "w3.weight" in p["expert_tensors"]

    def test_deepseek_pattern(self):
        """DeepSeek-V2 has 64 experts, top-6 gating."""
        p = EXPERT_PATTERNS["deepseek-v2"]
        assert p["num_experts"] == 64
        assert p["active_experts"] == 6
        assert p["shared_prefix"] is not None  # Has shared expert

    def test_all_patterns_have_required_fields(self):
        """Every pattern has required fields."""
        required = [
            "expert_prefix", "gate_prefix", "num_layers",
            "num_experts", "active_experts", "expert_tensors",
        ]
        for name, pattern in EXPERT_PATTERNS.items():
            for field in required:
                assert field in pattern, f"{name} missing {field}"

    def test_dbrx_pattern(self):
        """DBRX has 16 experts, 4-way gating."""
        p = EXPERT_PATTERNS["dbrx"]
        assert p["num_experts"] == 16
        assert p["active_experts"] == 4


class TestExpertExtractor:
    """Test ExpertExtractor initialization and path resolution."""

    def test_extractor_creates_cache_dir(self, tmp_path):
        """Extractor creates the cache directory if it doesn't exist."""
        cache = tmp_path / "test_cache"
        extractor = ExpertExtractor(cache_dir=cache)
        assert cache.exists()

    def test_arch_key_mapping(self):
        """Architecture key maps model names correctly."""
        assert ExpertExtractor._arch_key("mixtral-8x7b") == "mixtral"
        assert ExpertExtractor._arch_key("deepseek-v2-lite") == "deepseek-v2"
        assert ExpertExtractor._arch_key("qwen1.5-moe-a2.7b") == "qwen1.5-moe"
        assert ExpertExtractor._arch_key("dbrx") == "dbrx"

    def test_arch_key_unknown_raises(self):
        """Unknown model name raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported model"):
            ExpertExtractor._arch_key("unknown-model")

    def test_list_shards_empty(self, tmp_path):
        """list_shards returns empty for model with no shards."""
        extractor = ExpertExtractor(cache_dir=tmp_path)
        shards = extractor.list_shards("nonexistent")
        assert shards == []

    def test_extract_unknown_model_raises(self):
        """Extracting an unknown model raises ValueError."""
        extractor = ExpertExtractor()
        with pytest.raises(ValueError):
            extractor.extract("nonexistent-model")

    def test_load_shard_not_found_raises(self, tmp_path):
        """Loading a nonexistent shard raises FileNotFoundError."""
        extractor = ExpertExtractor(cache_dir=tmp_path)
        with pytest.raises(FileNotFoundError, match="Expert shard not found"):
            extractor.load_shard("mixtral-8x7b", expert_id=99)

    def test_save_manifest(self, tmp_path):
        """Manifest is saved with correct metadata."""
        tensors = {
            "w1": torch.randn(10, 10, dtype=torch.float16),
        }
        shard = ExpertShard(
            model_name="mixtral-8x7b",
            expert_id=0,
            architecture="mixtral",
            tensors=tensors,
        )

        manifest_path = tmp_path / "manifest.json"
        ExpertExtractor._save_manifest(manifest_path, "mixtral-8x7b", [shard])

        with open(manifest_path) as f:
            manifest = json.load(f)

        assert manifest["model_name"] == "mixtral-8x7b"
        assert manifest["num_shards"] == 1
        assert manifest["shards"][0]["expert_id"] == 0
        assert manifest["shards"][0]["num_tensors"] == 1


class TestFindTensor:
    """Test the fuzzy tensor name matching."""

    def test_exact_match(self):
        """Exact tensor name matches directly."""
        tensors = {"blk.0.ffn_exp.0.w1.weight": torch.tensor([1.0])}
        extractor = ExpertExtractor()
        result = extractor._find_tensor("blk.0.ffn_exp.0.w1.weight", tensors)
        assert result == "blk.0.ffn_exp.0.w1.weight"

    def test_no_match(self):
        """No match returns None."""
        tensors = {"some.other.tensor": torch.tensor([1.0])}
        extractor = ExpertExtractor()
        result = extractor._find_tensor("blk.0.ffn_exp.0.w1.weight", tensors)
        assert result is None

    def test_prefix_match(self):
        """Tensor found with model. prefix."""
        tensors = {"model.layers.0.ffn_exp.0.w1.weight": torch.tensor([1.0])}
        extractor = ExpertExtractor()
        result = extractor._find_tensor("0.ffn_exp.0.w1.weight", tensors)
        # Should find it via suffix matching
        assert result is not None