"""Sawyer Expert Extraction — split MoE models into per-expert weight shards.

Extracts individual expert weights from GGUF files so each node only needs
to load the experts assigned to it, not the entire model. A Mixtral 8x7B
model is ~24GB, but each expert is ~1.5GB — small enough for a single GPU.
"""

from sawyer.expert.extractor import ExpertExtractor
from sawyer.expert.shard import ExpertShard

__all__ = ["ExpertExtractor", "ExpertShard"]