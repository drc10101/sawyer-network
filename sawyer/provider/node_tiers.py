"""Sawyer Node Tiers — hardware-weighted earnings.

A node's earnings depend on what it contributes, not just how many
tokens it serves. A rig with a 4090 serving large experts is worth
more to the network than a laptop with a 1060 serving tiny ones.

Tier factors:
- GPU VRAM: determines which experts a node can host
- Expert size: larger experts = more compute per token
- Latency: faster responses = better UX = more valuable

Earnings formula:
    weighted_contribution = tokens_served * node_tier.multiplier
    provider_share = (weighted_contribution / total_weighted) * provider_pool

This means a Tier 4 node (24GB+ VRAM) earns 4x per token compared
to a Tier 1 node (4GB VRAM). The investment in hardware pays for itself.
"""

from dataclasses import dataclass
from enum import Enum


class NodeTier(Enum):
    """Hardware capability tiers for Sawyer nodes.

    Tiers are based on GPU VRAM — the primary factor in what experts
    a node can host and how fast it can run inference. More VRAM means
    larger experts, faster inference, and more valuable compute.

    Tier assignments:
        Tier 1 (4GB):    Can host Qwen1.5-MoE small experts (0.5GB)
        Tier 2 (8GB):    Can host DeepSeek-V2 Lite experts (0.8GB)
        Tier 3 (12GB):   Can host Mixtral experts (1.5GB) + run small models
        Tier 4 (24GB+):  Can host DBRX experts (2.5GB) + run full models locally
    """

    TIER_1 = "tier_1"  # 4GB VRAM — entry level
    TIER_2 = "tier_2"  # 8GB VRAM — mid range
    TIER_3 = "tier_3"  # 12GB VRAM — strong
    TIER_4 = "tier_4"  # 24GB+ VRAM — monster

    @classmethod
    def from_vram(cls, vram_gb: float) -> "NodeTier":
        """Classify a node based on available VRAM.

        Args:
            vram_gb: Available GPU VRAM in GB

        Returns:
            The appropriate NodeTier
        """
        if vram_gb >= 24:
            return cls.TIER_4
        elif vram_gb >= 12:
            return cls.TIER_3
        elif vram_gb >= 8:
            return cls.TIER_2
        else:
            return cls.TIER_1


@dataclass
class TierConfig:
    """Configuration for a node tier."""

    tier: NodeTier
    min_vram_gb: float
    max_expert_gb: float  # Largest expert this tier can host
    multiplier: float  # Earnings multiplier relative to Tier 1
    description: str
    can_host_models: list[str]  # Models this tier can serve experts for


# Tier definitions — higher tiers earn more per token because they
# bring more compute, can host larger experts, and enable the network
# to serve models that smaller nodes simply cannot.
TIER_CONFIGS: dict[NodeTier, TierConfig] = {
    NodeTier.TIER_1: TierConfig(
        tier=NodeTier.TIER_1,
        min_vram_gb=4,
        max_expert_gb=0.5,
        multiplier=1.0,
        description="Entry level — small experts, Qwen1.5-MoE only",
        can_host_models=["qwen1.5-moe-a2.7b"],
    ),
    NodeTier.TIER_2: TierConfig(
        tier=NodeTier.TIER_2,
        min_vram_gb=8,
        max_expert_gb=0.8,
        multiplier=2.0,
        description="Mid range — can host DeepSeek experts",
        can_host_models=["qwen1.5-moe-a2.7b", "deepseek-v2-lite"],
    ),
    NodeTier.TIER_3: TierConfig(
        tier=NodeTier.TIER_3,
        min_vram_gb=12,
        max_expert_gb=1.5,
        multiplier=3.0,
        description="Strong — Mixtral experts and small local models",
        can_host_models=["qwen1.5-moe-a2.7b", "deepseek-v2-lite", "mixtral-8x7b"],
    ),
    NodeTier.TIER_4: TierConfig(
        tier=NodeTier.TIER_4,
        min_vram_gb=24,
        max_expert_gb=2.5,
        multiplier=4.0,
        description="Monster — all experts, full local models",
        can_host_models=[
            "qwen1.5-moe-a2.7b",
            "deepseek-v2-lite",
            "mixtral-8x7b",
            "dbrx",
        ],
    ),
}


def get_tier_config(tier: NodeTier) -> TierConfig:
    """Get configuration for a node tier."""
    return TIER_CONFIGS[tier]


def classify_node(vram_gb: float, gpu_name: str = "") -> dict:
    """Classify a node's hardware tier and capabilities.

    Args:
        vram_gb: Available GPU VRAM in GB
        gpu_name: GPU model name (e.g., "NVIDIA GeForce RTX 4090")

    Returns:
        Dict with tier info, capabilities, and earnings multiplier
    """
    tier = NodeTier.from_vram(vram_gb)
    config = TIER_CONFIGS[tier]

    return {
        "tier": tier.value,
        "vram_gb": vram_gb,
        "multiplier": config.multiplier,
        "description": config.description,
        "max_expert_gb": config.max_expert_gb,
        "can_host_models": config.can_host_models,
        "gpu_name": gpu_name,
    }


def compute_weighted_contribution(
    tokens_served: int,
    vram_gb: float,
) -> float:
    """Compute weighted contribution for earnings distribution.

    Tokens served multiplied by the node's hardware tier multiplier.
    This means a Tier 4 node earns 4x per token compared to Tier 1.

    Args:
        tokens_served: Number of tokens this node served
        vram_gb: Available GPU VRAM for tier classification

    Returns:
        Weighted contribution value
    """
    tier = NodeTier.from_vram(vram_gb)
    config = TIER_CONFIGS[tier]
    return tokens_served * config.multiplier


def compute_earnings(
    tokens_served: int,
    vram_gb: float,
    provider_pool_usd: float,
    total_weighted: float,
    uptime_hours: float = 0.0,
    total_uptime_hours: float = 0.0,
    uptime_pool_pct: float = 0.10,
) -> dict:
    """Compute a node's earnings from the provider pool.

    Args:
        tokens_served: Tokens this node served
        vram_gb: Available GPU VRAM for tier classification
        provider_pool_usd: Total provider pool this period
        total_weighted: Sum of all nodes' weighted contributions
        uptime_hours: Hours this node was online
        total_uptime_hours: Sum of all nodes' uptime hours
        uptime_pool_pct: Percentage of pool reserved for uptime (default 10%)

    Returns:
        Dict with token_earnings, uptime_earnings, total_earnings, tier
    """
    tier = NodeTier.from_vram(vram_gb)
    config = TIER_CONFIGS[tier]

    # Token earnings: weighted share of the throughput pool
    throughput_pool = provider_pool_usd * (1 - uptime_pool_pct)
    if total_weighted > 0:
        weighted = tokens_served * config.multiplier
        token_share = (weighted / total_weighted) * throughput_pool
    else:
        token_share = 0.0

    # Uptime earnings: proportional share of uptime pool
    uptime_pool = provider_pool_usd * uptime_pool_pct
    if total_uptime_hours > 0:
        uptime_share = (uptime_hours / total_uptime_hours) * uptime_pool
    else:
        uptime_share = 0.0

    return {
        "tier": tier.value,
        "multiplier": config.multiplier,
        "weighted_contribution": tokens_served * config.multiplier,
        "token_earnings_usd": round(token_share, 4),
        "uptime_earnings_usd": round(uptime_share, 4),
        "total_earnings_usd": round(token_share + uptime_share, 4),
    }