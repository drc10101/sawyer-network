"""Tests for node tier hardware-weighted earnings."""

import pytest

from sawyer.provider.node_tiers import (
    NodeTier,
    TierConfig,
    classify_node,
    compute_earnings,
    compute_weighted_contribution,
    get_tier_config,
)


class TestNodeTierClassification:
    """Test GPU VRAM to tier classification."""

    def test_tier_1_4gb(self):
        """4GB VRAM = Tier 1 (entry level)."""
        info = classify_node(4.0)
        assert info["tier"] == "tier_1"
        assert info["multiplier"] == 1.0
        assert info["can_host_models"] == ["qwen1.5-moe-a2.7b"]

    def test_tier_2_8gb(self):
        """8GB VRAM = Tier 2 (mid range)."""
        info = classify_node(8.0)
        assert info["tier"] == "tier_2"
        assert info["multiplier"] == 2.0
        assert "deepseek-v2-lite" in info["can_host_models"]

    def test_tier_3_12gb(self):
        """12GB VRAM = Tier 3 (strong)."""
        info = classify_node(12.0)
        assert info["tier"] == "tier_3"
        assert info["multiplier"] == 3.0
        assert "mixtral-8x7b" in info["can_host_models"]

    def test_tier_4_24gb(self):
        """24GB VRAM = Tier 4 (monster)."""
        info = classify_node(24.0)
        assert info["tier"] == "tier_4"
        assert info["multiplier"] == 4.0
        assert "dbrx" in info["can_host_models"]

    def test_tier_4_48gb(self):
        """48GB VRAM (A6000) = Tier 4."""
        info = classify_node(48.0)
        assert info["tier"] == "tier_4"
        assert info["multiplier"] == 4.0

    def test_below_4gb_still_tier_1(self):
        """Even 2GB gets Tier 1 — they can still try."""
        info = classify_node(2.0)
        assert info["tier"] == "tier_1"

    def test_gpu_name_preserved(self):
        """GPU model name is passed through."""
        info = classify_node(24.0, gpu_name="NVIDIA GeForce RTX 4090")
        assert info["gpu_name"] == "NVIDIA GeForce RTX 4090"

    def test_boundary_just_under_8gb(self):
        """7.9GB is still Tier 1."""
        info = classify_node(7.9)
        assert info["tier"] == "tier_1"

    def test_boundary_exactly_8gb(self):
        """8.0GB is Tier 2."""
        info = classify_node(8.0)
        assert info["tier"] == "tier_2"


class TestWeightedContribution:
    """Test that hardware multipliers affect earnings."""

    def test_tier_4_earns_4x_tier_1(self):
        """A Tier 4 node earns 4x what a Tier 1 node earns per token."""
        tier1_weighted = compute_weighted_contribution(100_000, vram_gb=4.0)
        tier4_weighted = compute_weighted_contribution(100_000, vram_gb=24.0)

        assert tier1_weighted == 100_000 * 1.0
        assert tier4_weighted == 100_000 * 4.0
        assert tier4_weighted == tier1_weighted * 4

    def test_same_tokens_different_hardware(self):
        """Same token count, different hardware = different weighted contributions."""
        laptop = compute_weighted_contribution(50_000, vram_gb=8.0)
        monster = compute_weighted_contribution(50_000, vram_gb=24.0)

        # Monster earns 2x what mid-range earns for same tokens
        assert monster == laptop * 2

    def test_monster_pc_more_tokens_more_weight(self):
        """The kid with the monster PC serves more tokens AND gets a higher multiplier."""
        laptop = compute_weighted_contribution(100_000, vram_gb=8.0)
        monster = compute_weighted_contribution(500_000, vram_gb=24.0)

        # Laptop: 100K * 2.0 = 200K weighted
        # Monster: 500K * 4.0 = 2M weighted
        assert laptop == 200_000
        assert monster == 2_000_000
        # Monster earns 10x, not just 5x — because better hardware
        assert monster == laptop * 10


class TestEarningsDistribution:
    """Test full earnings computation with tier weighting."""

    def test_monster_vs_laptop(self):
        """Monster PC earns significantly more than laptop for the same pool."""
        # $1000 revenue, $700 provider pool
        # Monster: 500K tokens on a 4090 (Tier 4)
        # Laptop: 100K tokens on a 1060 (Tier 1)
        monster_weighted = compute_weighted_contribution(500_000, 24.0)
        laptop_weighted = compute_weighted_contribution(100_000, 4.0)
        total_weighted = monster_weighted + laptop_weighted

        monster_earnings = compute_earnings(
            tokens_served=500_000,
            vram_gb=24.0,
            provider_pool_usd=700.0,
            total_weighted=total_weighted,
            uptime_hours=720,
            total_uptime_hours=1440,
        )

        laptop_earnings = compute_earnings(
            tokens_served=100_000,
            vram_gb=4.0,
            provider_pool_usd=700.0,
            total_weighted=total_weighted,
            uptime_hours=720,
            total_uptime_hours=1440,
        )

        # Monster: weighted = 2M, laptop: weighted = 100K
        # Monster gets 95% of throughput pool
        assert monster_earnings["total_earnings_usd"] > laptop_earnings["total_earnings_usd"] * 5
        assert monster_earnings["tier"] == "tier_4"
        assert laptop_earnings["tier"] == "tier_1"

    def test_four_tier_network(self):
        """All four tiers contributing — each earns proportionally."""
        pool = 700.0

        t1 = compute_weighted_contribution(50_000, 4.0)    # Tier 1: 4GB
        t2 = compute_weighted_contribution(100_000, 8.0)    # Tier 2: 8GB
        t3 = compute_weighted_contribution(200_000, 12.0)   # Tier 3: 12GB
        t4 = compute_weighted_contribution(500_000, 24.0)   # Tier 4: 24GB
        total = t1 + t2 + t3 + t4

        e1 = compute_earnings(50_000, 4.0, pool, total, uptime_hours=200, total_uptime_hours=1700)
        e2 = compute_earnings(100_000, 8.0, pool, total, uptime_hours=300, total_uptime_hours=1700)
        e3 = compute_earnings(200_000, 12.0, pool, total, uptime_hours=500, total_uptime_hours=1700)
        e4 = compute_earnings(500_000, 24.0, pool, total, uptime_hours=700, total_uptime_hours=1700)

        # Each higher tier earns more
        assert e4["total_earnings_usd"] > e3["total_earnings_usd"]
        assert e3["total_earnings_usd"] > e2["total_earnings_usd"]
        assert e2["total_earnings_usd"] > e1["total_earnings_usd"]

        # Tier 4 monster gets the biggest slice
        assert e4["total_earnings_usd"] > e1["total_earnings_usd"] * 3

        # Total distributed equals pool
        total_distributed = sum(e["total_earnings_usd"] for e in [e1, e2, e3, e4])
        assert total_distributed == pytest.approx(pool, abs=0.01)

    def test_investment_pays_off(self):
        """A 4090 investment is rewarded with higher earnings."""
        # Two nodes serve the SAME number of tokens
        # One has a 1060 (Tier 1), one has a 4090 (Tier 4)
        same_tokens = 100_000

        tier1_weighted = compute_weighted_contribution(same_tokens, 4.0)
        tier4_weighted = compute_weighted_contribution(same_tokens, 24.0)
        total = tier1_weighted + tier4_weighted

        t1 = compute_earnings(same_tokens, 4.0, 700.0, total, uptime_hours=720, total_uptime_hours=1440)
        t4 = compute_earnings(same_tokens, 24.0, 700.0, total, uptime_hours=720, total_uptime_hours=1440)

        # Tier 4 earns 4x for the same work — that's the hardware investment paying off
        assert t4["token_earnings_usd"] == pytest.approx(t1["token_earnings_usd"] * 4, abs=0.01)

        # Uptime earnings are equal (same uptime)
        assert t4["uptime_earnings_usd"] == pytest.approx(t1["uptime_earnings_usd"], abs=0.01)

    def test_laptop_still_earns(self):
        """Even a Tier 1 laptop earns something — just less."""
        weighted = compute_weighted_contribution(10_000, 4.0)
        total = weighted + compute_weighted_contribution(500_000, 24.0)

        earnings = compute_earnings(10_000, 4.0, 700.0, total, uptime_hours=720, total_uptime_hours=1440)

        # They earn something — not zero
        assert earnings["total_earnings_usd"] > 0
        # But it's much less than the monster PC
        assert earnings["tier"] == "tier_1"


class TestTierConfigs:
    """Test tier configuration values."""

    def test_all_tiers_have_configs(self):
        """Every tier has a configuration."""
        for tier in NodeTier:
            config = get_tier_config(tier)
            assert config.multiplier > 0
            assert config.min_vram_gb > 0
            assert config.max_expert_gb > 0
            assert len(config.can_host_models) > 0

    def test_tier_multipliers_escalate(self):
        """Higher tiers have higher multipliers."""
        configs = [get_tier_config(t) for t in NodeTier]
        multipliers = [c.multiplier for c in configs]
        assert multipliers == sorted(multipliers)

    def test_tier_4_hosts_all_models(self):
        """Tier 4 can host all supported models."""
        config = get_tier_config(NodeTier.TIER_4)
        assert len(config.can_host_models) == 4

    def test_tier_1_only_qwen(self):
        """Tier 1 can only host Qwen1.5-MoE."""
        config = get_tier_config(NodeTier.TIER_1)
        assert config.can_host_models == ["qwen1.5-moe-a2.7b"]