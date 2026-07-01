"""Tests for the proportional revenue pool system."""

import pytest

from sawyer.provider.revenue_pool import (
    EarningsModel,
    PoolPeriod,
    ProviderDistribution,
    RevenuePool,
)


class TestRevenuePoolProportional:
    """Test proportional pool distribution."""

    def test_monster_contributor_gets_lion_share(self):
        """Provider doing 90% of work gets ~90% of the pool."""
        pool = RevenuePool(model=EarningsModel.PROPORTIONAL)
        period = pool.open_period(revenue_usd=1000.00)

        # Kid with monster PC serves 900K tokens
        # Someone with a laptop serves 100K tokens
        distributions = pool.close_period(
            period_id=period.period_id,
            provider_stats=[
                {"provider_id": "prov_monster", "tokens_served": 900_000, "uptime_hours": 720},
                {"provider_id": "prov_laptop", "tokens_served": 100_000, "uptime_hours": 720},
            ],
        )

        monster = next(d for d in distributions if d.provider_id == "prov_monster")
        laptop = next(d for d in distributions if d.provider_id == "prov_laptop")

        assert monster.share_percent == pytest.approx(90.0)
        assert laptop.share_percent == pytest.approx(10.0)

        # Provider pool is 70% of $1000 = $700
        # Token pool (90%) = $630, monster gets 90% of that = $567
        # Uptime pool (10%) = $70, split 50/50 = $35 each
        assert monster.token_earnings_usd == pytest.approx(567.0, abs=0.5)
        assert monster.is_monster is True  # >50% share

        # Laptop gets 10% from token pool ($63) + uptime share
        assert laptop.token_earnings_usd == pytest.approx(63.0, abs=0.5)
        assert laptop.is_monster is False

    def test_pool_split_is_70_30(self):
        """Platform gets 30%, providers get 70%."""
        pool = RevenuePool(model=EarningsModel.PROPORTIONAL)
        period = pool.open_period(revenue_usd=100.00)

        assert period.provider_pool_usd == 70.00
        assert period.platform_pool_usd == 30.00

        distributions = pool.close_period(
            period_id=period.period_id,
            provider_stats=[
                {"provider_id": "prov_1", "tokens_served": 50_000, "uptime_hours": 100},
            ],
        )

        # Total distributed should equal the provider pool
        total_distributed = sum(d.total_earnings_usd for d in distributions)
        assert total_distributed == pytest.approx(70.0, abs=0.01)

    def test_equal_contributors_equal_pay(self):
        """Two providers doing equal work get equal pay."""
        pool = RevenuePool(model=EarningsModel.PROPORTIONAL)
        period = pool.open_period(revenue_usd=100.00)

        distributions = pool.close_period(
            period_id=period.period_id,
            provider_stats=[
                {"provider_id": "prov_a", "tokens_served": 50_000, "uptime_hours": 100},
                {"provider_id": "prov_b", "tokens_served": 50_000, "uptime_hours": 100},
            ],
        )

        a = next(d for d in distributions if d.provider_id == "prov_a")
        b = next(d for d in distributions if d.provider_id == "prov_b")

        assert a.total_earnings_usd == pytest.approx(b.total_earnings_usd, abs=0.01)

    def test_uptime_bonus_distributed(self):
        """10% of pool goes to uptime, 90% to throughput."""
        pool = RevenuePool(model=EarningsModel.PROPORTIONAL)
        period = pool.open_period(revenue_usd=1000.00)

        distributions = pool.close_period(
            period_id=period.period_id,
            provider_stats=[
                {"provider_id": "prov_big", "tokens_served": 1_000_000, "uptime_hours": 720},
                {"provider_id": "prov_small", "tokens_served": 100_000, "uptime_hours": 720},
            ],
        )

        # Provider pool is $700
        # Token pool = 90% = $630
        # Uptime pool = 10% = $70
        # Both have equal uptime, so $35 each from uptime
        big = next(d for d in distributions if d.provider_id == "prov_big")
        small = next(d for d in distributions if d.provider_id == "prov_small")

        # Uptime earnings should be equal since both have same uptime
        assert big.uptime_earnings_usd == pytest.approx(small.uptime_earnings_usd, abs=0.5)

    def test_no_traffic_distribute_by_uptime(self):
        """When there's no traffic, distribute by uptime alone."""
        pool = RevenuePool(model=EarningsModel.PROPORTIONAL)
        period = pool.open_period(revenue_usd=100.00)

        distributions = pool.close_period(
            period_id=period.period_id,
            provider_stats=[
                {"provider_id": "prov_a", "tokens_served": 0, "uptime_hours": 500},
                {"provider_id": "prov_b", "tokens_served": 0, "uptime_hours": 500},
            ],
        )

        # No tokens served, so distribute $70 provider pool by uptime
        total_distributed = sum(d.total_earnings_usd for d in distributions)
        assert total_distributed == pytest.approx(70.0, abs=0.01)

    def test_period_summary(self):
        """Period summary includes all distribution details."""
        pool = RevenuePool(model=EarningsModel.PROPORTIONAL)
        period = pool.open_period(revenue_usd=500.00)
        pool.close_period(
            period_id=period.period_id,
            provider_stats=[
                {"provider_id": "prov_1", "tokens_served": 100_000, "uptime_hours": 200},
            ],
        )

        summary = pool.get_period_summary(period.period_id)
        assert summary is not None
        assert summary["total_revenue_usd"] == 500.00
        assert summary["provider_pool_usd"] == 350.00
        assert summary["platform_pool_usd"] == 150.00
        assert summary["total_tokens_served"] == 100_000
        assert summary["distributed"] is True
        assert len(summary["distributions"]) == 1

    def test_provider_earnings_history(self):
        """Providers can see their earnings across multiple periods."""
        pool = RevenuePool(model=EarningsModel.PROPORTIONAL)

        # Period 1
        period1 = pool.open_period(revenue_usd=100.00)
        pool.close_period(
            period_id=period1.period_id,
            provider_stats=[
                {"provider_id": "prov_1", "tokens_served": 50_000, "uptime_hours": 100},
            ],
        )

        # Period 2
        period2 = pool.open_period(revenue_usd=200.00)
        pool.close_period(
            period_id=period2.period_id,
            provider_stats=[
                {"provider_id": "prov_1", "tokens_served": 100_000, "uptime_hours": 200},
            ],
        )

        earnings = pool.get_provider_earnings("prov_1")
        assert len(earnings) == 2
        assert earnings[0]["period_id"] == period1.period_id
        assert earnings[1]["period_id"] == period2.period_id

    def test_three_providers_different_tiers(self):
        """Monster PC, mid-range, and laptop — proportional earnings."""
        pool = RevenuePool(model=EarningsModel.PROPORTIONAL)
        period = pool.open_period(revenue_usd=1000.00)

        distributions = pool.close_period(
            period_id=period.period_id,
            provider_stats=[
                # Monster PC: 70% of traffic
                {"provider_id": "prov_monster", "tokens_served": 700_000, "uptime_hours": 720},
                # Mid-range: 20% of traffic
                {"provider_id": "prov_mid", "tokens_served": 200_000, "uptime_hours": 500},
                # Laptop: 10% of traffic
                {"provider_id": "prov_laptop", "tokens_served": 100_000, "uptime_hours": 300},
            ],
        )

        monster = next(d for d in distributions if d.provider_id == "prov_monster")
        mid = next(d for d in distributions if d.provider_id == "prov_mid")
        laptop = next(d for d in distributions if d.provider_id == "prov_laptop")

        # Monster gets the biggest slice
        assert monster.total_earnings_usd > mid.total_earnings_usd
        assert mid.total_earnings_usd > laptop.total_earnings_usd

        # Monster is flagged as monster contributor
        assert monster.is_monster is True

        # All three earn something
        assert monster.total_earnings_usd > 0
        assert mid.total_earnings_usd > 0
        assert laptop.total_earnings_usd > 0

    def test_total_distributed_equals_pool(self):
        """Total distributed equals the provider pool exactly."""
        pool = RevenuePool(model=EarningsModel.PROPORTIONAL)
        period = pool.open_period(revenue_usd=500.00)

        distributions = pool.close_period(
            period_id=period.period_id,
            provider_stats=[
                {"provider_id": "p1", "tokens_served": 300_000, "uptime_hours": 400},
                {"provider_id": "p2", "tokens_served": 200_000, "uptime_hours": 300},
                {"provider_id": "p3", "tokens_served": 50_000, "uptime_hours": 200},
            ],
        )

        total_distributed = sum(d.total_earnings_usd for d in distributions)
        assert total_distributed == pytest.approx(period.provider_pool_usd, abs=0.01)


class TestRevenuePoolHybrid:
    """Test hybrid model — flat rate for low traffic, proportional for high."""

    def test_flat_rate_for_low_traffic(self):
        """Below threshold, use flat rate per token."""
        pool = RevenuePool(model=EarningsModel.HYBRID)
        period = pool.open_period(revenue_usd=100.00)

        # Only 5K tokens — below 10K threshold
        distributions = pool.close_period(
            period_id=period.period_id,
            provider_stats=[
                {"provider_id": "prov_1", "tokens_served": 5_000, "uptime_hours": 100},
            ],
        )

        # Should use flat rate: 5000 tokens * $0.002/1K * 70% = $0.007
        # Plus uptime bonus
        assert len(distributions) == 1
        assert distributions[0].total_earnings_usd > 0

    def test_proportional_for_high_traffic(self):
        """Above threshold, use proportional distribution."""
        pool = RevenuePool(model=EarningsModel.HYBRID)
        period = pool.open_period(revenue_usd=1000.00)

        # 500K tokens — well above 10K threshold
        distributions = pool.close_period(
            period_id=period.period_id,
            provider_stats=[
                {"provider_id": "prov_1", "tokens_served": 500_000, "uptime_hours": 200},
            ],
        )

        # Should use proportional distribution
        assert distributions[0].share_percent == 100.0


class TestPoolPeriod:
    """Test period lifecycle."""

    def test_period_starts_open(self):
        pool = RevenuePool()
        period = pool.open_period(revenue_usd=100.00)
        assert period.is_open is True
        assert period.distributed is False

    def test_period_closes_after_distribution(self):
        pool = RevenuePool()
        period = pool.open_period(revenue_usd=100.00)
        pool.close_period(
            period_id=period.period_id,
            provider_stats=[{"provider_id": "p1", "tokens_served": 10000, "uptime_hours": 100}],
        )
        assert period.is_open is False
        assert period.distributed is True

    def test_per_token_value(self):
        """Token value increases as revenue grows."""
        pool = RevenuePool()
        period = pool.open_period(revenue_usd=100.00)
        pool.close_period(
            period_id=period.period_id,
            provider_stats=[
                {"provider_id": "p1", "tokens_served": 50_000, "uptime_hours": 100},
            ],
        )

        # Provider pool = $70, 50K tokens
        # per_token_value = 70 / 50000 = $0.0014 per token
        assert period.per_token_value == pytest.approx(0.0014, abs=0.0001)

    def test_cannot_close_twice(self):
        pool = RevenuePool()
        period = pool.open_period(revenue_usd=100.00)
        pool.close_period(
            period_id=period.period_id,
            provider_stats=[{"provider_id": "p1", "tokens_served": 10000, "uptime_hours": 100}],
        )

        with pytest.raises(ValueError, match="already closed"):
            pool.close_period(
                period_id=period.period_id,
                provider_stats=[],
            )