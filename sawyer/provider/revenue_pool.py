"""Sawyer Revenue Pool — proportional earnings distribution.

Instead of a flat per-token rate, the 70% provider pool is distributed
proportionally based on each provider's contribution. This means:

- A kid with a monster PC doing 90% of the work gets 90% of the pool
- As the network grows, everyone's slice grows with it
- Small contributors still earn — just proportionally less

The pool model rewards contribution relative to total network output,
not just absolute volume. If you own a bigger share of a bigger pie,
your earnings grow even if your throughput stays the same.

Pool calculation:
    total_revenue = subscription_payments_this_period
    provider_pool = total_revenue * 0.70
    provider_share = (my_tokens_served / total_tokens_served) * provider_pool

If total_tokens_served is 0 (new network, no traffic yet), providers
with nodes online earn a baseline minimum per hour of uptime.
"""

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────

PROVIDER_POOL_PERCENT = 70.0  # 70% of revenue goes to providers
PLATFORM_POOL_PERCENT = 30.0  # 30% goes to platform

# Baseline earnings for uptime (per hour, per node)
# Even if no one is using the network, nodes that are online and available
# earn a small amount for being ready to serve.
UPTIME_RATE_PER_HOUR = 0.05  # $0.05/hr per node for being available

# Minimum tokens before proportional kicks in
# Below this threshold, providers earn the flat rate per token
MIN_TOKENS_FOR_PROPORTIONAL = 10_000  # 10K tokens


class EarningsModel(Enum):
    """How providers earn money."""

    FLAT = "flat"  # Fixed rate per 1K tokens ($0.002/1K)
    PROPORTIONAL = "proportional"  # Share of the 70% revenue pool
    HYBRID = "hybrid"  # Flat rate until threshold, then proportional


@dataclass
class PoolPeriod:
    """A billing period's revenue pool."""

    period_id: str
    start_time: float
    end_time: float = 0.0  # 0 = still open

    # Revenue
    total_revenue_usd: float = 0.0  # Total subscription revenue this period
    provider_pool_usd: float = 0.0  # 70% of total revenue
    platform_pool_usd: float = 0.0  # 30% of total revenue

    # Token throughput
    total_tokens_served: int = 0

    # Uptime
    total_node_hours: float = 0.0  # Total hours nodes were online

    # Distribution
    distributed: bool = False
    distributions: list[dict[str, Any]] = field(default_factory=list)

    @property
    def is_open(self) -> bool:
        """Whether this period is still accumulating."""
        return self.end_time == 0.0

    @property
    def per_token_value(self) -> float:
        """USD value per token in this period's provider pool.

        Only meaningful after the period closes and tokens are counted.
        """
        if self.total_tokens_served == 0:
            return 0.0
        return self.provider_pool_usd / self.total_tokens_served


@dataclass
class ProviderDistribution:
    """One provider's share of a pool period."""

    provider_id: str
    period_id: str
    tokens_served: int  # How many tokens this provider served
    share_percent: float  # What % of total tokens they handled
    uptime_hours: float  # How many hours their nodes were online
    token_earnings_usd: float  # Earnings from token throughput
    uptime_earnings_usd: float  # Earnings from uptime baseline
    total_earnings_usd: float  # Total this period

    @property
    def is_monster(self) -> bool:
        """Whether this provider is doing disproportionate work.

        A 'monster' provider handles >50% of total network traffic.
        These are the high-spec rigs that carry the network.
        """
        return self.share_percent >= 50.0


class RevenuePool:
    """Distributes the 70% provider pool proportionally.

    The pool model:
    1. Each billing period, total subscription revenue is collected
    2. 70% goes into the provider pool
    3. Each provider's share = (their tokens / total tokens) * pool
    4. Uptime bonus for nodes that stayed online even with low traffic
    5. Monster contributor bonus: providers doing >50% get a boost

    This means the kid with the monster PC doing most of the work
    gets most of the money. Not just a flat rate — a proportional
    share of the entire pool.
    """

    def __init__(self, model: EarningsModel = EarningsModel.PROPORTIONAL) -> None:
        self._model = model
        self._periods: list[PoolPeriod] = []
        self._period_counter = 0

    def open_period(
        self,
        revenue_usd: float,
        start_time: float | None = None,
    ) -> PoolPeriod:
        """Open a new billing period with known revenue.

        Args:
            revenue_usd: Total subscription revenue for this period
            start_time: Period start timestamp (default: now)
        """
        self._period_counter += 1
        period = PoolPeriod(
            period_id=f"pool-{self._period_counter:06d}",
            start_time=start_time or time.time(),
            total_revenue_usd=revenue_usd,
            provider_pool_usd=round(revenue_usd * PROVIDER_POOL_PERCENT / 100, 2),
            platform_pool_usd=round(revenue_usd * PLATFORM_POOL_PERCENT / 100, 2),
        )
        self._periods.append(period)
        logger.info(
            "Opened pool period %s: $%.2f revenue → $%.2f provider pool",
            period.period_id,
            revenue_usd,
            period.provider_pool_usd,
        )
        return period

    def close_period(
        self,
        period_id: str,
        provider_stats: list[dict[str, Any]],
        end_time: float | None = None,
    ) -> list[ProviderDistribution]:
        """Close a period and distribute the provider pool.

        Args:
            period_id: Period to close
            provider_stats: List of dicts with provider_id, tokens_served, uptime_hours
            end_time: Period end timestamp (default: now)

        Returns:
            List of ProviderDistribution with each provider's earnings
        """
        period = self._find_period(period_id)
        if period is None:
            raise ValueError(f"Period {period_id} not found")
        if not period.is_open:
            raise ValueError(f"Period {period_id} is already closed")

        period.end_time = end_time or time.time()

        # Count total tokens
        total_tokens = sum(s.get("tokens_served", 0) for s in provider_stats)
        total_uptime = sum(s.get("uptime_hours", 0.0) for s in provider_stats)
        period.total_tokens_served = total_tokens
        period.total_node_hours = total_uptime

        # Choose distribution model
        if total_tokens == 0:
            # No traffic yet — distribute by uptime only
            distributions = self._distribute_by_uptime(period, provider_stats)
        elif (
            self._model == EarningsModel.HYBRID
            and total_tokens < MIN_TOKENS_FOR_PROPORTIONAL
        ):
            # Low traffic — use flat rate
            distributions = self._distribute_flat(period, provider_stats)
        else:
            # Normal operation — proportional pool
            distributions = self._distribute_proportional(period, provider_stats)

        period.distributed = True
        period.distributions = [
            {
                "provider_id": d.provider_id,
                "tokens_served": d.tokens_served,
                "share_percent": round(d.share_percent, 2),
                "uptime_hours": round(d.uptime_hours, 2),
                "token_earnings_usd": round(d.token_earnings_usd, 4),
                "uptime_earnings_usd": round(d.uptime_earnings_usd, 4),
                "total_earnings_usd": round(d.total_earnings_usd, 4),
                "is_monster": d.is_monster,
            }
            for d in distributions
        ]

        logger.info(
            "Closed pool period %s: %d tokens across %d providers, "
            "$%.2f distributed",
            period_id,
            total_tokens,
            len(distributions),
            period.provider_pool_usd,
        )
        return distributions

    def _distribute_proportional(
        self,
        period: PoolPeriod,
        provider_stats: list[dict[str, Any]],
    ) -> list[ProviderDistribution]:
        """Distribute the provider pool proportionally by tokens served.

        Each provider gets: (my_tokens / total_tokens) * provider_pool
        Plus an uptime bonus for being available.
        """
        total_tokens = period.total_tokens_served
        total_uptime = period.total_node_hours
        pool = period.provider_pool_usd

        # Reserve 10% of pool for uptime bonus (being available matters)
        token_pool = pool * 0.90  # 90% distributed by throughput
        uptime_pool = pool * 0.10  # 10% distributed by uptime

        distributions = []
        for stats in provider_stats:
            provider_id = stats["provider_id"]
            tokens = stats.get("tokens_served", 0)
            uptime = stats.get("uptime_hours", 0.0)

            # Token share: proportion of total tokens
            if total_tokens > 0:
                share_percent = (tokens / total_tokens) * 100
                token_earnings = (tokens / total_tokens) * token_pool
            else:
                share_percent = 0.0
                token_earnings = 0.0

            # Uptime share: proportion of total uptime hours
            if total_uptime > 0:
                uptime_earnings = (uptime / total_uptime) * uptime_pool
            else:
                uptime_earnings = uptime * UPTIME_RATE_PER_HOUR

            distributions.append(
                ProviderDistribution(
                    provider_id=provider_id,
                    period_id=period.period_id,
                    tokens_served=tokens,
                    share_percent=share_percent,
                    uptime_hours=uptime,
                    token_earnings_usd=token_earnings,
                    uptime_earnings_usd=uptime_earnings,
                    total_earnings_usd=token_earnings + uptime_earnings,
                )
            )

        return distributions

    def _distribute_by_uptime(
        self,
        period: PoolPeriod,
        provider_stats: list[dict[str, Any]],
    ) -> list[ProviderDistribution]:
        """No traffic yet — distribute the pool by uptime alone.

        Early network: nodes that are online and available earn
        the full pool proportionally by how long they stayed up.
        """
        total_uptime = period.total_node_hours
        pool = period.provider_pool_usd

        distributions = []
        for stats in provider_stats:
            provider_id = stats["provider_id"]
            uptime = stats.get("uptime_hours", 0.0)

            if total_uptime > 0:
                share_percent = (uptime / total_uptime) * 100
                total_earnings = (uptime / total_uptime) * pool
            else:
                share_percent = 0.0
                total_earnings = 0.0

            distributions.append(
                ProviderDistribution(
                    provider_id=provider_id,
                    period_id=period.period_id,
                    tokens_served=0,
                    share_percent=share_percent,
                    uptime_hours=uptime,
                    token_earnings_usd=0.0,
                    uptime_earnings_usd=total_earnings,
                    total_earnings_usd=total_earnings,
                )
            )

        return distributions

    def _distribute_flat(
        self,
        period: PoolPeriod,
        provider_stats: list[dict[str, Any]],
    ) -> list[ProviderDistribution]:
        """Low-traffic mode — flat rate per token.

        Used when the network is small and proportional distribution
        would result in tiny payments. Falls back to $0.002 per 1K tokens.
        """
        from sawyer.provider.manager import EARNINGS_RATE_PER_1K_TOKENS, PROVIDER_SHARE_PERCENT

        total_tokens = period.total_tokens_served
        distributions = []

        for stats in provider_stats:
            provider_id = stats["provider_id"]
            tokens = stats.get("tokens_served", 0)
            uptime = stats.get("uptime_hours", 0.0)

            share_percent = (tokens / total_tokens * 100) if total_tokens > 0 else 0
            flat_earnings = (tokens / 1000.0) * EARNINGS_RATE_PER_1K_TOKENS
            provider_share = flat_earnings * (PROVIDER_SHARE_PERCENT / 100)
            uptime_bonus = uptime * UPTIME_RATE_PER_HOUR

            distributions.append(
                ProviderDistribution(
                    provider_id=provider_id,
                    period_id=period.period_id,
                    tokens_served=tokens,
                    share_percent=share_percent,
                    uptime_hours=uptime,
                    token_earnings_usd=provider_share,
                    uptime_earnings_usd=uptime_bonus,
                    total_earnings_usd=provider_share + uptime_bonus,
                )
            )

        return distributions

    def get_period(self, period_id: str) -> PoolPeriod | None:
        """Get a period by ID."""
        return self._find_period(period_id)

    def get_period_summary(self, period_id: str) -> dict[str, Any] | None:
        """Get a summary of a period including distributions."""
        period = self._find_period(period_id)
        if period is None:
            return None

        return {
            "period_id": period.period_id,
            "start_time": period.start_time,
            "end_time": period.end_time,
            "total_revenue_usd": period.total_revenue_usd,
            "provider_pool_usd": period.provider_pool_usd,
            "platform_pool_usd": period.platform_pool_usd,
            "total_tokens_served": period.total_tokens_served,
            "total_node_hours": period.total_node_hours,
            "distributed": period.distributed,
            "per_token_value": round(period.per_token_value, 6),
            "distributions": period.distributions,
        }

    def get_provider_earnings(
        self,
        provider_id: str,
    ) -> list[dict[str, Any]]:
        """Get all earnings for a provider across all periods."""
        earnings = []
        for period in self._periods:
            if not period.distributed:
                continue
            for dist in period.distributions:
                if dist["provider_id"] == provider_id:
                    earnings.append(
                        {
                            "period_id": period.period_id,
                            **dist,
                        }
                    )
        return earnings

    def _find_period(self, period_id: str) -> PoolPeriod | None:
        """Find a period by ID."""
        for period in self._periods:
            if period.period_id == period_id:
                return period
        return None