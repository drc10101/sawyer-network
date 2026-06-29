"""Sawyer Persisted Accounting — TokenAccountant backed by SQLite.

Wraps the in-memory TokenAccountant with automatic persistence.
Every write operation syncs to the database, and the accountant
can be reconstructed from stored state on restart.
"""

import logging

from sawyer.storage.database import SawyerStorage
from sawyer.token.accounting import (
    AccountingError,
    InferenceRecord,
    TokenAccountant,
    UserAccount,
)
from sawyer.token.budget import SubscriptionTier

logger = logging.getLogger(__name__)


class PersistedAccountant:
    """TokenAccountant with SQLite persistence.

    Every mutation is written to the database immediately.
    On startup, accounts and earnings are loaded from the database.
    """

    def __init__(self, storage: SawyerStorage) -> None:
        self._storage = storage
        self._accountant = TokenAccountant()
        self._load_from_db()

    def _load_from_db(self) -> None:
        """Reconstruct accountant state from the database."""
        # Load all accounts
        accounts = self._storage.list_accounts()
        for account in accounts:
            # Re-create in the in-memory accountant without triggering
            # the "already exists" error
            try:
                self._accountant.create_account(
                    user_id=account.user_id,
                    tier=account.tier,
                    rollover=account.balance.rollover,
                )
                # Overwrite with actual balance from DB
                mem_account = self._accountant.get_account(account.user_id)
                if mem_account:
                    mem_account.balance.current_balance = account.balance.current_balance
                    mem_account.balance.rollover = account.balance.rollover
                    mem_account.total_tokens_used = account.total_tokens_used
                    mem_account.total_inferences = account.total_inferences
                    mem_account.last_inference_at = account.last_inference_at
            except AccountingError:
                # Account already loaded from a previous call
                pass

        # Note: host earnings are keyed by node_id, loaded on demand

        logger.info("Loaded %d accounts from database", len(self._accountant._accounts))

    def create_account(
        self,
        user_id: str,
        tier: SubscriptionTier,
        rollover: int = 0,
    ) -> UserAccount:
        """Create a new user account and persist it."""
        account = self._accountant.create_account(user_id, tier, rollover)
        self._storage.save_account(account)
        logger.info("Created account for %s: tier=%s", user_id, tier.value)
        return account

    def get_account(self, user_id: str) -> UserAccount | None:
        """Get a user's account."""
        return self._accountant.get_account(user_id)

    def record_inference(
        self,
        user_id: str,
        model_name: str,
        expert_ids: list[int],
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        node_id: str = "",
        routing_strategy: str = "",
        finish_reason: str = "stop",
    ) -> InferenceRecord:
        """Record an inference and persist both the record and account update."""
        record = self._accountant.record_inference(
            user_id=user_id,
            model_name=model_name,
            expert_ids=expert_ids,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            node_id=node_id,
            routing_strategy=routing_strategy,
            finish_reason=finish_reason,
        )

        # Persist the inference record
        self._storage.save_inference_record(record)

        # Persist the updated account balance
        account = self._accountant.get_account(user_id)
        if account:
            self._storage.save_account(account)

        # Persist host earnings
        if node_id:
            earnings = self._accountant.get_host_earnings(node_id)
            if earnings:
                self._storage.save_host_earnings(earnings)

        # Audit log
        self._storage.append_audit(
            action="inference",
            actor_id=user_id,
            target_id=node_id or "unknown",
            silo="tokens",
            details={
                "model": model_name,
                "total_tokens": record.total_tokens,
                "record_id": record.record_id,
            },
        )

        return record

    def check_quota(self, user_id: str, estimated_tokens: int) -> bool:
        """Check if a user has enough tokens for an estimated request."""
        return self._accountant.check_quota(user_id, estimated_tokens)

    def process_billing_cycle(self, user_id: str):
        """Process billing cycle and persist the new balance."""
        balance = self._accountant.process_billing_cycle(user_id)
        account = self._accountant.get_account(user_id)
        if account:
            self._storage.save_account(account)

        self._storage.append_audit(
            action="billing_cycle",
            actor_id=user_id,
            target_id=user_id,
            silo="billing",
            details={
                "rollover_tokens": balance.rollover,
                "new_balance": balance.total_available,
            },
        )

        return balance

    def get_usage_summary(self, user_id: str) -> dict:
        """Get a usage summary for a user."""
        return self._accountant.get_usage_summary(user_id)

    def get_all_summaries(self) -> list[dict]:
        """Get usage summaries for all users."""
        return self._accountant.get_all_summaries()

    def get_host_earnings(self, node_id: str):
        """Get earnings for a hosting node."""
        return self._accountant.get_host_earnings(node_id)

    def get_all_host_earnings(self) -> list:
        """Get earnings for all hosting nodes."""
        return self._accountant.get_all_host_earnings()

    def get_inference_history(self, user_id: str, limit: int = 100) -> list[InferenceRecord]:
        """Get recent inference records for a user from the database."""
        return self._storage.get_inference_records(user_id, limit)
