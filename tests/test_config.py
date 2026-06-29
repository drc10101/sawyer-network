"""Tests for SawyerConfig — provider/payout settings and env overrides."""

import os
from unittest.mock import patch

import pytest

from sawyer.config import SawyerConfig


class TestConfigDefaults:
    """Test default configuration values."""

    def test_default_provider_share(self):
        config = SawyerConfig()
        assert config.provider_share_pct == 70.0
        assert config.platform_fee_pct == 30.0

    def test_default_earnings_rate(self):
        config = SawyerConfig()
        assert config.earnings_rate_per_1k == 0.002

    def test_default_min_payouts(self):
        config = SawyerConfig()
        assert config.min_payout_monthly == 10.0
        assert config.min_payout_quarterly == 25.0

    def test_default_payout_schedule(self):
        config = SawyerConfig()
        assert config.default_payout_schedule == "monthly"

    def test_default_stripe_keys_empty(self):
        config = SawyerConfig()
        assert config.stripe_secret_key == ""
        assert config.stripe_publishable_key == ""
        assert config.stripe_webhook_secret == ""


class TestConfigValidation:
    """Test configuration validation."""

    def test_valid_config(self):
        config = SawyerConfig()
        config.validate()  # Should not raise

    def test_invalid_share_pct(self):
        config = SawyerConfig()
        config.provider_share_pct = 60.0
        config.platform_fee_pct = 30.0
        with pytest.raises(ValueError, match="must equal 100"):
            config.validate()

    def test_invalid_payout_schedule(self):
        config = SawyerConfig()
        config.default_payout_schedule = "weekly"
        with pytest.raises(ValueError, match="monthly or quarterly"):
            config.validate()

    def test_valid_quarterly_schedule(self):
        config = SawyerConfig()
        config.default_payout_schedule = "quarterly"
        config.validate()  # Should not raise


class TestConfigEnvOverrides:
    """Test environment variable overrides."""

    def test_stripe_secret_key_from_env(self):
        with patch.dict(os.environ, {"SAWYER_STRIPE_SECRET_KEY": "sk_test_123"}):
            config = SawyerConfig()
            assert config.stripe_secret_key == "sk_test_123"

    def test_stripe_publishable_key_from_env(self):
        with patch.dict(os.environ, {"SAWYER_STRIPE_PUBLISHABLE_KEY": "pk_test_456"}):
            config = SawyerConfig()
            assert config.stripe_publishable_key == "pk_test_456"

    def test_stripe_webhook_secret_from_env(self):
        with patch.dict(os.environ, {"SAWYER_STRIPE_WEBHOOK_SECRET": "whsec_test"}):
            config = SawyerConfig()
            assert config.stripe_webhook_secret == "whsec_test"

    def test_provider_share_from_env(self):
        with patch.dict(os.environ, {"SAWYER_PROVIDER_SHARE_PCT": "80"}):
            config = SawyerConfig()
            assert config.provider_share_pct == 80.0
            # Platform fee stays default — validate would fail
            config.platform_fee_pct = 20.0
            config.validate()

    def test_earnings_rate_from_env(self):
        with patch.dict(os.environ, {"SAWYER_EARNINGS_RATE": "0.005"}):
            config = SawyerConfig()
            assert config.earnings_rate_per_1k == 0.005

    def test_min_payout_monthly_from_env(self):
        with patch.dict(os.environ, {"SAWYER_MIN_PAYOUT_MONTHLY": "25"}):
            config = SawyerConfig()
            assert config.min_payout_monthly == 25.0

    def test_min_payout_quarterly_from_env(self):
        with patch.dict(os.environ, {"SAWYER_MIN_PAYOUT_QUARTERLY": "50"}):
            config = SawyerConfig()
            assert config.min_payout_quarterly == 50.0

    def test_default_payout_schedule_from_env(self):
        with patch.dict(os.environ, {"SAWYER_DEFAULT_PAYOUT_SCHEDULE": "quarterly"}):
            config = SawyerConfig()
            assert config.default_payout_schedule == "quarterly"

    def test_env_overrides_dont_affect_defaults(self):
        """Without env vars, defaults should be unchanged."""
        # Clear any Sawyer env vars
        env_vars = {
            k: v for k, v in os.environ.items()
            if k.startswith("SAWYER_")
        }
        with patch.dict(os.environ, {}, clear=False):
            for key in env_vars:
                os.environ.pop(key, None)
            config = SawyerConfig()
            assert config.provider_share_pct == 70.0
            assert config.min_payout_monthly == 10.0
