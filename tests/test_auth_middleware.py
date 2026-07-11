"""Tests for Sawyer auth middleware — API key auth, CORS, input validation."""

import pytest
from unittest.mock import MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from sawyer.auth.middleware import (
    MAX_MAX_TOKENS,
    MAX_MESSAGE_LENGTH,
    MAX_MESSAGES,
    MAX_TEMPERATURE,
    MAX_TOP_P,
    VALID_MODEL_PATTERN,
    add_cors_middleware,
    validate_chat_request,
    verify_api_key,
)
from sawyer.auth.api import (
    APIKey,
    InvalidAPIKey,
    KeyStatus,
    RateLimitExceeded,
    SawyerAuth,
)
from fastapi import Depends


# --- Input validation tests ---

class TestValidateChatRequest:
    def test_valid_minimal_request(self):
        """Minimal valid request passes validation."""
        body = {
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "sawyer",
            "temperature": 0.7,
            "top_p": 0.9,
            "max_tokens": 512,
        }
        # Should not raise
        validate_chat_request(body)

    def test_missing_messages(self):
        """Missing messages field raises 422."""
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            validate_chat_request({"model": "sawyer"})
        assert exc_info.value.status_code == 422
        assert "messages is required" in str(exc_info.value.detail)

    def test_empty_messages(self):
        """Empty messages list raises 422."""
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            validate_chat_request({"messages": []})
        assert exc_info.value.status_code == 422

    def test_no_user_message(self):
        """Messages without user role raises 422."""
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            validate_chat_request({"messages": [{"role": "system", "content": "You are helpful"}]})
        assert exc_info.value.status_code == 422

    def test_too_many_messages(self):
        """More than MAX_MESSAGES messages raises 422."""
        from fastapi import HTTPException

        messages = [{"role": "user", "content": f"msg {i}"} for i in range(MAX_MESSAGES + 1)]
        with pytest.raises(HTTPException) as exc_info:
            validate_chat_request({"messages": messages})
        assert exc_info.value.status_code == 422

    def test_invalid_model_name(self):
        """Model name with special characters raises 422."""
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            validate_chat_request({
                "messages": [{"role": "user", "content": "hi"}],
                "model": "../../../etc/passwd",
            })
        assert exc_info.value.status_code == 422

    def test_temperature_out_of_range(self):
        """Temperature above 2.0 raises 422."""
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            validate_chat_request({
                "messages": [{"role": "user", "content": "hi"}],
                "temperature": 3.0,
            })
        assert exc_info.value.status_code == 422

    def test_max_tokens_out_of_range(self):
        """max_tokens above 32768 raises 422."""
        from fastapi import HTTPException

        with pytest.raises(HTTPException) as exc_info:
            validate_chat_request({
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 100000,
            })
        assert exc_info.value.status_code == 422

    def test_valid_model_names(self):
        """Various valid model name formats."""
        valid_names = ["sawyer", "llama-3.1-8b", "mixtral_8x7b", "gpt-4o-mini"]
        for name in valid_names:
            assert VALID_MODEL_PATTERN.match(name), f"Expected {name} to be valid"


# --- CORS middleware tests ---

class TestCORSMiddleware:
    def test_cors_headers_on_preflight(self):
        """OPTIONS request returns CORS headers."""
        app = FastAPI()

        @app.get("/test")
        async def test_route():
            return {"ok": True}

        add_cors_middleware(app)
        client = TestClient(app)

        response = client.options(
            "/test",
            headers={
                "Origin": "https://sawyer.infill.systems",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers

    def test_cors_headers_on_get(self):
        """GET request from allowed origin returns CORS headers."""
        app = FastAPI()

        @app.get("/test")
        async def test_route():
            return {"ok": True}

        add_cors_middleware(app)
        client = TestClient(app)

        response = client.get(
            "/test",
            headers={"Origin": "https://sawyer.infill.systems"},
        )
        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers


# --- API key auth dependency tests ---

class TestVerifyAPIKey:
    def test_missing_api_key_returns_401(self):
        """Request without API key returns 401."""
        app = FastAPI()

        @app.post("/v1/chat/completions")
        async def chat(request, auth=Depends(verify_api_key)):
            return {"status": "ok"}

        add_cors_middleware(app)
        client = TestClient(app)

        response = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
        assert response.status_code == 401

    def test_invalid_api_key_returns_401(self):
        """Request with invalid API key returns 401."""
        app = FastAPI()

        @app.post("/v1/chat/completions")
        async def chat(request, auth=Depends(verify_api_key)):
            return {"status": "ok"}

        add_cors_middleware(app)
        client = TestClient(app)

        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer sak_invalid_key_not_in_db"},
        )
        # Will return 401 because the key doesn't exist in the database
        assert response.status_code == 401

    def test_x_api_key_header_recognized(self):
        """X-API-Key header is recognized as an alternative to Authorization."""
        app = FastAPI()

        @app.post("/v1/chat/completions")
        async def chat(request, auth=Depends(verify_api_key)):
            return {"status": "ok"}

        add_cors_middleware(app)
        client = TestClient(app)

        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}]},
            headers={"X-API-Key": "sak_nonexistent"},
        )
        # Will return 401 because the key doesn't exist in the database,
        # but the point is it's recognized as an auth header, not as "missing"
        assert response.status_code == 401
        detail = response.json()
        assert detail.get("detail", {}).get("error") != "authentication_required"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])