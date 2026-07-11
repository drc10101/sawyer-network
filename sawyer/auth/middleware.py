"""Sawyer FastAPI security middleware — API key auth, CORS, input validation.

Wires SawyerAuth into the FastAPI app as:
- CORS middleware for the chat UI
- API key dependency on /v1/chat/completions
- Input validation for request bodies
"""

import logging
import re

from fastapi import Depends, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from sawyer.auth.api import InvalidAPIKey, RateLimitExceeded, SawyerAuth
from sawyer.storage.database import SawyerStorage

logger = logging.getLogger(__name__)

# Allowed origins for CORS (add production domains here)
ALLOWED_ORIGINS = [
    "https://sawyer.infill.systems",
    "https://api.sawyer.infill.systems",
    "http://localhost:3000",  # Local dev
    "http://localhost:8000",  # Local dev
]

# Input validation constraints
MAX_MESSAGES = 128
MAX_MESSAGE_LENGTH = 65536
MAX_MODEL_NAME_LENGTH = 128
VALID_MODEL_PATTERN = re.compile(r"^[a-zA-Z0-9._-]+$")
MAX_TEMPERATURE = 2.0
MAX_TOP_P = 1.0
MAX_MAX_TOKENS = 32768


def add_cors_middleware(app) -> None:
    """Add CORS middleware to the FastAPI app."""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=ALLOWED_ORIGINS,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-API-Key"],
    )


def get_auth(storage: SawyerStorage | None = None) -> SawyerAuth:
    """Get or create a SawyerAuth instance.

    Args:
        storage: Optional SawyerStorage instance. If None, creates a default one.

    Returns:
        SawyerAuth instance ready for key validation.
    """
    if storage is None:
        storage = SawyerStorage()
    return SawyerAuth(storage)


async def verify_api_key(
    request: Request,
) -> SawyerAuth:
    """FastAPI dependency that validates the API key from the request.

    Checks Authorization: Bearer <key> or X-API-Key header.
    Raises 401 for invalid/missing keys, 429 for rate limit exceeded.
    """
    from sawyer.storage.database import SawyerStorage

    # Extract API key from Authorization header or X-API-Key
    api_key = None

    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        api_key = auth_header[7:].strip()
    elif auth_header.startswith("sak_"):
        api_key = auth_header.strip()

    if not api_key:
        api_key = request.headers.get("X-API-Key", "").strip()

    if not api_key:
        raise HTTPException(
            status_code=401,
            detail={
                "error": "authentication_required",
                "message": "API key required. Use Authorization: Bearer <key> or X-API-Key header.",
            },
        )

    # Create auth instance — uses app state storage if available
    storage = getattr(request.app.state, "sawyer_storage", None) or SawyerStorage()
    auth = SawyerAuth(storage)

    try:
        auth.validate_key(api_key)
    except InvalidAPIKey as e:
        logger.warning("Invalid API key attempt from %s", request.client.host if request.client else "unknown")
        raise HTTPException(
            status_code=401,
            detail={
                "error": "invalid_api_key",
                "message": str(e),
            },
        ) from e
    except RateLimitExceeded as e:
        logger.warning("Rate limit exceeded for key from %s", request.client.host if request.client else "unknown")
        raise HTTPException(
            status_code=429,
            detail={
                "error": "rate_limit_exceeded",
                "message": str(e),
            },
        ) from e

    return auth


def validate_chat_request(body: dict) -> None:
    """Validate a chat completions request body.

    Raises HTTPException with 400/422 for invalid input.
    """
    errors = []

    # Messages validation
    messages = body.get("messages")
    if not messages or not isinstance(messages, list):
        errors.append("messages is required and must be an array")
    elif len(messages) > MAX_MESSAGES:
        errors.append(f"messages must have at most {MAX_MESSAGES} entries, got {len(messages)}")
    elif not any(m.get("role") == "user" for m in messages):
        errors.append("messages must contain at least one user message")
    else:
        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                errors.append(f"messages[{i}] must be an object")
                continue
            role = msg.get("role")
            if role not in ("system", "user", "assistant", "tool"):
                errors.append(f"messages[{i}].role must be system/user/assistant/tool, got '{role}'")
            content = msg.get("content", "")
            if isinstance(content, str) and len(content) > MAX_MESSAGE_LENGTH:
                errors.append(f"messages[{i}].content exceeds {MAX_MESSAGE_LENGTH} characters")

    # Model validation
    model = body.get("model", "sawyer")
    if not isinstance(model, str):
        errors.append("model must be a string")
    elif len(model) > MAX_MODEL_NAME_LENGTH:
        errors.append(f"model name exceeds {MAX_MODEL_NAME_LENGTH} characters")
    elif not VALID_MODEL_PATTERN.match(model):
        errors.append("model name contains invalid characters (allowed: a-z, A-Z, 0-9, ., _, -)")

    # Numeric parameter validation
    temperature = body.get("temperature", 0.7)
    if not isinstance(temperature, (int, float)) or temperature < 0 or temperature > MAX_TEMPERATURE:
        errors.append(f"temperature must be between 0 and {MAX_TEMPERATURE}")

    top_p = body.get("top_p", 0.9)
    if not isinstance(top_p, (int, float)) or top_p < 0 or top_p > MAX_TOP_P:
        errors.append(f"top_p must be between 0 and {MAX_TOP_P}")

    max_tokens = body.get("max_tokens", 512)
    if not isinstance(max_tokens, int) or max_tokens < 1 or max_tokens > MAX_MAX_TOKENS:
        errors.append(f"max_tokens must be between 1 and {MAX_MAX_TOKENS}")

    if errors:
        raise HTTPException(
            status_code=422,
            detail={
                "error": "invalid_request",
                "message": "Request validation failed",
                "errors": errors,
            },
        )