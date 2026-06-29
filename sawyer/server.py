"""Sawyer Server — orchestration layer tying model download, serving, and inference.

SawyerServer is the main entry point for running a Sawyer node:
1. Downloads GGUF weights via WeightLoader
2. Starts llama.cpp server via LlamaCppBackend
3. Registers with the router via SawyerNode
4. Provides a unified start/stop lifecycle
5. Reports health and metrics
"""

import logging
import signal
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sawyer.config import SawyerConfig
from sawyer.node.agent import SawyerNode
from sawyer.node.inference import BackendMode, LlamaCppBackend
from sawyer.node.weights import WeightLoader

logger = logging.getLogger(__name__)


@dataclass
class ServerStatus:
    """Current server status."""

    running: bool = False
    model_loaded: str | None = None
    node_id: str | None = None
    vram_used_gb: float = 0.0
    vram_total_gb: float = 0.0
    total_inferences: int = 0
    uptime_seconds: float = 0.0


class SawyerServer:
    """Orchestrates model download, serving, and inference.

    Usage:
        server = SawyerServer(config)
        server.download("mixtral-8x7b")  # Download GGUF weights
        server.serve("mixtral-8x7b")       # Start llama.cpp + register
        ...
        result = server.infer("Hello, world!")
        ...
        server.stop()                       # Clean shutdown
    """

    def __init__(
        self,
        config: SawyerConfig | None = None,
        backend_mode: BackendMode = BackendMode.SUBPROCESS,
        server_url: str | None = None,
        llama_cpp_path: str | None = None,
    ) -> None:
        self.config = config or SawyerConfig()
        self.backend_mode = backend_mode
        self.server_url = server_url
        self.llama_cpp_path = llama_cpp_path

        self._backend: LlamaCppBackend | None = None
        self._node: SawyerNode | None = None
        self._weight_loader = WeightLoader(self.config)
        self._start_time: float = 0.0
        self._running = False

    # ── Model Management ──────────────────────────────────────────

    def download(self, model_name: str, force: bool = False) -> Path:
        """Download a model's GGUF weights.

        Args:
            model_name: Model identifier (e.g., "mixtral-8x7b")
            force: Re-download even if already cached

        Returns:
            Path to the downloaded GGUF file
        """
        logger.info("Downloading model weights for %s", model_name)

        if not force and self._weight_loader.is_cached(model_name):
            logger.info("Model %s already cached", model_name)
            return self._weight_loader.get_cached_path(model_name)

        path = self._weight_loader.download_weight(model_name=model_name, force=force)
        logger.info("Downloaded %s to %s", model_name, path.path)
        return path.path

    def list_available_models(self) -> list[dict[str, Any]]:
        """List models available in cache and their status."""
        from sawyer.model.registry import list_models

        models = list_models()
        result = []
        for m in models:
            cached = self._weight_loader.is_cached(m.name)
            result.append(
                {
                    "name": m.name,
                    "display_name": m.display_name,
                    "expert_count": m.expert_count,
                    "parameter_count": m.parameter_count,
                    "quantization": m.quantization,
                    "size_gb": m.size_gb_q4,
                    "cached": cached,
                    "path": str(self._weight_loader.get_cached_path(m.name)) if cached else None,
                }
            )
        return result

    # ── Serving Lifecycle ──────────────────────────────────────────

    def serve(
        self,
        model_name: str,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        n_threads: int | None = None,
        extra_args: list[str] | None = None,
    ) -> None:
        """Start serving a model: load weights, start backend, register.

        Args:
            model_name: Model identifier to serve
            n_ctx: Context window size
            n_gpu_layers: GPU layers (-1 for all)
            n_threads: Thread count (None for auto)
            extra_args: Additional llama-server arguments
        """
        if self._running:
            raise RuntimeError("Server is already running. Stop it first.")

        # Ensure model is downloaded
        if not self._weight_loader.is_cached(model_name):
            logger.info("Model %s not cached, downloading...", model_name)
            self.download(model_name)

        model_path = self._weight_loader.get_cached_path(model_name)
        logger.info("Serving model %s from %s", model_name, model_path)

        # Start inference backend
        self._backend = LlamaCppBackend(
            config=self.config,
            mode=self.backend_mode,
            server_url=self.server_url,
            llama_cpp_path=self.llama_cpp_path,
        )
        self._backend.start_server(
            model_path=model_path,
            model_name=model_name,
            n_ctx=n_ctx,
            n_gpu_layers=n_gpu_layers,
            n_threads=n_threads,
            extra_args=extra_args,
        )

        # Register with network
        self._node = SawyerNode(self.config)
        self._node._backend = self._backend

        self._start_time = time.time()
        self._running = True
        logger.info("Sawyer server serving %s", model_name)

    def stop(self) -> None:
        """Stop serving: deregister, stop backend."""
        if not self._running:
            return

        logger.info("Stopping Sawyer server")

        if self._node and self._node._router_client:
            try:
                self._node._router_client.deregister()
                self._node._router_client.close()
            except Exception as e:
                logger.warning("Error deregistering: %s", e)

        if self._backend:
            self._backend.close()

        self._running = False
        self._node = None
        self._backend = None
        logger.info("Sawyer server stopped")

    # ── Inference ──────────────────────────────────────────────────

    def infer(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run inference on the served model.

        Returns:
            Dict with text, token counts, latency
        """
        if not self._running or not self._backend:
            raise RuntimeError("Server is not running. Call serve() first.")

        result = self._backend.infer(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs,
        )

        return {
            "text": result.text,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "latency_ms": result.latency_ms,
            "model": result.model_name,
            "finish_reason": result.finish_reason,
        }

    def chat(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
        temperature: float = 0.7,
    ) -> dict[str, Any]:
        """Run chat-completion inference.

        Args:
            messages: List of {"role": "...", "content": "..."} dicts
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Dict with text, token counts, latency
        """
        if not self._running or not self._backend:
            raise RuntimeError("Server is not running. Call serve() first.")

        result = self._backend.chat(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        return {
            "text": result.text,
            "input_tokens": result.input_tokens,
            "output_tokens": result.output_tokens,
            "latency_ms": result.latency_ms,
            "model": result.model_name,
            "finish_reason": result.finish_reason,
        }

    # ── Status ─────────────────────────────────────────────────────

    def get_status(self) -> ServerStatus:
        """Get current server status."""
        backend_status = self._backend.get_status() if self._backend else None

        return ServerStatus(
            running=self._running,
            model_loaded=backend_status.model_loaded if backend_status else None,
            node_id=self._node.node_id if self._node else None,
            vram_used_gb=backend_status.vram_used_gb if backend_status else 0.0,
            vram_total_gb=backend_status.vram_total_gb if backend_status else 0.0,
            total_inferences=(backend_status.total_inferences if backend_status else 0),
            uptime_seconds=time.time() - self._start_time if self._start_time else 0.0,
        )

    # ── Context Manager ────────────────────────────────────────────

    def __enter__(self) -> "SawyerServer":
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop()

    # ── Signal Handling ─────────────────────────────────────────────

    def install_signal_handlers(self) -> None:
        """Install SIGINT/SIGTERM handlers for graceful shutdown."""

        def _shutdown(signum: int, frame: Any) -> None:
            logger.info("Received signal %s, shutting down...", signal.Signals(signum).name)
            self.stop()

        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)
