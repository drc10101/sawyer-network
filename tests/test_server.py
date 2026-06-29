"""Tests for SawyerServer — orchestration layer."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sawyer.config import SawyerConfig
from sawyer.node.inference import InferenceResult
from sawyer.server import SawyerServer


class TestSawyerServerInit:
    """Test SawyerServer initialization."""

    def test_default_config(self):
        """Server initializes with default config."""
        server = SawyerServer()
        assert server.config is not None
        assert not server._running
        assert server._backend is None

    def test_custom_config(self):
        """Server accepts custom config."""
        config = SawyerConfig(node_name="test-node")
        server = SawyerServer(config=config)
        assert server.config.node_name == "test-node"


class TestSawyerServerDownload:
    """Test model download orchestration."""

    def setup_method(self):
        self.config = SawyerConfig(
            cache_dir=tempfile.mkdtemp(), expert_cache_dir=tempfile.mkdtemp()
        )
        self.server = SawyerServer(config=self.config)

    def test_download_calls_weight_loader(self):
        """Download delegates to WeightLoader."""
        mock_wf = MagicMock()
        mock_wf.path = Path("/fake/model.gguf")
        with patch.object(self.server._weight_loader, "download_weight") as mock_dl:
            mock_dl.return_value = mock_wf
            with patch.object(self.server._weight_loader, "is_cached", return_value=False):
                result = self.server.download("mixtral-8x7b")
                assert result == Path("/fake/model.gguf")
                mock_dl.assert_called_once()

    def test_download_skips_if_cached(self):
        """Download skips if model already cached."""
        with (
            patch.object(self.server._weight_loader, "is_cached", return_value=True),
            patch.object(self.server._weight_loader, "get_cached_path") as mock_path,
        ):
            mock_path.return_value = Path("/cached/model.gguf")
            result = self.server.download("mixtral-8x7b")
            assert result == Path("/cached/model.gguf")


class TestSawyerServerServe:
    """Test serving lifecycle."""

    def setup_method(self):
        self.config = SawyerConfig(
            cache_dir=tempfile.mkdtemp(), expert_cache_dir=tempfile.mkdtemp()
        )
        self.server = SawyerServer(config=self.config)

    def test_serve_starts_backend_and_registers(self):
        """Serve downloads model, starts backend, registers node."""
        with (
            patch.object(self.server._weight_loader, "is_cached", return_value=True),
            patch.object(
                self.server._weight_loader,
                "get_cached_path",
                return_value=Path("/fake/model.gguf"),
            ),
            patch("sawyer.server.LlamaCppBackend") as MockBackend,
        ):
            mock_backend = MagicMock()
            MockBackend.return_value = mock_backend

            with patch("sawyer.server.SawyerNode") as MockNode:
                mock_node = MagicMock()
                MockNode.return_value = mock_node

                self.server.serve("mixtral-8x7b")

                assert self.server._running
                mock_backend.start_server.assert_called_once()
                mock_node._backend = mock_backend

    def test_serve_raises_if_already_running(self):
        """Serve raises if server already running."""
        self.server._running = True
        with pytest.raises(RuntimeError, match="already running"):
            self.server.serve("mixtral-8x7b")


class TestSawyerServerInfer:
    """Test inference through SawyerServer."""

    def setup_method(self):
        self.config = SawyerConfig()
        self.server = SawyerServer(config=self.config)
        self.server._running = True
        self.server._backend = MagicMock()
        self.server._backend.infer.return_value = InferenceResult(
            text="Hello!",
            input_tokens=5,
            output_tokens=10,
            latency_ms=50.0,
            model_name="mixtral-8x7b",
        )

    def test_infer_returns_dict(self):
        """Infer returns a structured dict."""
        result = self.server.infer("Hello, world!")
        assert result["text"] == "Hello!"
        assert result["input_tokens"] == 5
        assert result["output_tokens"] == 10
        assert result["model"] == "mixtral-8x7b"

    def test_infer_raises_if_not_running(self):
        """Infer raises if server not started."""
        server = SawyerServer()
        with pytest.raises(RuntimeError, match="not running"):
            server.infer("test")


class TestSawyerServerChat:
    """Test chat inference through SawyerServer."""

    def setup_method(self):
        self.config = SawyerConfig()
        self.server = SawyerServer(config=self.config)
        self.server._running = True
        self.server._backend = MagicMock()
        self.server._backend.chat.return_value = InferenceResult(
            text="Hi there!",
            input_tokens=10,
            output_tokens=20,
            latency_ms=75.0,
            model_name="mixtral-8x7b",
        )

    def test_chat_returns_dict(self):
        """Chat returns a structured dict."""
        messages = [{"role": "user", "content": "Hello!"}]
        result = self.server.chat(messages)
        assert result["text"] == "Hi there!"
        assert result["input_tokens"] == 10
        assert result["output_tokens"] == 20


class TestSawyerServerStatus:
    """Test server status reporting."""

    def test_status_when_not_running(self):
        """Status reflects not running."""
        server = SawyerServer()
        status = server.get_status()
        assert not status.running
        assert status.model_loaded is None
        assert status.total_inferences == 0

    def test_status_when_running(self):
        """Status reflects running state."""
        server = SawyerServer()
        server._running = True
        server._start_time = 1000.0
        server._backend = MagicMock()
        server._backend.get_status.return_value = MagicMock(
            model_loaded="mixtral-8x7b",
            vram_used_gb=4.0,
            vram_total_gb=8.0,
            total_inferences=42,
        )

        status = server.get_status()
        assert status.running
        assert status.model_loaded == "mixtral-8x7b"
        assert status.vram_used_gb == 4.0
        assert status.total_inferences == 42


class TestSawyerServerStop:
    """Test server shutdown."""

    def test_stop_cleans_up(self):
        """Stop deregisters and closes backend."""
        server = SawyerServer()
        server._running = True
        mock_backend = MagicMock()
        server._backend = mock_backend
        server._node = MagicMock()
        server._node._router_client = MagicMock()

        server.stop()

        assert not server._running
        mock_backend.close.assert_called_once()

    def test_stop_when_not_running(self):
        """Stop is a no-op when not running."""
        server = SawyerServer()
        server.stop()  # Should not raise


class TestSawyerServerContextManager:
    """Test context manager protocol."""

    def test_context_manager(self):
        """Server works as context manager."""
        with SawyerServer() as server:
            assert server is not None
            assert not server._running
        # Stop called on exit
