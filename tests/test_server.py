import socket
from pathlib import Path
from string import Template
from typing import List
from unittest.mock import Mock, patch, MagicMock
import pytest

from reality_defender_mcp_server.web_server.server import (
    pretty_size,
    UploadMetadata,
    WebServerConfig,
    TemplateEngine,
    find_free_port,
    create_server,
)


def test_pretty_size_bytes() -> None:
    """Test pretty_size with bytes."""
    assert pretty_size(512) == "512B"
    assert pretty_size(1023) == "1023B"


def test_pretty_size_kilobytes() -> None:
    """Test pretty_size with kilobytes."""
    assert pretty_size(1024) == "1.0KB"
    assert pretty_size(2048) == "2.0KB"
    assert pretty_size(5120) == "5.0KB"


def test_upload_metadata_creation() -> None:
    """Test creating UploadMetadata with all fields."""
    metadata = UploadMetadata(
        created_at_timestamp="2023-01-01T00:00:00Z",
        file_extension=".jpg",
        file_id="test-uuid",
        file_path="/path/to/file.jpg",
        file_size=1024,
        mime_type="image/jpeg",
        source_filename="test.jpg",
        source_type="user_upload",
        source_url=None,
    )

    assert metadata.created_at_timestamp == "2023-01-01T00:00:00Z"
    assert metadata.file_extension == ".jpg"
    assert metadata.file_id == "test-uuid"
    assert metadata.file_path == "/path/to/file.jpg"
    assert metadata.file_size == 1024
    assert metadata.mime_type == "image/jpeg"
    assert metadata.source_filename == "test.jpg"
    assert metadata.source_type == "user_upload"
    assert metadata.source_url is None


def test_upload_metadata_with_url() -> None:
    """Test creating UploadMetadata with URL source."""
    metadata = UploadMetadata(
        created_at_timestamp="2023-01-01T00:00:00Z",
        file_extension=".png",
        file_id="test-uuid",
        file_path="/path/to/file.png",
        file_size=2048,
        mime_type="image/png",
        source_filename=None,
        source_type="url_download",
        source_url="https://example.com/image.png",
    )

    assert metadata.source_type == "url_download"
    assert metadata.source_url == "https://example.com/image.png"
    assert metadata.source_filename is None


def test_web_server_config_defaults() -> None:
    """Test WebServerConfig with default values."""
    config = WebServerConfig(
        bind_address=("127.0.0.1", 8080), upload_dir=Path("uploads")
    )

    assert config.bind_address == ("127.0.0.1", 8080)
    assert config.upload_dir == Path("uploads")
    assert config.allowed_extensions == {".png", ".jpg", ".jpeg", ".gif", ".webp"}
    assert config.max_file_size == 1024 * 1024  # 1MB


def test_web_server_config_custom_values() -> None:
    """Test WebServerConfig with custom values."""
    config = WebServerConfig(
        bind_address=("0.0.0.0", 9000),
        upload_dir=Path("/tmp/uploads"),
        allowed_extensions={".jpg", ".png"},
        max_file_size=2048,
    )

    assert config.bind_address == ("0.0.0.0", 9000)
    assert config.upload_dir == Path("/tmp/uploads")
    assert config.allowed_extensions == {".jpg", ".png"}
    assert config.max_file_size == 2048


def test_template_engine_init() -> None:
    """Test TemplateEngine initialization."""
    templates_dir = Path("/templates")
    engine = TemplateEngine(templates_dir)

    assert engine.templates_dir == templates_dir
    assert engine.cache == {}


@pytest.mark.asyncio
async def test_template_engine_render_with_cache() -> None:
    """Test TemplateEngine render method with cached template."""
    templates_dir = Path("/templates")
    engine = TemplateEngine(templates_dir)

    # Pre-populate cache
    engine.cache["test"] = Template("Cached $message")

    with patch("aiofiles.open") as mock_file:
        result = await engine.render("test", {"message": "content"})

        mock_file.assert_not_called()  # Should not read file when cached
        assert result == "Cached content"


def test_find_free_port_returns_integer() -> None:
    """Test that find_free_port returns a valid port number."""
    port = find_free_port()
    assert isinstance(port, int)
    assert 1024 <= port <= 65535


def test_find_free_port_returns_available_port() -> None:
    """Test that find_free_port returns an available port."""
    port = find_free_port()

    # Try to bind to the returned port to verify it's available
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # This should not raise an exception if port is available
        s.bind(("", port))
        s.listen(1)


def test_find_free_port_multiple_calls() -> None:
    """Test that multiple calls to find_free_port return different ports."""
    ports: List[int] = [find_free_port() for _ in range(5)]

    # Not strictly guaranteed, but very likely to be different
    assert len(set(ports)) >= 3  # At least 3 unique ports out of 5


@patch("reality_defender_mcp_server.web_server.server.uvicorn.Server")
@patch("reality_defender_mcp_server.web_server.server.uvicorn.Config")
@patch("reality_defender_mcp_server.web_server.server.TemplateEngine")
def test_create_server_basic(
    mock_template_engine: MagicMock, mock_config: MagicMock, mock_server: MagicMock
) -> None:
    """Test create_server with basic configuration."""
    web_config = WebServerConfig(
        bind_address=("127.0.0.1", 8080), upload_dir=Path("uploads")
    )

    mock_template_instance = Mock()
    mock_template_engine.return_value = mock_template_instance

    mock_config_instance = Mock()
    mock_config.return_value = mock_config_instance

    mock_server_instance = Mock()
    mock_server.return_value = mock_server_instance

    with patch("reality_defender_mcp_server.web_server.server.app"):
        result = create_server(web_config)

        mock_template_engine.assert_called_once()
        mock_config.assert_called_once()
        mock_server.assert_called_once_with(mock_config_instance)

        assert result == mock_server_instance


@patch("reality_defender_mcp_server.web_server.server.uvicorn.Config")
def test_create_server_uvicorn_config(
    mock_config: MagicMock
) -> None:
    """Test create_server passes correct configuration to uvicorn."""
    web_config = WebServerConfig(
        bind_address=("localhost", 3000), upload_dir=Path("uploads")
    )

    with patch("reality_defender_mcp_server.web_server.server.app") as mock_app:
        create_server(web_config)

        mock_config.assert_called_once_with(
            mock_app, host="localhost", port=3000, log_config=None, access_log=False
        )
