import asyncio
from typing import Any, Optional
from unittest.mock import AsyncMock, Mock, patch

import pytest

from reality_defender_mcp_server.mcp_server import (
    AppContext,
    Error,
    GenerateUploadUrlOutput,
    GetFileInfoOutput,
    RealityDefenderAnalysisRequest,
    RealityDefenderAnalysisResponse,
    RealityDefenderClientHarness,
    RealityDefenderDetectFileResponse,
    RealityDefenderGetResultResponse,
    RealityDefenderModel,
    RealityDefenderUploadResponse,
    reality_defender_generate_upload_url,
    reality_defender_get_file_info,
    reality_defender_request_file_analysis,
    reality_defender_request_to_check_authenticity_of_hosted_file,
)


# Helper function to create async file mock
def create_async_file_mock(content: str) -> AsyncMock:
    """Create a properly configured async file mock."""
    mock_file_handle = AsyncMock()
    mock_file_handle.read = AsyncMock(return_value=content)
    mock_file_handle.write = AsyncMock()

    mock_aiofiles_open = AsyncMock()
    mock_aiofiles_open.return_value.__aenter__.return_value = mock_file_handle
    mock_aiofiles_open.return_value.__aexit__ = AsyncMock(return_value=None)

    return mock_aiofiles_open


def create_async_context_manager_mock(return_value: Any) -> AsyncMock:
    """Create a generic async context manager mock."""
    mock_cm = AsyncMock()
    mock_cm.__aenter__ = AsyncMock(return_value=return_value)
    mock_cm.__aexit__ = AsyncMock(return_value=None)
    return mock_cm


def test_reality_defender_analysis_request_valid_file_path() -> None:
    """Test RealityDefenderAnalysisRequest with valid file_path."""
    request = RealityDefenderAnalysisRequest(
        file_path="/path/to/file.jpg", file_url=None, expected_file_type="image"
    )
    assert request.file_path == "/path/to/file.jpg"
    assert request.file_url is None
    assert request.expected_file_type == "image"


def test_reality_defender_analysis_request_valid_file_url() -> None:
    """Test RealityDefenderAnalysisRequest with valid file_url."""
    request = RealityDefenderAnalysisRequest(
        file_url="https://example.com/file.jpg", file_path=None, expected_file_type="image"
    )
    assert request.file_url == "https://example.com/file.jpg"
    assert request.file_path is None
    assert request.expected_file_type == "image"


def test_reality_defender_analysis_request_validation_error() -> None:
    """Test RealityDefenderAnalysisRequest validation fails with both inputs."""
    with pytest.raises(
        ValueError, match="Exactly one of file_path or file_url must be provided"
    ):
        RealityDefenderAnalysisRequest(
            file_path="/path/to/file.jpg",
            file_url="https://example.com/file.jpg",
            expected_file_type="image",
        )


def test_reality_defender_analysis_request_validation_error_no_input() -> None:
    """Test RealityDefenderAnalysisRequest validation fails with no inputs."""
    with pytest.raises(
        ValueError, match="Exactly one of file_path or file_url must be provided"
    ):
        RealityDefenderAnalysisRequest(file_path=None, file_url=None, expected_file_type="image")


def test_error_model() -> None:
    """Test Error model creation."""
    error = Error(error="Test error message")
    assert error.error == "Test error message"


def test_reality_defender_model() -> None:
    """Test RealityDefenderModel creation."""
    model = RealityDefenderModel(name="test_model", status="AUTHENTIC", score=85.5)
    assert model.name == "test_model"
    assert model.status == "AUTHENTIC"
    assert model.score == 85.5


def test_reality_defender_detect_file_response() -> None:
    """Test RealityDefenderDetectFileResponse creation."""
    response = RealityDefenderDetectFileResponse(
        status="AUTHENTIC", score=90.0, models=[], request_id="test-request-id"
    )
    assert response.status == "AUTHENTIC"
    assert response.score == 90.0
    assert response.models == []
    assert response.request_id == "test-request-id"


def test_reality_defender_get_result_response() -> None:
    """Test RealityDefenderGetResultResponse creation."""
    response = RealityDefenderGetResultResponse(
        status="MANIPULATED", score=15.0, models=[], request_id="test-request-id"
    )
    assert response.status == "MANIPULATED"
    assert response.score == 15.0
    assert response.models == []
    assert response.request_id == "test-request-id"


def test_reality_defender_upload_response() -> None:
    """Test RealityDefenderUploadResponse creation."""
    response = RealityDefenderUploadResponse(
        media_id="test-media-id", request_id="test-request-id"
    )
    assert response.media_id == "test-media-id"
    assert response.request_id == "test-request-id"


def test_reality_defender_analysis_response() -> None:
    """Test RealityDefenderAnalysisResponse creation."""
    response = RealityDefenderAnalysisResponse(
        status="AUTHENTIC", score=95.0, models=[], file_id="test-file-id"
    )
    assert response.status == "AUTHENTIC"
    assert response.score == 95.0
    assert response.models == []
    assert response.file_id == "test-file-id"


def test_generate_upload_url_output() -> None:
    """Test GenerateUploadUrlOutput model creation."""
    output = GenerateUploadUrlOutput(
        upload_url="http://localhost:8080/upload/123", request_id="123"
    )
    assert output.upload_url == "http://localhost:8080/upload/123"
    assert output.request_id == "123"


def test_get_file_info_output() -> None:
    """Test GetFileInfoOutput model creation."""
    output = GetFileInfoOutput(
        created_at_timestamp="2023-01-01T00:00:00Z",
        file_extension=".jpg",
        file_path="/path/to/file.jpg",
        file_size=1024,
        mime_type="image/jpeg",
        request_id="test-request-id",
        source_filename="original.jpg",
        source_type="upload",
        source_url="https://example.com/original.jpg",
    )
    assert output.created_at_timestamp == "2023-01-01T00:00:00Z"
    assert output.file_extension == ".jpg"
    assert output.file_path == "/path/to/file.jpg"
    assert output.file_size == 1024
    assert output.mime_type == "image/jpeg"
    assert output.request_id == "test-request-id"
    assert output.source_filename == "original.jpg"
    assert output.source_type == "upload"
    assert output.source_url == "https://example.com/original.jpg"


# RealityDefenderClientHarness tests
def test_reality_defender_client_harness_init() -> None:
    """Test RealityDefenderClientHarness initialization."""
    harness = RealityDefenderClientHarness("test-api-key")
    assert harness.api_key == "test-api-key"
    assert harness.client is None


@patch("reality_defender_mcp_server.mcp_server.RealityDefender")
def test_reality_defender_client_harness_get_client_success(
    mock_reality_defender: Mock,
) -> None:
    """Test RealityDefenderClientHarness get_client returns client successfully."""
    mock_client = Mock()
    mock_reality_defender.return_value = mock_client

    harness = RealityDefenderClientHarness("test-api-key")
    result = harness.get_client()

    assert result == mock_client
    assert harness.client == mock_client
    mock_reality_defender.assert_called_once_with(api_key="test-api-key")


def test_reality_defender_client_harness_get_client_no_api_key() -> None:
    """Test RealityDefenderClientHarness get_client returns error when no API key."""
    harness = RealityDefenderClientHarness("")
    result = harness.get_client()

    assert isinstance(result, Error)
    assert "API key not provided" in result.error


@patch("reality_defender_mcp_server.mcp_server.RealityDefender")
def test_reality_defender_client_harness_get_client_cached(
    mock_reality_defender: Mock,
) -> None:
    """Test RealityDefenderClientHarness get_client returns cached client."""
    mock_client = Mock()
    mock_reality_defender.return_value = mock_client

    harness = RealityDefenderClientHarness("test-api-key")
    result1 = harness.get_client()
    result2 = harness.get_client()

    assert result1 == mock_client
    assert result2 == mock_client
    mock_reality_defender.assert_called_once()


def test_app_context_creation() -> None:
    """Test AppContext dataclass creation."""
    config = Mock()
    reality_defender = Mock()
    web_server_url = "http://localhost:8080"

    context = AppContext(
        config=config, reality_defender=reality_defender, web_server_url=web_server_url
    )

    assert context.config == config
    assert context.reality_defender == reality_defender
    assert context.web_server_url == web_server_url


# Function tests
def test_reality_defender_request_to_check_authenticity_of_hosted_file() -> None:
    """Test reality_defender_request_to_check_authenticity_of_hosted_file function."""
    url = "https://example.com/test.jpg"

    with patch("reality_defender_mcp_server.mcp_server.logger") as mock_logger:
        messages = reality_defender_request_to_check_authenticity_of_hosted_file(url)

        mock_logger.info.assert_called_once_with(
            f"Creating prompt to check authenticity of a file: {url}"
        )
        assert len(messages) == 1
        assert (
            "Download this file and use the Reality Defender API"
            in messages[0].content.text
        )


@pytest.mark.asyncio
@patch("reality_defender_mcp_server.mcp_server.uuid.uuid4")
async def test_reality_defender_generate_upload_url_success(mock_uuid: Mock) -> None:
    """Test reality_defender_generate_upload_url success case."""
    mock_uuid.return_value = Mock()
    mock_uuid.return_value.__str__ = Mock(return_value="test-uuid")

    mock_context = Mock()
    mock_context.request_context.lifespan_context.web_server_url = (
        "http://localhost:8080"
    )

    result = await reality_defender_generate_upload_url(mock_context)

    assert isinstance(result, GenerateUploadUrlOutput)
    assert result.upload_url == "http://localhost:8080/upload/test-uuid"
    assert result.request_id == "test-uuid"


@pytest.mark.asyncio
async def test_reality_defender_generate_upload_url_error() -> None:
    """Test reality_defender_generate_upload_url error case."""
    mock_context = Mock()
    mock_context.request_context.lifespan_context.web_server_url = Error(
        error="Web server error"
    )

    result = await reality_defender_generate_upload_url(mock_context)

    assert isinstance(result, Error)
    assert result.error == "Web server error"


@pytest.mark.asyncio
@patch("reality_defender_mcp_server.mcp_server.uuid.uuid4")
async def test_reality_defender_generate_upload_url_exception(mock_uuid: Mock) -> None:
    """Test reality_defender_generate_upload_url when exception occurs."""
    mock_uuid.side_effect = Exception("UUID generation failed")

    mock_context = Mock()
    mock_context.request_context.lifespan_context.web_server_url = (
        "http://localhost:8080"
    )

    result = await reality_defender_generate_upload_url(mock_context)

    assert isinstance(result, Error)
    assert "Failed to generate upload URL" in result.error


@pytest.mark.asyncio
async def test_reality_defender_get_file_info_file_not_found() -> None:
    """Test reality_defender_get_file_info when file directory doesn't exist."""
    mock_context = Mock()
    mock_uploads_dir = Mock()
    mock_file_dir = Mock()
    mock_file_dir.exists.return_value = False
    mock_uploads_dir.__truediv__ = Mock(return_value=mock_file_dir)
    mock_context.request_context.lifespan_context.config.web_server_uploads_dir = (
        mock_uploads_dir
    )

    result = await reality_defender_get_file_info("non-existent-id", mock_context)

    assert isinstance(result, Error)
    assert "No file directory found" in result.error


@pytest.mark.asyncio
async def test_reality_defender_get_file_info_metadata_not_found() -> None:
    """Test reality_defender_get_file_info when metadata file doesn't exist."""
    mock_context = Mock()
    mock_uploads_dir = Mock()
    mock_file_dir = Mock()
    mock_file_dir.exists.return_value = True
    mock_metadata_path = Mock()
    mock_metadata_path.exists.return_value = False
    mock_file_dir.__truediv__ = Mock(return_value=mock_metadata_path)
    mock_uploads_dir.__truediv__ = Mock(return_value=mock_file_dir)
    mock_context.request_context.lifespan_context.config.web_server_uploads_dir = (
        mock_uploads_dir
    )

    result = await reality_defender_get_file_info("test-id", mock_context)

    assert isinstance(result, Error)
    assert "No metadata file found" in result.error


@pytest.mark.asyncio
@patch("builtins.open")
@patch("reality_defender_mcp_server.mcp_server.UploadMetadata")
async def test_reality_defender_get_file_info_success(
    mock_upload_metadata: Mock, mock_open: Mock
) -> None:
    """Test reality_defender_get_file_info success case."""
    # Mock file content
    mock_file_content = '{"test": "data"}'
    mock_file_handle = Mock()
    mock_file_handle.read.return_value = mock_file_content
    mock_open.return_value.__enter__.return_value = mock_file_handle

    # Mock metadata
    mock_metadata = Mock()
    mock_metadata.created_at_timestamp = "2023-01-01T00:00:00Z"
    mock_metadata.file_extension = ".jpg"
    mock_metadata.file_size = 1024
    mock_metadata.mime_type = "image/jpeg"
    mock_metadata.source_filename = "original.jpg"
    mock_metadata.source_type = "upload"
    mock_metadata.source_url = "https://example.com/original.jpg"
    mock_upload_metadata.model_validate_json.return_value = mock_metadata

    # Mock context
    mock_context = Mock()
    mock_uploads_dir = Mock()
    mock_file_dir = Mock()
    mock_file_dir.exists.return_value = True
    mock_file_dir.__truediv__ = Mock(return_value=Mock(exists=Mock(return_value=True)))
    mock_uploads_dir.__truediv__ = Mock(return_value=mock_file_dir)
    mock_context.request_context.lifespan_context.config.web_server_uploads_dir = (
        mock_uploads_dir
    )

    result = await reality_defender_get_file_info("test-id", mock_context)

    assert isinstance(result, GetFileInfoOutput)
    assert result.created_at_timestamp == "2023-01-01T00:00:00Z"
    assert result.file_extension == ".jpg"
    assert result.file_size == 1024
    assert result.mime_type == "image/jpeg"
    assert result.source_filename == "original.jpg"
    assert result.source_type == "upload"
    assert result.source_url == "https://example.com/original.jpg"


@pytest.mark.asyncio
@patch("builtins.open")
async def test_reality_defender_get_file_info_read_error(mock_open: Mock) -> None:
    """Test reality_defender_get_file_info when file read fails."""
    mock_open.side_effect = IOError("File read error")

    mock_context = Mock()
    mock_uploads_dir = Mock()
    mock_file_dir = Mock()
    mock_file_dir.exists.return_value = True
    mock_file_dir.__truediv__ = Mock(return_value=Mock(exists=Mock(return_value=True)))
    mock_uploads_dir.__truediv__ = Mock(return_value=mock_file_dir)
    mock_context.request_context.lifespan_context.config.web_server_uploads_dir = (
        mock_uploads_dir
    )

    result = await reality_defender_get_file_info("test-id", mock_context)

    assert isinstance(result, Error)
    assert "File metadata read failed" in result.error


@pytest.mark.asyncio
async def test_reality_defender_request_file_analysis_no_inputs() -> None:
    """Test reality_defender_request_file_analysis with no file_path or file_url."""
    # Create a request that bypasses validation
    request = Mock()
    request.file_path = None
    request.file_url = None
    request.model_dump_json.return_value = '{"file_path": null, "file_url": null}'

    mock_context = Mock()

    result = await reality_defender_request_file_analysis(request, mock_context)

    assert isinstance(result, Error)
    assert "file_path and file_url are both absent" in result.error


@pytest.mark.asyncio
async def test_reality_defender_request_file_analysis_client_error() -> None:
    """Test reality_defender_request_file_analysis when client creation fails."""
    request = Mock()
    request.file_path = "/valid/path/file.jpg"
    request.file_url = None
    request.model_dump_json.return_value = '{"file_path": "/valid/path/file.jpg"}'

    mock_context = Mock()
    mock_reality_defender = Mock()
    mock_reality_defender.get_client.return_value = Error(error="Client error")
    mock_context.request_context.lifespan_context.reality_defender = (
        mock_reality_defender
    )

    result = await reality_defender_request_file_analysis(request, mock_context)

    assert isinstance(result, Error)
    assert result.error == "Client error"


@pytest.mark.asyncio
@patch("reality_defender_mcp_server.mcp_server.os.path.exists")
@patch("reality_defender_mcp_server.mcp_server.Path")
async def test_reality_defender_request_file_analysis_file_not_found(
    mock_path: Mock, mock_exists: Mock
) -> None:
    """Test reality_defender_request_file_analysis when file doesn't exist."""
    request = Mock()
    request.file_path = "/valid/path/file.jpg"
    request.file_url = None
    request.model_dump_json.return_value = '{"file_path": "/valid/path/file.jpg"}'

    mock_exists.return_value = False
    mock_path_instance = Mock()
    mock_path_instance.is_relative_to.return_value = True
    mock_path.return_value = mock_path_instance

    mock_context = Mock()
    mock_reality_defender = Mock()
    mock_reality_defender.get_client.return_value = Mock()
    mock_context.request_context.lifespan_context.reality_defender = (
        mock_reality_defender
    )
    mock_context.request_context.lifespan_context.config.web_server_uploads_dir = (
        "/uploads"
    )

    result = await reality_defender_request_file_analysis(request, mock_context)

    assert isinstance(result, Error)
    assert "Image file not found" in result.error


@pytest.mark.asyncio
@patch("reality_defender_mcp_server.mcp_server.urlparse")
async def test_reality_defender_request_file_analysis_invalid_url(
    mock_urlparse: Mock,
) -> None:
    """Test reality_defender_request_file_analysis with invalid URL."""
    request = Mock()
    request.file_path = None
    request.file_url = "invalid-url"
    request.model_dump_json.return_value = '{"file_url": "invalid-url"}'

    mock_parsed = Mock()
    mock_parsed.scheme = ""
    mock_parsed.netloc = ""
    mock_urlparse.return_value = mock_parsed

    mock_context = Mock()
    mock_reality_defender = Mock()
    mock_reality_defender.get_client.return_value = Mock()
    mock_context.request_context.lifespan_context.reality_defender = (
        mock_reality_defender
    )

    result = await reality_defender_request_file_analysis(request, mock_context)

    assert isinstance(result, Error)
    assert "Invalid image URL format" in result.error


@pytest.mark.parametrize(
    "file_type,expected_type",
    [
        ("image", "image"),
        ("video", "video"),
        ("audio", "audio"),
        ("text", "text"),
    ],
)
def test_reality_defender_analysis_request_file_types(
    file_type: str, expected_type: str
) -> None:
    """Test RealityDefenderAnalysisRequest with different file types."""
    request = RealityDefenderAnalysisRequest(  # type: ignore
        file_path="/path/to/file", expected_file_type=file_type
    )
    assert request.expected_file_type == expected_type


@pytest.mark.parametrize(
    "status,score",
    [
        ("AUTHENTIC", 95.0),
        ("MANIPULATED", 15.0),
        ("ANALYZING", None),
    ],
)
def test_reality_defender_analysis_response_statuses(
    status: str, score: Optional[float]
) -> None:
    """Test RealityDefenderAnalysisResponse with different statuses."""
    response = RealityDefenderAnalysisResponse(status=status, score=score, models=[])  # type: ignore
    assert response.status == status
    assert response.score == score


@pytest.mark.asyncio
@patch("reality_defender_mcp_server.mcp_server.tempfile.NamedTemporaryFile")
@patch("reality_defender_mcp_server.mcp_server.aiohttp.ClientSession")
@patch("reality_defender_mcp_server.mcp_server.urlparse")
async def test_reality_defender_request_file_analysis_download_timeout(
    mock_urlparse: Mock, mock_client_session: Mock, mock_temp_file: Mock
) -> None:
    """Test reality_defender_request_file_analysis with download timeout."""
    request = Mock()
    request.file_path = None
    request.file_url = "https://example.com/test.jpg"
    request.expected_file_type = "image"
    request.model_dump_json.return_value = (
        '{"file_url": "https://example.com/test.jpg"}'
    )

    # Mock URL parsing
    mock_parsed = Mock()
    mock_parsed.scheme = "https"
    mock_parsed.netloc = "example.com"
    mock_urlparse.return_value = mock_parsed

    # Mock temp file
    mock_temp_file_instance = Mock()
    mock_temp_file_instance.name = "/tmp/test_file"
    mock_temp_file.return_value.__enter__.return_value = mock_temp_file_instance

    # Mock session to raise timeout during session creation
    mock_client_session.side_effect = asyncio.TimeoutError()

    mock_context = Mock()
    mock_reality_defender = Mock()
    mock_reality_defender.get_client.return_value = Mock()
    mock_context.request_context.lifespan_context.reality_defender = (
        mock_reality_defender
    )

    result = await reality_defender_request_file_analysis(request, mock_context)

    assert isinstance(result, Error)
    assert "download timed out" in result.error


@pytest.mark.asyncio
@patch("reality_defender_mcp_server.mcp_server.Path")
async def test_reality_defender_request_file_analysis_invalid_file_path(
    mock_path: Mock
) -> None:
    """Test reality_defender_request_file_analysis with invalid file path."""
    request = Mock()
    request.file_path = "/invalid/path/file.jpg"
    request.file_url = None
    request.model_dump_json.return_value = '{"file_path": "/invalid/path/file.jpg"}'

    mock_path_instance = Mock()
    mock_path_instance.is_relative_to.return_value = False
    mock_path.return_value = mock_path_instance

    mock_context = Mock()
    mock_reality_defender = Mock()
    mock_reality_defender.get_client.return_value = Mock()
    mock_context.request_context.lifespan_context.reality_defender = (
        mock_reality_defender
    )
    mock_context.request_context.lifespan_context.config.web_server_uploads_dir = (
        "/uploads"
    )

    result = await reality_defender_request_file_analysis(request, mock_context)

    assert isinstance(result, Error)
    assert "Invalid file path provided" in result.error


def test_reality_defender_client_harness_none_api_key() -> None:
    """Test RealityDefenderClientHarness with None API key."""
    harness = RealityDefenderClientHarness(None)  # type: ignore
    result = harness.get_client()

    assert isinstance(result, Error)
    assert "API key not provided" in result.error
