import os
import logging
from pathlib import Path
from unittest.mock import patch, MagicMock
from reality_defender_mcp_server.config import Config, load_config, setup_logging


def test_config_with_all_fields() -> None:
    config_data = {
        "REALITY_DEFENDER_API_KEY": "test-api-key",
        "WEB_SERVER_HOST": "test-host",
        "WEB_SERVER_PORT": 1234,
        "WEB_SERVER_UPLOADS_DIR": "test-uploads-dir",
        "LOG_LEVEL": "INFO",
    }
    config = Config.model_validate(config_data)

    assert config.reality_defender_api_key == "test-api-key"
    assert config.web_server_host == "test-host"
    assert config.web_server_port == 1234
    assert config.web_server_uploads_dir == Path("test-uploads-dir")
    assert config.log_level == "INFO"


def test_config_with_custom_log_level() -> None:
    config_data = {
        "REALITY_DEFENDER_API_KEY": "test-api-key",
        "WEB_SERVER_HOST": "test-host",
        "WEB_SERVER_PORT": 1234,
        "WEB_SERVER_UPLOADS_DIR": "test-uploads-dir",
        "LOG_LEVEL": "DEBUG",
    }
    config = Config.model_validate(config_data)

    assert config.log_level == "DEBUG"


def test_load_config_with_env_dict() -> None:
    env_dict = {
        "REALITY_DEFENDER_API_KEY": "test-api-key",
        "WEB_SERVER_HOST": "test-host",
        "WEB_SERVER_PORT": 1234,
        "WEB_SERVER_UPLOADS_DIR": "test-uploads-dir",
        "LOG_LEVEL": "DEBUG",
    }
    config = load_config(env_dict)

    assert config.reality_defender_api_key == "test-api-key"
    assert config.web_server_host == "test-host"
    assert config.web_server_port == 1234
    assert config.web_server_uploads_dir == Path("test-uploads-dir")
    assert config.log_level == "DEBUG"


@patch.dict(
    os.environ,
    {
        "REALITY_DEFENDER_API_KEY": "test-api-key",
        "WEB_SERVER_HOST": "test-host",
        "WEB_SERVER_PORT": "1234",
        "WEB_SERVER_UPLOADS_DIR": "test-uploads-dir",
        "LOG_LEVEL": "DEBUG",
    },
)
def test_load_config_from_os_environ() -> None:
    """Test load_config from os.environ."""
    config = load_config()

    assert config.reality_defender_api_key == "test-api-key"
    assert config.web_server_host == "test-host"
    assert config.web_server_port == 1234
    assert config.web_server_uploads_dir == Path("test-uploads-dir")
    assert config.log_level == "DEBUG"


@patch("reality_defender_mcp_server.config.logging.basicConfig")
def test_setup_logging_default_level(mock_basic_config: MagicMock) -> None:
    setup_logging()

    mock_basic_config.assert_called_once_with(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@patch("reality_defender_mcp_server.config.logging.basicConfig")
def test_setup_logging_debug_level(mock_basic_config: MagicMock) -> None:
    """Test setup_logging with DEBUG level."""
    setup_logging("DEBUG")

    mock_basic_config.assert_called_once_with(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@patch("reality_defender_mcp_server.config.logging.basicConfig")
def test_setup_logging_invalid_level(mock_basic_config: MagicMock) -> None:
    """Test setup_logging with invalid level defaults to INFO."""
    setup_logging("INVALID")

    mock_basic_config.assert_called_once_with(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@patch("reality_defender_mcp_server.config.logging.basicConfig")
def test_setup_logging_with_all_levels(mock_basic_config: MagicMock) -> None:
    """Test setup_logging with all valid levels."""
    levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    expected_levels = [
        logging.DEBUG,
        logging.INFO,
        logging.WARNING,
        logging.ERROR,
        logging.CRITICAL,
    ]

    for level, expected in zip(levels, expected_levels):
        mock_basic_config.reset_mock()
        setup_logging(level)
        mock_basic_config.assert_called_once_with(
            level=expected,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
