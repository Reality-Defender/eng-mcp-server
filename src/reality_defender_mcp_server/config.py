import logging
from pathlib import Path
import os

from pydantic import BaseModel, Field


from dotenv import load_dotenv

load_dotenv()


class Config(BaseModel):
    reality_defender_api_key: str = Field(
        alias="REALITY_DEFENDER_API_KEY",
        description="API key for the Reality Defender API",
    )

    web_server_host: str = Field(
        "127.0.0.1",
        alias="WEB_SERVER_HOST",
        description="The host address to bind the web server for local uploads to.",
    )
    web_server_port: int = Field(
        0,
        alias="WEB_SERVER_PORT",
        description="The port to bind the web server for local uploads to (0, the default, means find an open port).",
    )
    web_server_uploads_dir: Path = Field(
        Path("uploads"),
        alias="WEB_SERVER_UPLOADS_DIR",
        description="The path to store files that are uploaded to the web server",
    )
    log_level: str = Field("INFO", alias="LOG_LEVEL", description="Current log level")


def load_config(env: dict[str, str] | None = None) -> Config:
    env = env or dict(os.environ)

    return Config.model_validate(env)


def setup_logging(log_level: str = "INFO") -> None:
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    level = level_map.get(log_level.upper(), logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set specific loggers
    logging.getLogger("fastapi").setLevel(logging.WARNING)
    logging.getLogger("mcp").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
