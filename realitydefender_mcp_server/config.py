from pathlib import Path
import os

from pydantic import BaseModel, Field


class Config(BaseModel):
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
    debug: bool = Field(False, alias="DEBUG", description="Enable debug mode")


def load_config(env: dict[str, str] | None = None) -> Config:
    env = env or dict(os.environ)

    return Config.model_validate(env)
