"""
FastAPI web server for handling Reality Defender image uploads.
"""

import json
import logging
import math
import socket
from string import Template
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, cast

import aiofiles
import uvicorn
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel


def pretty_size(size_bytes: int) -> str:
    for mag, unit in ((3, "GB"), (2, "MB"), (1, "KB")):
        byte_count = math.pow(1024, mag)

        if size_bytes % byte_count == 0:
            return str(size_bytes // byte_count) + unit

    return str(size_bytes) + "B"


class UploadMetadata(BaseModel):
    """Metadata for uploaded files."""

    created_at_timestamp: str
    file_extension: str
    file_id: str
    file_path: str
    file_size: int
    mime_type: str
    source_filename: str | None
    source_type: str
    source_url: str | None


class WebServerConfig(BaseModel):
    """Configuration for the web server."""

    bind_address: tuple[str, int]
    upload_dir: Path
    allowed_extensions: set[str] = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
    max_file_size: int = 1024 * 1024  # 1MB


logger = logging.getLogger(__name__)


app: FastAPI = FastAPI(
    title="Reality Defender Image Upload",
    description="Web interface for uploading images for Reality Defender analysis",
)


class TemplateEngine:
    cache: dict[str, Template]
    templates_dir: Path

    def __init__(self, templates_dir: Path):
        self.cache = {}
        self.templates_dir = templates_dir

    async def render(self, name: str, data: dict[str, object] | None = None) -> str:
        template = self.cache.get(name)

        if not template:
            async with aiofiles.open(self.templates_dir.joinpath(f"{name}.html")) as fp:
                template_content = await fp.read()

            template = Template(template_content)

            self.cache[name] = template

        return template.substitute(data or {})


def get_template_engine() -> TemplateEngine:
    raise NotImplementedError()


def get_web_server_config() -> WebServerConfig:
    raise NotImplementedError()


@app.get("/upload/{file_id}", response_class=HTMLResponse)
async def get_upload_form(
    file_id: str,
    request: Request,
    templates: Annotated[TemplateEngine, Depends(get_template_engine)],
    web_server_config: Annotated[WebServerConfig, Depends(get_web_server_config)],
) -> str:
    """Serve upload form."""

    request_id = str(uuid.uuid4())[:8]
    client_ip = request.client.host if request.client else "unknown"
    logger.info(
        f"Serving upload form for UUID: {file_id} [request_id: {request_id}, client_ip: {client_ip}]"
    )
    logger.debug(
        f"Upload form requested - UUID: {file_id}, User-Agent: {request.headers.get('user-agent', 'unknown')} [request_id: {request_id}]"
    )
    return await templates.render(
        "upload_file",
        {
            "file_id": file_id,
            "max_file_size_label": pretty_size(web_server_config.max_file_size),
            "max_file_size_bytes": web_server_config.max_file_size,
        },
    )


@app.post("/upload/{file_id}", response_class=HTMLResponse)
async def upload_file(
    file_id: str,
    request: Request,
    file: Annotated[UploadFile, File(...)],
    web_server_config: Annotated[WebServerConfig, Depends(get_web_server_config)],
    templates: Annotated[TemplateEngine, Depends(get_template_engine)],
) -> str:
    """Handle file upload."""

    request_id = str(uuid.uuid4())[:8]
    client_ip = request.client.host if request and request.client else "unknown"
    logger.info(
        f"Processing file upload for UUID: {file_id}, filename: {file.filename} [request_id: {request_id}, client_ip: {client_ip}]"
    )
    logger.debug(
        f"Upload request received - UUID: {file_id}, Content-Type: {file.content_type}, User-Agent: {request.headers.get('user-agent', 'unknown') if request else 'unknown'} [request_id: {request_id}]"
    )
    upload_start_time = time.time()
    logger.info(
        f"Starting file upload processing for UUID: {file_id} [request_id: {request_id}]"
    )

    # Validate UUID format
    try:
        _ = uuid.UUID(file_id)
        logger.debug(
            f"UUID validation passed for: {file_id} [request_id: {request_id}]"
        )
    except ValueError:
        logger.error(f"Invalid UUID format: {file_id} [request_id: {request_id}]")
        raise HTTPException(status_code=400, detail="Invalid UUID")

    if not file.filename:
        logger.error(
            f"No filename provided in upload for UUID: {file_id} [request_id: {request_id}]"
        )
        raise HTTPException(status_code=400, detail="No file uploaded")

    logger.debug(
        f"Processing file: {file.filename}, Content-Type: {file.content_type} [request_id: {request_id}]"
    )

    # Check file size
    content_read_start = time.time()
    content = await file.read()
    file_size = len(content)
    content_read_duration = time.time() - content_read_start
    logger.info(
        f"File size validation - received {file_size:,} bytes (limit: {web_server_config.max_file_size:,} bytes) in {content_read_duration:.2f}s [request_id: {request_id}]"
    )

    if file_size > web_server_config.max_file_size:
        logger.error(
            f"File too large: {file_size:,} bytes > {web_server_config.max_file_size:,} bytes for UUID: {file_id} [request_id: {request_id}]"
        )
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {web_server_config.max_file_size // 1024}KB.",
        )

    # Validate file extension
    file_ext = Path(file.filename).suffix.lower()
    logger.debug(f"File extension: {file_ext} [request_id: {request_id}]")
    if file_ext not in web_server_config.allowed_extensions:
        logger.error(
            f"Invalid file extension: {file_ext} for UUID: {file_id} [request_id: {request_id}]"
        )
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(sorted(web_server_config.allowed_extensions))}",
        )

    # Create upload directory
    upload_uuid_dir = web_server_config.upload_dir / file_id
    logger.debug(
        f"Creating upload directory: {upload_uuid_dir} [request_id: {request_id}]"
    )
    try:
        upload_uuid_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(
            f"Failed to create upload directory: {str(e)} [request_id: {request_id}]"
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to create upload directory: {str(e)}"
        )

    # Save file data to blob with extension
    blob_path = upload_uuid_dir / f"blob{file_ext}"
    logger.debug(f"Saving file to: {blob_path} [request_id: {request_id}]")

    try:
        file_write_start = time.time()
        with open(blob_path, "wb") as f:
            _ = f.write(content)
        file_write_duration = time.time() - file_write_start
    except Exception as e:
        logger.error(
            f"Failed to create upload directory: {str(e)} [request_id: {request_id}]"
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to create upload directory: {str(e)}"
        )

    # Create metadata
    metadata = UploadMetadata(
        created_at_timestamp=datetime.now(timezone.utc).isoformat(),
        file_extension=file_ext,
        file_id=file_id,
        file_path=str(blob_path),
        file_size=file_size,
        mime_type=file.content_type or "application/octet-stream",
        source_filename=file.filename,
        source_type="user_upload",
        source_url=None,
    )

    try:
        metadata_path = upload_uuid_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata.model_dump(), f, indent=2)

    except Exception as e:
        logger.error(
            f"Failed to create upload directory: {str(e)} [request_id: {request_id}]"
        )
        raise HTTPException(
            status_code=500, detail=f"Failed to create upload directory: {str(e)}"
        )

    total_duration = time.time() - upload_start_time
    write_speed = file_size / file_write_duration if file_write_duration > 0 else 0
    logger.info(
        f"File upload successful: {file.filename} ({file_size:,} bytes) saved to {blob_path} in {total_duration:.2f}s (write: {file_write_duration:.2f}s, {write_speed / 1024:.1f} KB/s) [request_id: {request_id}]"
    )

    return await templates.render(
        "success",
        {
            "file_id": file_id,
            "blob_path": blob_path,
            "metadata_path": metadata_path,
        },
    )


@app.get("/health", response_class=JSONResponse)
async def health_check(request: Request) -> dict[str, str]:
    """Health check endpoint."""

    client_ip = request.client.host if request.client else "unknown"
    logger.debug(f"Health check requested [client_ip: {client_ip}]")

    return {
        "status": "healthy",
        "service": "reality-defender-upload",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/", response_class=HTMLResponse)
async def root(
    request: Request, templates: Annotated[TemplateEngine, Depends(get_template_engine)]
) -> str:
    """Home page with upload creation."""

    client_ip = request.client.host if request.client else "unknown"
    logger.debug(f"Home page accessed [client_ip: {client_ip}]")

    return await templates.render("home")


@app.post("/create-upload")
async def create_upload(request: Request) -> JSONResponse:
    """Generate a new upload UUID and return redirect info."""

    file_id = str(uuid.uuid4())
    client_ip = request.client.host if request.client else "unknown"
    logger.info(f"Created new file ID: {file_id} [client_ip: {client_ip}]")

    return JSONResponse({"file_id": file_id, "redirect_url": f"/upload/{file_id}"})


def find_free_port() -> int:
    """Find an available port on the system."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        port = cast(int, s.getsockname()[1])
    return port


def create_server(web_server_config: WebServerConfig) -> uvicorn.Server:
    """Run FastAPI web server synchronously (blocking)."""
    templates_dir: Path = Path(__file__).parent.joinpath("templates")
    template_engine: TemplateEngine = TemplateEngine(templates_dir)

    app.dependency_overrides[get_web_server_config] = lambda: web_server_config
    app.dependency_overrides[get_template_engine] = lambda: template_engine

    host, port = web_server_config.bind_address

    logger.info(f"Starting web server on {host}:{port}")

    return uvicorn.Server(
        uvicorn.Config(app, host=host, port=port, log_config=None, access_log=False)
    )
