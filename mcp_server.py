#!/usr/bin/env python3

import asyncio
import logging
import mimetypes
import os
import tempfile
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import override
from urllib.parse import urlparse

import aiofiles
import aiohttp
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.prompts.base import UserMessage, Message as PromptMessage
from mcp.server.session import ServerSession
from pydantic import BaseModel, Field, ValidationError
from realitydefender import RealityDefender

from realitydefender_mcp_server.config import load_config, Config
from realitydefender_mcp_server.web_server.server import UploadMetadata, WebServerConfig, create_server, find_free_port


def ellispis(text: str, max_length: int, replacement: str = "...") -> str:
    if len(text) < max_length:
        return text
    else:
        return text[0:max_length] + " " + replacement


class AppServerSession(ServerSession):
    pass


class RealityDefenderAnalysisRequest(BaseModel):
    """Request authenticity analysis on a file from Reality Defender"""

    file_path: str | None = Field(None, description="Local path to file")
    file_url: str | None = Field(None, description="URL to file to validate")
    expected_file_type: str = Field(..., description="The type of file expected: video, image, audio or text")

    @override
    def model_post_init(self, __context: object) -> None:
        inputs = [self.file_path, self.file_url]
        if sum(1 for x in inputs if x is not None) != 1:
            raise ValueError("Exactly one of file_path or file_url must be provided")


class Error(BaseModel):
    error: str


class RealityDefenderModel(BaseModel):
    name: str
    status: str
    score: float | None = None


class RealityDefenderDetectFileResponse(BaseModel):
    status: str
    score: float | None = None
    models: list[RealityDefenderModel] = Field(default_factory=list)
    request_id: str | None = None


class RealityDefenderGetResultResponse(BaseModel):
    status: str
    score: float | None = None
    models: list[RealityDefenderModel] = Field(default_factory=list)
    request_id: str | None = None


class RealityDefenderUploadResponse(BaseModel):
    media_id: str
    request_id: str


class RealityDefenderAnalysisResponse(BaseModel):
    status: str = Field(description="Overall authenticity status (ARTIFICIAL, AUTHENTIC, ANALYZING)")
    score: float | None = Field(description="Confidence score (0-100)")
    models: list[RealityDefenderModel] = Field(description="Individual model results")
    file_id: str | None = Field(None, description="Request ID for processed file metadata (present for URL downloads)")


class RealityDefenderClientHarness:
    api_key: str | None
    client: RealityDefender | None

    def __init__(self, env: dict[str, str]):
        self.api_key = env.get("REALITY_DEFENDER_API_KEY")
        self.client = None

    def get_client(self) -> RealityDefender | Error:
        if self.client:
            return self.client

        if not self.api_key:
            return Error(error="API key not provided. REALITY_DEFENDER_API_KEY is not defined")

        self.client = RealityDefender({"api_key": self.api_key})

        return self.client


@dataclass
class AppContext:
    config: Config
    reality_defender: RealityDefenderClientHarness
    web_server_url: str


logger = logging.getLogger(__name__)


@asynccontextmanager
async def app_lifespan(_: FastMCP) -> AsyncIterator[AppContext]:
    config = load_config()

    logger.root.setLevel(logging.DEBUG if config.debug else logging.INFO)

    logger.info("Starting MCP server lifespan")

    logger.info(f"Upload directory configured: {config.web_server_uploads_dir.absolute()}")

    logger.info("Starting web server for file uploads")

    web_server_host = config.web_server_host
    web_server_port = config.web_server_port

    if web_server_port == 0:
        web_server_port = find_free_port()

    web_server_config = WebServerConfig(
        bind_address=(web_server_host, web_server_port), upload_dir=config.web_server_uploads_dir
    )

    match web_server_config.bind_address:
        case ("0.0.0.0" | "127.0.0.1" | "localhost", port):
            web_server_url = f"http://localhost:{port}"
        case (host, port):
            web_server_url = f"http://{host}:{port}"

    async def run_web_server():
        try:
            web_server = create_server(web_server_config)
        except Exception:
            logger.exception("Create server failed")
            raise

        try:
            await web_server.serve()
        except Exception as e:
            logger.error("An unexpected error occurred", exc_info=e)

    web_server_task = asyncio.create_task(run_web_server())

    logger.info(f"Web server started at {web_server_url}")

    logger.info("MCP server initialization complete")

    try:
        yield AppContext(
            config=config,
            reality_defender=RealityDefenderClientHarness(dict(os.environ)),
            web_server_url=web_server_url,
        )
    except Exception as e:
        logger.error("An unexpected error occurred", exc_info=e)
    finally:
        logger.info("MCP server lifespan ending")

        __ = web_server_task.cancel()


mcp = FastMCP(
    "realitydefender",
    lifespan=app_lifespan,
    instructions="""
# Reality Defender: MCP Server

## Detecting AI generated media
For file analysis via Reality Defender, there are two flows: user uploaded and direct download.

### user_upload
A user indicates they have a file they would like to upload for analysis themselves. They might 
have been sent a file or have already downloaded it from a web site. In this case, the user will need to be
sent to a custom file upload page where they must upload the file themselves so it can be analyzed.

 * step 1: reality_defender_generate_upload_url - Create a URL so the user can upload the file
 * step 2: reality_defender_get_upload_info - Get the details of the upload after the user affirms upload
 * step 3: reality_defender_validate_image_authenticity - Ask Reality Defender to analyze the file

### direct_download
The user has the URL to a file hosted externally. In this case, the file can be downloaded
directly to the filesystem and processed without any additional steps necessary by the user.

 * step 1: reality_defender_validate_image_authenticity - Ask Reality Defender to analyze the file
 * step 2: reality_defender_get_upload_info - Using the response from step 1, retrieve additional file metadata for final response to user.

Note: direct_download is preferred to user_upload as it's more convenient for the user, but user_upload works in more scenarios

## Error Handling

For ANY error returned by a tool:
* Explain the specific issue clearly to the user
* If the error is an HTTP failure, clearly explain why this could happen and indicate which host returned the error.
    * Note: 404 errors during file download could be due to authentication failure depending on how the server handles the request.
* Suggest using the user_upload workflow as an alternative to direct_download when it might work
* Be helpful and specific in your guidance

Always follow the appropriate sequence based on user input and error conditions.
""",
)


@mcp.prompt()
def reality_defender_request_to_check_authenticity_of_hosted_file(url: str) -> list[PromptMessage]:
    logger.info(f"Creating prompt to check authenticity of a file: {url}")
    return [
        UserMessage(
            f"Download this file and use the Reality Defender API to determine if it was generated by AI: {url}"
        ),
    ]


class GenerateUploadUrlOutput(BaseModel):
    upload_url: str
    request_id: str


@mcp.tool()
async def reality_defender_generate_upload_url(
    ctx: Context[AppServerSession, AppContext],
) -> GenerateUploadUrlOutput | Error:
    """
    Generate a unique upload URL for Reality Defender image validation.

    This is the first step in the image verification workflow for the user_upload
    flow. Show the returned URL to the user and ask them to upload their image, then
    tell you when complete.

    Returns:
        A GenerateUploadUrlOutput object with information needed to direct the user
        to upload the image or an Error
    """
    web_server_url = ctx.request_context.lifespan_context.web_server_url
    if isinstance(web_server_url, Error):
        return web_server_url

    try:
        request_id = str(uuid.uuid4())
        upload_url = f"{web_server_url}/upload/{request_id}"

        logger.info(f"Generated upload URL with request_id: {request_id}")
        logger.debug(f"Upload URL: {upload_url}")

        return GenerateUploadUrlOutput(upload_url=upload_url, request_id=request_id)
    except Exception as e:
        logger.error(f"Failed to generate upload URL: {str(e)}")
        return Error(error=f"Failed to generate upload URL: {str(e)}")


class GetFileInfoOutput(BaseModel):
    created_at_timestamp: str
    file_extension: str
    file_path: str
    file_size: int
    mime_type: str
    request_id: str
    source_filename: str | None
    source_type: str
    source_url: str | None = None


@mcp.tool()
async def reality_defender_get_file_info(
    file_id: str, ctx: Context[AppServerSession, AppContext]
) -> GetFileInfoOutput | Error:
    """
    Get metadata about a uploaded file (user_upload) or downloaded file (direct_download).

    The input to this tool, `file_id` is returned from calls to tools in steps before this
    tool is called. It's also possible the user directly asks for information about a file
    by `file_id`, in which case you can use this tool as well. This returns complete file
    information including the file path needed for validation.

    Args:
        file_id: The file_id to retrieve metadata for

    Returns:
        GetFileInfoOutput object with details about the upload or an Error
    """
    logger.info(f"Retrieving file metadata for file_id: {file_id}")

    files_dir = ctx.request_context.lifespan_context.config.web_server_uploads_dir
    file_dir = files_dir / file_id

    if not file_dir.exists():
        return Error(error=f"No file directory found for file_id: {file_id}")

    metadata_path = file_dir / "metadata.json"

    if not metadata_path.exists():
        return Error(error=f"No metadata file found for file_id: {file_id}")

    try:
        with open(metadata_path, "r") as f:
            metadata_json = f.read()
    except Exception as e:
        logger.error(f"Failed to read metadata file for file {file_id}", exc_info=e)
        return Error(error=f"File metadata read failed: {e}")

    try:
        metadata = UploadMetadata.model_validate_json(metadata_json)
    except Exception as e:
        return Error(error=f"File metadata read failed: invalid format - {e}")

    logger.info(f"Retrieved metadata for file: (source: {metadata.source_type})")

    file_path = str(file_dir / ("blob" + metadata.file_extension))

    return GetFileInfoOutput(
        created_at_timestamp=metadata.created_at_timestamp,
        file_extension=metadata.file_extension,
        file_path=file_path,
        file_size=metadata.file_size,
        mime_type=metadata.mime_type,
        request_id=file_id,
        source_filename=metadata.source_filename,
        source_type=metadata.source_type,
        source_url=metadata.source_url,
    )


@mcp.tool()
async def reality_defender_request_file_analysis(
    request: RealityDefenderAnalysisRequest, ctx: Context[AppServerSession, AppContext]
) -> RealityDefenderAnalysisResponse | Error:
    """
    Validate file analysis using Reality Defender API. Reality Defender is used
    to determine if a file (text, video, image, or audio) shows signs of being AI generated.

    Either a file path or url is needed to invoke this tool. This might be called
    directly with a user provided URL (direct_download flow) or this tool could be
    called to the path to a file after the user has uploaded it and you've got the
    file path from a call to reality_defender_get_upload_info (user_upload flow).

    Args:
        request: RealityDefenderAnalysisRequest with information on the file

    Returns:
        RealityDefenderAnalysisResponse with status (ARTIFICIAL/AUTHENTIC), confidence score,
        individual model results, and analysis details, or Error on failure
    """
    logger.info(f"Starting Reality Defender image validation: {request.model_dump_json()}")

    reality_defender_result = ctx.request_context.lifespan_context.reality_defender.get_client()
    if isinstance(reality_defender_result, Error):
        logger.error(f"Reality Defender client not available: {reality_defender_result}")
        return reality_defender_result

    reality_defender = reality_defender_result

    source_url: str | None = None
    source_filename: str | None = None

    upload_dir = ctx.request_context.lifespan_context.config.web_server_uploads_dir

    if request.file_path:
        if not Path(request.file_path).is_relative_to(upload_dir):
            return Error(error="Invalid file path provided: it must be in the filesystem of this project")

        logger.info(f"Processing local image file: {request.file_path}")

        if not os.path.exists(request.file_path):
            return Error(error=f"Image file not found: {request.file_path}")

        file_path = request.file_path
        file_id = Path(request.file_path).parent.name

    elif request.file_url:
        source_url = request.file_url

        download_start_time = time.time()

        logger.info(f"Downloading image from URL: {request.file_url}")

        parsed_url = urlparse(request.file_url)
        if not parsed_url.scheme or not parsed_url.netloc:
            return Error(
                error="Invalid image URL format. The URL must include 'http://' or 'https://' and a valid domain. Please check the URL and try again, or upload the image directly using the upload workflow."
            )

        with tempfile.NamedTemporaryFile(mode="wb") as temp_file:
            try:
                timeout = aiohttp.ClientTimeout(total=30)

                async with aiohttp.ClientSession(timeout=timeout) as session:
                    logger.debug(f"Making HTTP request to: {request.file_url}")

                    request_start_time = time.time()

                    async with session.get(request.file_url) as response:
                        request_duration = time.time() - request_start_time

                        logger.debug(
                            f"HTTP response received in {request_duration:.2f}s - Status: {response.status}, Content-Type: {response.headers.get('content-type', 'unknown')}"
                        )

                        if response.status != 200:
                            response_text = ellispis(
                                await response.text(encoding="latin-1", errors="replace"), 64, "..."
                            )

                            logger.error(f"Download failed with status code: {response.status}. {response_text}")

                            return Error(
                                error=f"An non successful HTTP status code was received when downloading image: {response.status}."
                            )

                        content_type = response.headers.get("content-type", "").split(";")[0].lower()
                        if not content_type.startswith(f"{request.expected_file_type}/"):
                            logger.error(f"URL does not return '{request.expected_file_type}' content: {content_type}")

                            return Error(
                                error=f"Download was successful (status code: 200) but the result was not of type '{request.expected_file_type}': Content-Type is {response.headers.get('content-type')}."
                            )

                        for content_disposition_key_value in response.headers.get("content-disposition", "").split(";"):
                            if content_disposition_key_value.lower().startswith("filename="):
                                source_filename = content_disposition_key_value.split("=", maxsplit=1)[1]

                        total_size = int(response.headers.get("content-length", 0))
                        downloaded_size = 0
                        download_start = time.time()

                        if total_size > 0:
                            logger.info(f"Starting download of {total_size} bytes")
                        else:
                            logger.info("Starting download (size unknown)")

                        try:
                            async with aiofiles.open(temp_file.name, "wb") as f:
                                async for chunk in response.content.iter_chunked(8192):
                                    _ = await f.write(chunk)

                                    downloaded_size += len(chunk)

                                    if total_size > 0 and downloaded_size % (100 * 1024) == 0:  # Every 100KB
                                        progress = downloaded_size / total_size * 100
                                        logger.debug(
                                            f"Download progress: {downloaded_size:,}/{total_size:,} bytes ({progress:.1f}%)"
                                        )
                        except Exception as e:
                            logger.error(f"Download failed: {request.file_url} -> {temp_file.name}", exc_info=e)

                            return Error(error=f"Failed to write the downloaded content to the temporary file: {e}")
            except asyncio.TimeoutError:
                logger.error(f"Download timeout after 30 seconds: {request.file_url}")

                return Error(error="The image download timed out")

            except aiohttp.ClientConnectorError as e:
                logger.error(f"Connection error downloading image: {str(e)}")

                return Error(error=f"Could not connect to the server hosting the image: {e}")

            except aiohttp.ClientError as e:
                logger.error(f"HTTP client error downloading image: {str(e)}")

                return Error(error=f"An unknown error occurred while downloading the image: {e}")

            download_duration = time.time() - download_start
            download_speed = downloaded_size / download_duration if download_duration > 0 else 0
            logger.info(
                f"Download completed: {downloaded_size:,} bytes in {download_duration:.2f}s ({download_speed / 1024:.1f} KB/s)"
            )

            # Create upload metadata for the downloaded file
            file_id = str(uuid.uuid4())
            upload_dir = ctx.request_context.lifespan_context.config.web_server_uploads_dir
            file_dir = upload_dir / file_id
            file_dir.mkdir(parents=True, exist_ok=True)

            # Determine file extension from content type or URL
            file_ext = mimetypes.guess_extension(content_type)
            if file_ext is None:
                guessed_type, _ = mimetypes.guess_type(parsed_url.path)
                if guessed_type:
                    file_ext = mimetypes.guess_extension(guessed_type)

            if not file_ext:
                logger.error(f"Unable to resolve file extension from the downloaded file: {content_type} [{file_id=}]")

                return Error(
                    error=f"Invalid/Unsupported file: Content-Type is '{content_type}'. File extension resolution failed."
                )

            file_path = str(file_dir / f"blob{file_ext}")

            try:
                async with aiofiles.open(temp_file.name, "rb") as src:
                    async with aiofiles.open(file_path, "wb") as dst:
                        async for chunk in src:
                            _ = await dst.write(chunk)
            except Exception as e:
                logger.error(f"Copy file failed: {temp_file.name} -> {file_path}", exc_info=e)

                return Error(error=f"Writing final file unexpectedly failed: {e}")

            # Create metadata
            metadata = UploadMetadata(
                created_at_timestamp=datetime.now(timezone.utc).isoformat(),
                file_extension=file_ext,
                file_id=file_id,
                file_path=file_path,
                file_size=downloaded_size,
                mime_type=content_type,
                source_filename=source_filename,
                source_type="direct_download",
                source_url=source_url,
            )

            metadata_path = file_dir / "metadata.json"

            try:
                async with aiofiles.open(metadata_path, "w") as f:
                    _ = await f.write(metadata.model_dump_json())
            except Exception as e:
                logger.error(f"Writing metadata failed: {metadata_path}. {metadata}", exc_info=e)

                return Error(error=f"Writing file metadata unexpectedly failed: {e}")

            total_duration = time.time() - download_start_time
            logger.info(f"Image download and metadata creation completed in {total_duration:.2f}s [{file_id=}]")
    else:
        return Error(error="Invalid request: file_path and file_url are both absent")

    logger.info(f"Analyzing image with Reality Defender: {file_path}")

    file_size = os.path.getsize(file_path)

    logger.debug(f"File size for analysis: {file_size} bytes")

    try:
        logger.debug("Uploading file to Reality Defender API")

        upload_result = RealityDefenderUploadResponse.model_validate(
            await reality_defender.upload({"file_path": file_path})
        )
    except ValidationError as e:
        logger.error("Reality Defender upload returned an invalid result", exc_info=e)
        return Error(error=f"Failed to parse upload file result from Reality Defender: {str(e)}")
    except Exception as e:
        logger.error("Reality Defender upload failed", exc_info=e)
        return Error(error=f"Failed to upload file to Reality Defender: {str(e)}")

    logger.info(
        f"Reality Defender upload successful - media_id: {upload_result.media_id}, request_id: {upload_result.request_id}"
    )

    attempt_count = 0
    max_attempt_count = 10

    while True:
        try:
            detect_result = RealityDefenderGetResultResponse.model_validate(
                await asyncio.wait_for(
                    reality_defender.get_result(upload_result.request_id, {"max_attempts": 1}), timeout=1.0
                )
            )
        except ValidationError as e:
            logger.error("Reality Defender detection result returned an invalid result", exc_info=e)
            return Error(error=f"Failed to parse detection result from Reality Defender: {str(e)}")
        except asyncio.TimeoutError:
            pass
        except Exception as e:
            logger.error("Reality Defender detection result returned an invalid result", exc_info=e)
            return Error(error=f"Failed to parse detection result from Reality Defender: {str(e)}")
        else:
            if detect_result.status != "ANALYZING":
                break

        attempt_count += 1

        logger.info(
            f"Waiting for image detection to complete - request_id: {upload_result.request_id}. Completed attempt {attempt_count} of {max_attempt_count}"
        )

        await ctx.report_progress(
            progress=float(attempt_count),
            total=float(max_attempt_count),
            message="Waiting for Realty Defender to process the image. Continuing to poll for result.",
        )

        try:
            await asyncio.sleep(5.0)
        except Exception as e:
            return Error(error=f"An unknown error occurred during sleep: {str(e)}")

    logger.debug(f"Parsed Reality Defender response: {detect_result.model_dump()}")

    response = RealityDefenderAnalysisResponse(
        file_id=file_id,
        status=detect_result.status,
        score=detect_result.score,
        models=detect_result.models,
    )

    logger.info(f"Reality Defender analysis complete - Status: {response.status}, Score: {response.score}")

    return response


if __name__ == "__main__":
    mcp.run()
