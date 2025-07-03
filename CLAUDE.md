# Reality Defender MCP Server

This document outlines key information for working with the Reality Defender MCP Server project.

## Documentation Guidelines

### Documentation Maintenance

- **Keep Documentation Updated**: All code changes must be accompanied by corresponding updates to CLAUDE.md and README.md
- **Document New Features**: Any new functionality should be documented with usage examples
- **Update Configuration**: Ensure any new environment variables or configuration options are documented
- **Update Troubleshooting**: Add common issues and solutions as they are discovered

### Structured Data Guidelines

- **Always Use Pydantic Models**: All structured data, especially from external sources like APIs, must use Pydantic models
- **Input Validation**: Define explicit types and validation rules in Pydantic models for all external inputs
- **Output Serialization**: Use model_dump() for all data sent to external systems
- **Type Annotations**: Ensure all function parameters and return values have proper type annotations

### Python 3.13 Standards & Best Practices

- **Modern Type Annotations**: Use Python 3.13 union syntax (`str | None` instead of `Optional[str]`)
- **Strict Type Hints**: Apply type hints to all functions, parameters, and return values
- **Prefer `object` over `Any`**: Use `object` instead of `Any` for better type safety (similar to TypeScript's `unknown` over `any`)
- **Type Parameter Syntax**: Use the new `Class[T]` type parameter syntax for generics
- **Pattern Matching**: Utilize structural pattern matching for complex conditionals
- **PEP 695 Type Aliases**: Use the new type alias syntax: `type Vec = list[float]`
- **Self Type**: Use the `Self` type for methods that return instances of their own class
- **Type Guards**: Implement type guards to narrow types in conditional blocks

### Error Handling Pattern

MCP tools use a consistent error handling pattern:

- **Return Union Types**: All tools return `SuccessType | Error` instead of raising exceptions
- **Error Model**: Use `Error(error="message")` pydantic model with single `error: str` property
- **Wrap Operations**: Surround potentially failing operations with try/catch blocks
- **Consistent Logging**: Log errors before returning Error objects for debugging

### URL Download Error Handling

Enhanced error handling for file URL downloads:

- **404 Not Found**: User-friendly message explaining URL may be incorrect/moved
- **403/401 Forbidden/Unauthorized**: Clear explanation about access restrictions
- **429 Rate Limited**: Information about temporary server blocking
- **5xx Server Errors**: Explanation of server-side problems
- **Timeout Errors**: Guidance about slow downloads or large files
- **Connection Errors**: Help with network connectivity issues

All URL errors suggest using the upload workflow as an alternative with specific guidance.

### Async-First Development

- **Async by Default**: Use async/await patterns wherever possible
- **aiohttp over requests**: Use aiohttp for HTTP requests instead of requests
- **aiofiles**: Use aiofiles for file operations in async contexts
- **AsyncIterator**: Use proper async iteration patterns

### Code Quality & Formatting

- **Ruff**: Use ruff for code formatting and linting
- **Format Command**: Run `ruff format` to reformat files when possible

### Git & Development Workflow

- **Git Commands**: Use `git log --patch` for combined log and diff viewing
- **Lock Files**: Never edit lock files directly - ask for updates after adding dependencies

## Configuration

The application uses environment variables for configuration:

| Variable | Description | Required |
|----------|-------------|----------|
| `REALITY_DEFENDER_API_KEY` | API key for Reality Defender deepfake detection service | Yes |
| `DEBUG` | Enable debug mode (set to "true" to enable) | No |
| `WEB_SERVER_HOST` | Host for the web server (default: 127.0.0.1) | No |
| `WEB_SERVER_PORT` | Port for the web server (default: 8080) | No |
| `WEB_SERVER_UPLOADS_DIR` | Directory for file uploads (default: ./uploads) | No |

## Key Components

### Reality Defender Integration

Web-based upload system with structured file organization for AI-generated media detection.

#### MCP Tools

All MCP tools return union types with Error objects instead of raising exceptions:

- **`reality_defender_generate_upload_url`** - Returns `GenerateUploadUrlOutput | Error`
- **`reality_defender_get_file_info`** - Returns `GetFileInfoOutput | Error` 
- **`reality_defender_request_file_analysis`** - Returns `RealityDefenderAnalysisResponse | Error`

Error handling uses a consistent `Error` pydantic model with a single `error: str` property.

#### Upload Structure

Each upload creates `uploads/{uuid}/`:
- `blob{ext}` - Raw file data with proper extension
- `metadata.json` - Upload info (filename, size, timestamp, MIME type, source)

#### Web Server

FastAPI server on `localhost:8080` with 1MB limit, supports PNG/JPG/JPEG/GIF/WebP.

**Package Structure:**
- `liar_liar.web_server.server` - Core FastAPI application with WebServer class and threading utilities
- `web_server.py` - CLI interface (standalone runner)

**API Endpoints:**
- `GET /` - Service information
- `GET /health` - Health check
- `GET /upload/{uuid}` - Upload form
- `POST /upload/{uuid}` - File upload
- `GET /docs` - OpenAPI documentation

**Running Standalone:**
```bash
# Basic usage
python web_server.py --port 8080

# Basic with debugging
python web_server.py --debug

# Custom settings
python web_server.py --upload-dir ./custom_uploads --host 0.0.0.0 --port 9000
```

**Programmatic Usage:**
```python
from liar_liar.web_server import start_web_server, create_web_server_app
from pathlib import Path

# Start in background thread
thread = start_web_server(Path("uploads"))

# Create app for testing
app = create_web_server_app(Path("uploads"))
```

**File Structure:**
```
uploads/
└── {uuid}/
    ├── blob          # Raw file data
    └── metadata.json # Upload metadata with filename, size, timestamp, MIME type
```

#### User Flow

**Primary Workflow (Recommended - User Upload):**

1. **Generate Upload URL**: LLM calls `reality_defender_generate_upload_url()`
2. **User Upload**: LLM shows URL to user, asks them to upload image and confirm completion
3. **Get Upload Info**: LLM calls `reality_defender_get_upload_info(uuid)` with UUID from URL
4. **Validate Image**: LLM calls `reality_defender_validate_image_authenticity()` with file path
5. **Present Results**: LLM shows upload metadata and authenticity analysis to user

**Alternative Workflow (Direct URL):**

1. **Validate Image**: LLM calls `reality_defender_validate_image_authenticity()` with image_url
2. **Handle Results**:
   - If successful: Response includes `download_uuid` field
   - If error: Explain specific issue and suggest upload workflow
3. **Get Download Info**: LLM calls `reality_defender_get_upload_info(download_uuid)` 
4. **Present Results**: Show download metadata and authenticity analysis

**URL Error Handling Examples:**
- 404: "Image not found - URL may be incorrect, moved, or deleted. Please try uploading the image directly."
- 403: "Access denied - authentication or permissions required. Please upload the image directly."
- Timeout: "Download took too long - image may be too large. Please upload a smaller image directly."

**Implementation Notes:**
- Uses `aiohttp` for async HTTP requests and `aiofiles` for file operations
- FastAPI web framework for HTTP upload handling with testable architecture
- Proper MIME type and file extension validation
- Modern Python 3.13 union syntax throughout
- Structured as a proper package with CLI interface
- Enhanced logging with request IDs, timing, and detailed error context
- URL downloads create upload metadata for transparency and subsequent retrieval

## Running the Application

### Server Mode

To run the application in server mode:

```bash
python server.py [options]
```

Server options:
- `--debug`: Enable debug logging
- `--host`: Server host (default: 127.0.0.1)
- `--port`: Server port (default: 8000)
- `--polling-interval`: Polling interval in seconds (default: 5)
- `--max-messages-per-poll`: Max number of messages to fetch per poll (default: 100)

### Development Setup

Prerequisites:
- Python 3.13
- Poetry (dependency management)

Setup steps:
1. Clone the repository
2. Run `poetry install` to install dependencies
3. Set up the required environment variables
4. Run the server with `poetry run python server.py`

## Common Tasks

### Adding New Agents

1. Create a new agent class in the `lie_detector/slack_bot/` directory
2. Follow the pattern in `word_reverser.py` by implementing:
   - Processing logic for messages
   - A main run loop
3. Update `server.py` to initialize and run your new agent

### Linting and Type Checking

Run the following commands:

```bash
# Run type checking
poetry run pyright

# Run linting
poetry run ruff check . 
```

### Running Tests

```bash
poetry run pytest
```

## Troubleshooting

- **Authentication Errors**: Check that your `SLACK_BOT_TOKEN` is valid and has the required scopes
- **Channel Not Found**: Verify the `SLACK_TARGET_CHANNEL_ID` is correct and the bot has been invited to the channel
- **Message Processing Issues**: Enable debug logging with the `--debug` flag to see detailed logs
