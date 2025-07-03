#!/usr/bin/env python3
"""
CLI for running the Reality Defender web server directly.
"""

import argparse
import logging
from pathlib import Path

from realitydefender_mcp_server.web_server.server import run_web_server_sync


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Reality Defender Web Server")

    parser.add_argument(
        "--upload-dir", type=Path, default=Path("uploads"), help="Directory to store uploaded files (default: uploads)"
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host to bind to (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=None, help="Port to bind to (default: auto-detect available port)")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    if args.port:
        logging.info(f"Starting Reality Defender web server on {args.host}:{args.port}")
    else:
        logging.info(f"Starting Reality Defender web server on {args.host} (auto-detecting port)")
    logging.info(f"Upload directory: {args.upload_dir.absolute()}")

    try:
        run_web_server_sync(args.upload_dir, args.host, args.port)
    except KeyboardInterrupt:
        logging.info("Shutting down web server")


if __name__ == "__main__":
    main()
