FROM ghcr.io/astral-sh/uv:python3.11-alpine

ENV REALITY_DEFENDER_API_KEY=""

ADD . /app

WORKDIR /app
RUN uv sync --locked

# Run the application
CMD ["uv", "run", "./src/realitydefender_mcp_server/mcp_server.py"]
