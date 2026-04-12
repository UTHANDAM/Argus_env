"""FastAPI application for ARGUS."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from fastapi.responses import HTMLResponse, RedirectResponse
from openenv.core.env_server.http_server import create_app

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import ArgusAction, ArgusObservation
from server.argus_env_environment import ArgusEnvironment


def _create_environment() -> ArgusEnvironment:
    return ArgusEnvironment()


app = create_app(
    _create_environment,
    ArgusAction,
    ArgusObservation,
    env_name="argus_env",
    max_concurrent_envs=8,
)


@app.get("/", include_in_schema=False)
def home() -> RedirectResponse:
    return RedirectResponse("/docs")


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
