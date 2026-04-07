"""FastAPI application for ARGUS."""

from __future__ import annotations

import argparse

from fastapi.responses import HTMLResponse
from openenv.core.env_server.http_server import create_app

try:
    from ..models import ArgusAction, ArgusObservation
    from .argus_env_environment import ArgusEnvironment
except ImportError:  # pragma: no cover - direct source-tree execution
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
def home() -> HTMLResponse:
        return HTMLResponse(
                """
                <!doctype html>
                <html lang="en">
                <head>
                    <meta charset="utf-8" />
                    <meta name="viewport" content="width=device-width, initial-scale=1" />
                    <title>ARGUS</title>
                    <style>
                        :root {
                            color-scheme: dark;
                            --bg: #0f1117;
                            --panel: #171b26;
                            --text: #e7ecf5;
                            --muted: #aab4c5;
                            --accent: #f6c177;
                            --accent-2: #8bd5ca;
                        }
                        * { box-sizing: border-box; }
                        body {
                            margin: 0;
                            min-height: 100vh;
                            display: grid;
                            place-items: center;
                            background:
                                radial-gradient(circle at top, rgba(139, 213, 202, 0.12), transparent 34%),
                                radial-gradient(circle at bottom right, rgba(246, 193, 119, 0.12), transparent 28%),
                                var(--bg);
                            color: var(--text);
                            font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
                            padding: 24px;
                        }
                        main {
                            width: min(760px, 100%);
                            background: linear-gradient(180deg, rgba(23, 27, 38, 0.95), rgba(17, 20, 28, 0.96));
                            border: 1px solid rgba(255, 255, 255, 0.08);
                            border-radius: 20px;
                            padding: 32px;
                            box-shadow: 0 20px 70px rgba(0, 0, 0, 0.35);
                        }
                        .eyebrow {
                            letter-spacing: 0.18em;
                            text-transform: uppercase;
                            color: var(--accent-2);
                            font-size: 12px;
                            margin-bottom: 10px;
                        }
                        h1 {
                            margin: 0 0 12px;
                            font-size: clamp(2rem, 4vw, 3.5rem);
                            line-height: 1.05;
                        }
                        p { color: var(--muted); line-height: 1.65; font-size: 1rem; }
                        .cta {
                            display: inline-flex;
                            align-items: center;
                            justify-content: center;
                            gap: 10px;
                            margin-top: 12px;
                            padding: 14px 20px;
                            border-radius: 999px;
                            background: linear-gradient(135deg, #f6c177, #8bd5ca);
                            color: #0f1117;
                            font-weight: 700;
                            letter-spacing: 0.08em;
                            text-transform: uppercase;
                            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.22);
                        }
                        .cta:hover {
                            text-decoration: none;
                            transform: translateY(-1px);
                        }
                        .grid {
                            display: grid;
                            gap: 12px;
                            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
                            margin-top: 24px;
                        }
                        .card {
                            background: rgba(255, 255, 255, 0.03);
                            border: 1px solid rgba(255, 255, 255, 0.08);
                            border-radius: 16px;
                            padding: 16px;
                        }
                        .card strong { display: block; margin-bottom: 6px; color: var(--accent); }
                        a { color: var(--accent-2); text-decoration: none; }
                        a:hover { text-decoration: underline; }
                    </style>
                </head>
                <body>
                    <main>
                        <div class="eyebrow">OpenEnv Space</div>
                        <h1>ARGUS</h1>
                        <p>
                            ML evaluation integrity environment for missing baselines, cherry-picked ablations,
                            and benchmark contamination checks.
                        </p>
                        <p>
                            Use <strong>POST /reset</strong> to start an episode and <strong>POST /step</strong> to submit an action.
                            The full API is in the main page below.
                        </p>
                        <a class="cta" href="/docs">MAIN PAGE LINK</a>
                        <div class="grid">
                            <div class="card"><strong>Task types</strong>easy, medium, hard</div>
                            <div class="card"><strong>Deployment</strong>Hugging Face Docker Space</div>
                            <div class="card"><strong>Validation</strong>deterministic grading</div>
                        </div>
                    </main>
                </body>
                </html>
                """.strip(),
        )


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
