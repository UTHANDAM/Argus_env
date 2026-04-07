"""
ARGUS — FastAPI Server
======================
Wires the ArgusEnvEnvironment into the OpenEnv HTTP server.
create_app() handles all routing (/reset, /step, /state, /health).

Design decision: try/except import pattern handles both package mode
(when installed as argus_env) and direct execution (python server/app.py).
"""

from openenv.core.env_server import create_app

try:
    from ..models import ArgusAction, ArgusObservation
    from .argus_env_environment import ArgusEnvEnvironment
except ImportError:
    from models import ArgusAction, ArgusObservation
    from server.argus_env_environment import ArgusEnvEnvironment

app = create_app(
    ArgusEnvEnvironment,
    ArgusAction,
    ArgusObservation,
    env_name="argus_env"
)

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()