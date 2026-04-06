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