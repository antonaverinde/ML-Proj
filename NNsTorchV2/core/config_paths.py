import os
from pathlib import Path


def _load_dotenv() -> None:
    """Load .env from project root (2 levels up) without overwriting existing env vars."""
    env_path = Path(__file__).parent.parent / '.env'
    if not env_path.exists():
        return
    with env_path.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            key, value = line.split('=', 1)
            os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()

_LOAD_PATH_KEYS = {
    'GPU':      'LOAD_PATH_GPU',
    'Thermo10': 'LOAD_PATH_THERMO10',
    'Linux':    'LOAD_PATH_LINUX',
    'Windows':  'LOAD_PATH_WINDOWS',
}

_BASE_PATH_KEYS = {
    'GPU':      'BASE_PATH_GPU',
    'Thermo10': 'BASE_PATH_THERMO10',
    'Linux':    'BASE_PATH_LINUX',
    'Windows':  'BASE_PATH_WINDOWS',
}


def _resolve_path(sys: str, key_map: dict, label: str) -> str:
    """Look up path from env, falling back to GPU key; raise if not set."""
    env_key = key_map.get(sys, key_map['GPU'])
    value = os.environ.get(env_key)
    if value is None:
        gpu_key = key_map['GPU']
        value = os.environ.get(gpu_key)
        if value is None:
            raise EnvironmentError(
                f"{label}: env var '{env_key}' (and fallback '{gpu_key}') are not set. "
                f"Copy .env.example to .env and fill in your paths."
            )
    return value


def set_load_path(sys: str) -> str:
    """Return the measurement data root for the given system name."""
    return _resolve_path(sys, _LOAD_PATH_KEYS, 'set_load_path')


def set_base_path(sys: str) -> str:
    """Return the project base path for the given system name."""
    return _resolve_path(sys, _BASE_PATH_KEYS, 'set_base_path')


def get_full_load_path(sys: str, subfolder_name: str) -> str:
    """Join set_load_path with a subfolder."""
    return os.path.join(set_load_path(sys), subfolder_name)
