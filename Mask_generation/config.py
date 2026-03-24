"""
Configuration module for loading environment variables.
Handles path loading from .env file with validation and error handling.

This module loads paths from a .env file to keep sensitive file paths
out of version control when publishing this code as open source.

Environment Variables:
    BINARY_MASKS_BASE: Base directory for all mask processing
    IR_DATA_BASE: Base directory for IR measurement data
    PROJECT_BASE: (Optional) Parent project directory

Usage:
    from config import BINARY_MASKS_PATH, IR_DATA_PATH

    image_path = BINARY_MASKS_PATH / "subfolder" / "image.jpg"
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from the directory where this config.py is located
config_dir = Path(__file__).parent
env_path = config_dir / '.env'
load_dotenv(dotenv_path=env_path)


def get_path(env_var_name, required=True):
    """
    Get path from environment variable with validation.

    Args:
        env_var_name: Name of environment variable
        required: Whether the path is required (raises error if missing)

    Returns:
        Path object or None

    Raises:
        ValueError: If required path is not set
    """
    path_str = os.getenv(env_var_name)

    if path_str is None:
        if required:
            raise ValueError(
                f"Environment variable '{env_var_name}' is not set.\n"
                f"Please copy .env.example to .env and configure your paths.\n"
                f"Expected location: {config_dir / '.env'}"
            )
        return None

    return Path(path_str)


# Pre-load commonly used paths
try:
    BINARY_MASKS_PATH = get_path('BINARY_MASKS_BASE', required=True)
    IR_DATA_PATH = get_path('IR_DATA_BASE', required=True)
    PROJECT_PATH = get_path('PROJECT_BASE', required=False)

    # Print confirmation (can be disabled by setting QUIET=1 in .env)
    if os.getenv('QUIET') != '1':
        print("[OK] Configuration loaded successfully")
        print(f"  BINARY_MASKS_PATH: {BINARY_MASKS_PATH}")
        print(f"  IR_DATA_PATH: {IR_DATA_PATH}")
        if PROJECT_PATH:
            print(f"  PROJECT_PATH: {PROJECT_PATH}")

except ValueError as e:
    print(f"\n[ERROR] Configuration Error: {e}")
    print("\nPlease set up your .env file:")
    print("1. Copy .env.example to .env")
    print(f"2. Edit .env with your local paths")
    print(f"   Location: {config_dir}")
    raise
