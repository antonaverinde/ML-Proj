import os

def set_load_path(sys: str) -> str:
    paths = {
        "Thermo10": '',
        "Linux": '',
        "Windows": r"",
        "GPU": ''
    }
    return paths.get(sys, paths["GPU"])


def set_base_path(sys: str) -> str:

    paths = {
        "Thermo10": '',
        "Linux": '',
        "Windows": r"",
        "GPU": ''
    }
    return paths.get(sys, paths["GPU"])


def get_full_load_path(sys: str, subfolder_name: str) -> str:
    return os.path.join(set_load_path(sys), subfolder_name)
