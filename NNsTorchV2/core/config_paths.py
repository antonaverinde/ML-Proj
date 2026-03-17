import os

def set_load_path(sys: str) -> str:
    paths = {
        "Thermo10": '/home/aaverin/RZ-Dienste/Thermo_Daten-MX2/2025/2025-11-04-Av-ZIKA-Mirko-Taris-Hologen-2kw-measurements',
        "Linux": '/mnt/daten-mx2/2025/2025-11-04-Av-ZIKA-Mirko-Taris-Hologen-2kw-measurements',
        "Windows": r"\\Gfs01\\g71\\Thermo_Daten-MX2\\2025\\2025-11-04-Av-ZIKA-Mirko-Taris-Hologen-2kw-measurements",
        "GPU": '/home/aaverin/RZ-Dienste/hpc-user/aaverin/2025/2025-11-04-Av-ZIKA-Mirko-Taris-Hologen-2kw-measurements'
    }
    return paths.get(sys, paths["GPU"])


def set_base_path(sys: str) -> str:

    paths = {
        "Thermo10": '/home/aaverin/RZ-Dienste/Thermo-MX1/Mitarbeiter/Averin/Python/Sample4Fasseteil/Laser/Pulse/KerasKIv2',
        "Linux": '/mnt/thermo/Mitarbeiter/Averin/Python/Sample4Fasseteil/Laser/Pulse/KerasKIv2',
        "Windows": r"\\gfs03\\G33a\\Thermo-MX1\\Mitarbeiter\\Averin\\Python\\Sample4Fasseteil\\Laser\\Pulse\\KerasKIv2",
        "GPU": '/home/aaverin/RZ-Dienste/hpc-user/aaverin/Python/Pulse/KIprojV2_Claude'
    }
    return paths.get(sys, paths["GPU"])


def get_full_load_path(sys: str, subfolder_name: str) -> str:
    return os.path.join(set_load_path(sys), subfolder_name)
