import pathlib as pl
MODULEROOT = pl.Path(__file__).parent
MODEL_PATH = f"{MODULEROOT}/dim192_5splits"
DB_PATH = f"{MODULEROOT}/data/metal_surfaces.db"
DFT_DB_PATH = f"{MODULEROOT}/data/FG2dataset.db"

METAL_STRUCT_DICT = {
    "Ag": "fcc",
    "Au": "fcc",
    "Cd": "hcp",
    "Co": "hcp",
    "Cu": "fcc",
    "Fe": "bcc",
    "Ir": "fcc",
    "Ni": "fcc",
    "Os": "hcp",
    "Pd": "fcc",
    "Pt": "fcc",
    "Rh": "fcc",
    "Ru": "hcp",
    "Zn": "hcp",
}

FACET_DICT = {
    "fcc": ["111", "110", "100"],
    "hcp": ["0001", "10m10", "10m11"],
    "bcc": ["111", "110", "100"],
}

METALS = [
    "Ag",
    "Au",
    "Cd",
    "Co",
    "Cu",
    "Fe",
    "Ir",
    "Ni",
    "Os",
    "Pd",
    "Pt",
    "Rh",
    "Ru",
    "Zn",
]

ADSORBATE_ELEMS = ["C", "H", "O", "N", "S"]

from care.evaluators.gamenet_uq.interface import GameNetUQInter, GameNetUQRxn

__all__ = [
    "GameNetUQInter",
    "GameNetUQRxn"]
