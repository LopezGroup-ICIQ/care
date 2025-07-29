import pathlib as pl

import numpy as np
from sklearn.preprocessing import OneHotEncoder


MODULEROOT = pl.Path(__file__).parent
MODEL_PATH = f"{MODULEROOT}/model"
DFT_DB_PATH = f"{MODULEROOT}/data/fg.db"

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

ONE_HOT_ENCODER_NODES = OneHotEncoder().fit(np.array(ADSORBATE_ELEMS + METALS).reshape(-1, 1))
ELEMENT_DOMAIN = list(ONE_HOT_ENCODER_NODES.categories_[0])

from care.evaluators.gamenet_uq.interface import GameNetUQInter, GameNetUQRxn

__all__ = [
    "GameNetUQInter",
    "GameNetUQRxn"]
