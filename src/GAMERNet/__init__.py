import pathlib as plb


MODULEROOT = plb.Path(__file__).parent
DB_PATH = f"{MODULEROOT}/rnet/data/metal_surfaces.db"
H_PATH = f"{MODULEROOT}/rnet/data/CONTCAR"
MODEL_PATH = f"{MODULEROOT}/gnn_eads/models/GAME-Net"
DOS_PATH = f"{MODULEROOT}/gnn_eads/new_web/dockonsurf/dockonsurf.py"
DOCK_DATA = f"{MODULEROOT}/gnn_eads/new_web/adsurf/data"