import pathlib as plb


MODULEROOT = plb.Path(__file__).parent
DB_PATH = f"{MODULEROOT}/rnet/data/metal_surfaces.db"
MODEL_PATH = f"{MODULEROOT}/gnns/gamenet_uq"
# DOS_PATH = f"{MODULEROOT}/gnn_eads/new_web/dockonsurf/dockonsurf.py"
DOCK_DATA = f"{MODULEROOT}/rnet/adsurf/data"