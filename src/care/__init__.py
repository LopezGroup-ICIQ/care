import pathlib as plb


MODULEROOT = plb.Path(__file__).parent
DB_PATH = f"{MODULEROOT}/rnet/data/metal_surfaces.db"
MODEL_PATH = f"{MODULEROOT}/gnns/gamenet_uq"