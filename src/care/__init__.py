import pathlib as plb


MODULEROOT = plb.Path(__file__).parent
DB_PATH = f"{MODULEROOT}/data/metal_surfaces.db"
MODEL_PATH = f"{MODULEROOT}/gnn/gamenet_uq_dim192"