import os

from ase.db import connect

from care import Surface
from care.constants import METAL_STRUCT_DICT

def load_surface(db_path: str, 
                 metal: str, 
                 hkl: str) -> Surface:
    """
    Load surface from ASE database.

    Args:
        db_path (str): Path to the ASE database.
        metal (str): Metal symbol (e.g., "Ag")
        hkl (str): Miller index (e.g., "111", "0001")

    Note:
        The database should contain a surface with the given metal and Miller index.
        For hcp metals, the Miller index should be in the form "hkil", negative indices
        should be written as "mh-kil" (e.g. "10m11" stands for 10-11).
    """
    metal_db = connect(os.path.abspath(db_path))
    metal_structure = f"{METAL_STRUCT_DICT[metal]}({hkl})"
    try:
        surface_ase = metal_db.get_atoms(
            calc_type="surface", metal=metal, facet=metal_structure)
    except:
        raise ValueError(f"Surface {metal_structure} not found in the database.")

    return Surface(surface_ase, hkl)