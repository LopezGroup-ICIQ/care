import json

from care import Surface
from .utils import serialize_complex_data, deserialize_complex_data, ChemJSONEncoder

def surface_to_dict(surface: Surface) -> dict:
    return {
        "slab": serialize_complex_data(surface.slab),
        "facet": surface.facet,
        "mp_id": surface.mp_id,
    }

def surface_from_dict(data: dict) -> Surface:
    slab = deserialize_complex_data(data.get("slab"))
    facet = data.get("facet", None)
    mp_id = data.get("mp_id", None)
    return Surface(ase_atoms_slab=slab, facet=facet, mp_id=mp_id)

def save_surface(surface: Surface, path):
    """Saves a Surface object to a JSON file."""
    data = surface_to_dict(surface)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, cls=ChemJSONEncoder, indent=4)

def load_surface(path: str) -> Surface:
    """Loads a Surface object from a JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return surface_from_dict(data)