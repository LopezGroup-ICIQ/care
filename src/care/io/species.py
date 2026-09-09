import json

from ase import Atoms
from rdkit import Chem

from care import Intermediate
import care.crn.intermediate as inter_module
from .utils import ChemJSONEncoder, serialize_complex_data, deserialize_complex_data

def save_intermediate(inter: Intermediate, path):
    data = intermediate_to_dict(inter)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, cls=ChemJSONEncoder, indent=4)

def load_intermediate(filename: str) -> Intermediate:
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return intermediate_from_dict(data)

def intermediate_to_dict(inter: Intermediate) -> dict:
    molecule_block = serialize_complex_data(getattr(inter, 'molecule', None))
        
    return {
        "class_name": inter.__class__.__name__,
        "code": inter.code,
        "phase": inter.phase,
        "molecule": molecule_block,
        "ads_configs": serialize_complex_data(getattr(inter, 'ads_configs', {})),
        "E": getattr(inter, 'E', None)
    }

def intermediate_from_dict(data: dict) -> Intermediate:
    mol_text = data.get("molecule")
    class_name = data.get("class_name", "Intermediate")
    molecule = deserialize_complex_data(mol_text) if mol_text else Atoms()

    cls = getattr(inter_module, class_name, Intermediate)

    if class_name == "SurfaceSite":
        inter = cls(code=data.get("code"))
        inter.E = data.get("E", None)
    else:
        inter = cls(code=data.get("code"), molecule=molecule)
    
    if class_name == "GasSpecies":
        inter.E = data.get("E", None)

    if class_name in ("AdsorbedSpecies"):
        raw_ads = data.get("ads_configs", {})
        inter.ads_configs = deserialize_complex_data(raw_ads) if raw_ads else {}
        
    return inter