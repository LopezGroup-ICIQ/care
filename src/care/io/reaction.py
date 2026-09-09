import json

from care import ElementaryReaction
import care.crn.templates as rxn_module
from .utils import serialize_complex_data, deserialize_complex_data, ChemJSONEncoder
from .species import intermediate_to_dict, intermediate_from_dict

def reaction_to_dict(rxn: "ElementaryReaction") -> dict:
    """Converts an ElementaryReaction into a serializable dictionary."""
    class_name = rxn.__class__.__name__
    
    serialized_components = []
    for comp_set in rxn.components:
        serialized_components.append([intermediate_to_dict(inter) for inter in comp_set])
    
    return {
        "class_name": class_name,
        "r_type": rxn.r_type,
        "stoic": rxn.stoic,
        "components": serialized_components,
        "energies": {
            "ts": getattr(rxn, "_e_ts", None), 
        },
        "kinetics": {
            "k_dir": rxn.k_dir,
            "k_rev": rxn.k_rev,
            "k_eq": rxn.k_eq,
            "rate": getattr(rxn, "rate", None)
        },
        "extra_intermediates_codes": list(rxn.extra_intermediates.keys()),
        "neb_images": serialize_complex_data(rxn.neb_images),
        "neb_energies": serialize_complex_data(rxn.neb_energies),
        "is_atoms": serialize_complex_data(rxn.is_atoms),
        "fs_atoms": serialize_complex_data(rxn.fs_atoms),
    }

def reaction_from_dict(data: dict, registry: dict) -> ElementaryReaction:
    """Reconstructs the specific child class instance."""
    
    # Map component codes strictly using the network registry as frozensets
    obj_components = [
        frozenset([registry[code] for code in comp_list])
        for comp_list in data["components"]
    ]
    
    # Dynamic Class Lookup
    class_name = data.get("class_name", "ElementaryReaction")
    cls = getattr(rxn_module, class_name, rxn_module.ElementaryReaction)
    
    # Instantiate child class
    rxn = cls(
        components=tuple(obj_components),
        r_type=data["r_type"],
        stoic=data.get("stoic")
    )
    
    # Restore explicit TS energy ONLY via the clean setter block
    energies = data.get("energies", {})
    rxn.e_ts = energies.get("ts")
    
    # Restore Kinetics
    kinetics = data.get("kinetics", {})
    rxn.k_dir = kinetics.get("k_dir")
    rxn.k_rev = kinetics.get("k_rev")
    rxn.k_eq = kinetics.get("k_eq")
    rxn.rate = kinetics.get("rate")

    # Restore Complex/Nested data
    rxn.neb_images = deserialize_complex_data(data.get("neb_images"))
    rxn.neb_energies = deserialize_complex_data(data.get("neb_energies"))
    rxn.is_atoms = deserialize_complex_data(data.get("is_atoms"))
    rxn.fs_atoms = deserialize_complex_data(data.get("fs_atoms"))
    
    # Restore extra intermediates
    extra_codes = data.get("extra_intermediates_codes", [])
    rxn.extra_intermediates = {
        code: registry[code] for code in extra_codes if code in registry
    }
    
    return rxn

def save_reaction(rxn: "ElementaryReaction", path: str) -> None:
    """Saves an ElementaryReaction object to a JSON file."""
    data = reaction_to_dict(rxn)
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, cls=ChemJSONEncoder, indent=4)

def load_reaction(path: str) -> "ElementaryReaction":
    """Loads an ElementaryReaction object from a JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # 1. Build a mini-registry from the standalone reaction's components
    registry = {}
    for comp_set in data.get("components", []):
        for inter_data in comp_set:
            if isinstance(inter_data, dict) and "code" in inter_data:
                registry[inter_data["code"]] = intermediate_from_dict(inter_data)
                
    # 2. Convert the nested dictionaries back into simple lists of codes 
    # to perfectly match the expectations of reaction_from_dict
    component_codes = []
    for comp_set in data.get("components", []):
        component_codes.append([inter_data["code"] for inter_data in comp_set])
        
    data["components"] = component_codes
    return reaction_from_dict(data, registry=registry)