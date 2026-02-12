import gzip
import json
import numpy as np
from ase import Atoms
import ase.constraints as ase_constraints
from ase.constraints import dict2constraint 
from rdkit import Chem

from care import Intermediate, ElementaryReaction, ReactionNetwork, Surface
import care.crn.templates as rxn_module

import json
import numpy as np

class ChemJSONEncoder(json.JSONEncoder):
    """Custom encoder to handle NumPy types within ASE data."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.int64, np.int32, np.int16)):
            return int(obj)
        if isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        return super().default(obj)

def save_intermediate(inter: Intermediate, path):
    """Saves an Intermediate object to a JSON file."""
    data = intermediate_to_dict(inter)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, cls=ChemJSONEncoder, indent=4)

def load_intermediate(filename: str) -> Intermediate:
    """Loads an Intermediate object from a JSON file."""
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return intermediate_from_dict(data)

def intermediate_to_dict(inter: Intermediate) -> dict:
    if inter.phase in ("gas", "ads"):
        molecule_block = Chem.MolToMolBlock(inter.rdkit, forceV3000=True)
    else:
        molecule_block = serialize_complex_data(inter.molecule)
    return {
        "code": inter.code,
        "phase": inter.phase,
        "molecule": molecule_block,
        "ads_configs": serialize_complex_data(inter.ads_configs)
    }

def intermediate_from_dict(data: dict) -> Intermediate:
    """Reconstructs the object, handling the nested ads_configs."""
    phase = data.get("phase")
    mol_text = data.get("molecule")
    if phase in ("gas", "ads"):
        molecule = Chem.MolFromMolBlock(mol_text, removeHs=False, sanitize=True)
    else:
        molecule = Atoms()
    inter = Intermediate(
        code=data.get("code"),
        molecule=molecule,
        phase=data.get("phase")
    )
    raw_ads = data.get("ads_configs", {})
    inter.ads_configs = deserialize_complex_data(raw_ads) if raw_ads else {}
    return inter

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

def serialize_complex_data(item):
    """Recursively handles ASE Atoms, Constraints, and numpy types."""
    
    # Handle ASE Objects with a .todict() method (Atoms and most Constraints)
    if hasattr(item, 'todict'):
        name = item.__class__.__name__
        return {
            f"__ase_{name}__": True, 
            "data": serialize_complex_data(item.todict())
        }

    if isinstance(item, (np.floating, float)):
        return float(item)
    if isinstance(item, (np.integer, int)):
        return int(item)
    if isinstance(item, np.ndarray):
        return item.tolist()

    # Recursive dictionary walk
    if isinstance(item, dict):
        return {k: serialize_complex_data(v) for k, v in item.items()}
    
    # Recursive list/tuple walk
    if isinstance(item, (list, tuple)):
        return [serialize_complex_data(i) for i in item]
    
    return item

def normalize_ase_dict(dct, class_name):
    """Ensures a dictionary has the 'name' and 'kwargs' structure ASE expects."""
    if not isinstance(dct, dict):
        return dct
        
    # If it's already in the correct format, just return it
    if "name" in dct and "kwargs" in dct:
        return dct

    # Otherwise, rebuild it
    actual_name = dct.get("name", class_name)
    # Extract everything that isn't metadata into kwargs
    metadata_keys = ["name", "__ase_Atoms__", "__ase_FixAtoms__", "data"]
    kwargs = {k: v for k, v in dct.items() if k not in metadata_keys}
    
    return {"name": actual_name, "kwargs": kwargs}

def deserialize_complex_data(item):
    if isinstance(item, dict):
        # Handle ASE wrappers
        special_key = next((k for k in item.keys() if k.startswith("__ase_")), None)
        
        if special_key:
            class_name = special_key.replace("__ase_", "").replace("__", "")
            raw_data = item["data"]
            
            # Case A: Rebuilding Atoms
            if class_name == "Atoms":
                # Convert list to arrays for ASE
                for key, value in raw_data.items():
                    if isinstance(value, list):
                        raw_data[key] = np.array(value)
                
                # Normalize nested constraints before Atoms.fromdict sees them
                if "constraints" in raw_data and raw_data["constraints"]:
                    normalized_constraints = []
                    for c in raw_data["constraints"]:
                        # If the constraint itself was wrapped in our __ase_ format
                        if isinstance(c, dict) and any(k.startswith("__ase_") for k in c.keys()):
                            c_key = next(k for k in c.keys() if k.startswith("__ase_"))
                            c_name = c_key.replace("__ase_", "").replace("__", "")
                            normalized_constraints.append(normalize_ase_dict(c["data"], c_name))
                        else:
                            normalized_constraints.append(c)
                    raw_data["constraints"] = normalized_constraints
                
                return Atoms.fromdict(raw_data)
            
            # Case B: Rebuilding loose Constraints
            if hasattr(ase_constraints, class_name):
                # Recursively clean the inner data (like indices)
                clean_inner = deserialize_complex_data(raw_data)
                normalized = normalize_ase_dict(clean_inner, class_name)
                return dict2constraint(normalized)

        # Recursive walk
        return {k: deserialize_complex_data(v) for k, v in item.items()}
    
    elif isinstance(item, list):
        return [deserialize_complex_data(i) for i in item]
    
    return item

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
            "is": rxn.e_is,
            "ts": rxn.e_ts,
            "fs": rxn.e_fs,
            "rxn": rxn.e_rxn,
            "act": rxn.e_act,
        },
        "kinetics": {
            "k_dir": rxn.k_dir,
            "k_rev": rxn.k_rev,
            "k_eq": rxn.k_eq,
        },
        "extra_intermediates_codes": list(rxn.extra_intermediates.keys()),
        "neb_images": serialize_complex_data(rxn.neb_images),
        "neb_energies": serialize_complex_data(rxn.neb_energies),
        "is_atoms": serialize_complex_data(rxn.is_atoms),
        "fs_atoms": serialize_complex_data(rxn.fs_atoms),
    }

def reaction_from_dict(data: dict, registry: dict) -> ElementaryReaction:
    """Reconstructs the specific child class instance."""
    
    # 1. Map components using the registry
    obj_components = [
        [registry[code] for code in comp_list]
        for comp_list in data["components"]
    ]
    
    # 2. Dynamic Class Lookup
    class_name = data.get("class_name", "ElementaryReaction")
    cls = getattr(rxn_module, class_name, rxn_module.ElementaryReaction)
    
    # 3. Instantiate child class
    rxn = cls(
        components=obj_components,
        r_type=data["r_type"],
        stoic=data["stoic"]
    )
    
    # 4. Restore Energies
    energies = data.get("energies", {})
    rxn.e_is = tuple(energies.get("is")) if energies.get("is") else None
    rxn.e_ts = tuple(energies.get("ts")) if energies.get("ts") else None
    rxn.e_fs = tuple(energies.get("fs")) if energies.get("fs") else None
    rxn.e_rxn = tuple(energies.get("rxn")) if energies.get("rxn") else None
    rxn.e_act = tuple(energies.get("act")) if energies.get("act") else None
    
    # 5. Restore Kinetics
    kinetics = data.get("kinetics", {})
    rxn.k_dir = kinetics.get("k_dir")
    rxn.k_rev = kinetics.get("k_rev")
    rxn.k_eq = kinetics.get("k_eq")

    # 6. Restore Complex/Nested data
    rxn.neb_images = deserialize_complex_data(data.get("neb_images"))
    rxn.neb_energies = deserialize_complex_data(data.get("neb_energies"))
    rxn.is_atoms = deserialize_complex_data(data.get("is_atoms"))
    rxn.fs_atoms = deserialize_complex_data(data.get("fs_atoms"))
    
    # 7. Restore extra intermediates
    extra_codes = data.get("extra_intermediates_codes", [])
    rxn.extra_intermediates = {
        code: registry[code] for code in extra_codes if code in registry
    }
    
    return rxn

def network_to_dict(network: ReactionNetwork) -> dict:
    """Serializes the entire network using a reference-based approach."""
    
    # Collect all unique intermediates into a registry (code -> dict)
    inters_dict = {
        node.code: intermediate_to_dict(node) 
        for node in network.nodes 
        if isinstance(node, Intermediate)
    }
    
    # Serialize reactions using species codes instead of full objects
    serialized_reactions = []
    for rxn in network.reactions:
        rxn_data = reaction_to_dict(rxn)
        rxn_data["components"] = [
            [inter.code for inter in comp_set] 
            for comp_set in rxn.components
        ]
        rxn_data["extra_intermediates"] = list(rxn.extra_intermediates.keys())
        
        serialized_reactions.append(rxn_data)

    return {
        "oc": network.oc,
        "surface": surface_to_dict(network.surface) if network.surface else None,
        "species_registry": inters_dict,
        "reactions": serialized_reactions
    }

def network_from_dict(data: dict) -> ReactionNetwork:
    """Reconstructs the ReactionNetwork from a reference-based dictionary."""

    registry = {
        code: intermediate_from_dict(inter_data)
        for code, inter_data in data["species_registry"].items()
    }
    
    rebuilt_reactions = []
    for rxn_data in data["reactions"]:
        codes_components = rxn_data["components"]
        obj_components = [
            [registry[code] for code in comp_list]
            for comp_list in codes_components
        ]

        rxn_data_copy = rxn_data.copy()
        rxn_data_copy["components"] = obj_components

        rxn = reaction_from_dict(rxn_data_copy, registry=registry)
        rebuilt_reactions.append(rxn)

    return ReactionNetwork(
        reactions=rebuilt_reactions,
        surface=surface_from_dict(data["surface"]) if data.get("surface") else None,
        oc=data.get("oc")
    )

def save_network(network: ReactionNetwork, filepath: str, compress=True):
    data = network_to_dict(network)
    if compress and not filepath.endswith('.gz'):
        filepath += '.gz'
    if filepath.endswith('.gz'):
        with gzip.open(filepath, 'wt', encoding='utf-8') as f:
            json.dump(data, f, cls=ChemJSONEncoder)
    else:
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, cls=ChemJSONEncoder, indent=4)
    print(f"ReactionNetwork saved to {filepath}")

def load_network(filepath):
    """Loads network, automatically detecting if it is gzipped."""
    if filepath.endswith('.gz'):
        with gzip.open(filepath, 'rt', encoding='utf-8') as f:
            data = json.load(f)
    else:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    return network_from_dict(data)
