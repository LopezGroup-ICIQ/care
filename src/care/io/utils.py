import json
import numpy as np
from ase import Atoms
import ase.constraints as ase_constraints
from ase.constraints import dict2constraint 

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
    
def serialize_complex_data(item):
    """Recursively handles ASE Atoms, Constraints, and numpy types."""
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

    if isinstance(item, dict):
        return {k: serialize_complex_data(v) for k, v in item.items()}
    
    if isinstance(item, (list, tuple)):
        return [serialize_complex_data(i) for i in item]
    
    return item

def deserialize_complex_data(item):
    if isinstance(item, dict):
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