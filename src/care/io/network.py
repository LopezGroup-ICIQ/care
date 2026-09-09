import json
import gzip

from care import Intermediate, ReactionNetwork
from .utils import ChemJSONEncoder
from .species import intermediate_from_dict, intermediate_to_dict
from .reaction import reaction_from_dict, reaction_to_dict
from .surface import surface_from_dict, surface_to_dict

def network_to_dict(network: ReactionNetwork) -> dict:
    """Serializes the entire network using a reference-based approach."""
    inters_dict = {
        node.code: intermediate_to_dict(node) 
        for node in network.nodes 
        if isinstance(node, Intermediate)
    }
    
    serialized_reactions = []
    for rxn in network.reactions:
        rxn_data = reaction_to_dict(rxn)

        rxn_data["components"] = [
            [inter.code for inter in comp_set] 
            for comp_set in rxn.components
        ]
        
        serialized_reactions.append(rxn_data)

    return {
        "oc": network.oc,
        "surface": surface_to_dict(network.surface) if getattr(network, 'surface', None) else None,
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
        rxn = reaction_from_dict(rxn_data, registry=registry)
        rebuilt_reactions.append(rxn)

    crn = ReactionNetwork(reactions=rebuilt_reactions)
    crn.add_catalyst(surface_from_dict(data["surface"]) if data.get("surface") else None)
    return crn

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
