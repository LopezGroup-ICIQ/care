import os
from typing import Union

from ase.db import connect
from ase.build import surface
from ase.constraints import FixAtoms
from ase.io import read
from mp_api.client import MPRester
from pymatgen.io.ase import AseAtomsAdaptor

from care import Surface

def parse_hkl_string(hkl_str):
    """
    Parse an hkl or hkil string where negative signs are denoted with 'm'.
    Examples:
        '111'     -> (1, 1, 1)
        '0001'    -> (0, 0, 1)
        '10m10'   -> (1, 0, 0)
        '10m11'   -> (1, 0, 1)
        '2m1m12'  -> (2, -1, 2)
    """
    def parse_index_block(s):
        """Convert a string like '2m1m1' into a list of integers: [2, -1, -1]"""
        result = []
        i = 0
        while i < len(s):
            if s[i] == 'm':
                result.append(-int(s[i + 1]))
                i += 2
            else:
                result.append(int(s[i]))
                i += 1
        return result

    indices = parse_index_block(hkl_str)

    if len(indices) == 3:
        return tuple(indices)  # Standard (hkl)
    elif len(indices) == 4:
        h, k, _, l = indices  # Drop i index
        return (h, k, l)
    else:
        raise ValueError(f"Invalid hkl string format: {hkl_str}")
    
def load_surface(metal: str = None,
                 hkl: Union[str, list[int]] = None,
                 mpid: str = None,
                 path: str = None,
                 bulk_path: str = None,
                 num_layers: Union[int, float] = 3,
                 xy_repeat: int = 1,
                 vacuum: float = 15.0) -> Surface:
    """
    Load catalyst surface. Four options:
    - metal and hkl. Get surface from the ASE database intergrated in CARE for metals.
    - mp-id and hkl. This option requires API Key for Materials Project.
    - path to VASP CONTCAR file. Load surface from the CONTCAR file.
    - bulk_path and hkl to generate a surface from a bulk structure provided in the bulk_path as a VASP CONTCAR file.

    Args:
        metal (str): Metal symbol (e.g., "Ag")
        hkl (str or list[int]): Miller index (e.g., "111" or [1, 1, 1])
        mp_id (str): Materials Project ID (e.g., "mp-1234")
        path (str): Path to VASP CONTCAR file representing a surface.
        bulk_path (str): Path to VASP CONTCAR file representing a bulk structure.
        num_layers (int or float): If integer, refers to number of equivalent layers in the slab.
            If float, refers to the minimum height of the material slab.
        xy_repeat (int): Number of times to repeat the slab in the x and y directions
        vacuum (float): Vacuum spacing in Angstroms. defaults to 10 Angstroms.

    Note:
        The database should contain a surface with the given metal and Miller index.
        For hcp metals, the Miller index should be in the form "hkil", negative indices
        should be written as "mh-kil" (e.g. "10m11" stands for 10-11).
    """
    if path:
        slab = read(path)
        return Surface(ase_atoms_slab=slab, facet=hkl, from_mp=False)
    if hkl is None:
        raise ValueError("Miller index hkl not provided.")
    if isinstance(hkl, list):
        if len(hkl) != 3 and not all(isinstance(i, int) for i in hkl):
            raise ValueError("Miller index hkl must be a list of length 3 integers.")
        h, k, l = hkl
    else:
        if not isinstance(hkl, str):
            raise ValueError("Miller index hkl must be a string or a list of integers.")
        h, k, l = parse_hkl_string(hkl)
    if bulk_path:
        bulk = read(bulk_path)
        if isinstance(num_layers, int):
            slab = surface(bulk, (h, k, l), num_layers, vacuum=0.0, periodic=False)
        elif isinstance(num_layers, float):
            layers = 1
            while True:
                slab = surface(bulk, (h, k, l), layers, vacuum=0.0, periodic=False)
                highest_z = max([atom.position[2] for atom in slab])
                if highest_z > num_layers:
                    break
                layers += 1
        else:
            raise ValueError("num_layers must be an int or float.")
        z = {atom.index:atom.position[2] for atom in slab}
        layers_z = list(set(z.values()))
        layers_z.sort()
        num_layers = len(layers_z)
        slab.set_constraint(FixAtoms(indices=[atom.index for atom in slab if atom.position[2] in layers_z[:int(num_layers/2)]]))
        xy_repeat = xy_repeat if xy_repeat else 3
        slab = slab.repeat((xy_repeat, xy_repeat, 1))
        delta_vacuum = vacuum if vacuum else 10.0
        slab.set_cell([slab.cell[0], slab.cell[1], slab.cell[2] + [0, 0, delta_vacuum]], scale_atoms=False)
        return Surface(ase_atoms_slab=slab, facet=hkl, from_mp=False)
    if mpid and not metal:
        if not os.environ.get("MP_API_KEY"):
            raise ValueError("Materials Project API key not set. Please set your MP_API_KEY environment variable.")
        with MPRester(os.environ.get("MP_API_KEY")) as mpr:
            bulk = mpr.get_structure_by_material_id(mpid, final=True, conventional_unit_cell=True)
            # bulk to ASE
            ase_adaptor = AseAtomsAdaptor()
            bulk = ase_adaptor.get_atoms(bulk, msonable=False)
        if isinstance(num_layers, int):
            slab = surface(bulk, (h, k, l), num_layers, vacuum=0.0, periodic=False)
        elif isinstance(num_layers, float):
            layers = 1
            while True:
                slab = surface(bulk, (h, k, l), layers, vacuum=0.0, periodic=False)
                highest_z = max([atom.position[2] for atom in slab])
                if highest_z > num_layers:
                    break
                layers += 1
        else:
            raise ValueError("num_layers must be an int or float.")
        z = {atom.index:atom.position[2] for atom in slab}
        layers_z = list(set(z.values()))
        layers_z.sort()
        num_layers = len(layers_z)
        slab.set_constraint(FixAtoms(indices=[atom.index for atom in slab if atom.position[2] in layers_z[:int(num_layers/2)]]))
        xy_repeat = xy_repeat if xy_repeat else 3
        slab = slab.repeat((xy_repeat, xy_repeat, 1))
        delta_vacuum = vacuum if vacuum else 10.0
        slab.set_cell([slab.cell[0], slab.cell[1], slab.cell[2] + [0, 0, delta_vacuum]], scale_atoms=False)
        return Surface(ase_atoms_slab=slab, facet=hkl, from_mp=True)
    elif not mpid and metal:
        metal_db = connect(DB_PATH)
        metal_structure = f"{METAL_STRUCT_DICT[metal]}({hkl})"
        try:
            surface_ase = metal_db.get_atoms(
                calc_type="surface", metal=metal, facet=metal_structure, add_additional_information=True
            )
        except:
            # Generate surface from scratch (possible with current implementation!!!)
            raise ValueError(f"{metal} surface {metal_structure} not found in the database.")
        return Surface(ase_atoms_slab=surface_ase, facet=hkl, from_mp=False)

from care.evaluators.energy_estimator import IntermediateEnergyEstimator, ReactionEnergyEstimator
from care.evaluators.gamenet_uq import GameNetUQInter, GameNetUQRxn
from care.evaluators.ocp import OCPIntermediateEvaluator, OCPReactionEvaluator
from care.evaluators.mace import MACEIntermediateEvaluator, MACEReactionEvaluator
from care.evaluators.petmad import PETMADIntermediateEvaluator, PETMADReactionEvaluator
from care.evaluators.orb import ORBIntermediateEvaluator, ORBReactionEvaluator
from care.evaluators.sevennet import SevenNetIntermediateEvaluator, SevenNetReactionEvaluator
from care.evaluators.gamenet_uq import DB_PATH, METAL_STRUCT_DICT  # here for a reason

eval_dict = {
    "gamenetuq": (GameNetUQInter, GameNetUQRxn),
    "ocp": (OCPIntermediateEvaluator, OCPReactionEvaluator),
    "mace": (MACEIntermediateEvaluator, MACEReactionEvaluator),
    "petmad": (PETMADIntermediateEvaluator, PETMADReactionEvaluator),
    "orb": (ORBIntermediateEvaluator, ORBReactionEvaluator),
    "sevennet": (SevenNetIntermediateEvaluator, SevenNetReactionEvaluator),
}

def get_available_evaluators():
    """
    Show available energy evaluators in CARE.
    """
    return list(eval_dict.keys())

def load_inter_evaluator(model: str, surface, **kwargs) -> IntermediateEnergyEstimator:
    """
    Load the intermediate evaluator.

    Args:
        name (str): The name of the intermediate evaluator.

    Returns:
        IntermediateEnergyEstimator: The intermediate evaluator.
    """
    return eval_dict[model][0](surface, **kwargs)

def load_reaction_evaluator(model: str, intermediates, **kwargs) -> ReactionEnergyEstimator:
    """
    Load the reaction evaluator.

    Args:
        name (str): The name of the reaction evaluator.

    Returns:
        ReactionEnergyEstimator: The reaction evaluator.
    """
    return eval_dict[model][1](intermediates, **kwargs)

__all__ = [
    "IntermediateEnergyEstimator",
    "ReactionEnergyEstimator",
    "load_inter_evaluator",
    "load_reaction_evaluator",
    "load_surface",
]
