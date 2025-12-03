import os
from typing import Union

from ase import Atoms
from ase.db import connect
from ase.build import surface
from ase.constraints import FixAtoms
from ase.io import read
from mp_api.client import MPRester
from pymatgen.io.ase import AseAtomsAdaptor
import numpy as np

from care.constants import METAL_STRUCT_DICT, METAL_SURFACES_DB_PATH


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
    
def bottom_half_indices(slab: Atoms) -> np.ndarray[int]:
    z = slab.positions[:, 2]
    sorted_indices = np.argsort(z)
    n = len(slab)
    bottom_half = sorted_indices[: n // 2]
    return bottom_half
    
def load_surface(metal: str = None,
                 hkl: Union[str, list[int]] = None,
                 mp_id: str = None,
                 mp_api_key: str = None,
                 surface_path: str = None,
                 bulk_path: str = None,
                 num_layers: Union[int, float] = 3,
                 xy_repeat: int = 1,
                 vacuum: float = 15.0) -> "Surface":
    if metal and hkl:
        return Surface.from_metal_db(metal, hkl)
    elif bulk_path:
        return Surface.from_bulk_poscar(bulk_path, hkl, num_layers=num_layers, xy_repeat=xy_repeat, vacuum=vacuum)
    elif mp_id:
        return Surface.from_mp(mp_id, hkl, mp_api_key=mp_api_key, num_layers=num_layers, xy_repeat=xy_repeat, vacuum=vacuum)
    elif surface_path:
        return Surface.from_poscar(surface_path)
    else:
        raise ValueError("At least one of metal and hkl, bulk_path, mp_id, or surface_path must be provided to load a Surface object.")


class Surface:
    """
    Class for representing catalyst surfaces.
    """

    def __init__(
        self,
        ase_atoms_slab: Atoms,
        facet: str = None,
        mp_id: str = None,
    ):
        self.slab = ase_atoms_slab
        try:
            self.metal = ase_atoms_slab.get_chemical_formula()[:2] if not mp_id else ase_atoms_slab.get_chemical_formula()
            self.crystal_structure = METAL_STRUCT_DICT[self.metal] if not mp_id else "Unknown"
        except:
            self.metal = ase_atoms_slab.get_chemical_formula()
            self.crystal_structure = "Unknown"
        self.facet = facet
        self.num_atoms = len(ase_atoms_slab)
        self.mp_id = mp_id 
        self.energy = None
        self.slab.new_array("atom_tags", [0] * len(self.slab), dtype=int)  # 0=surface atom, 1=adsorbate atom

    def __repr__(self) -> str:
        return f"{self.metal}({self.facet})"
    
    @classmethod
    def from_mp(cls, 
                mp_id: str, 
                hkl: Union[str, list[int]], 
                mp_api_key: str,
                num_layers: Union[int, float] = 3,
                xy_repeat: int = 1, 
                vacuum: float = 15.0) -> "Surface":
        """
        Create a Surface object from a bulk in the Materials Project database.
        """
        os.environ["MP_API_KEY"] = mp_api_key
        with MPRester(os.environ.get("MP_API_KEY")) as mpr:
            bulk = mpr.get_structure_by_material_id(mp_id, final=True, conventional_unit_cell=True)
            ase_adaptor = AseAtomsAdaptor()
            bulk = ase_adaptor.get_atoms(bulk, msonable=False)
        return cls.from_bulk_poscar(bulk, hkl, num_layers=num_layers, xy_repeat=xy_repeat, vacuum=vacuum, mp_id=mp_id)

    @classmethod
    def from_bulk_poscar(cls, 
                         bulk_poscar_path: Union[str, Atoms], 
                         hkl: Union[str, list[int]], 
                         num_layers: Union[int, float] = 3,
                         xy_repeat: int = 1, 
                         vacuum: float = 15.0, 
                         mp_id: str = None) -> "Surface":
        """
        Create a Surface object from a bulk material stored as POSCAR.
        """
        if isinstance(hkl, list):
            if len(hkl) != 3 and not all(isinstance(i, int) for i in hkl):
                raise ValueError("Miller index hkl must be a list of length 3 integers.")
            h, k, l = hkl
        else:
            if not isinstance(hkl, str):
                raise ValueError("Miller index hkl must be a string or a list of integers.")
            h, k, l = parse_hkl_string(hkl)
        bulk = read(bulk_poscar_path, format="vasp") if isinstance(bulk_poscar_path, str) else bulk_poscar_path

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
        slab.set_constraint(FixAtoms(indices=bottom_half_indices(slab)))  # Fix bottom half of the slab
        slab = slab.repeat((xy_repeat, xy_repeat, 1))
        slab.set_cell([slab.cell[0], slab.cell[1], slab.cell[2] + [0, 0, vacuum]], scale_atoms=False)
        return cls(ase_atoms_slab=slab, facet=hkl, mp_id=mp_id)

    @classmethod
    def from_poscar(cls, surface_poscar_path: str) -> "Surface":
        """
        Create a Surface object from an ASE Atoms object and a facet string.
        """
        if surface_poscar_path:
            slab = read(surface_poscar_path, format="vasp")
            return cls(ase_atoms_slab=slab)
        else:
            raise ValueError("Surface POSCAR path is required to create a Surface object.")

    @classmethod
    def from_metal_db(cls, 
                      metal: str, 
                      hkl: Union[str, list[int]]) -> "Surface":
        """
        Create a Surface object loading it from the ASE database for 
        metal surfaces in CARE.
        """
        metal_db = connect(METAL_SURFACES_DB_PATH)
        metal_structure = f"{METAL_STRUCT_DICT[metal]}({hkl})"
        try:
            surface_ase = metal_db.get_atoms(
                calc_type="surface", metal=metal, facet=metal_structure, add_additional_information=True
            )
            surface_ase.set_constraint(FixAtoms(indices=bottom_half_indices(surface_ase)))
        except:
            raise ValueError(f"{metal} surface {metal_structure} not in the database. Generate it from the Materials Project with Surface.from_mp().")
        return cls(ase_atoms_slab=surface_ase, facet=hkl)

    @property
    def num_layers(self) -> int:
        z = {atom.index: round(atom.position[2], 1) for atom in self.slab}
        layers_z = list(set(z.values()))
        return len(layers_z)

    @property
    def slab_height(self) -> float:
        z_atoms = self.slab.get_positions()[:, 2]
        return max(z_atoms)

    @property
    def vacuum_height(self) -> float:
        return self.slab.get_cell()[2,2] - self.slab_height

    @property
    def slab_diag(self) -> float:
        a, b, _ = self.slab.get_cell()
        return np.linalg.norm(a + b)

    @property
    def shortest_side(self) -> float:
        a, b, _ = self.slab.get_cell()
        return min(np.linalg.norm(a), np.linalg.norm(b))

    @property
    def area(self) -> float:
        """
        Calculate area in Angstrom^2 of the surface.
        """
        a, b, _ = self.slab.get_cell()
        return np.linalg.norm(np.cross(a, b))
    
    @property
    def fixed_atoms(self) -> list[int]:
        """
        Get indices of fixed atoms in the slab.
        """
        for constraint in self.slab._get_constraints():
            if isinstance(constraint, FixAtoms):
                return list(constraint.index)
        return []
    
    @property
    def surface_atoms(self) -> list[int]:
        """
        Get indices of surface atoms in the slab.
        """
        fixed = set(self.fixed_atoms)
        return [atom.index for atom in self.slab if atom.index not in fixed]
