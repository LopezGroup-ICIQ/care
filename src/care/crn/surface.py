from ase import Atoms
import numpy as np

from care.constants import METAL_STRUCT_DICT


class Surface:
    """
    Class for representing catalyst surfaces.
    """

    def __init__(
        self,
        ase_atoms_slab: Atoms,
        facet: str,
        from_mp: bool = False,
    ):
        self.slab = ase_atoms_slab
        try:
            self.metal = ase_atoms_slab.get_chemical_formula()[:2] if not from_mp else ase_atoms_slab.get_chemical_formula()
            self.crystal_structure = METAL_STRUCT_DICT[self.metal] if not from_mp else "Unknown"
        except:
            self.metal = ase_atoms_slab.get_chemical_formula()
            self.crystal_structure = "Unknown"
        self.facet = facet
        self.num_atoms = len(ase_atoms_slab)
        self.from_mp = from_mp

    def __repr__(self) -> str:
        return f"{self.metal}({self.facet})"

    @property
    def num_layers(self) -> int:
        z = {atom.index: atom.position[2] for atom in self.slab}
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
