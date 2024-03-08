from collections import defaultdict

import numpy as np
from acat.adsorption_sites import SlabAdsorptionSites
from ase import Atoms

from care import Intermediate
from care.constants import METAL_STRUCT_DICT


class Surface(Intermediate):
    """
    Class for representing transition metal surfaces models.
    """

    def __init__(
        self,
        ase_atoms_slab: Atoms = Atoms(),
        facet: str = None,
    ):
        super().__init__(code='*', molecule=Atoms(), is_surface=True, phase='surf')
        self.slab = ase_atoms_slab
        self.metal = ase_atoms_slab.get_chemical_formula()[:2]
        self.crystal_structure = METAL_STRUCT_DICT[self.metal]
        self.facet = facet
        self.num_atoms = len(ase_atoms_slab)

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
    def active_sites(self) -> list[dict]:
        surf = self.crystal_structure + self.facet
        if self.facet == "10m10":
            surf += "h"
        tol_dict = defaultdict(lambda: 0.5)
        tol_dict["Cd"] = 1.5
        tol_dict["Co"] = 0.75
        tol_dict["Os"] = 0.75
        tol_dict["Ru"] = 0.75
        tol_dict["Zn"] = 1.25
        if self.facet == "10m11" or (self.crystal_structure == "bcp" and self.facet in ("111", "100")):
            tol = 2.0
            sas = SlabAdsorptionSites(
                self.slab,
                surface=surf,
                tol=tol,
                label_sites=True)
        elif self.crystal_structure == "fcc" and self.facet == "110":
            tol = 1.5
            sas = SlabAdsorptionSites(
                self.slab,
                surface=surf,
                tol=tol,
                label_sites=True)
        else:
            sas = SlabAdsorptionSites(
                self.slab,
                surface=surf,
                tol=tol_dict[self.metal],
                label_sites=True,
                optimize_surrogate_cell=True,
            )
        sas = sas.get_unique_sites()
        sas = [site for site in sas if site["position"][2] > 0.65 * self.slab_height]
        return sas
