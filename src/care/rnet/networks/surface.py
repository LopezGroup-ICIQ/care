from collections import defaultdict

from ase import Atoms
import numpy as np
from acat.adsorption_sites import SlabAdsorptionSites

from GAMERNet.rnet.networks.utils import metal_structure_dict

class Surface:
    """
    Class for representing transition metal surfaces models.
    """
    def __init__(self, 
                 ase_atoms_slab: Atoms,
                 facet: str, 
                 ):
        self.slab = ase_atoms_slab
        self.metal = ase_atoms_slab.get_chemical_formula()[:2]
        self.crystal_structure = metal_structure_dict[self.metal]
        self.facet = facet
        self.num_atoms = len(ase_atoms_slab)
        self.num_layers = self.get_num_layers()
        self.slab_height = self.get_slab_height()
        self.slab_diag = self.get_slab_diag()
        self.area = self.get_area()
        self.active_sites = self.find_active_sites()

    def __repr__(self) -> str:
        return f"{self.slab.get_chemical_formula()}({self.facet})"

    def get_num_layers(self) -> int:
        z = {atom.index:atom.position[2] for atom in self.slab}
        layers_z = list(set(z.values()))
        return len(layers_z)
    
    def get_slab_height(self) -> float:
        z_atoms = self.slab.get_positions()[:,2]
        return max(z_atoms)
    
    def get_slab_diag(self) -> float:
        a, b, _ = self.slab.get_cell()
        return np.linalg.norm(a + b)

    def get_area(self) -> float:
        """
        Calculate area in Angstrom^2 of the surface.
        """
        a, b, _ = self.slab.get_cell()
        return np.linalg.norm(np.cross(a, b))

    def find_active_sites(self) -> list[dict]:
        surf = self.crystal_structure + self.facet
        if self.facet == "10m10":
            surf += "h"
        tol_dict = defaultdict(lambda: 0.5)
        tol_dict["Cd"] = 1.5
        tol_dict["Co"] = 0.75
        tol_dict["Os"] = 0.75
        tol_dict["Ru"] = 0.75
        tol_dict["Zn"] = 1.25
        sas = SlabAdsorptionSites(self.slab,
                                  surface=surf, 
                                  tol=tol_dict[self.metal], 
                                  label_sites=True, 
                                  optimize_surrogate_cell=True)
        sas = sas.get_unique_sites()
        sas = [site for site in sas if site['position'][2] > 0.75 * self.slab_height]  
        return sas