from collections import defaultdict

from ase import Atoms
import numpy as np
from acat.adsorption_sites import SlabAdsorptionSites
from pymatgen.core.periodic_table import Element
from pymatgen.core.structure import Structure
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.io.ase import AseAtomsAdaptor

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
        self.area = self.get_area()
        self.active_sites_dict_acat = self.find_active_sites_acat()
        self.active_sites_dict_pmg = self.find_active_sites_pmg()

    def __repr__(self) -> str:
        return f"{self.metal}({self.facet})"

    def get_num_layers(self) -> int:
        z = {atom.index:atom.position[2] for atom in self.slab}
        layers_z = list(set(z.values()))
        return len(layers_z)
    
    def get_slab_height(self) -> float:
        z_atoms = self.slab.get_positions()[:,2]
        return max(z_atoms)

    def get_area(self) -> float:
        """
        Calculate area in Angstrom^2 of the surface.
        """
        a, b, _ = self.slab.get_cell()
        return np.linalg.norm(np.cross(a, b))

    def find_active_sites_acat(self) -> list[dict]:
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
                                  label_sites=False, 
                                  optimize_surrogate_cell=True)
        sas = sas.get_unique_sites()
        sas = [site for site in sas if site['position'][2] > 0.75 * self.slab_height]  
        return sas
    
    def find_active_sites_pmg(self):
        """
        Get unique active sites of the surface with Pymatgen.
        """
        surface_pmg = AseAtomsAdaptor.get_structure(self.slab)
        surf_sites = AdsorbateSiteFinder(surface_pmg, selective_dynamics=True)
        most_active_sites = surf_sites.find_adsorption_sites()
        return most_active_sites