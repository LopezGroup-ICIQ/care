from abc import ABC
from typing import Union
from copy import deepcopy, copy
import warnings

from ase import Atoms
from ase.mep import NEB
from ase.optimize import BFGS, LBFGS
from torch.cuda import empty_cache

from care import Intermediate, ElementaryReaction, Surface, silent_context
from care.crn.intermediate import SurfaceSite, GasSpecies, AdsorbedSpecies
from care.adsorption import place_adsorbate
from care.crn.utils.graph import atoms_to_graph

class UniversalEvaluator(ABC):
    """
    Unified base class handling both species relaxations and reaction NEB calculations.
    """
    def __init__(
        self,
        # Electrochemical params
        T: float = 298.0,
        ref_electrode: str = "SHE",
        pH: float = 7.0,
        U: float = 0.0,
        # NEB params
        num_images: int = 3,
        climb: bool = True,
        k: float = 0.1,
        max_steps: int = 100,
        dx: float = 1.5,
        tol: float = 0.0,
        optimizer: str = "BFGS",
        parallel: bool = False,
        remove_rotation_and_translation: bool = True,
        neb_method: str = "aseneb",
        interpolation_method: str = "linear",
        allow_shared_calculator: bool = False,
        # Relaxation params
        fmax: float = 0.05,
        num_configs: int = 1,
        del_traj: bool = True,
        logfile: str = None,
        patience: int = 3,
        **kwargs
    ):
        self.T = T
        self.ref_electrode = ref_electrode
        self.pH = pH
        self.U = U

        # General simulation settings
        self.max_steps = max_steps
        self.fmax = fmax
        self.optimizer = optimizer
        self.optimizer_class = BFGS if optimizer == "BFGS" else LBFGS
        
        # TS-search settings
        self.num_images = num_images
        self.climb = climb
        self.k = k
        self.dx = dx
        self.tol = tol
        self.parallel = parallel
        self.remove_rotation_and_translation = remove_rotation_and_translation
        self.neb_method = neb_method
        self.interpolation_method = interpolation_method
        self.allow_shared_calculator = allow_shared_calculator

        # Relaxation settings
        self.num_configs = num_configs
        self.del_traj = del_traj
        self.logfile = logfile
        self.patience = patience * num_configs

    def __call__(self, x: Union[Intermediate, ElementaryReaction, Atoms], **kwargs) -> None:
        self.eval(x, **kwargs)

    def eval(self, x: Union[Intermediate, ElementaryReaction, Atoms, list], **kwargs) -> None:
        original_attrs = {}
        for key, value in kwargs.items():
            if hasattr(self, key):
                original_attrs[key] = getattr(self, key)
                setattr(self, key, value)
                
        try:
            if isinstance(x, Intermediate):
                self._eval_species(x, **kwargs)
            elif isinstance(x, ElementaryReaction):
                self._eval_reaction(x, **kwargs)
            elif isinstance(x, Atoms):
                self._eval_atoms(x, **kwargs)
            elif isinstance(x, list):
                if hasattr(self, "_eval_reaction_batch"):
                    self._eval_reaction_batch(x, **kwargs)
                else:
                    for item in x:
                        self.eval(item, **kwargs)
            else:
                raise TypeError("Input must be an Intermediate, ElementaryReaction, Atoms object, or a list.")
        finally:
            for key, value in original_attrs.items():
                setattr(self, key, value)

    def _eval_atoms(self, atoms: Atoms, **kwargs) -> None:
        """Raw ASE Atoms structural relaxation (Model-Agnostic)."""
        self._assign_calculator(atoms)
        opt = self.optimizer_class(atoms, logfile=self.logfile)
        opt.run(fmax=self.fmax, steps=self.max_steps)
        if self.del_traj:
            atoms.calc = None

    def _eval_gas_species(self, intermediate: GasSpecies, **kwargs) -> None:
        """Default: Relax gas molecule using the MLIP and store potential energy."""
        molec_eval = intermediate.molecule.copy()
        self._assign_calculator(molec_eval)
        opt = self.optimizer_class(molec_eval, logfile=self.logfile)
        opt.run(fmax=self.fmax, steps=self.max_steps)
        intermediate.E = molec_eval.get_potential_energy()
        intermediate.molecule = molec_eval
        if self.del_traj:
            molec_eval.calc = None

    def _ensure_surface_evaluated(self, surface: Surface) -> None:
        """Default: Relax bare slab if its energy is not yet computed."""
        if getattr(surface, "energy", None) is None:
            self._assign_calculator(surface.slab)
            opt = self.optimizer_class(surface.slab, logfile=self.logfile)
            opt.run(fmax=self.fmax, steps=self.max_steps)
            surface.energy = surface.slab.get_potential_energy()
            if self.del_traj:
                surface.slab.calc = None

    def _calc_adsorbed_mu(self, current_energy: float, intermediate: Intermediate, surface: Surface, adsorption: Atoms) -> float:
        """Default: E_tot - E_slab."""
        n_slab = len(surface.slab)
        scaling_factor = (len(adsorption) - len(intermediate.molecule)) / n_slab
        return current_energy - (scaling_factor * surface.energy)

    def _calc_ts_energy(self, energy_TS: float, reaction: ElementaryReaction) -> float:
        """Default: E_TS - E_slab."""
        surface_energy = getattr(reaction.catalyst, "energy", 0.0) if reaction.catalyst else 0.0
        return energy_TS - surface_energy

    def _eval_species(self, intermediate: Intermediate, **kwargs) -> None:
        if not all([elem in self.adsorbate_domain for elem in intermediate.molecule.get_chemical_symbols()]):
            raise ValueError(f'{self.__class__.__name__} cannot evaluate elements outside its domain.')

        if isinstance(intermediate, GasSpecies):
            self._eval_gas_species(intermediate, **kwargs)

        elif isinstance(intermediate, AdsorbedSpecies):
            surface = getattr(intermediate, "catalyst", None)
            if not isinstance(surface, Surface):
                raise ValueError(f"No surface attached to intermediate {intermediate.code}.")
            
            self._ensure_surface_evaluated(surface)

            ads_config_dict = {}
            adsorptions = place_adsorbate(intermediate, surface, -1)
            attempts = 0
            best_broken_ads = None
            lowest_broken_mu = float('inf')
            last_attempt_energy = None
            
            for i, adsorption in enumerate(adsorptions):
                if len(ads_config_dict) == self.num_configs or attempts >= self.patience:
                    break
                attempts += 1
                self._assign_calculator(adsorption)
                opt = self.optimizer_class(adsorption, logfile=self.logfile)
                opt.run(fmax=self.fmax, steps=self.max_steps)
                
                current_energy = adsorption.get_potential_energy()
                last_attempt_energy = current_energy
                g = atoms_to_graph(adsorption, atom_tags=adsorption.get_array("atom_tags"), surface_order=-1, filter=True)
                
                if g is None:
                    if current_energy < lowest_broken_mu:
                        lowest_broken_mu = current_energy
                        best_broken_ads = adsorption.copy()
                    if self.del_traj:
                        adsorption.calc = None
                    continue
                
                ads_config_dict[str(i)] = {
                    'ase': adsorption,
                    'mu': self._calc_adsorbed_mu(current_energy, intermediate, surface, adsorption),
                    'connectivity': True,
                }
                if self.del_traj:
                    adsorption.calc = None
                    
            if len(ads_config_dict) == 0:
                warnings.warn(f"Failed to find intact configuration for {intermediate.formula}. Falling back to broken structure.")
                fallback_ads = best_broken_ads if best_broken_ads is not None else adsorptions[-1]
                fallback_energy = lowest_broken_mu if best_broken_ads is not None else last_attempt_energy
                
                if fallback_energy is not None:
                    ads_config_dict["0"] = {
                        'ase': fallback_ads,
                        'mu': self._calc_adsorbed_mu(fallback_energy, intermediate, surface, fallback_ads),
                        'connectivity': False
                    }

            intermediate.ads_configs = ads_config_dict
        elif isinstance(intermediate, SurfaceSite):
            pass

    def _eval_reaction(self, reaction: ElementaryReaction, **kwargs) -> None:
        for species in list(reaction.reactants) + list(reaction.products):
            if not isinstance(species, SurfaceSite) and getattr(species, 'E', None) is None:
                self._eval_species(species, **kwargs)

        if reaction.requires_neb:
            try:
                reaction.get_states(self)
            except Exception as e:
                print(f"Error occurred while defining initial and final states for reaction {reaction.repr_hr}: {e}")
                return

            try:
                images = [reaction.is_atoms] + [reaction.is_atoms.copy() for _ in range(self.num_images)] + [reaction.fs_atoms]
                neb = NEB(
                    images, 
                    k=self.k, 
                    climb=self.climb, 
                    parallel=self.parallel, 
                    remove_rotation_and_translation=self.remove_rotation_and_translation, 
                    method=self.neb_method, 
                    allow_shared_calculator=self.allow_shared_calculator
                )
                        
                neb.interpolate(method=self.interpolation_method, mic=True, apply_constraint=True)
                                
                for image in images[1:self.num_images + 1]:  # only internal images
                    self._assign_calculator(image)

                optimizer = LBFGS(neb, logfile=None) if self.optimizer == "LBFGS" else BFGS(neb, logfile=None)
                optimizer.run(fmax=0.05, steps=self.max_steps)

                final_NEB_frames = []
                final_NEB_energies = []
                for i, image in enumerate(neb.images):
                    if i == 0 or i == len(neb.images) - 1:
                        self._assign_calculator(image)
                    energy_image = image.get_potential_energy()
                    final_NEB_energies.append(energy_image)
                    image.calc = None
                    final_NEB_frames.append(image)
                    
                energy_TS = max(final_NEB_energies)
                reaction.neb_images = final_NEB_frames
                reaction.neb_energies = final_NEB_energies
                reaction.e_ts = self._calc_ts_energy(energy_TS, reaction)
                
            except Exception as e:
                print(f"Error occurred while running NEB for reaction {reaction.repr_hr}: {e}")
                return
        else:
            pass

        empty_cache()

    def _assign_calculator(self, atoms):
        with silent_context():
            if self.allow_shared_calculator:
                atoms.calc = self.calc
            elif hasattr(self, "get_calculator"):
                atoms.calc = self.get_calculator()
            else:
                try:
                    atoms.calc = deepcopy(self.calc)
                except Exception:
                    atoms.calc = copy(self.calc)

from care.evaluators.gamenet_uq import GameNetUQevaluator, GAMENETUQ_AVAILABLE
from care.evaluators.fairchemv1 import FairChemV1evaluator, FAIRCHEMV1_AVAILABLE
from care.evaluators.fairchemv2 import FairChemV2evaluator, FAIRCHEMV2_AVAILABLE
from care.evaluators.mace import MACEevaluator, MACE_AVAILABLE
from care.evaluators.upet import UPETevaluator, UPET_AVAILABLE
from care.evaluators.orb import ORBevaluator, ORB_AVAILABLE
from care.evaluators.sevennet import SevenNetevaluator, SEVENNET_AVAILABLE

eval_dict = {
    "gamenetuq": (GameNetUQevaluator, GAMENETUQ_AVAILABLE),
    "fairchemv1": (FairChemV1evaluator, FAIRCHEMV1_AVAILABLE),
    "fairchemv2": (FairChemV2evaluator, FAIRCHEMV2_AVAILABLE),
    "mace": (MACEevaluator, MACE_AVAILABLE),
    "upet": (UPETevaluator, UPET_AVAILABLE),
    "orb": (ORBevaluator, ORB_AVAILABLE),
    "sevennet": (SevenNetevaluator, SEVENNET_AVAILABLE),
}

def get_available_evaluators(installed_only=False) -> list:
    """Show available energy evaluators in CARE."""
    if installed_only:
        return [key for key, (_, available) in eval_dict.items() if available]
    return list(eval_dict.keys())

def load_evaluator(model: str, **kwargs) -> "UniversalEvaluator":
    """Load the energy evaluator."""
    if model not in eval_dict:
        raise ValueError(f"Model '{model}' not recognized. Choose from {list(eval_dict.keys())}")
    return eval_dict[model][0](**kwargs)

__all__ = [
    "UniversalEvaluator",
    "load_evaluator",
    "get_available_evaluators",
]