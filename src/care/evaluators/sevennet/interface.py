"""
Interface to SevenNet potentials.
"""

from copy import deepcopy
import logging
from typing import Union
import warnings

from ase import Atoms
from ase.optimize import BFGS, LBFGS
from ase.data import chemical_symbols

from care import Intermediate, Surface
from care.evaluators import IntermediateEnergyEstimator
from care.adsorption import place_adsorbate
from care.evaluators.utils import atoms_to_data

try:
    from sevenn.calculator import SevenNetCalculator, SevenNetD3Calculator
    SEVENNET_AVAILABLE = True
except ImportError:    
    SEVENNET_AVAILABLE = False   

class SevenNetIntermediateEvaluator(IntermediateEnergyEstimator):
    def __init__(
        self,
        surface: Surface,
        model: str = "7net-mf-ompa",
        modal: str = "mpa",
        file_type: str = "checkpoint",
        device: str = "cpu",
        fmax: float = 0.05,
        max_steps: int = 100,
        num_configs: int = 1,
        dispersion: bool = False,
        del_traj: bool = True,
        logfile: str = None,
        optimizer: str = "BFGS",
        patience: int = 3,
        **kwargs
    ):
        """Interface to the SevenNet potentials.

        Args:
            surface (Surface): The surface on which the reaction network is adsorbed.
            model (str): The version of the employed SevenNet potential. Default to 7net-mf-ompa.
            modal (str): The modal of the employed SevenNet potential. Default to mpa.
                Can be omitted if the model is not multi-fidelity trained.
            file_type (str): The type of file to load. Default to checkpoint.
            device (str): The device to use for the calculation. Default is "cpu".
                For D3 calculations, use "cuda" (GPU) as currently CPU+D3 is not supported.
                Note: As of 02-04-2025, the SevenNet ASE calculator does not support parallelization.  
            fmax (float): The maximum force allowed on the atoms. Default is 0.05 eV/Angstrom.
            max_steps (int): The maximum number of steps for the relaxation. Default is 100.
            num_configs (int): The number of configurations to consider for the adsorbed phase. Default is 1.
            dispersion (bool): If True, use the D3 dispersion correction. Default is False.
                If True, the device must be "cuda" (GPU) as currently CPU+D3 is not supported.
            del_traj (bool): If True, keep relaxation trajectory and calculator for each intermediate configuration; 
                             note that this option may imply 10e6x larger CRN files!
            logfile (str): The path to the logfile for relaxation trajectories. Default is None. Use '-' for stdout.
            optimizer (str): The optimizer to use for the relaxation. Default is "BFGS". Other options are "LBFGS".
            patience (int): The number of steps to wait before considering a relaxation as failed. Default is 3.
        """
        if not SEVENNET_AVAILABLE:
            raise ImportError("SevenNet not installed. "
            "Install it using pip install care-crn[sevennet]")

        self.model = model
        self.modal = modal
        self.file_type = file_type
        self.surface = surface
        self.n_slab = len(surface.slab)  # required in cases when slab is resized to fit larger adsorbates
        self.slab_energy = 0.0
        self.device = device
        self.dispersion = dispersion
        if self.dispersion:
            if self.device == "cuda":
                self.calc = SevenNetD3Calculator(
                    model=self.model,
                    device=self.device,
                    modal=self.modal,
                    file_type=self.file_type,
                    dispersion=True
                )
            else:
                raise ValueError("D3 correction only available for GPU (cuda) device. Please set device to 'cuda'.")
        else:
            self.calc = SevenNetCalculator(
                model=self.model,
                device=self.device,
                modal=self.modal,
                file_type=self.file_type,
                dispersion=False
            )

        self.num_params = sum([p.numel() for p in self.calc.model.parameters()])
        self.fmax = fmax
        self.max_steps = max_steps
        self.num_configs = num_configs
        self.del_traj = del_traj
        self.logfile = logfile
        self.patience = patience * num_configs
        self.optimizer = BFGS if optimizer == "BFGS" else LBFGS
        self.is_mlp = True
        self.get_slab_energy()

    def __repr__(self) -> str:
        return f'SevenNet potential ({self.model}, {round(self.num_params/1e6, 1)}M params, {self.device})'

    def __call__(self,
                 intermediate: Intermediate,
                 **kwargs) -> None:
        if isinstance(intermediate, (Intermediate, Atoms)):
            self.eval(intermediate, **kwargs)
        else:
            return NotImplementedError("Input must be a CARE Intermediate or ASE Atoms object.")

    def get_slab_energy(self):
        self.surface.slab.calc = self.calc
        opt = BFGS(self.surface.slab, 
                   logfile=self.logfile)
        opt.run(fmax=self.fmax, steps=self.max_steps)
        self.slab_energy = self.surface.slab.get_potential_energy()
        if self.del_traj:
            self.surface.slab.calc = None
        self.surface.energy = self.slab_energy

    @property
    def adsorbate_domain(self):
        return chemical_symbols[1:]

    @property
    def surface_domain(self):
        return chemical_symbols[1:]
    
    def get_calculator(self):
        return SevenNetD3Calculator(
            model=self.model,
            device=self.device,
            modal=self.modal,
            file_type=self.file_type,
            dispersion=self.dispersion)

    def eval(
        self,
        intermediate: Union[Intermediate, Atoms],
    ):

        """
        Given the surface and the intermediate, return the properties of the intermediate as attributes of the intermediate object.
        """
        if isinstance(intermediate, Intermediate):
            if not all([elem in self.adsorbate_domain for elem in intermediate.molecule.get_chemical_symbols()]):
                raise ValueError(
                    f'SevenNet can only evaluate molecules with {", ".join(self.adsorbate_domain)} elements.'
                )
            if intermediate.phase == 'gas':  # gas
                molec_eval = deepcopy(intermediate.molecule)
                molec_eval.set_cell([10, 10, 10])  # TODO: Should be function of molecule size

                molec_eval.calc = self.calc
                opt = BFGS(molec_eval, 
                        logfile=self.logfile)
                opt.run(fmax=self.fmax, steps=self.max_steps)
                intermediate.ads_configs = {
                    intermediate.phase: {
                        "ase": molec_eval,
                        "mu": molec_eval.get_potential_energy(),  # eV
                        "s": 0.0,  # eV
                    }
                }
                if self.del_traj:
                    molec_eval.calc = None
            elif intermediate.phase == "ads":  # adsorbed
                if self.surface is None:
                    raise ValueError("Surface must be provided for adsorbed phase evaluation.")
                ads_config_dict = {}
                adsorptions = place_adsorbate(intermediate, self.surface, -1)
                attempts = 0
                best_broken_ads = None
                lowest_broken_mu = float('inf')
                for i, adsorption in enumerate(adsorptions):
                    if len(ads_config_dict) == self.num_configs or len(ads_config_dict) == len(adsorptions):
                        break
                    attempts += 1
                    adsorption.calc = self.calc
                    opt = self.optimizer(adsorption, 
                            logfile=self.logfile)
                    opt.run(fmax=self.fmax, steps=self.max_steps)
                    current_energy = adsorption.get_potential_energy()
                    g = atoms_to_data(adsorption, atom_tags=adsorption.get_array("atom_tags"), surface_order=-1, filter=True)
                    if g is None:
                        if current_energy < lowest_broken_mu:
                            lowest_broken_mu = current_energy
                            best_broken_ads = adsorption.copy()
                        continue
                    ads_config_dict[str(i)] = {
                        'ase': adsorption,
                        'mu': current_energy - ((len(adsorption) - len(intermediate.molecule))/self.n_slab) * self.slab_energy,  # eV
                        's': 0.0,
                        'connectivity': True,
                        # 'converged': opt.converged()
                    }
                    if self.del_traj:
                        adsorption.calc = None
                if len(ads_config_dict) == 0:
                    warnings.warn(
                        f"Failed to find intact configuration for {intermediate.formula} "
                        f"after {attempts} attempts. Falling back to the lowest energy broken structure."
                    )
                    fallback_ads = best_broken_ads if best_broken_ads is not None else adsorptions[-1]
                    
                    ads_config_dict["0"] = {
                        'ase': fallback_ads,
                        'mu': (lowest_broken_mu if best_broken_ads is not None else fallback_ads.get_potential_energy()) - self.slab_energy,
                        's': 0.0,
                        'connectivity': False
                    }
                elif len(ads_config_dict) < self.num_configs:
                    logging.info(
                        f"Requested {self.num_configs} configs for {intermediate.formula}, "
                        f"but only found {len(ads_config_dict)} valid ones before hitting the attempt limit."
                    )
                intermediate.ads_configs = ads_config_dict
            else:
                raise ValueError("Phase not supported by the current estimator.")
        elif isinstance(intermediate, Atoms):
            intermediate.calc = self.calc
            opt = self.optimizer(intermediate,
                       logfile=self.logfile)
            opt.run(fmax=self.fmax, steps=self.max_steps)
            if self.del_traj:
                intermediate.calc = None
        else:
            return NotImplementedError("Input must be an Intermediate or Atoms object.")
