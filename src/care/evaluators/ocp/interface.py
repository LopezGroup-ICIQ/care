"""
Interface to Open Catalyst Project (OCP) models.
"""
from typing import Union

from ase import Atoms
from ase.optimize import BFGS
from ase.data import chemical_symbols

from care import Intermediate, Surface
from care.evaluators import IntermediateEnergyEstimator
from care.adsorption import place_adsorbate
from care.evaluators.utils import atoms_to_data

class OCPIntermediateEvaluator(IntermediateEnergyEstimator):
    def __init__(
        self,
        surface: Surface = None,
        name: str = 'EquiformerV2-31M-S2EF-OC20-All+MD',
        device: str = 'cpu',
        fmax: float = 0.05,
        max_steps: int = 5,
        num_configs: int = 1,
        del_traj: bool = True,
        logfile: str = None,
        **kwargs
    ):
        """Interface for the models from the Open Catalyst Project
        (OCP) for predicting the energy of an intermediate on a surface.

        Args:
            surface (Surface): The surface on which the reaction network is adsorbed.
            name (str): The name of the model to use among the checkpoints available in fairchem (OC20 and OC22)
            device (str): The device to use for the calculation. 'cpu' or 'cuda'.
            fmax (float): The maximum force allowed on the atoms. Default is 0.05 eV/Angstrom.
            max_steps (int): The maximum number of steps for the relaxation. Default is 100.
            num_configs (int): The number of configurations to consider for the adsorbed phase. Default to 1.
            del_traj (bool): If True, keep relaxation trajectory and calculator for each intermediate configuration; 
                            note that this option may imply 10e6x larger CRN files!
            logfile (str): The path to the logfile for relaxation trajectories. Default is None. Use '-' for stdout.
        
        Note:

        - The intermediate energy is stored as E_tot - E_slab in eV.
        """
        from fairchem.core.models.model_registry import model_name_to_local_file
        from fairchem.core.common.relaxation.ase_utils import OCPCalculator

        self.model_name = name
        self.checkpoint_path = model_name_to_local_file(name, local_cache='/tmp/fairchem_checkpoints/')
        self.surface = surface
        self.device = device
        cpu = True if device == 'cpu' else False
        self.calc = OCPCalculator(checkpoint_path=self.checkpoint_path, cpu=cpu, seed=42)
        self.num_params = sum([p.numel() for p in self.calc.trainer.model.parameters()])
        self.fmax = fmax
        self.max_steps = max_steps
        self.num_configs = num_configs
        self.eref = {'C': -7.282, 'H': -3.477, 'O': -7.204, 'N': -8.083}  # eV
        self.is_mlp = True
        self.del_traj = del_traj
        self.logfile = logfile

    def __repr__(self) -> str:
        return f'{self.model_name} from Meta FairChemV1 models'

    def __call__(self,
                 intermediate: Intermediate,
                 **kwargs) -> None:
        if isinstance(intermediate, (Intermediate, Atoms)):
            self.eval(intermediate, **kwargs)
        else:
            return NotImplementedError("Input must be a CARE Intermediate or ASE Atoms object.")
        
    @property
    def adsorbate_domain(self):
        """Returns the list of adsorbate elements that your model can handle."""
        return ['C', 'H', 'O', 'N']
    
    @property
    def surface_domain(self):
        """Returns the list of surface elements that your model can handle."""
        return chemical_symbols[1:]

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
                    f'OCP models can only evaluate adsorbates/molecules with {", ".join(self.adsorbate_domain)} elements.'
                )
            
            gas_energy = sum(intermediate[el] * self.eref[el] for el in ['C', 'H', 'O', 'N'])

            if intermediate.phase == "gas":  # gas phase
                intermediate.ads_configs = {
                    "gas": {
                        "ase": intermediate.molecule,
                        "mu": gas_energy,  # eV
                        "s": 0.0,  # eV
                    }
                }
            elif intermediate.phase == "ads":  # adsorbed
                if self.surface is None:
                    raise ValueError("Surface must be provided for adsorbed phase evaluation.")
                ads_config_dict = {}
                adsorptions = place_adsorbate(intermediate, self.surface, -1)
                for i, adsorption in enumerate(adsorptions):
                    if len(ads_config_dict) == self.num_configs or len(ads_config_dict) == len(adsorptions):
                        break
                    adsorption.calc = self.calc
                    opt = BFGS(adsorption, 
                            logfile=self.logfile)
                    opt.run(fmax=self.fmax, steps=self.max_steps)
                    g = atoms_to_data(adsorption, adsorption.get_array("atom_tags"), -1, True)
                    if g is None:
                        continue
                    ads_config_dict[str(i)] = {}
                    ads_config_dict[str(i)]['ase'] = adsorption
                    # Note: OCP output is Eads, so to get Etot - Eslab, we need to add the gas-phase energy of the adsorbate
                    ads_config_dict[str(i)]['mu'] = adsorption.get_potential_energy() + gas_energy
                    ads_config_dict[str(i)]['s'] = 0.0
                    if self.del_traj:
                        adsorption.calc = None
                if len(ads_config_dict) == 0:
                    Warning(f"No valid adsorption configuration found for {intermediate.formula}, keep the last one.")
                    ads_config_dict["0"] = {}
                    ads_config_dict["0"]['ase'] = adsorption
                    ads_config_dict["0"]['mu'] = adsorption.get_potential_energy() + gas_energy
                    ads_config_dict["0"]['s'] = 0.0
                    ads_config_dict["0"]['converged'] = opt.converged()
                    ads_config_dict["0"]['connectivity'] = False
                else:
                    intermediate.ads_configs = ads_config_dict
            else:
                raise ValueError("Phase not supported by the current estimator.")
        elif isinstance(intermediate, Atoms):
            intermediate.calc = self.calc
            opt = BFGS(intermediate,
                       logfile=self.logfile)
            opt.run(fmax=self.fmax, steps=self.max_steps)
            if self.del_traj:
                intermediate.calc = None
        else:
            return NotImplementedError("Input must be an Intermediate or Atoms object.")