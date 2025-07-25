"""
Interface to MACE models.
"""

from copy import deepcopy

from ase.optimize import BFGS
from ase.data import chemical_symbols

from care import Intermediate, Surface
from care.evaluators import IntermediateEnergyEstimator, ReactionEnergyEstimator
from care.adsorption import place_adsorbate

class MACEIntermediateEvaluator(IntermediateEnergyEstimator):
    def __init__(
        self,
        surface: Surface,
        size: str = "large",
        device: str = "cpu",
        fmax: float = 0.05,
        max_steps: int = 100,
        dtype: str = "float32",
        num_configs: int = 1,
        dispersion: bool=True,
        del_traj: bool = True,
        **kwargs
    ):
        """Interface to the MACE models family.

        Args:
            surface (Surface): The surface on which the reaction network is adsorbed.
            size (str): The size of the model to use among the mace models. Default is "large", available are "small", "medium", and "large".
            device (str): The device to use for the calculation. Default is "cpu".
            fmax (float): The maximum force allowed on the atoms. Default is 0.05 eV/Angstrom.
            max_steps (int): The maximum number of steps for the relaxation. Default is 100.
            dtype (str): The data type to use for the calculation. Default is "float32".
            num_configs (int): The number of configurations to consider for the adsorbed phase. Default is 1.
            dispersion (bool): Include dispersion correction. Defaults to True.
            del_traj (bool): If True, keep relaxation trajectory and calculator for each intermediate configuration; 
                             note that this option may imply 10e6x larger CRN files!
        """
        from mace.calculators import mace_mp, MACECalculator

        self.surface = surface
        self.slab_energy = 0.0
        self.size = size
        self.dtype = dtype
        self.device = device
        self.dispersion = dispersion
        self.calc = mace_mp(model=self.size, device=self.device, default_dtype=dtype, dispersion=dispersion)
        if dispersion:
            dummy_calc = mace_mp(model=self.size, device=self.device, default_dtype=dtype, dispersion=False)
            self.num_params = sum([p.numel() for p in dummy_calc.models[0].parameters()])
            del dummy_calc
        else:
            self.num_params = sum([p.numel() for p in self.calc.models[0].parameters()])
        self.fmax = fmax
        self.max_steps = max_steps
        self.num_configs = num_configs
        self.del_traj = del_traj
        self.get_slab_energy()

    def __repr__(self) -> str:
        return f'MACE-MP-0 potential ({self.size}, {round(self.num_params/1e6, 1)}M params, {self.device}, {self.dtype})'

    def __call__(self,
                 intermediate: Intermediate,
                 **kwargs) -> None:
        if isinstance(intermediate, Intermediate):
            self.eval(intermediate, **kwargs)
        else:
            return NotImplementedError("Input must be an Intermediate object.")

    def get_slab_energy(self):
        self.surface.slab.calc = self.calc
        opt = BFGS(self.surface.slab)
        opt.run(fmax=self.fmax, steps=self.max_steps)
        self.slab_energy = self.surface.slab.get_potential_energy()
        if self.del_traj:
            self.surface.slab.calc = None
        self.surface.energy = self.slab_energy
        print('self.slab_energy: ', self.slab_energy)

    @property
    def adsorbate_domain(self):
        """Returns the list of adsorbate elements that your model can handle."""
        try:
            return [chemical_symbols[i] for i in self.calc.z_table.zs]
        except:
            return chemical_symbols

    @property
    def surface_domain(self):
        """Returns the list of surface elements that your model can handle."""
        try:
            return [chemical_symbols[i] for i in self.calc.z_table.zs]
        except:
            return chemical_symbols

    def eval(
        self,
        intermediate: Intermediate,
    ):

        """
        Given the surface and the intermediate, return the properties of the intermediate as attributes of the intermediate object.
        """

        if not all([elem in self.adsorbate_domain for elem in intermediate.molecule.get_chemical_symbols()]):
            raise ValueError(
                f'MACE can only evaluate molecules with {", ".join(self.adsorbate_domain)} elements.'
            )
        if intermediate.phase == 'gas':  # gas
            molec_eval = deepcopy(intermediate.molecule)
            molec_eval.set_cell([10, 10, 10])  # TODO: Should be function of molecule size

            molec_eval.calc = self.calc
            opt = BFGS(molec_eval)
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
            print(intermediate.ads_configs)
        elif intermediate.phase == "ads":  # adsorbed
            ads_config_dict = {}
            adsorptions = place_adsorbate(intermediate, self.surface, self.num_configs)
            for i, adsorption in enumerate(adsorptions):
                ads_config_dict[str(i)] = {}
                adsorption.calc = self.calc
                opt = BFGS(adsorption)
                opt.run(fmax=self.fmax, steps=self.max_steps)
                ads_config_dict[str(i)]['ase'] = adsorption
                ads_config_dict[str(i)]['mu'] = adsorption.get_potential_energy() - self.slab_energy # eV
                ads_config_dict[str(i)]['s'] = 0.0
                if self.del_traj:
                    adsorption.calc = None
            intermediate.ads_configs = ads_config_dict
        else:
            raise ValueError("Phase not supported by the current estimator.")


class MACEReactionEvaluator(ReactionEnergyEstimator):
    def __init__(
        self,
        intermediates: dict[str, Intermediate], 
        T: float = 298.0,
        ref_electrode: str = "SHE",
        pH: float = 7.0,
        U: float = 0.0,
        **kwargs
    ):
        super().__init__(
            intermediates=intermediates,
            T=T,
            ref_electrode=ref_electrode,
            pH=pH,
            U=U,
            **kwargs
        )    

    @property
    def adsorbate_domain(self):
        """Returns the list of adsorbate elements that your model can handle."""
        return chemical_symbols[1:]

    @property
    def surface_domain(self):
        """Returns the list of surface elements that your model can handle."""
        return chemical_symbols[1:]
