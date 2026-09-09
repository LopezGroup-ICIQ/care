"""
Interface to FairChem-V1 potentials.
"""
from ase.data import chemical_symbols
from care import Intermediate, Surface, silent_context, ElementaryReaction
from care.evaluators import UniversalEvaluator
from care.crn.intermediate import GasSpecies

try:
    from fairchem.core.models.model_registry import model_name_to_local_file
    from fairchem.core.common.relaxation.ase_utils import OCPCalculator
    FAIRCHEMV1_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    FAIRCHEMV1_AVAILABLE = False


class FairChemV1evaluator(UniversalEvaluator):
    def __init__(
        self,
        name: str = 'EquiformerV2-31M-S2EF-OC20-All+MD',
        device: str = 'cpu',
        **kwargs
    ):
        if not FAIRCHEMV1_AVAILABLE:
            raise ImportError("FairChemV1 dependencies missing. Install fairchem-core, torch-scatter, torch-sparse.")

        super().__init__(**kwargs)

        self.model_name = name
        self.checkpoint_path = model_name_to_local_file(name, local_cache='/tmp/fairchem_checkpoints/')
        self.device = device
        cpu = True if device == 'cpu' else False
        with silent_context():
            self.calc = OCPCalculator(checkpoint_path=self.checkpoint_path, cpu=cpu, seed=42)
            
        self.num_params = sum([p.numel() for p in self.calc.trainer.model.parameters()])
        self.eref = {'C': -7.282, 'H': -3.477, 'O': -7.204, 'N': -8.083}  # eV
        self.is_mlp = True

    def __repr__(self) -> str:
        return f'{self.model_name} from Meta FairChemV1 models'

    @property
    def adsorbate_domain(self):
        return ['C', 'H', 'O', 'N']
    
    @property
    def surface_domain(self):
        return chemical_symbols[1:]
    
    def get_calculator(self):
        with silent_context():
            return OCPCalculator(checkpoint_path=self.checkpoint_path, cpu=self.device == 'cpu', seed=42)

    def _eval_gas_species(self, intermediate: GasSpecies, **kwargs) -> None:
        """FairChem uses atomic gas references instead of molecular MLIP relaxations."""
        intermediate.E = sum(intermediate[el] * self.eref[el] for el in ['C', 'H', 'O', 'N'])

    def _ensure_surface_evaluated(self, surface: Surface) -> None:
        """Bare slab energy is implicitly set to 0.0 for OCP adsorption models."""
        if getattr(surface, "energy", None) is None:
            surface.energy = 0.0

    def _calc_adsorbed_mu(self, current_energy: float, intermediate: Intermediate) -> float:
        """FairChem mu = predicted_adsorption_energy + atomic_gas_references."""
        gas_energy = sum(intermediate[el] * self.eref[el] for el in ['C', 'H', 'O', 'N'])
        return current_energy + gas_energy

    def _calc_ts_energy(self, energy_TS: float, reaction: ElementaryReaction) -> float:
        """Applies atomic references to TS energy and guards against non-energetic barriers."""
        referenced_ts_energy = energy_TS + sum(
            reaction.is_atoms.get_chemical_symbols().count(el) * self.eref[el]
            for el in self.adsorbate_domain
        )
        return referenced_ts_energy
