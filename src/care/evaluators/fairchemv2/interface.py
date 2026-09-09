"""
Interface to FairChemV2 UMA models.
"""

from ase.data import chemical_symbols

from care.evaluators import UniversalEvaluator

try:
    from fairchem.core import pretrained_mlip, FAIRChemCalculator
    FAIRCHEMV2_AVAILABLE = True
except ImportError:
    FAIRCHEMV2_AVAILABLE = False

class FairChemV2evaluator(UniversalEvaluator):
    def __init__(
        self,
        name: str = 'uma-s-1p1',
        device: str = 'cpu',
        task_name: str = 'oc20',
        **kwargs
    ):
        """
        Interface to FairChemV2 UMA models. 
        To use this class of models, you need to have the permissions to access the fairchem models from Meta.

        Args:

        name(str): The name of the model to use among the checkpoints available in fairchem-v2 (OC20 and OC22)
        device(bool): cpu or cuda. Default is cpu
        task_name(str): Default to 'oc20'.
        
        """
        if not FAIRCHEMV2_AVAILABLE:
            raise ImportError("The FairChemV2IntermediateEvaluator requires 'fairchem-core' to be installed. "
                "Please install it using pip install fairchem-core==2.19.0.")
        super().__init__(**kwargs)
        self.model_name = name
        self.device = device
        self.task_name = task_name
        self.predictor = pretrained_mlip.get_predict_unit(name, device=device)
        self.calc = FAIRChemCalculator(self.predictor, task_name=task_name)
        self.num_params = sum([p.numel() for p in self.calc.predictor.model.parameters()])
        self.is_mlp = True

    def __repr__(self) -> str:
        return f'{self.model_name} from Meta FairChem-v2 models'
        
    @property
    def adsorbate_domain(self):
        """Returns the list of adsorbate elements that your model can handle."""
        return chemical_symbols[1:]
    
    @property
    def surface_domain(self):
        """Returns the list of surface elements that your model can handle."""
        return chemical_symbols[1:]
    
    def get_calculator(self):
        return FAIRChemCalculator(self.predictor, task_name=self.task_name)
