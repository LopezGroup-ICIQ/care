"""
Interface to UPET potentials.
"""
from ase.data import chemical_symbols

from care.evaluators import UniversalEvaluator

try:
    from upet.calculator import UPETCalculator
    UPET_AVAILABLE = True
except ImportError:
    UPET_AVAILABLE = False


class UPETevaluator(UniversalEvaluator):
    def __init__(
        self,
        model: str = "pet-mad-s",
        version: str = "latest",
        device: str = "cpu",
        dtype: str = "float32",
        **kwargs
    ):
        """Interface to the UPET family of MLIPs.

        Args:
            model(str): Model name.
            version (str): UPET model version. Default is "latest".
            device (str): The device to use for the calculation. Default is "cpu".
            dtype (str): The data type to use for the calculation. Default is "float32".
            **kwargs: UniversalEvaluator parameters.
        """
        if not UPET_AVAILABLE:
            raise ImportError(
                "UPET not installed. "
                "Install it using pip install care-crn[upet]"
            )
        super().__init__(**kwargs)

        self.model = model
        self.version = version
        self.dtype = dtype
        self.device = device
        self.calc = UPETCalculator(model=model, version=version, device=device)
        self.num_params = 0  # sum([p.numel() for p in self.calc._model.parameters()]) #TODO adapt
        self.is_mlp = True
        self.supports_batching = False

    def __repr__(self) -> str:
        return f'UPET ({self.model} v{self.version}, {round(self.num_params/1e6, 1)}M params, {self.device}, {self.dtype})'

    @property
    def adsorbate_domain(self):
        """Returns the list of adsorbate elements that your model can handle."""
        return chemical_symbols[1:]

    @property
    def surface_domain(self):
        """Returns the list of surface elements that your model can handle."""
        return chemical_symbols[1:]
    
    def get_calculator(self):
        return UPETCalculator(model=self.model, version=self.version, device=self.device)
