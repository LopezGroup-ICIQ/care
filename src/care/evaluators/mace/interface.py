"""
Interface to MACE models.
"""
from ase.data import chemical_symbols

from care.evaluators import UniversalEvaluator

try:
    from mace.calculators import mace_mp
    MACE_AVAILABLE = True
except ImportError:
    MACE_AVAILABLE = False


class MACEevaluator(UniversalEvaluator):
    def __init__(
        self,
        size: str = "large",
        device: str = "cpu",
        dtype: str = "float32",
        dispersion: bool = True,
        **kwargs
    ):
        """Interface to the MACE models family.

        Args:
            size (str): The size of the model to use among the mace models. Default is "large", available are "small", "medium", and "large".
            device (str): The device to use for the calculation. Default is "cpu".
            dtype (str): The data type to use for the calculation. Default is "float32".
            dispersion (bool): Include dispersion correction. Defaults to True.
        """
        if not MACE_AVAILABLE:
            raise ImportError(
                "The MACEEvaluator requires 'mace-torch' to be installed. "
                "Please install it using pip install mace-torch."
            )

        super().__init__(**kwargs)
        
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

        self.is_mlp = True
        self.supports_batching = False

    def __repr__(self) -> str:
        return f'MACE-MP-0 ({self.size}, {round(self.num_params/1e6, 1)}M params, {self.device}, {self.dtype})'

    @property
    def adsorbate_domain(self):
        """Returns the list of adsorbate elements that your model can handle."""
        try:
            return [chemical_symbols[i] for i in self.calc.z_table.zs]
        except Exception:
            return chemical_symbols

    @property
    def surface_domain(self):
        """Returns the list of surface elements that your model can handle."""
        try:
            return [chemical_symbols[i] for i in self.calc.z_table.zs]
        except Exception:
            return chemical_symbols
        
    def get_calculator(self):
        return mace_mp(model=self.size, device=self.device, default_dtype=self.dtype, dispersion=self.dispersion)
