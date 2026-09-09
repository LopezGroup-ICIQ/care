"""
Interface to ORB potentials.
"""

from ase.data import chemical_symbols

from care.evaluators import UniversalEvaluator

try:
    from orb_models.forcefield.pretrained import ORB_PRETRAINED_MODELS
    from orb_models.forcefield.inference.calculator import ORBCalculator
    ORB_AVAILABLE = True
except:
    ORB_AVAILABLE = False


class ORBevaluator(UniversalEvaluator):
    def __init__(
        self,
        version: str = "orb-v3-conservative-inf-omat",
        device: str = "cpu",
        max_num_neighbors: int = None,
        half_supercell: bool = None,
        dtype: str = "float32-high",
        **kwargs
    ):
        """Interface to the ORB potentials.

        Args:
            surface (Surface): The surface on which the reaction network is adsorbed.
            version (str): The version of the employed ORB potential. Default to orb-v3-conservative-inf-omat.
            device (str): The device to use for the calculation. Default is "cpu".
            max_num_neighbors (int): The maximum number of neighbors to consider for the k-nearest neighbors method. Default is 20.
            dtype (str): The data type to use for the calculation "float32-high", "float32-highest", "float64". Default is "float32-high".
        """
        if not ORB_AVAILABLE:
            raise ImportError("Orb not installed. "
                "Install it using pip install care-crn[orb]")

        if version not in ORB_PRETRAINED_MODELS:
            raise ValueError(f"Version {version} not existing. Choose from {list(ORB_PRETRAINED_MODELS.keys())}.")
        
        super().__init__(**kwargs)
        self.version = version
        self.dtype = dtype
        self.device = device
        self.model, self.atoms_adapter = ORB_PRETRAINED_MODELS[version](device=device, precision=dtype)
        self.max_num_neighbors = max_num_neighbors
        self.half_supercell = half_supercell
        self.calc = ORBCalculator(model=self.model,
                                  atoms_adapter=self.atoms_adapter,
                                  max_num_neighbors=self.max_num_neighbors,
                                  half_supercell=self.half_supercell, 
                                  device=self.device)
        self.num_params = sum([p.numel() for p in self.calc.model.parameters()])
        self.is_mlp = True

    def __repr__(self) -> str:
        return f'ORB ({self.version}, {round(self.num_params/1e6, 1)}M params, {self.device}, {self.dtype})'

    @property
    def adsorbate_domain(self):
        """Returns the list of adsorbate elements that your model can handle."""
        return chemical_symbols[1:]

    @property
    def surface_domain(self):
        """Returns the list of surface elements that your model can handle."""
        return chemical_symbols[1:]
    
    def get_calculator(self):
        return ORBCalculator(model=self.model, 
                            atoms_adapter=self.atoms_adapter,
                            half_supercell=self.half_supercell,
                            device=self.device)

