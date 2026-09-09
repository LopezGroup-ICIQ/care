from ase.data import chemical_symbols
from care.evaluators import UniversalEvaluator

try:
    from sevenn.calculator import SevenNetCalculator, SevenNetD3Calculator
    SEVENNET_AVAILABLE = True
except ImportError:    
    SEVENNET_AVAILABLE = False   

class SevenNetevaluator(UniversalEvaluator):
    def __init__(
        self,
        model: str = "7net-mf-ompa",
        modal: str = "mpa",
        file_type: str = "checkpoint",
        device: str = "cpu",
        dispersion: bool = False,
        enable_cueq: bool = False,
        enable_flash: bool = False,
        enable_oeq: bool = False,
        **kwargs
    ):
        if not SEVENNET_AVAILABLE:
            raise ImportError("SevenNet not installed. "
            "Install it using pip install care-crn[sevenn]")
        
        super().__init__(**kwargs)
        self.model = model
        self.modal = modal
        self.file_type = file_type
        self.device = device
        self.dispersion = dispersion
        self.eval_cueq = enable_cueq
        self.eval_flash = enable_flash
        self.eval_oeq = enable_oeq
        
        if self.dispersion:
            if self.device == "cuda":
                self.calc = SevenNetD3Calculator(
                    model=self.model,
                    device=self.device,
                    modal=self.modal,
                    file_type=self.file_type, 
                    enable_cueq=self.eval_cueq,
                    enable_flash=self.eval_flash,
                    enable_oeq=self.eval_oeq
                )
                sevennet_base = self.calc.mixer.calcs[0]
            else:
                raise ValueError("D3 correction only available for GPU (cuda) device. Please set device to 'cuda'.")
        else:
            self.calc = SevenNetCalculator(
                model=self.model,
                device=self.device,
                modal=self.modal,
                file_type=self.file_type, 
                enable_cueq=self.eval_cueq,
                enable_flash=self.eval_flash,
                enable_oeq=self.eval_oeq
            )
            sevennet_base = self.calc

        self.num_params = sum([p.numel() for p in sevennet_base.model.parameters()])
        self.is_mlp = True

    def __repr__(self) -> str:
        return f'SevenNet ({self.model}, {round(self.num_params/1e6, 1)}M params, {self.device})'

    @property
    def adsorbate_domain(self):
        return chemical_symbols[1:]

    @property
    def surface_domain(self):
        return chemical_symbols[1:]
    
    def get_calculator(self):
        if self.dispersion:
            return SevenNetD3Calculator(
                model=self.model,
                device=self.device,
                modal=self.modal,
                file_type=self.file_type, 
                enable_cueq=self.eval_cueq,
                enable_flash=self.eval_flash,
                enable_oeq=self.eval_oeq
            )
        else:
            return SevenNetCalculator(
                model=self.model,
                device=self.device,
                modal=self.modal,
                file_type=self.file_type, 
                enable_cueq=self.eval_cueq,
                enable_flash=self.eval_flash,
                enable_oeq=self.eval_oeq
            )