from care import Intermediate
from ase import Atoms

class Electron(Intermediate):
    """
    Electron species (e-)  
    """
    def __init__(self):
        super().__init__("e-", molecule=Atoms())
        self.is_surface = False
        self.closed_shell = False
        self.phase = 'electro'
        self.mass = 9.10938356e-31  # kg
        self.electrons = 1
        self.bader = -1  # ne (where e is the elementary charge, 1.602176634 × 10^-19 C)
        self.formula = 'e-'

    def __str__(self) -> str:
        return "Electron(e-)"
    

class Proton(Intermediate):
    """
    Proton species (H+)  
    """
    def __init__(self):
        super().__init__("H+", molecule=Atoms('H', positions=[(0, 0, 0)]))
        self.is_surface = False
        self.closed_shell = False
        self.phase = "solv"
        self.mass = 1.6726219e-27  # kg
        self.electrons = 0
        self.bader = 1  # ne (where e is the elementary charge, 1.602176634 × 10^-19 C)
        self.formula = 'H+'

    def __str__(self) -> str:
        return "Proton(H+)"



class Hydroxide(Intermediate):
    """
    Hydroxide species (OH-)  
    """
    def __init__(self):
        super().__init__("OH-", molecule=Atoms('HO', positions=[(0, 0, 0), (0, 0, 0.96)]))
        self.is_surface = False
        self.closed_shell = False
        self.phase = 'solv'
        self.mass = 3.3496e-26  # kg
        self.electrons = 0
        self.bader = -1  # ne (where e is the elementary charge, 1.602176634 × 10^-19 C)
        self.formula = 'OH-'

    def __str__(self) -> str:
        return "Hydroxide(OH-)"
    
class Water(Intermediate):
    """
    Water species (H2O)  
    """
    def __init__(self):
        super().__init__("H2O", molecule=Atoms('H2O', positions=[(0, 0, 0), (0.96, 0, 0), (0.48, 0.83, 0)]))
        self.is_surface = False
        self.closed_shell = False
        self.phase = 'solv'
        self.mass = 2.991e-26  # kg
        self.electrons = 0
        self.bader = 0  # ne (where e is the elementary charge, 1.602176634 × 10^-19 C)
        self.formula = 'H2O'

    def __str__(self) -> str:
        return "Water(H2O)"