from care import Intermediate

class Electron(Intermediate):
    """
    Electron species (e-)  
    """
    def __init__(self):
        super().__init__("e-")
        self.is_surface = False
        self.closed_shell = False
        self.phase = None
        self.mass = 9.10938356e-31  # kg
        self.electrons = 1
        self.charge = -1.602176634e-19  # C

    def __str__(self) -> str:
        return "Electron species (e-)"
    

class Proton(Intermediate):
    """
    Proton species (H+)  
    """
    def __init__(self):
        super().__init__("H+")
        self.is_surface = False
        self.closed_shell = False
        self.phase = None
        self.mass = 1.6726219e-27  # kg
        self.electrons = 0
        self.charge = 1.602176634e-19  # C

    def __str__(self) -> str:
        return "Proton species (H+)"
    

class Hydroxide(Intermediate):
    """
    Hydroxide species (OH-)  
    """
    def __init__(self):
        super().__init__("OH-")
        self.is_surface = False
        self.closed_shell = False
        self.phase = None
        self.mass = 3.3496e-26  # kg
        self.electrons = 0
        self.charge = -1.602176634e-19  # C

    def __str__(self) -> str:
        return "Hydroxide species (OH-)"