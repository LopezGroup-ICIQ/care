"""Module containing charged species and solvent for electrocatalysis simulations."""

from care import Intermediate
from ase import Atoms


class Electron(Intermediate):
    """
    Electron species (e-)
    """

    def __init__(self):
        super().__init__(code="e-", molecule=Atoms(), phase="electro")
        self.is_surface = False
        self.closed_shell = False
        self._mass = 9.10938356e-31  # kg
        self._electrons = 1
        self._charge = (
            -1
        )  # ne (where e is the elementary charge, 1.602176634 × 10^-19 C)
        self._formula = "e-"

    def __str__(self) -> str:
        return "Electron(e-)"


class Proton(Intermediate):
    """
    Proton species (H+)
    """

    def __init__(self):
        super().__init__(
            code="H+", molecule=Atoms("H", positions=[(0, 0, 0)]), phase="solv"
        )
        self.is_surface = False
        self.closed_shell = False
        self._mass = 1.6726219e-27  # kg
        self._electrons = 0
        self._charge = 1  # ne (where e is the elementary charge, 1.602176634 × 10^-19 C)
        self._formula = "H+"

    def __str__(self) -> str:
        return "Proton(H+)"


class Hydroxide(Intermediate):
    """
    Hydroxide species (OH-)
    """

    def __init__(self):
        super().__init__(
            code="OH-",
            molecule=Atoms("HO", positions=[(0, 0, 0), (0, 0, 0.96)]),
            phase="solv",
        )
        self.is_surface = False
        self.closed_shell = False
        self._mass = 3.3496e-26  # kg
        self._electrons = 0
        self._charge = (
            -1
        )  # ne (where e is the elementary charge, 1.602176634 × 10^-19 C)
        self._formula = "OH-"

    def __str__(self) -> str:
        return "Hydroxide(OH-)"


class Water(Intermediate):
    """
    Water solvent species (H2O)
    """

    def __init__(self):
        super().__init__(
            code="H2O(aq)",
            molecule=Atoms("H2O", positions=[(0, 0, 0), (0.96, 0, 0), (0.48, 0.83, 0)]),
            phase="solv",
        )
        self.is_surface = False
        self.closed_shell = False
        self._mass = 2.991e-26  # kg
        self._electrons = 0
        self._charge = 0  # ne (where e is the elementary charge, 1.602176634 × 10^-19 C)
        self._formula = "H2O"

    def __str__(self) -> str:
        return "Water(H2O)"


class Cation(Intermediate):
    """
    Cation species (e.g. K+, Na+, Li+)
    """

    def __init__(self, metal: str, charge: int):
        super().__init__(
            code=f"{metal}+", molecule=Atoms(metal, positions=[(0, 0, 0)]), phase="solv"
        )
        self.is_surface = False
        self.closed_shell = False
        self.mass = 1.6726219e-27  # kg
        self.electrons = 0
        self.charge = (
            charge  # ne (where e is the elementary charge, 1.602176634 × 10^-19 C)
        )
        self.formula = f"{metal}" + "+" * charge

    def __str__(self) -> str:
        return f"{self.code}({self.formula})"
