"""Module containing charged species and solvent for electrocatalysis simulations."""

from care import Intermediate
from ase import Atoms

# Constant for converting AMU (atomic mass units) to kilograms
AMU_TO_KG = 1.66053906660e-27


class ElectroSpecies(Intermediate):
    """
    Base class bridging the Intermediate ABC with specialized electrochemical 
    and solvent species that do not require graph-topology generation.
    """
    __slots__ = ("_formula", "_mass", "_electrons", "molecule", "closed_shell")

    def __init__(
        self, 
        code: str, 
        phase: str, 
        formula: str, 
        charge: int, 
        mass: float, 
        electrons: int, 
        molecule: Atoms
    ):
        super().__init__(code=code, phase=phase)
        self._formula = formula
        self.charge = charge
        self._mass = mass
        self._electrons = electrons
        self.molecule = molecule
        self.closed_shell = False  # Retaining original logic for electro species

    @property
    def formula(self) -> str:
        return self._formula

    @property
    def mass(self) -> float:
        return self._mass

    @property
    def electrons(self) -> int:
        return self._electrons

    def _get_elem_count(self, key: str) -> int:
        if self.molecule is not None:
            return self.molecule.get_chemical_symbols().count(key)
        return 0


class Electron(ElectroSpecies):
    """Electron e-"""

    def __init__(self):
        super().__init__(
            code="e-", 
            phase="electro",
            formula="e-",
            charge=-1,
            mass=9.10938356e-31,  # kg
            electrons=1,
            molecule=Atoms()
        )

    def __str__(self) -> str:
        return "Electron(e-)"


class Proton(ElectroSpecies):
    """Proton H+"""

    def __init__(self):
        super().__init__(
            code="H+", 
            phase="solv",
            formula="H+",
            charge=1,
            mass=1.6726219e-27,  # kg
            electrons=0,
            molecule=Atoms("H", positions=[(0, 0, 0)])
        )

    def __str__(self) -> str:
        return "Proton(H+)"


class Hydroxide(ElectroSpecies):
    """Hydroxide species (OH-)"""

    def __init__(self):
        super().__init__(
            code="OH-", 
            phase="solv",
            formula="OH-",
            charge=-1,
            mass=3.3496e-26,  # kg
            electrons=0,
            molecule=Atoms("HO", positions=[(0, 0, 0), (0, 0, 0.96)])
        )

    def __str__(self) -> str:
        return "Hydroxide(OH-)"


class Water(ElectroSpecies):
    """Water solvent species (H2O)"""

    def __init__(self):
        super().__init__(
            code="H2O(aq)", 
            phase="solv",
            formula="H2O",
            charge=0,
            mass=2.991e-26,  # kg
            electrons=0,
            molecule=Atoms("H2O", positions=[(0, 0, 0), (0.96, 0, 0), (0.48, 0.83, 0)])
        )

    def __str__(self) -> str:
        return "Water(H2O)"


class Cation(ElectroSpecies):
    """Cation species (e.g. K+, Na+, Li+)"""

    def __init__(self, metal: str, charge: int):
        mol = Atoms(metal, positions=[(0, 0, 0)])
        
        # Calculate accurate mass in kg instead of hardcoding the proton mass
        actual_mass_kg = mol.get_masses().sum() * AMU_TO_KG
        
        super().__init__(
            code=f"{metal}+", 
            phase="solv",
            formula=f"{metal}" + "+" * charge,
            charge=charge,
            mass=actual_mass_kg,
            electrons=0,
            molecule=mol
        )

    def __str__(self) -> str:
        return f"{self.code}({self.formula})"