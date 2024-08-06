from copy import deepcopy
from typing import Optional, Union

import numpy as np
from rdkit import Chem
from scipy.linalg import null_space
from torch_geometric.data import Data

from care.crn.intermediate import Intermediate
from care.constants import INTER_ELEMS, R_TYPES, K_B, H, K_BU


class ElementaryReaction:
    """Class for representing elementary reactions.

    Attributes:
        code (str): Code associated with the elementary reaction.
        components (list of frozensets): List containing the frozensets.
            with the components of the reaction.
        r_type (str): Elementary reaction type.
    """

    r_types: list[str] = R_TYPES

    def __init__(
        self,
        code: str = None,
        components: Union[
            tuple[frozenset[Chem.rdchem.Mol]], tuple[frozenset[Intermediate]]
        ] = None,
        r_type: str = None,
        stoic: dict[str, float] = None,
    ):
        self._code = code
        self._components = None
        self.components = components

        type_component = type(list(self.components[0])[0])

        if type_component == Chem.rdchem.Mol:
            self.components = (
                [
                    Intermediate(
                        code=(
                            "*"
                            if comp.GetNumAtoms() == 0
                            else Chem.inchi.MolToInchiKey(comp)
                        ),
                        molecule=comp,
                        phase="surf" if comp.GetNumAtoms() == 0 else "ads",
                        is_surface=True if comp.GetNumAtoms() == 0 else False,
                    )
                    for comp in self.components[0]
                ],
                [
                    Intermediate(
                        code=(
                            "*"
                            if comp.GetNumAtoms() == 0
                            else Chem.inchi.MolToInchiKey(comp)
                        ),
                        molecule=comp,
                        phase="surf" if comp.GetNumAtoms() == 0 else "ads",
                        is_surface=True if comp.GetNumAtoms() == 0 else False,
                    )
                    for comp in self.components[1]
                ],
            )
        elif type_component == Intermediate:
            self.components = components
        self.reactants = self.components[0]
        self.products = self.components[1]

        # enthalpy attributes (mu, std)
        self.e_is: Optional[tuple[float, float]] = None  # initial state
        self.e_ts: Optional[tuple[float, float]] = None  # transition state
        self.e_fs: Optional[tuple[float, float]] = None  # final state
        self.e_rxn: Optional[tuple[float, float]] = None  # reaction energy
        self.e_act: Optional[tuple[float, float]] = None  # activation energy

        # entropy attributes (mu, std)
        self.s_is: Optional[tuple[float, float]] = 0, 0
        self.s_ts: Optional[tuple[float, float]] = 0, 0
        self.s_fs: Optional[tuple[float, float]] = 0, 0
        self.s_rxn: Optional[tuple[float, float]] = 0, 0
        self.s_act: Optional[tuple[float, float]] = 0, 0

        # Gibbs free energy attributes (mu, std)
        self.g_is: Optional[tuple[float, float]] = None
        self.g_ts: Optional[tuple[float, float]] = None
        self.g_fs: Optional[tuple[float, float]] = None
        self.g_rxn: Optional[tuple[float, float]] = None
        self.g_act: Optional[tuple[float, float]] = None

        # Kinetic constants
        self.k_dir: Optional[float] = None  # direct rate constant
        self.k_rev: Optional[float] = None  # reverse rate constant
        self.k_eq: Optional[float] = None  # equilibrium constant

        # Reaction rate
        self.rate: Optional[float] = None

        self.ts_graph: Optional[Data] = None
        self.r_type: str = r_type
        if self.r_type not in self.r_types:
            raise ValueError(f"Invalid reaction type: {self.r_type}")
        self.stoic = stoic
        if self.r_type != "pseudo" and self.stoic is None:
            self.stoic = self.solve_stoichiometry()

    def __lt__(self, other):
        return self.code < other.code

    def __repr__(self) -> str:
        out_str = ""

        lhs, rhs = [], []
        for inter in self.components[0]:
            if inter.phase == "surf":
                out_str = "[{}]".format(str(abs(self.stoic[inter.code]))) + "*"
            else:
                out_str = (
                    "[{}]".format(str(abs(self.stoic[inter.code]))) + inter.__str__()
                )
            lhs.append(out_str)
        for inter in self.components[1]:
            if inter.phase == "surf":
                out_str = "[{}]".format(str(abs(self.stoic[inter.code]))) + "*"
            else:
                out_str = (
                    "[{}]".format(str(abs(self.stoic[inter.code]))) + inter.__str__()
                )
            rhs.append(out_str)        
        lhs.sort(), rhs.sort()  # sort alphabetically
        return " + ".join(lhs) + " <-> " + " + ".join(rhs)

    @property
    def repr_hr(self) -> str:
        comps_str = []
        for component in self.components:
            inters_str = []
            for inter in component:
                if inter.phase == "surf":
                    out_str = "[{}]".format(str(abs(self.stoic[inter.code]))) + "*"
                elif inter.phase == "gas":
                    out_str = (
                        "[{}]".format(str(abs(self.stoic[inter.code])))
                        + inter.molecule.get_chemical_formula()
                        + "(g)"
                    )
                else:
                    out_str = (
                        "[{}]".format(str(abs(self.stoic[inter.code])))
                        + inter.molecule.get_chemical_formula()
                        + "*"
                    )
                inters_str.append(out_str)
            comp_str = " + ".join(inters_str)
            comps_str.append(comp_str)
        return " <-> ".join(comps_str)

    def __str__(self) -> str:
        return self.__repr__()

    def __eq__(self, other):
        if isinstance(other, ElementaryReaction):
            return frozenset(self.components) == frozenset(other.components)
        return False

    def __hash__(self):
        return hash((self.components))

    def __getitem__(self, key):
        pass

    def __iter__(self):
        return iter(list(self.reactants) + list(self.products))

    def __add__(self, other) -> "ReactionMechanism":
        """
        The result of adding two elementary reactions is a new elementary reaction with type 'pseudo'
        """
        if isinstance(other, ElementaryReaction):
            species = (
                set(self.reactants)
                | set(self.products)
                | set(other.reactants)
                | set(other.products)
            )
            stoic_dict = {}
            for k, v in self.stoic.items():
                stoic_dict[k] = v
            for k, v in other.stoic.items():
                if k in stoic_dict.keys():
                    stoic_dict[k] += v
                else:
                    stoic_dict[k] = v
            for k, v in list(stoic_dict.items()):
                if v == 0:
                    del stoic_dict[k]
            reactants, products = [], []
            for specie in species:
                if specie.code not in stoic_dict.keys():
                    pass
                elif stoic_dict[specie.code] > 0:
                    products.append(specie)
                else:
                    reactants.append(specie)
            step = ReactionMechanism(components=[reactants, products], r_type="pseudo")
            step.stoic = stoic_dict
            if self.e_rxn is None or other.e_rxn is None:
                step.e_rxn = None
            else:
                step.e_rxn = (
                    self.e_rxn[0] + other.e_rxn[0],
                    (self.e_rxn[1] ** 2 + other.e_rxn[1] ** 2) ** 0.5,
                )
            return step
        else:
            raise TypeError("The object is not an ElementaryReaction")

    def __mul__(self, other) -> "ReactionMechanism":
        """
        The result of multiplying an elementary reaction by a scalar
        is a new elementary reaction with type 'pseudo'
        """
        if isinstance(other, (float, int)):
            if other > 0:
                step = ReactionMechanism(
                    components=(self.reactants, self.products), r_type="pseudo"
                )
                step.stoic = {}
                for k, v in self.stoic.items():
                    step.stoic[k] = v * other
                if self.e_rxn is None:
                    step.e_rxn = None
                else:
                    step.e_rxn = self.e_rxn[0] * other, abs(other) * self.e_rxn[1]
                return step
            else:
                rev = deepcopy(self)
                rev.reverse()
                step = ReactionMechanism(
                    components=(rev.reactants, rev.products), r_type="pseudo"
                )
                step.stoic = {}
                for k, v in rev.stoic.items():
                    step.stoic[k] = v * abs(other)
                if rev.e_rxn is None:
                    step.e_rxn = None
                else:
                    step.e_rxn = rev.e_rxn[0] * abs(other), abs(other) * rev.e_rxn[1]
                return step
        else:
            raise TypeError("other is not a scalar value")

    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __sub__(self, other) -> "ReactionMechanism":
        """
        The result of subtracting one elementary reaction from another equals 
        the sum of the first reaction and the reverse of the second reaction.
        """
        if isinstance(other, ElementaryReaction):
            return self + (-1) * other
        else:
            raise TypeError("The object is not an ElementaryReaction")

    @property
    def components(self):
        return self._components

    @components.setter
    def components(self, other):
        if other is None:
            self._components = []
        else:
            _ = []
            for item in other:
                _.append(frozenset(item))
            self._components = tuple(_)

    @property
    def code(self):
        if self._code is None:
            self._code = self.__repr__()
        return self._code

    @code.setter
    def code(self, other):
        self._code = other

    def solve_stoichiometry(self) -> dict[str, float]:
        """Solve the stoichiometry of the elementary reaction.
        sum_i nu_i * S_i = 0 (nu_i are the stoichiometric coefficients and S_i are the species)

        Returns:
            dict containing the stoichiometry of the elementary reaction.
        """
        reactants = [specie for specie in self.reactants]
        products = [specie for specie in self.products]
        species = reactants + products
        stoic_dict = {
            specie.code: -1 if specie in reactants else 1 for specie in species
        }  # initial guess (correct for most of the steps)
        matrix = np.zeros((len(species), len(INTER_ELEMS)), dtype=np.int8)
        for i, inter in enumerate(species):
            for j, element in enumerate(INTER_ELEMS):
                if element == "*" and inter.phase not in ("gas", "solv", "electro"):
                    matrix[i, j] = 1
                elif element == "q":
                    matrix[i, j] = species[i].charge
                else:
                    matrix[i, j] = species[i][element]
        y = np.zeros((len(INTER_ELEMS), 1))
        for i, _ in enumerate(INTER_ELEMS):
            y[i] = np.dot(
                matrix[:, i], np.array([stoic_dict[specie.code] for specie in species])
            )
        if np.all(y == 0):
            return stoic_dict
        else:
            stoic = null_space(matrix.T)
            stoic = stoic[:, np.all(np.abs(stoic) > 1e-9, axis=0)]
            min_abs = min([abs(x) for x in stoic])
            stoic = np.round(stoic / min_abs).astype(int)
            if stoic[0] > 0:
                stoic = [-x for x in stoic]
            stoic = [int(x[0]) for x in stoic]
            for i, specie in enumerate(species):
                stoic_dict[specie.code] = stoic[i]
        return stoic_dict

    def reverse(self):
        """
        Reverse the elementary reaction in-place.
        Example: A + B <-> C + D becomes C + D <-> A + B
        """

        self.components = self.components[::-1]
        for k, v in self.stoic.items():
            self.stoic[k] = -v
        self.reactants, self.products = self.products, self.reactants
        if self.e_rxn:
            self.e_rxn = -self.e_rxn[0], self.e_rxn[1]
            self.e_is, self.e_fs = self.e_fs, self.e_is

        if self.e_act:
            self.e_act = (
                self.e_act[0] + self.e_rxn[0], # should be minus as e_rxn is already the reverse, not the direct!
                (self.e_act[1] ** 2 + self.e_rxn[1] ** 2) ** 0.5,
            )
            
        self.code = self.__repr__()

    def bb_order(self):
        """
        Set the elementary reaction in the bond-breaking direction, e.g.:
        CH4 + * -> CH3 + H*
        """
        pass

    def bb(self):
        """Set reaction to bond-breaking direction."""
        self.bb_order()

    def bf(self):
        """
        Set reaction to bond-forming direction.
        """
        self.bb_order()
        self.reverse()

    def get_kinetic_constants(
        self, t: float, uq: bool = False, thermo: bool = False
    ) -> tuple:
        """
        Evaluate the kinetic constants of the reactions in the network
        with transition state theory and Hertz-Knudsen equation.

        Args:
            t (float): Temperature in Kelvin.
            uq (bool, optional): If True, the uncertainty of the activation
                energy and the reaction energy will be considered. Defaults to
                False.
            thermo (bool, optional): If True, the activation barriers will be
                neglected and only the thermodynamic path is considered. Defaults
                to False.
        """
        if thermo:
            e_rxn = np.random.normal(self.e_rxn[0], self.e_rxn[1]) if uq else self.e_rxn[0]
            e_act = max(0, e_rxn)
        else:
            e_act = np.random.normal(self.e_act[0], self.e_act[1]) if uq else self.e_act[0]
            e_rxn = np.random.normal(self.e_rxn[0], self.e_rxn[1]) if uq else self.e_rxn[0]
        k_eq = np.exp(-e_rxn / t / K_B)
        if self.r_type == "adsorption":
            k_dir = 1e-18 / (2 * np.pi * self.adsorbate_mass * K_BU * t) ** 0.5
        else:
            k_dir = (K_B * t / H) * np.exp(-e_act / t / K_B)

        return k_dir, k_dir / k_eq
    

class ReactionMechanism(ElementaryReaction):
    """
    Reaction mechanism class.

    A reaction mechanism is defined here as a linear combination of elementary reactions. 
    """

    def __init__(self, components, r_type, r_dict=None):
        """
        Initialize a reaction mechanism object.

        Args:
            reactions (list): List of elementary reactions.
        """
        super().__init__(components=components, r_type=r_type)
        self.r_dict = r_dict

    # def get_rate_equation(self):
    #     """
    #     Get the rate equation of the reaction mechanism.

    #     Returns:
    #         str: Rate equation of the reaction mechanism.
    #     """
    #     rate_equation = ""
    #     for reaction in self.reactions:
    #         rate_equation += f"{reaction.get_rate_equation()} + "
    #     return rate_equation[:-3]

    # def get_rate_constant(self):
    #     """
    #     Get the rate constant of the reaction mechanism.

    #     Returns:
    #         float: Rate constant of the reaction mechanism.
    #     """
    #     rate_constant = 0
    #     for reaction in self.reactions:
    #         rate_constant += reaction.get_rate_constant()
    #     return rate_constant
