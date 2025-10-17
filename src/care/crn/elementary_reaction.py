from copy import deepcopy
import re
from typing import Optional

import numpy as np
from scipy.linalg import null_space

from care import Intermediate, format_reaction
from care.constants import INTER_ELEMS, R_TYPES, K_B, H


class ElementaryReaction:
    """Base class for representing elementary reactions.

    Attributes:
        components (list of frozensets): List containing the frozensets.
            with the components of the reaction.
        r_type (str): Elementary reaction type.
    """
    __slots__ = (
        "_components", "r_type", "stoic",
        "e_is", "e_ts", "e_fs", "e_rxn", "e_act",
        "k_dir", "k_rev", "k_eq", "rate",
        "_repr_str", "extra_intermediates",
        "neb_images", "neb_energies",
        "is_graph", "ts_graph", "fs_graph",
        "is_atoms", "fs_atoms", "_code", "_repr_hr"
    )
    r_types: list[str] = R_TYPES

    def __init__(
        self,
        components: tuple[frozenset[Intermediate]] = None,
        r_type: str = None,
        stoic: dict[str, float] = None,
    ):
        self._components = None
        self.components = components
        self._code = None

        # enthalpy attributes (mu, std)
        self.e_is: Optional[tuple[float, float]] = None  # initial state
        self.e_ts: Optional[tuple[float, float]] = None  # transition state
        self.e_fs: Optional[tuple[float, float]] = None  # final state
        self.e_rxn: Optional[tuple[float, float]] = None  # reaction energy
        self.e_act: Optional[tuple[float, float]] = None  # activation energy

        # Kinetic constants
        self.k_dir: Optional[float] = None  # direct rate constant
        self.k_rev: Optional[float] = None  # reverse rate constant
        self.k_eq: Optional[float] = None  # equilibrium constant

        self.r_type: str = r_type
        if self.r_type not in self.r_types:
            raise ValueError(f"Invalid reaction type: {self.r_type}")
        self.stoic = stoic
        if self.r_type != "pseudo" and self.stoic is None:
            self.stoic = self.solve_stoichiometry()

        self.neb_images = None
        self.neb_energies = None
        self.is_graph = None
        self.ts_graph = None
        self.fs_graph = None
        self.is_atoms = None
        self.fs_atoms = None
        self.extra_intermediates = {}

    @property
    def reactants(self):
        return self.components[0] if self.components else []

    @property
    def products(self):
        return self.components[1] if self.components else []

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
        return " + ".join(lhs) + " \u27F9 " + " + ".join(rhs)

    def get_repr_hr(self) -> str:
        def sort_key(s):
            if re.fullmatch(r"\[\d+\]\*", s):
                return (1, 0)  # [#]* group
            elif re.fullmatch(r"\[\d+\]H\+\(solv\)", s):
                return (2, 0)  # [#]H+(solv) group
            elif re.fullmatch(r"\[\d+\]e-", s):
                return (3, 0)  # [#]e- group
            else:
                return (0, 0)  # normal entries
        comps_str = []
        for component in self.components:
            inters_str = []
            for inter in component:
                if inter.phase == "surf":
                    out_str = "[{}]".format(str(abs(self.stoic[inter.code]))) + "*"
                elif inter.phase == "gas":
                    out_str = (
                        "[{}]".format(str(abs(self.stoic[inter.code])))
                        + inter.formula
                        + "(g)"
                    )
                elif inter.phase == "solv":
                    out_str = (
                        "[{}]".format(str(abs(self.stoic[inter.code])))
                        + inter.formula
                        + "(solv)"
                    )
                elif inter.phase == "electro":
                    out_str = (
                        "[{}]".format(str(abs(self.stoic[inter.code])))
                        + inter.formula
                    )
                else:
                    out_str = (
                        "[{}]".format(str(abs(self.stoic[inter.code])))
                        + inter.formula
                        + "*"
                    )
                inters_str.append(out_str)
            inters_str_sorted = sorted(inters_str)
            inters_str_sorted = sorted(inters_str_sorted, key=sort_key)
            comp_str = " + ".join(inters_str_sorted)
            comps_str.append(comp_str)
        return format_reaction(" \u27F9 ".join(comps_str))

    def __str__(self) -> str:
        return self.__repr__()

    def __eq__(self, other):
        if isinstance(other, ElementaryReaction):
            return frozenset(self.components) == frozenset(other.components)
        return False

    def __hash__(self):
        return id(self)
    
    def __getitem__(self, key):
        all_species = list(self.reactants) + list(self.products)
        return all_species[key]

    def __len__(self):
        return len(self.reactants) + len(self.products)

    def __iter__(self):
        return iter(list(self.reactants) + list(self.products))

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
        self._code = None
        self._repr_hr = None

    @property
    def code(self):
        if not hasattr(self, "_code") or self._code is None:
            self._code = self.__repr__()
        return self._code
    
    @property
    def repr_hr(self):
        if not hasattr(self, "_repr_hr") or self._repr_hr is None:
            self._repr_hr = self.get_repr_hr()
        return self._repr_hr
    
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
        if self.e_rxn:
            self.e_rxn = -self.e_rxn[0], self.e_rxn[1]
            self.e_is, self.e_fs = self.e_fs, self.e_is

        if self.e_act:
            self.e_act = (
                self.e_act[0] + self.e_rxn[0], # As e_rxn already stores the reverse rxn energy, we add, not substract!
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
        self, t: float, uq: bool = False, clip_eact: float = -1.0
    ) -> tuple:
        """
        Evaluate the kinetic constants of the reactions in the network
        with transition state theory and Hertz-Knudsen equation.

        Args:
            t (float): Temperature in Kelvin.
            uq (bool, optional): If True, the uncertainty of the activation
                energy and the reaction energy will be considered. Defaults to
                False.
            clip_eact (float, optional): If > 0.0, the activation energy will be clipped, only if 
                both the forward and reverse activation energies are > clip_eact.
                if zero, the reaction will be assumed to be barrierless.
        """
        e_act = np.random.normal(self.e_act[0], self.e_act[1]) if uq else self.e_act[0]
        e_rxn = np.random.normal(self.e_rxn[0], self.e_rxn[1]) if uq else self.e_rxn[0]
        e_act_rev = e_act - e_rxn
        
        if isinstance(clip_eact, (float, int)):
            if clip_eact > 0.0 and e_act > 0 and e_act_rev > 0:
                if e_act > clip_eact and e_act_rev > clip_eact:
                    if e_act >= e_act_rev:
                        e_act = clip_eact + e_rxn
                    else:
                        e_act = clip_eact
            if clip_eact == 0.0:
                e_act = max(0.0, e_rxn)
        elif isinstance(clip_eact, dict):
            x = self.r_type
            alpha, beta = clip_eact.get(x, (1, 0))
            if "BondFormation" in self.__class__.__name__:
                e_act = beta - self.e_rxn[0] * alpha + self.e_rxn[0]
            elif "BondBreaking" in self.__class__.__name__:
                e_act = beta + self.e_rxn[0] * alpha
            else:
                pass
            e_act = max(0, e_act)
        else:
            pass
        
        k_dir = (K_B * t / H) * np.exp(-e_act / t / K_B)
        k_eq = np.exp(-e_rxn / t / K_B)
        return k_dir, k_dir / k_eq

    def update_intermediates(self, evaluated_dict: dict[str, Intermediate]):
        """
        Update the intermediates of the elementary reaction with evaluated ones.

        Args:
            evaluated_dict (dict): Dictionary mapping Intermediate codes to
                                evaluated Intermediate objects.
        """
        for component in self.components:
            for inter in component:
                if inter.code in evaluated_dict:
                    inter.ads_configs = evaluated_dict[inter.code].ads_configs
        if self.r_type == "PCET":
            self.extra_intermediates["XLYOFNOQVPJJNP-UHFFFAOYSA-Ng"] = evaluated_dict.get("XLYOFNOQVPJJNP-UHFFFAOYSA-Ng")  # H2O
            self.extra_intermediates["UFHFLCQGNIYNRP-UHFFFAOYSA-N"] = evaluated_dict.get("UFHFLCQGNIYNRP-UHFFFAOYSA-Ng")  # H2

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
