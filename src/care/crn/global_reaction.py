import re
from typing import Optional

import numpy as np
from scipy.linalg import null_space

from care import Intermediate, format_reaction
from care.constants import INTER_ELEMS


class GlobalReaction:
    """Base class for representing global reactions.

    Attributes:
        components (list of frozensets): List containing the frozensets.
            with the components of the reaction.
        stoic (dict): Dictionary containing the stoichiometry of the reaction.
    """
    __slots__ = (
        "_components", "_reactants", "_products", "stoic",
        "e_is", "e_ts", "e_fs", "e_rxn", "e_act",
        "k_dir", "k_rev", "k_eq", "rate",
        "_repr_str",
        "is_graph", "fs_graph",
        "is_atoms", "fs_atoms", "_code", "_repr_hr"
    )

    def __init__(
        self,
        components: tuple[frozenset[Intermediate]] = None,
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

        self.stoic = stoic
        if self.stoic is None:
            self.stoic = self.solve_stoichiometry()

        self.is_graph = None
        self.fs_graph = None
        self.is_atoms = None
        self.fs_atoms = None

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
    
    @property
    def _signature(self):
        r_codes = tuple(sorted(i.code for i in self.reactants))
        p_codes = tuple(sorted(i.code for i in self.products))
        sides = [r_codes, p_codes]
        sides.sort()
        return tuple(sides)

    def __str__(self) -> str:
        return self.__repr__()
    
    def __eq__(self, other):
        if not isinstance(other, GlobalReaction):
            return NotImplemented
        return self._signature == other._signature

    def __hash__(self):
        return hash(self._signature)
    
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
        """
        Solve the stoichiometry of the elementary reaction.
        sum_i nu_i * S_i = 0 (nu_i are the stoichiometric coefficients and S_i are the species)

        Returns:
            dict containing the stoichiometry of the elementary reaction.
        """
        reactants = [specie for specie in self.reactants]  # self.reactants is a frozenset
        products = [specie for specie in self.products]  # self.products is a frozenset
        species = reactants + products
        matrix = np.zeros((len(species), len(INTER_ELEMS)))
        for i, inter in enumerate(species):
            for j, element in enumerate(INTER_ELEMS):
                matrix[i, j] = inter[element]
        default_coeffs = np.array([-1] * len(self.reactants) + [1] * len(self.products))  # Initial guess
        if np.allclose(matrix.T @ default_coeffs, 0):
            return dict(zip([s.code for s in species], default_coeffs.tolist()))
        ns = null_space(matrix.T)
        if ns.size == 0:
            raise ValueError("No stoichiometric solution found (Null space is empty).")
        stoic = ns[:, 0]
        min_val = np.min(np.abs(stoic[np.abs(stoic) > 1e-9]))
        stoic = np.round(stoic / min_val)
        if stoic[0] > 0:
            stoic *= -1
        return dict(zip([s.code for s in species], stoic.astype(int).tolist()))

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

    @property
    def is_evaluated(self) -> bool:
        """
        Check if all intermediates in the reaction have been energetically evaluated.
        """
        for attr in [self.e_is, self.e_fs, self.e_rxn, self.e_act, self.e_ts]:
            if attr is None:
                return False
        return True
