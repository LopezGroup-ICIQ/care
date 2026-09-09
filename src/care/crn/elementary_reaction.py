from copy import deepcopy
import re
from typing import Optional

import numpy as np
from scipy.linalg import null_space

from care import Intermediate, format_reaction
from care.crn.intermediate import AdsorbedSpecies, GasSpecies, SurfaceSite
from care.constants import INTER_ELEMS, R_TYPES, K_B, H


class ElementaryReaction:
    """Base class for representing elementary reactions.

    Attributes:
        components (list of frozensets): List containing the frozensets.
            with the components of the reaction.
        r_type (str): Elementary reaction type.
    """
    __slots__ = (
        "_components", "_reactants", "_products", "r_type", "stoic",
        "k_dir", "k_rev", "k_eq", "rate",
        "_repr_str", "extra_intermediates",
        "neb_images", "neb_energies",
        "is_graph", "ts_graph", "fs_graph",
        "is_atoms", "fs_atoms", "_code", "_repr_hr", "_catalyst", "requires_neb", "_e_ts", 
        "_e_is", "_e_fs", "_e_rxn", "_e_act"
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

        # Kinetic constants
        self.k_dir: Optional[float] = None  # direct rate constant
        self.k_rev: Optional[float] = None  # reverse rate constant
        self.k_eq: Optional[float] = None  # equilibrium constant

        self.r_type: str = r_type
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

        self._catalyst = None
        self.requires_neb = False
        self._e_ts = None
        self._e_is = None 
        self._e_fs = None 
        self._e_rxn = None
        self._e_act = None

    @property
    def reactants(self):
        return self.components[0] if self.components else []

    @property
    def products(self):
        return self.components[1] if self.components else []
    
    @property
    def catalyst(self):
        return self._catalyst
    
    @catalyst.setter
    def catalyst(self, catalyst):
        self._catalyst = catalyst
        for species in self:
            if isinstance(species, AdsorbedSpecies):
                species.catalyst = catalyst

    @property
    def e_is(self) -> float:
        """Dynamically sums the energies of the reactants or returns explicit override."""
        if self._e_is is not None:
            return self._e_is
            
        for species in self.reactants:
            if getattr(species, 'phase', None) != "surf" and getattr(species, 'E', None) is None:
                return None
                
        return sum(
            abs(min(0, self.stoic[species.code])) * species.E 
            for species in self.reactants if getattr(species, 'phase', None) != "surf"
        )

    @e_is.setter
    def e_is(self, value: float):
        self._e_is = value
        self._e_rxn = None  # Reset reaction energy when IS energy is set
        self._e_act = None  # Reset activation energy when IS energy is set

    @property
    def e_fs(self) -> float:
        """Dynamically sums the energies of the products or returns explicit override."""
        if self._e_fs is not None:
            return self._e_fs
            
        for species in self.products:
            if getattr(species, 'phase', None) != "surf" and getattr(species, 'E', None) is None:
                return None
                
        return sum(
            abs(max(0, self.stoic[species.code])) * species.E 
            for species in self.products if getattr(species, 'phase', None) != "surf"
        )

    @e_fs.setter
    def e_fs(self, value: float):
        self._e_fs = value
        self._e_rxn = None  # Reset reaction energy when FS energy is set
        self._e_act = None  # Reset activation energy when FS energy is set

    @property
    def e_rxn(self) -> float:
        """Thermodynamic reaction energy: E_FS - E_IS or explicit override."""
        if self._e_rxn is not None:
            return self._e_rxn
        if self.e_fs is None or self.e_is is None:
            return None
        return self.e_fs - self.e_is

    @e_rxn.setter
    def e_rxn(self, value):
        if self.e_is is not None and self.e_fs is not None:
            raise ValueError(
                "Cannot explicitly set e_rxn when both e_is and e_fs are already defined. "
                "Modify the absolute state energies to update the reaction energy."
            )
        self._e_rxn = value
        self._e_is = None
        self._e_fs = None
    
    @property
    def e_ts(self) -> float:
        """
        Transition state energy.
        Returns the explicitly set TS energy (from NEB/GNN), 
        or falls back to the maximum of IS/FS energies (barrierless).
        """
        if self.e_is is None or self.e_fs is None:
            return None

        if self._e_ts is None or self._e_ts < max(self.e_is, self.e_fs):
            return self.e_is if self.e_is > self.e_fs else self.e_fs

        return self._e_ts
    
    @e_ts.setter
    def e_ts(self, value: float):
        self._e_ts = value
        self._e_act = None

    @property
    def e_act(self) -> float:
        """Activation energy: E_TS - E_IS or explicit override."""
        if self._e_act is not None:
            return self._e_act
        if self.e_ts is None or self.e_is is None:
            return None            
        return self.e_ts - self.e_is
    
    @e_act.setter
    def e_act(self, value: float):
        if self.e_is is not None and self.e_ts is not None:
            raise ValueError(
                "Cannot explicitly set e_act when both e_is and e_ts are already defined. "
                "Modify the absolute state energies to update the activation barrier."
            )
        self._e_act = value
        self._e_ts = None
    
    @property 
    def e_act_rev(self) -> float:
        return self.e_act - self.e_rxn
    
    @property
    def e_rxn_rev(self) -> float:
        return -self.e_rxn

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
        if not isinstance(other, ElementaryReaction):
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
        self._code = self.__repr__()
        self.is_atoms, self.fs_atoms = self.fs_atoms, self.is_atoms
        self.is_graph, self.fs_graph = self.fs_graph, self.is_graph
        if self.neb_images is not None:
            self.neb_images = self.neb_images[::-1]
        if self.neb_energies is not None:
            self.neb_energies = self.neb_energies[::-1]

        self.k_dir, self.k_rev = self.k_rev, self.k_dir
        
        if self.k_eq is not None:
            self.k_eq = 1.0 / self.k_eq if self.k_eq != 0 else float('inf')
            
        if getattr(self, "rate", None) is not None:
            self.rate = -self.rate

        self._e_is, self._e_fs = self._e_fs, self._e_is
        if self._e_rxn is not None:
            self._e_rxn = -self._e_rxn

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
        self, t: float, clip_eact: float = -1.0
    ) -> tuple:
        """
        Evaluate the kinetic constants of the reactions in the network
        with transition state theory.
        """
        e_act = self.e_act
        e_rxn = self.e_rxn
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
                e_act = beta - e_rxn * alpha + e_rxn
            elif "BondBreaking" in self.__class__.__name__:
                e_act = beta + e_rxn * alpha
            else:
                pass
            e_act = max(0, e_act)
            
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
                    if isinstance(inter, AdsorbedSpecies):
                        inter.ads_configs = evaluated_dict[inter.code].ads_configs
                    elif isinstance(inter, GasSpecies):
                        inter.molecule = evaluated_dict[inter.code].molecule
                        inter.E = evaluated_dict[inter.code].E
                    elif isinstance(inter, SurfaceSite):
                        inter.E = evaluated_dict[inter.code].E
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
                step.e_rxn = self.e_rxn + other.e_rxn
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
                    step.e_rxn = self.e_rxn * other
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
                    step.e_rxn = rev.e_rxn * abs(other)
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
    def is_thermo_evaluated(self) -> bool:
        """Check if thermodynamic states (IS, FS, reaction energy) are evaluated."""
        return all(attr is not None for attr in [self.e_is, self.e_fs])

    @property
    def is_kinetic_evaluated(self) -> bool:
        """Check if kinetic states are evaluated."""
        return self._e_ts is not None if self.requires_neb else True

    @property    
    def is_evaluated(self) -> bool:
        """Check if both thermodynamics and kinetics are fully evaluated."""
        return self.is_thermo_evaluated and self.is_kinetic_evaluated


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
