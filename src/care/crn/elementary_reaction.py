from typing import Optional, Union

import numpy as np
from rdkit import Chem
from scipy.linalg import null_space
from torch_geometric.data import Data

from care import Intermediate
from care.constants import INTER_ELEMS, R_TYPES


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
        is_electro: bool = False,
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
                        code="*"
                        if comp.GetNumAtoms() == 0
                        else Chem.inchi.MolToInchiKey(comp),
                        molecule=comp,
                        phase="surf" if comp.GetNumAtoms() == 0 else "ads",
                        is_surface=True if comp.GetNumAtoms() == 0 else False,
                    )
                    for comp in self.components[0]
                ],
                [
                    Intermediate(
                        code="*"
                        if comp.GetNumAtoms() == 0
                        else Chem.inchi.MolToInchiKey(comp),
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
        self._bader_energy = None
        self.r_type: str = r_type
        if self.r_type not in self.r_types:
            raise ValueError(f"Invalid reaction type: {self.r_type}")
        self.is_electro = is_electro
        self.stoic = stoic
        if self.r_type != "pseudo" and self.stoic is None:
            self.stoic = self.solve_stoichiometry()

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        out_str = ""
        for component in self.components:
            for inter in component:
                if inter.phase == "surf":
                    out_str += (
                        "[{}]".format(str(abs(self.stoic[inter.code]))) + "*" + "+"
                    )
                else:
                    out_str += (
                        "[{}]".format(str(abs(self.stoic[inter.code])))
                        + inter.__str__()
                        + "+"
                    )
            out_str = out_str[:-1]
            out_str += "<->"
        return out_str[:-3]

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

    def __eq__(self, other):
        if isinstance(other, ElementaryReaction):
            return frozenset(self.components) == frozenset(other.components)
        return False

    def __hash__(self):
        return hash((self.components))

    def __getitem__(self, key):
        pass

    def __add__(self, other) -> "ElementaryReaction":
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
            step = ElementaryReaction(components=[reactants, products], r_type="pseudo")
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

    def __mul__(self, other) -> "ElementaryReaction":
        """
        The result of multiplying an elementary reaction by a scalar is a new elementary reaction with type 'pseudo'
        """
        if isinstance(other, float) or isinstance(other, int):
            step = ElementaryReaction(
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
            raise TypeError("The object is not an ElementaryReaction")

    def __rmul__(self, other):
        return self.__mul__(other)

    @property
    def bader_energy(self):
        if self._bader_energy is None:
            return self.energy
        return self._bader_energy

    @bader_energy.setter
    def bader_energy(self, other):
        self._bader_energy = other

    def bb_order(self):
        """
        Set the elementary reaction in the bond-breaking direction, e.g.:
        CH4 + * -> CH3 + H*
        If is not in the bond-breaking direction, reverse it
        Adsorption steps are reversed to desorption steps, while desorption steps are preserved
        """
        if self.r_type in ("adsorption", "desorption"):
            if self.r_type == "adsorption":
                self.reverse()
            else:
                pass
        else:
            size_reactants, size_products = [], []
            for reactant in self.reactants:
                if not reactant.is_surface:
                    size_reactants.append(len(reactant.molecule))
            for product in self.products:
                if not product.is_surface:
                    size_products.append(len(product.molecule))
            if max(size_reactants) < max(size_products):
                self.reverse()
            else:
                pass

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
        }  # guess (correct for most of the steps)
        elements = INTER_ELEMS
        nc, na = len(species), len(elements)
        matrix = np.zeros((nc, na))
        for i, inter in enumerate(species):
            for j, element in enumerate(elements):
                if element == "*" and inter.phase != "gas":
                    matrix[i, j] = 1
                else:
                    matrix[i, j] = (
                        species[i].molecule.get_chemical_symbols().count(element)
                    )
        y = np.zeros((na, 1))
        for i, _ in enumerate(elements):
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
        Reverse the elementary reaction inplace.
        Example: A + B <-> C + D becomes C + D <-> A + B
        reaction energy and barrier are also reversed
        """
        self.components = self.components[::-1]
        for k, v in self.stoic.items():
            self.stoic[k] = -v
        if self.r_type in ("adsorption", "desorption"):
            self.r_type = "desorption" if self.r_type == "adsorption" else "adsorption"
        self.reactants, self.products = self.products, self.reactants
        if self.e_rxn != None:
            self.e_rxn = -self.e_rxn[0], self.e_rxn[1]
            self.e_is, self.e_fs = self.e_fs, self.e_is
        if "-" not in self.r_type:
            self.e_act = max(0, self.e_rxn[0]), self.e_rxn[1]
        else:
            self.e_act = (
                self.e_ts[0] - self.e_is[0],
                (self.e_ts[1] ** 2 + self.e_is[1] ** 2) ** 0.5,
            )
            if self.e_act[0] < 0:
                self.e_act = 0, self.e_rxn[1]
            if (
                self.e_act[0] < self.e_rxn[0]
            ):  # Barrier lower than self energy
                self.e_act = self.e_rxn[0], self.e_rxn[1]
        self.code = self.__repr__()
