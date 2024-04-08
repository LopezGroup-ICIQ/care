from typing import Optional, Union

import numpy as np
from rdkit import Chem
from scipy.linalg import null_space
from torch_geometric.data import Data

from care import Intermediate
from care.constants import INTER_ELEMS, R_TYPES, K_B, H, K_BU
from care.crn.utilities.electro import Electron, Proton, Hydroxide, Water

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

        self.alpha = 0.5

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
        self.stoic = stoic
        if self.r_type != "pseudo" and self.stoic is None:
            self.stoic = self.solve_stoichiometry()

        if self.r_type in ("adsorption", "desorption"):
            adsorbate = [inter for inter in self.reactants if inter.phase == "gas"][0]
            self.adsorbate_mass = adsorbate.mass
        else:
            self.adsorbate_mass = None
    
    def __lt__(self, other):
        return self.code < other.code


    def __repr__(self) -> str:
        out_str = ""

        lhs, rhs = [], []
        for inter in self.components[0]:
            if inter.phase == "surf":
                out_str = (
                    "[{}]".format(str(abs(self.stoic[inter.code]))) + "*"
                )
            else:
                out_str = (
                    "[{}]".format(str(abs(self.stoic[inter.code])))
                    + inter.__str__()
                )
            lhs.append(out_str)
        for inter in self.components[1]:
            if inter.phase == "surf":
                out_str = (
                    "[{}]".format(str(abs(self.stoic[inter.code]))) + "*"
                )
            else:
                out_str = (
                    "[{}]".format(str(abs(self.stoic[inter.code])))
                    + inter.__str__()
                )
            rhs.append(out_str)
        # sort alphabetically
        lhs.sort()
        rhs.sort()
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
        y = self.__repr__() + "\n"
        y += self.repr_hr + "\n"
        y += f"Type: {self.r_type}\n"
        y += f"Reaction energy (eV): {self.e_rxn}\n"
        y += f"Activation energy (eV): {self.e_act}\n"    
        
        return y

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
        The result of multiplying an elementary reaction by a scalar
        is a new elementary reaction with type 'pseudo'
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

        if self.e_act:
            if "-" not in self.r_type:
                if self.r_type == "PCET":
                    self.e_act = (
                        self.e_act[0] - self.e_rxn[0],
                        (self.e_act[1] ** 2 + self.e_rxn[1] ** 2) ** 0.5,
                    )
                    if self.e_act[0] < 0:
                        self.e_act = 0, self.e_rxn[1]
                    if self.e_act[0] < self.e_rxn[0]:
                        self.e_act = self.e_rxn[0], self.e_rxn[1]
                        
                else:
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


    def bb(self):
        """Set reaction to bond-breaking direction."""
        self.bb_order()
        

    def bf(self):
        """        
        Set reaction to bond-forming direction.
        """
        self.bb_order()
        self.reverse()

    def electro_rxn(self, pH: float, h2o_gas: Intermediate, oh_code: str, surface_inter: Intermediate):
        """
        Adjust the elementary reaction to electrochemical nomenclature.

        Parameters:
        ----------
        pH: float
            pH of the system.
        h2o_gas: Intermediate
            Water molecule in the gas phase.
        oh_code: str
            Code of the hydroxide intermediate.
        surface_inter: Intermediate
            Intermediate representing the surface.

        Returns:
        -------
        None
        """

        if self.r_type in ("C-H", "H-O"):

            reactants = [inter for inter in self.reactants]
            products = [inter for inter in self.products]

            new_reactants, new_products = [], []
            for reactant in reactants:
                if reactant.is_surface:
                    if pH <= 7:
                        continue
                    else:
                        new_reactants.append(Hydroxide())
                elif reactant.formula == 'H':
                    if pH <= 7:
                        new_reactants.append(Proton())
                        new_reactants.append(Electron())
                    else:
                        new_reactants.append(Water())
                        new_reactants.append(Electron())
                else:
                    if not reactant.is_surface:
                        new_reactants.append(reactant)
            for product in products:
                if product.is_surface:
                    if pH <= 7:
                        continue
                    else:
                        new_reactants.append(Hydroxide())
                elif product.formula == 'H':
                    if pH <= 7:
                        new_products.append(Proton())
                        new_products.append(Electron())
                    else:
                        new_products.append(Water())
                        new_products.append(Electron())
                else:
                    if not product.is_surface:
                        new_products.append(product)
            
            self.components = [new_reactants, new_products]
            self.reactants = new_reactants
            self.products = new_products
            self.r_type = "PCET"
            self.stoic = self.solve_stoichiometry()
            self.code = self.__repr__()

        elif self.r_type in ("C-O", "O-O"):
            for component_set in self.components:
                if oh_code in [component.code for component in component_set]:
                    if len(self.products) == 1:
                            continue
                    else:
                        new_reactants, new_products = [], []
                        for reactant in self.reactants:
                            if reactant.is_surface:
                                new_reactants.append(Electron())
                                if pH <= 7:
                                    new_reactants.append(Proton())
                                else:
                                    new_reactants.append(Water())
                            elif reactant.formula == 'HO':
                                if pH <= 7:
                                    new_reactants.append(h2o_gas)
                                    new_reactants.append(surface_inter)
                                else:
                                    new_reactants.append(h2o_gas)
                                    new_reactants.append(Hydroxide())
                            else:
                                new_reactants.append(reactant)
                        for product in self.products:
                            if product.is_surface:
                                new_products.append(Electron())
                                if pH <= 7:
                                    new_products.append(Proton())
                                else:
                                    new_products.append(Water())
                            elif product.formula == 'HO':
                                if pH <= 7:
                                    new_products.append(h2o_gas)
                                else:
                                    new_products.append(h2o_gas)
                                    new_products.append(Hydroxide()) 
                                if len(self.products) == 1:
                                    new_products.append(surface_inter)
                            else:
                                new_products.append(product)

                        self.components = [new_reactants, new_products]
                        self.reactants = new_reactants
                        self.products = new_products
                        self.r_type = "PCET"
                        self.stoic = self.solve_stoichiometry()
                        self.code = self.__repr__()
        else:
            pass

    def get_kinetic_constants(self, 
                              t: float, 
                              uq: bool = False, 
                              thermo: bool = False) -> tuple:
        """
        Evaluate the kinetic constants of the reactions in the network.

        Args:
            t (float, optional): Temperature in Kelvin. Defaults to None.
            uq (bool, optional): If True, the uncertainty of the activation
                energy and the reaction energy will be considered. Defaults to
                False.
            thermo (bool, optional): If True, the activation barriers will be 
                neglected and only the thermodynamic path is considered. Defaults
                to False.
        """
        if thermo:
            if uq:
                e_rxn = np.random.normal(self.e_rxn[0], self.e_rxn[1])
                e_act = 0 if e_rxn < 0 else e_rxn
            else:
                e_rxn = self.e_rxn[0]
                e_act = 0 if e_rxn < 0 else e_rxn
        else:
            if uq:
                e_act = np.random.normal(self.e_act[0], self.e_act[1])
                e_rxn = np.random.normal(self.e_rxn[0], self.e_rxn[1])
            else:
                e_act = self.e_act[0]
                e_rxn = self.e_rxn[0]
        k_eq = np.exp(-e_rxn / t / K_B)
        if self.r_type == "adsorption":  
            k_dir = 1e-18 / (2 * np.pi * self.adsorbate_mass * K_BU * t) ** 0.5
        else: 
            k_dir = (K_B * t / H) * np.exp(-e_act / t / K_B)
        k_rev = k_dir / k_eq

        return k_dir, k_rev


