import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import null_space

from GAMERNet.rnet.networks.intermediate import Intermediate

REACTION_TYPES = ['desorption', 'C-O', 'C-OH', 'C-H', 'H-H', 'O-O', 'C-C', 'O-H', 'O-OH', 'eley_rideal', 'adsorption', 'pseudo']

class ElementaryReaction:
    """Class for representing elementary reactions.

    Attributes:
        code (str): Code associated with the elementary reaction.
        components (list of frozensets): List containing the frozensets.
            with the components of the reaction.
        r_type (str): Elementary reaction type.
    """
    def __init__(self, 
                 code: str=None, 
                 components: tuple[list[Intermediate]]=None, 
                 r_type: str=None, 
                 is_electro: bool=False):
        self._code = code
        self._components = None
        self.components = components
        self.reactants = self.components[0]
        self.products = self.components[1]
        self.energy = None
        self.e_act = None
        self._bader_energy = None
        self.r_type = r_type
        if self.r_type not in REACTION_TYPES:
            raise ValueError(f'Invalid reaction type: {self.r_type}')
        self.is_electro = is_electro
        if self.r_type != 'pseudo':
            self.stoic = self.solve_stoichiometry()
        else:
            self.stoic = None      

    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        out_str = ''
        for i, comp in enumerate(self.components):
            for j, inter in enumerate(comp):
                out_str += '[{}]'.format(str(abs(self.stoic[i][j]))) + inter.__str__() + '+'
            out_str = out_str[:-1]
            out_str += '<->'
        return out_str[:-3]        

    def __eq__(self, other):
        if isinstance(other, ElementaryReaction):
            return self.bb_order() == other.bb_order()
        return False

    def __hash__(self):
        return hash((self.components)) 

    def __add__(self, other):
        if isinstance(other, ElementaryReaction):
            stoic1_dict, stoic2_dict, stoic_dict = {}, {}, {}
            for i, inter in enumerate(self.reactants):
                stoic1_dict[inter] = self.stoic[0][i]
            for i, inter in enumerate(self.products):
                stoic1_dict[inter] = self.stoic[1][i]
            for i, inter in enumerate(other.reactants):
                stoic2_dict[inter] = other.stoic[0][i]
            for i, inter in enumerate(other.products):
                stoic2_dict[inter] = other.stoic[1][i]
            species = set(self.reactants) | set(self.products) | set(other.reactants) | set(other.products)
            for specie in species:
                stoic_dict[specie] = 0
            for specie in stoic1_dict.keys():
                stoic_dict[specie] += stoic1_dict[specie]
            for specie in stoic2_dict.keys():
                stoic_dict[specie] += stoic2_dict[specie]
            reactants, products, stoic = [], [], [[], []]
            for specie in list(stoic_dict.keys()):
                if stoic_dict[specie] == 0:
                    pass
                elif stoic_dict[specie] > 0:
                    products.append(specie)
                    stoic[1].append(stoic_dict[specie])
                else:
                    reactants.append(specie)
                    stoic[0].append(stoic_dict[specie])
            step = ElementaryReaction(components=[reactants, products], r_type='pseudo')
            step.stoic = stoic
            if step.energy is None or other.energy is None:
                step.energy = None
            else:
                step.energy = self.energy[0] + other.energy[0], (self.energy[1]**2 + other.energy[1]**2)**0.5
            return step
        else:
            raise TypeError('The object is not an ElementaryReaction')

    def __mul__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            step = ElementaryReaction(components=(self.reactants, self.products), r_type='pseudo')
            step.stoic = [[], []]
            for v in range(len(self.stoic[0])):
                step.stoic[0].append(self.stoic[0][v] * other)
            for v in range(len(self.stoic[1])):
                step.stoic[1].append(self.stoic[1][v] * other)
            if step.energy is None:
                step.energy = None
            else:
                step.energy = self.energy[0] * other, self.energy[1]
            return step
        else:
            raise TypeError('The object is not an ElementaryReaction')
        
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
        if self.r_type in ('adsorption', 'desorption'):
            if self.r_type == 'adsorption':
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

    def calc_reaction_energy(self, bader=False, min_state=True):
        """
        Get the reaction energy of the elementary reaction.

        Args:
            bader (bool, optional): If True, the reaction energy will be
                calculated using the Bader energies of the intermediates.
                Defaults to False.
            min_state (bool, optional): If True, the reaction energy will be
                calculated using the minimum energy of the intermediates.
                Defaults to True.

        Returns:
            float: Reaction energy of the elementary reaction in eV.
        """
        if min_state:
            v_h = []
            var_h = []
            for i, reactant in enumerate(self.reactants):
                energy_list = [config['energy'] for _, config in reactant.ads_configs.items()]
                std_list = [config['std'] for _, config in reactant.ads_configs.items()]
                e_min_config = min(energy_list)
                std_min_config = std_list[energy_list.index(e_min_config)]
                v_h.append(self.stoic[0][i] * e_min_config)
                var_h.append(self.stoic[0][i] * std_min_config**2)
            for i, product in enumerate(self.products):
                energy_list = [config['energy'] for _, config in product.ads_configs.items()]
                std_list = [config['std'] for _, config in product.ads_configs.items()]
                e_min_config = min(energy_list)
                std_min_config = std_list[energy_list.index(e_min_config)]
                v_h.append(self.stoic[1][i] * e_min_config)
                var_h.append(self.stoic[1][i] * std_min_config**2)
        else:
            pass
        self.energy = sum(v_h), sum(var_h)**0.5

    def calc_reaction_barrier(self, bader: bool=False, bep_params: list[float]=[0.4, 0.7], min_state: bool=True):
        """
        Get elementary reaction barrier with BEP theory.

        """
        self.calc_reaction_energy(bader=bader, min_state=min_state)
        q, m = bep_params
        if self.energy[0] < 0:
            # return q + (m - 1) * rxn_energy, (m * std**2)**0.5
            self.e_act = q + (m - 1) * self.energy[0], (m * self.energy[1]**2)**0.5
        else: 
            # return q + m * rxn_energy, (m * std**2)**0.5
            self.e_act = q + m * self.energy[0], (m * self.energy[1]**2)**0.5
            if self.e_act[0] < self.energy[0]:
                # raise ValueError('The reaction barrier is lower than the reaction energy')
                # warning
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
                _.append(list(item))
            self._components = tuple(_)

    @property
    def code(self):
        if self._code is None:
            self._code = self.__repr__()
        return self._code

    @code.setter
    def code(self, other):
        self._code = other

    def add_components(self, pair):
        """Add components to the elementary reaction.

        Args:
            pair (list of Intermediate): Intermediates that will be added to
                the components.
        """
        new_pair = list(pair)
        self.components.append(new_pair)
    
    def solve_stoichiometry(self) -> list[list[int]]:
        """Solve the stoichiometry of the elementary reaction.
        sum_i nu_i * S_i = 0 (nu_i are the stoichiometric coefficients and S_i are the species)

        Returns:
            dict containing the stoichiometry of the elementary reaction.
        """
        reactants = [specie for specie in self.reactants]
        products = [specie for specie in self.products]
        species = reactants + products
        elements = set()
        for specie in species:
            if specie.phase != 'gas':
                elements.add('*')
            if not specie.is_surface:
                elements.update(specie.molecule.get_chemical_symbols())
        elements = list(elements)
        nc, na = len(species), len(elements)
        matrix = np.zeros((nc, na))
        for i, inter in enumerate(species):
            for j, element in enumerate(elements):
                if element == '*' and inter.phase != 'gas':
                    matrix[i, j] = 1
                else:
                    matrix[i, j] = species[i].molecule.get_chemical_symbols().count(element)
        stoic = null_space(matrix.T)
        min_abs = min([abs(x) for x in stoic])
        stoic = np.round(stoic / min_abs).astype(int)
        if stoic[0] > 0:
            stoic = [-x for x in stoic]
        stoic = [int(x[0]) for x in stoic]        
        return [stoic[:len(reactants)], stoic[len(reactants):]]
    
    def reverse(self):
        """
        Reverse the elementary reaction.
        Example: A + B <-> C + D becomes C + D <-> A + B
        reaction energy and barrier are also reversed 
        """
        self.components = self.components[::-1]
        self.stoic = self.stoic[::-1]
        for v in range(len(self.stoic[0])):
            self.stoic[0][v] *= -1
        for v in range(len(self.stoic[1])):
            self.stoic[1][v] *= -1
        if self.r_type in ('adsorption', 'desorption'):
            self.r_type = 'desorption' if self.r_type == 'adsorption' else 'adsorption'
        self.reactants, self.products = self.products, self.reactants
        if self.energy != None:
            self.energy = -self.energy[0], self.energy[1]
        if self.e_act != None:
            self.e_act = self.e_act[0] - self.energy[0], self.e_act[1]
        self.__repr__()