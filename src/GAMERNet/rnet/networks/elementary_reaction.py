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

    r_types = REACTION_TYPES
    
    def __init__(self, 
                 code: str=None, 
                 components: tuple[frozenset[Intermediate]]=None, 
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
        for component in self.components:
            for inter in component:
                out_str += '[{}]'.format(str(abs(self.stoic[inter]))) + inter.__str__() + '+'
            out_str = out_str[:-1]
            out_str += '<->'
        return out_str[:-3]        

    def __eq__(self, other):
        if isinstance(other, ElementaryReaction):
            return self.bb_order() == other.bb_order()
        return False

    def __hash__(self):
        return hash((self.components))

    def __getitem__(self, key): 
        pass

    def __add__(self, other):
        """
        The result of adding two elementary reactions is a new elementary reaction with type 'pseudo'
        """
        if isinstance(other, ElementaryReaction):
            species = set(self.reactants) | set(self.products) | set(other.reactants) | set(other.products)
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
            step = ElementaryReaction(components=[reactants, products], r_type='pseudo')
            step.stoic = stoic_dict
            if step.energy is None or other.energy is None:
                step.energy = None
            else:
                step.energy = self.energy[0] + other.energy[0], (self.energy[1]**2 + other.energy[1]**2)**0.5
            return step
        else:
            raise TypeError('The object is not an ElementaryReaction')

    def __mul__(self, other):
        """
        The result of multiplying an elementary reaction by a number is a new elementary reaction with type 'pseudo'
        """
        if isinstance(other, float) or isinstance(other, int):
            step = ElementaryReaction(components=(self.reactants, self.products), r_type='pseudo')
            step.stoic = {}
            for k, v in self.stoic.items():
                step.stoic[k] = v * other
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
            for reactant in self.reactants:
                energy_list = [config['energy'] for _, config in reactant.ads_configs.items()]
                std_list = [config['std'] for _, config in reactant.ads_configs.items()]
                e_min_config = min(energy_list)
                std_min_config = std_list[energy_list.index(e_min_config)]
                v_h.append(self.stoic[reactant.code] * e_min_config)
                var_h.append(self.stoic[reactant.code] * std_min_config**2)
            for product in self.products:
                energy_list = [config['energy'] for _, config in product.ads_configs.items()]
                std_list = [config['std'] for _, config in product.ads_configs.items()]
                e_min_config = min(energy_list)
                std_min_config = std_list[energy_list.index(e_min_config)]
                v_h.append(self.stoic[product.code] * e_min_config)
                var_h.append(self.stoic[product.code] * std_min_config**2)
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
            self.e_act = q + (m - 1) * self.energy[0], (m * self.energy[1]**2)**0.5
        else: 
            self.e_act = q + m * self.energy[0], (m * self.energy[1]**2)**0.5
            if self.e_act[0] < self.energy[0]:
                print('The reaction barrier is lower than the reaction energy')
                self.e_act = 0.0, 0.0
            

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
        # initialize stoic_dict as zeros for all species
        stoic_dict = {}
        for specie in species:
            stoic_dict[specie.code] = 0
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
        # print(elements, species, matrix)
        # print("matrix rank:", np.linalg.matrix_rank(matrix.T))
        # print("matrix null space:", null_space(matrix.T), rcond=1e-9)
        rank = np.linalg.matrix_rank(matrix.T)
        if rank < nc -1: # inifinite² solutions
            for reactant in reactants:
                stoic_dict[reactant.code] = -1
            for product in products:
                stoic_dict[product.code] = 1
            return stoic_dict
        else:
            stoic = null_space(matrix.T)
            # remove columns that contain zeros or values close to zero
            stoic = stoic[:, np.all(np.abs(stoic) > 1e-9, axis=0)]
            # print(stoic)
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
        Reverse the elementary reaction.
        Example: A + B <-> C + D becomes C + D <-> A + B
        reaction energy and barrier are also reversed 
        """
        self.components = self.components[::-1]
        for k, v in self.stoic.items():
            self.stoic[k] = -v        
        if self.r_type in ('adsorption', 'desorption'):
            self.r_type = 'desorption' if self.r_type == 'adsorption' else 'adsorption'
        self.reactants, self.products = self.products, self.reactants
        if self.energy != None:
            self.energy = -self.energy[0], self.energy[1]
        if self.e_act != None:
            self.e_act = self.e_act[0] - self.energy[0], self.e_act[1]
        self.__repr__()