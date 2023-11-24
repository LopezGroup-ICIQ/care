import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import null_space

from care.rnet.networks.intermediate import Intermediate

INTERPOL = {'O-H' : {'alpha': 0.39, 'beta': 0.89}, 
            'C-H' : {'alpha': 0.63, 'beta': 0.81},
            'H-C' : {'alpha': 0.63, 'beta': 0.81},
            'C-C' : {'alpha': 1.00, 'beta': 0.64},
            'C-O' : {'alpha': 1.00, 'beta': 1.24},
            'C-OH': {'alpha': 1.00, 'beta': 1.48}, 
            'O-O' : {'alpha': 1.00, 'beta': 1.00}, # from these, values are random
            'H-H' : {'alpha': 1.00, 'beta': 1.00},
            'O-OH' : {'alpha': 1.00, 'beta': 1.00},
            'default': {'alpha': 0.00, 'beta': 0.00}} 

class ElementaryReaction:
    """Class for representing elementary reactions.

    Attributes:
        code (str): Code associated with the elementary reaction.
        components (list of frozensets): List containing the frozensets.
            with the components of the reaction.
        r_type (str): Elementary reaction type.
    """

    r_types = ['desorption',
               'C-O',
               'C-OH', 
               'C-H',
               'H-H',
               'O-O',
               'C-C', 
               'H-O', 
               'O-OH', 
               'eley_rideal', 
               'adsorption', 
               'pseudo', 
               'rearrangement']
    
    def __init__(self, 
                 code: str=None, 
                 components: tuple[frozenset[Intermediate]]=None, 
                 r_type: str=None, 
                 is_electro: bool=False,
                 stoic: dict[str, float]=None):
        self._code = code
        self._components = None
        self.components = components
        self.reactants = self.components[0]
        self.products = self.components[1]
        self.energy = None
        self.e_act = None
        self._bader_energy = None
        self.r_type = r_type
        if self.r_type not in self.r_types:
            raise ValueError(f'Invalid reaction type: {self.r_type}')
        self.is_electro = is_electro
        if self.r_type in ('C-H', 'C-OH', 'H-O', 'O-OH', 'H-H'):
            self.is_electro = True
        self.stoic = stoic
        if self.r_type != 'pseudo' and self.stoic is None:
             self.stoic = self.solve_stoichiometry()    


    def __str__(self) -> str:
        return self.__repr__()

    def __repr__(self) -> str:
        out_str = ''
        for component in self.components:
            for inter in component:
                if inter.phase == 'surf':
                    out_str += '[{}]'.format(str(abs(self.stoic[inter.code]))) + "*" + '+'
                else:
                    out_str += '[{}]'.format(str(abs(self.stoic[inter.code]))) + inter.__str__() + '+'
            out_str = out_str[:-1]
            out_str += '<->'
        return out_str[:-3]        

    def __eq__(self, other):
        if isinstance(other, ElementaryReaction):
            return frozenset(self.components) == frozenset(other.components)
        return False

    def __hash__(self):
        return hash((self.components))

    def __getitem__(self, key): 
        pass

    def __add__(self, other) -> 'ElementaryReaction':
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
            if self.energy is None or other.energy is None:
                step.energy = None
            else:
                step.energy = self.energy[0] + other.energy[0], (self.energy[1]**2 + other.energy[1]**2)**0.5
            return step
        else:
            raise TypeError('The object is not an ElementaryReaction')

    def __mul__(self, other) -> 'ElementaryReaction':
        """
        The result of multiplying an elementary reaction by a scalar is a new elementary reaction with type 'pseudo'
        """
        if isinstance(other, float) or isinstance(other, int):
            step = ElementaryReaction(components=(self.reactants, self.products), r_type='pseudo')
            step.stoic = {}
            for k, v in self.stoic.items():
                step.stoic[k] = v * other
            if self.energy is None:
                step.energy = None
            else:
                step.energy = self.energy[0] * other, abs(other) * self.energy[1]
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
            mu_is, var_is, mu_fs, var_fs = 0, 0, 0, 0
            for reactant in self.reactants:
                energy_list = [config['energy'] for _, config in reactant.ads_configs.items()]
                std_list = [config['std'] for _, config in reactant.ads_configs.items()]
                e_min_config = min(energy_list)
                std_min_config = std_list[energy_list.index(e_min_config)]
                mu_is += abs(self.stoic[reactant.code]) * e_min_config
                var_is += self.stoic[reactant.code]**2 * std_min_config**2
            for product in self.products:
                energy_list = [config['energy'] for _, config in product.ads_configs.items()]
                std_list = [config['std'] for _, config in product.ads_configs.items()]
                e_min_config = min(energy_list)
                std_min_config = std_list[energy_list.index(e_min_config)]
                mu_fs += abs(self.stoic[product.code]) * e_min_config
                var_fs += self.stoic[product.code]**2 * std_min_config**2
            mu_dhr = mu_fs - mu_is
            var_dhr = var_fs + var_is
            std_dhr = var_dhr**0.5
        else:
            pass
        self.energy = mu_dhr, std_dhr

    def calc_reaction_barrier(self, bader: bool=False, bep_params: dict=INTERPOL, min_state: bool=True):
        """
        Get elementary reaction barrier with BEP theory.

        """
        self.calc_reaction_energy(bader=bader, min_state=min_state)
        if self.r_type not in ('adsorption', 'desorption', 'eley_rideal', 'pseudo'):
            alpha = bep_params[self.r_type]['alpha']
            beta = bep_params[self.r_type]['beta']
        else: 
            alpha = bep_params['default']['alpha']
            beta = bep_params['default']['beta']
        self.bb_order()
        self.e_act = alpha * self.energy[0] + beta, alpha * self.energy[1]
        if self.e_act[0] < 0:
            self.e_act = 0, 0
            

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
        stoic_dict = {specie.code: -1 if specie in reactants else 1 for specie in species} # guess (correct for most of the steps)
        elements = Intermediate.elements
        nc, na = len(species), len(elements)
        matrix = np.zeros((nc, na))
        for i, inter in enumerate(species):
            for j, element in enumerate(elements):
                if element == '*' and inter.phase != 'gas':
                    matrix[i, j] = 1
                else:
                    matrix[i, j] = species[i].molecule.get_chemical_symbols().count(element)
        y = np.zeros((na, 1))
        for i, _ in enumerate(elements):
            y[i] = np.dot(matrix[:, i], np.array([stoic_dict[specie.code] for specie in species]))
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
            energy_old = self.energy
            self.energy = -energy_old[0], energy_old[1]
        if self.e_act != None:
            self.e_act = self.e_act[0] - energy_old[0], self.e_act[1]
        self.code = self.__repr__()

