import matplotlib.pyplot as plt

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
        out_str = ''
        for comp in self.components:
            for inter in comp:
                out_str += inter.repr + '+'
            out_str = out_str[:-1]
            out_str += '<->'
        self.repr = out_str[:-3]
        self.stoic = self.solve_stoichiometry()

    def __str__(self):
        return self.repr

    def __repr__(self):
        return self.repr + f' [{self.r_type}]'        

    def __eq__(self, other):
        if isinstance(other, ElementaryReaction):
            return self.bb_order() == other.bb_order()
        return False

    def __hash__(self):
        return hash((self.components)) 

    def __add__(self, other):
        if isinstance(other, ElementaryReaction):
            step = ElementaryReaction(components=(self.components[0] | other.components[0], self.components[1] | other.components[1]), r_type='pseudo')
            if step.energy is None or other.energy is None:
                step.energy = None
            else:
                step.energy = self.energy[0] + other.energy[0], (self.energy[1]**2 + other.energy[1]**2)**0.5
            # TODO: fix stoichiometric coefficients
            return step
        else:
            raise TypeError('The object is not an ElementaryReaction')   

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
                _.append(frozenset(item))
            self._components = tuple(_)

    @property
    def code(self):
        if self._code is None:
            self._code = self.repr
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
        new_pair = frozenset(pair)
        self.components.append(new_pair)

    # def draft(self):
    #     """Draw a draft of the transition state using the drafts of the
    #     components.

    #     Returns:
    #         obj:`matplotlib.pyplot.Figure` containing the draft.
    #     """
    #     counter = 1
    #     for component in self.components:
    #         for intermediate in component:
    #             plt.subplot(2, 2, counter)
    #             intermediate.draft()
    #             counter += 1
    #     return plt.show()
    
    def solve_stoichiometry(self):
        """Solve the stoichiometry of the elementary reaction.

        Returns:
            dict containing the stoichiometry of the elementary reaction.
        """

        stoic = [[], []]
        react_n_atoms = 0
        for inter in self.components[0]:
            react_n_atoms += int(inter.molecule.get_chemical_symbols().count('C') + inter.molecule.get_chemical_symbols().count('H') + inter.molecule.get_chemical_symbols().count('O'))
        prod_n_atoms = 0
        for inter in self.components[1]:
            prod_n_atoms += int(inter.molecule.get_chemical_symbols().count('C') + inter.molecule.get_chemical_symbols().count('H') + inter.molecule.get_chemical_symbols().count('O'))
        stoich_relation = int(react_n_atoms/prod_n_atoms)

        # Adding the stoichiometry of the reactants
        for idx, inter in enumerate(self.components[0]):
            copied_list = list(self.components[0].copy())
            copied_list.pop(idx)
            remaining_elem = copied_list[0]
            if stoich_relation != 1 and inter.is_surface and remaining_elem.phase == 'gas':
                stoic[0].append(-stoich_relation)
            else:
                stoic[0].append(-1)
        # Adding the stoichiometry of the products
        for inter in self.components[1]:
            stoic[1].append(stoich_relation)

        return stoic
    
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
        self.energy = -self.energy if self.energy is not None else None
        if self.r_type in ('adsorption', 'desorption'):
            self.r_type = 'desorption' if self.r_type == 'adsorption' else 'adsorption'
        out_str = ''
        for comp in self.components:
            for inter in comp:
                out_str += inter.repr + '+'
            out_str = out_str[:-1]
            out_str += '<->'
        self.repr = out_str[:-3]
        self.reactants, self.products = self.products, self.reactants
        if self.energy != None:
            self.energy = -self.energy[0], self.energy[1]
        if self.e_act != None:
            self.e_act = self.e_act[0] - self.energy[0], self.e_act[1]