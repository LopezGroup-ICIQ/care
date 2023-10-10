import numpy as np
from math import ceil
import copy

from GAMERNet.rnet.networks.intermediate import Intermediate

REACTION_TYPES = ['desorption', 'C-O', 'C-OH', 'C-H', 'H-H', 'O-O', 'C-C', 'O-H', 'O-OH', 'eley_rideal', 'adsorption']

class ElementaryReaction:
    """Class for representing elementary reactions.

    Attributes:
        code (str): Code associated with the elementary reaction.
        components (list of frozensets): List containing the frozensets.
            with the components of the reaction.
        energy (float): Transition state energy of the elementary reaction.
        r_type (str): Elementary reaction type.
    """
    def __init__(self, 
                 code: str=None, 
                 components: tuple[frozenset[Intermediate]]=None, 
                 energy: float=0.0,
                 r_type: str=None, 
                 is_electro: bool=False):
        self._code = code
        self._components = None
        self.components = components
        self.reactants = self.components[0]
        self.products = self.components[1]
        self.energy = energy
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

    def __repr__(self):
        return self.repr        

    def __eq__(self, other):
        if isinstance(other, ElementaryReaction):
            return self.bb_order() == other.bb_order()
        return False

    def __hash__(self):
        return hash((self.components))

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
        Order the components of the elementary reaction in the direction of the
        bond breaking reaction, e.g.:
        CH4 + * -> CH3 + H*

        Returns:
            list of frozensets containing the reactants before the bond
            breaking in the firs position and the products before the breakage.
        """
        new_list = list(self.components)
        flatten = [list(comp) for comp in self.components]
        max_numb = 0
        for index, item in enumerate(flatten):
            for species in item:
                if species.is_surface:
                    new_list.insert(0, new_list.pop(index))
                    break
                tmp_numb = len(species.molecule)
                if tmp_numb > max_numb:
                    max_numb = tmp_numb
                    index_numb = index
            else:
                continue
            break
        else:
            new_list.insert(0, new_list.pop(index_numb))
        return new_list

    def full_order(self):
        """
        missing docstring
        """
        ordered = self.bb_order()
        react, prod = ordered
        react = list(react)
        prod = list(prod)
        order = []
        for item in [react, prod]:
            try:
                mols = [item[0],
                        item[1]]
            except IndexError:
                mols = [item[0], item[0]]
                
            if mols[0].is_surface:
                order.append(mols[::-1])
            elif not mols[1].is_surface and (len(mols[0].molecule) <
                                             len(mols[1].molecule)):
                order.append(mols[::-1])
            else:
                order.append(mols)
        return order

    # def full_label(self):
    #     """
    #     xxx
    #     """
    #     order = self.full_order()
    #     full_label = ''
    #     for item in order:
    #         for inter in item:
    #             if inter.phase == ['cat']:
    #                 full_label += 'i'
    #             elif inter.code == 'e-':
    #                 full_label += 'xxxxxxx'
    #                 continue
    #             else:
    #                 full_label += 'g'
    #             full_label += str(inter.code)
    #     return full_label

    def calc_activation_energy(self, reverse=False, bader=False):
        components = [list(item) for item in self.bb_order()]
        if reverse:
            components = components[1]
        else:
            components = components[0]

        if len(components) == 1:
            components = components * 2

        if bader:
            inter_ener = sum([inter.bader_energy for inter in components])
            out_ener = self.bader_energy - inter_ener
        else:
            inter_ener = sum([inter.energy for inter in components])
            out_ener = self.energy - inter_ener
        return out_ener

    def calc_delta_energy(self, reverse=False, bader=False):
        components = [list(item) for item in self.bb_order()]
        start_comp, end_comp = components

        if len(end_comp) == 1:
            end_comp = end_comp * 2

        if reverse:
            start_comp, end_comp = end_comp, start_comp

        if bader:
            start_ener = sum([inter.bader_energy for inter in start_comp])
            end_ener = sum([inter.bader_energy for inter in end_comp])
        else:
            start_ener = sum([inter.energy for inter in start_comp])
            end_ener = sum([inter.energy for inter in end_comp])
        out_ener = end_ener - start_ener
        return out_ener

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
        """Add components to the transition state.

        Args:
            pair (list of Intermediate): Intermediates that will be added to
                the components.
        """
        new_pair = frozenset(pair)
        self.components.append(new_pair)

    # def get_code(self, option='g'):
    #     """Automatically generate a code for the transition state using the
    #     code of the intermediates.

    #     Args:
    #         option (str, optional): 'g' or 'i'. If 'g' the position of the
    #             intermediates in the code is inverted. Defaults to None.
    #     """
    #     end_str = ''
    #     in_str = 'i'
    #     out_str = 'f'
    #     act_comp = self.bb_order()
    #     if option in ['inverse', 'i']:
    #         act_comp = self.components[::-1]
    #     elif option in ['general', 'g']:
    #         out_str = 'i'

    #     for species in act_comp[0]:
    #         end_str += in_str + str(species.code)
    #     for species in act_comp[1]:
    #         end_str += out_str + str(species.code)
    #     return end_str

    # def draft(self):
    #     """Draw a draft of the transition state using the drafts of the
    #     components.

    #     Returns:
    #         obj:`matplotlib.pyplot.Figure` containing the draft.
    #     """
    #     counter = 1
    #     for item in self.components:
    #         for component in item:
    #             plt.subplot(2, 2, counter)
    #             component.draft()
    #             counter += 1
    #     return plt.show()

    # def solve_stoichiometry(self):
    #     """Solve the stoichiometry of the elementary reaction.

    #     Returns:
    #         dict containing the stoichiometry of the elementary reaction.
    #     """

    #     s_matrix = np.zeros((4, len(self.components[0])+len(self.components[1])))
    #     for index, inter in enumerate(self.components[0]):
    #         s_matrix[0, index] = inter.molecule.get_chemical_symbols().count('C')
    #         s_matrix[1, index] = inter.molecule.get_chemical_symbols().count('H')
    #         s_matrix[2, index] = inter.molecule.get_chemical_symbols().count('O')
    #         if inter.phase != 'gas':
    #             s_matrix[3, index] = 1
    #     for index, inter in enumerate(self.components[1]):
    #         s_matrix[0, index+len(self.components[0])] = inter.molecule.get_chemical_symbols().count('C')
    #         s_matrix[1, index+len(self.components[0])] = inter.molecule.get_chemical_symbols().count('H')
    #         s_matrix[2, index+len(self.components[0])] = inter.molecule.get_chemical_symbols().count('O')
    #         if inter.phase != 'gas':
    #             s_matrix[3, index+len(self.components[0])] = 1

    #     U, S, Vt = np.linalg.svd(s_matrix)

    #     rank = np.sum(S > 1e-10)

    #     null_space = Vt[rank:].T

    #     smallest_vector = null_space[:, 0]  # In this case, this will be the only vector in the null space
        
    #     entire_list = [abs(v) for v in smallest_vector]
    #     min_val = min(entire_list)
    #     slist_lhs = [int(abs(v/min_val)) for v in smallest_vector[:len(self.components[0])]]
    #     slist_rhs = [int(abs(v/min_val)) for v in smallest_vector[len(self.components[0]):]]
    #     if slist_lhs[0] > 0:
    #         slist_lhs = [-v for v in slist_lhs]
    #     return [slist_lhs, slist_rhs]

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