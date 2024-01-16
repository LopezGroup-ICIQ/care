import numpy as np
from ase import Atoms


class Bond:
    def __init__(self, 
                 atoms_obj: Atoms,
                 index_1: int, 
                 index_2: int):
        """
        Class for representing chemical bonds from ASE Atoms objects.
        The atoms object must contain the connectivity list as atoms_obj.array['conn_pairs'].

        Parameters
        ----------
        atoms_obj : Atoms
            _description_
        index_1 : int
            _description_
        index_2 : int
            _description_
        """
        self.atom_1 = atoms_obj[index_1]
        self.atom_2 = atoms_obj[index_2]
        self.atoms = frozenset((self.atom_1, self.atom_2))
        self.elements = frozenset((self.atom_1.symbol, self.atom_2.symbol))
        # self.distance = atoms_obj.get_distance(index_1, index_2)
        self.num_connections_1 = np.count_nonzero(atoms_obj.arrays['conn_pairs'] == index_1)
        self.num_connections_2 = np.count_nonzero(atoms_obj.arrays['conn_pairs'] == index_2)
        self.bond_order = frozenset(((self.atom_1.symbol, self.num_connections_1),
                                     (self.atom_2.symbol, self.num_connections_2)))
    def __repr__(self):
        rtr_str = '{}({})-{}({})'.format(self.atom_1.symbol,
                                                  self.num_connections_1,
                                                  self.atom_2.symbol,
                                                  self.num_connections_2)
                                                #   self.distance)
        return rtr_str

class BondPackage:
    """
    Class for representing a collection of bonds.
    """
    def __init__(self):
        self.name = None
        self.bonds = []
        self.bond_types = []
        self.bond_elements = {}
        self.orders = {}

    def __contains__(self, item):
        new_set = frozenset(item)
        for sg_bond in self.bonds:
            if sg_bond.atoms == new_set:
                return True
        else:
            return False

    def __getitem__(self, choice):
        if isinstance(choice, int):
            selection = self.bonds[choice]
        elif isinstance(choice, str):
            selection = self.element_search(choice)
        elif isinstance(choice, (list, tuple)):
            selection = self.bond_search(choice)
        return selection

    def __len__(self):
        return len(self.bonds)

    def bond_add(self, bond):
        if isinstance(bond, (list, tuple)):
            for item in bond:
                self._bond_add(item)
        elif isinstance(bond, Bond):
            self._bond_add(bond)
        else:
            raise NotImplementedError

    def _bond_add(self, bond: Bond):
        if bond.elements not in self.bond_elements:
            self.bond_elements[bond.elements] = []
        if bond.bond_order not in self.bond_types:
            self.bond_types.append(bond.bond_order)
            self.bond_elements[bond.elements].append(bond.bond_order)
        for _, atom in enumerate(list(bond.atoms)):
            element_type = [t[0] for t in list(bond.bond_order) if t[0] == atom]
            if atom.symbol not in self.orders:
                self.orders[atom.symbol] = []
            elif element_type in self.orders[atom.symbol]:
                continue
            self.orders[atom.symbol].append(element_type)
        self.bonds.append(bond)

    def sum_bonds(self, other):
        bonds = other.bonds.copy()
        self.bond_add(bonds)

    # def _compute_average(self, sub_pack):
    #     dist_arr = np.asarray([sg_bond.distance for sg_bond in sub_pack])
    #     return np.average(dist_arr)

    def type_average(self, elements):
        return np.average(np.asarray([sg_bond for sg_bond in self.bonds if
                                      sg_bond.elements == frozenset(elements)]))

    def element_search(self, element: str):
        selection = [sg_bond for sg_bond in self.bonds if
                     element in sg_bond.elements]
        return selection

    def element_order_search(self, element_type):
        selection = [sg_bond for sg_bond in self.bonds if
                     element_type in sg_bond.bond_order]
        return selection

    def bond_search(self, elements):
        frz_set = frozenset(elements)
        selection = [sg_bond for sg_bond in self.bonds if
                     frz_set == sg_bond.elements]
        return selection

    def bond_order_search(self, bond_type):
        frz_set = frozenset(bond_type)
        selection = [sg_bond for sg_bond in self.bonds if
                     frz_set == sg_bond.bond_order]
        return selection

    def analysis_element(self):
        rtr_dic = {}
        for key, value in self.orders.items():
            sub_typ = []
            for item in value:
                arr_pvt = self.element_order_search(item)
                sub_typ.append({'order': item,
                                'average': self._compute_average(arr_pvt)})

            avg_el = np.average(np.array([elem['average'] for elem in sub_typ]))
            rtr_dic[key] = {'sub_types': sub_typ,
                            'total_average': avg_el}
        return rtr_dic

    def analysis_bond(self):
        rtr_dic = {}
        for key, value in self.bond_elements.items():
            sub_typ = []
            avg_tot = 0
            for item in value:
                arr_pvt = self.bond_order_search(item)
                lst_pvt = list(item)
                weight = len(arr_pvt)
                sub_typ.append({'elements': lst_pvt,
                                'average': self._compute_average(arr_pvt),
                                'weight': weight})
                avg_tot += weight

            avg_el = np.array([elem['average'] * float(elem['weight'])
                               for elem in sub_typ])
            avg_el = np.sum(avg_el / avg_tot)
            avg_norm = np.average(np.array([elem['average'] for elem in sub_typ]))
            lst_pvt = list(key)
            rtr_dic[key] = {'sub_types': sub_typ,
                            'weighted_average': avg_el,
                            'total_average': avg_norm,
                            'bond_total': avg_tot}
        return rtr_dic