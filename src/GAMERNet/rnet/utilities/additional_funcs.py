from itertools import combinations
import pickle
import networkx as nx
import numpy as np
import networkx.algorithms.isomorphism as iso
from GAMERNet.rnet.networks.elementary_reaction import ElementaryReaction
from GAMERNet.rnet.networks.intermediate import Intermediate
from matplotlib.colors import to_hex
from ase import Atoms
from GAMERNet.rnet.utilities.functions import get_voronoi_neighbourlist
from GAMERNet.rnet.graphs.graph_fn import ase_coord_2_graph
from collections import Counter
# from GAMERNet.rnet.networks.reaction_network import ReactionNetwork

INTERPOL = {'O-H' : {'alpha': 0.39, 'beta': 0.89},
            'C-H' : {'alpha': 0.63, 'beta': 0.81},
            'H-C' : {'alpha': 0.63, 'beta': 0.81},
            'C-C' : {'alpha': 1.00, 'beta': 0.64},
            'C-O' : {'alpha': 1.00, 'beta': 1.24},
            'C-OH': {'alpha': 1.00, 'beta': 1.48}}

BOX_TMP = """<
<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="20">
  <TR>
    <TD COLSPAN="3" BGCOLOR="{}">{}</TD>
  </TR>
  <TR>
    <TD BGCOLOR="{}">{}</TD>
    <TD BGCOLOR="{}">{}</TD>
    <TD BGCOLOR="{}">{}</TD>
  </TR>
  <TR>
    <TD BGCOLOR="{}">{}</TD>
    <TD BGCOLOR="{}">{}</TD>
    <TD BGCOLOR="{}">{}</TD>
  </TR>
</TABLE>>
"""
BOX_TMP_3 = """<
<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="20">
  <TR>
    <TD COLSPAN="3" BGCOLOR="{}">{}</TD>
  </TR>
  <TR>
    <TD BGCOLOR="{}">{}</TD>
    <TD BGCOLOR="{}">{}</TD>
    <TD BGCOLOR="{}">{}</TD>
  </TR>
  <TR>
    <TD BGCOLOR="{}">{}</TD>
    <TD BGCOLOR="{}">{}</TD>
    <TD BGCOLOR="{}">{}</TD>
  </TR>
  <TR>
    <TD BGCOLOR="{}">{}</TD>
    <TD BGCOLOR="{}">{}</TD>
    <TD BGCOLOR="{}">{}</TD>
  </TR>
</TABLE>>
"""

BOX_TMP_0 = """<
<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="20">
  <TR>
    <TD BGCOLOR="{0}">{1}</TD>
  </TR>
</TABLE>>
"""

BOX_TMP_0_WOCODE = """<
<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="20">
  <TR>
    <TD BGCOLOR="{0}">{1}</TD>
  </TR>
</TABLE>>
"""

BOX_TMP_flat = """<
<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="20">
  <TR>
    <TD COLSPAN="3" BGCOLOR="{0}">{1}</TD>
  </TR>
  <TR>
    <TD BGCOLOR="{6}">{7}</TD>
    <TD BGCOLOR="{12}">{13}</TD>
    <TD BGCOLOR="{18}">{19}</TD>
  </TR>
</TABLE>>
"""

ELEM_WEIGTHS = {'H': 1., 'C': 12, 'O': 16}

#### pyRDTP.analysis

BOND_ORDER = {'C': 4,
              'O': 2}


class Bond:
    def __init__(self, 
                 atoms_obj: Atoms,
                 index_1: int, 
                 index_2: int):
        """The atoms object must contain the connectivity list as atoms_obj.array['conn_pairs'].

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
        self.distance = atoms_obj.get_distance(index_1, index_2)
        self.num_connections_1 = np.count_nonzero(atoms_obj.arrays['conn_pairs'] == index_1)
        self.num_connections_2 = np.count_nonzero(atoms_obj.arrays['conn_pairs'] == index_2)
        self.bond_order = frozenset(((self.atom_1.symbol, self.num_connections_1),
                                     (self.atom_2.symbol, self.num_connections_2)))
    def __repr__(self):
        rtr_str = '{}({})-{}({}) [{:.4f}]'.format(self.atom_1.symbol,
                                                  self.num_connections_1,
                                                  self.atom_2.symbol,
                                                  self.num_connections_2,
                                                  self.distance)
        return rtr_str

class BondPackage:
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
        for index, atom in enumerate(list(bond.atoms)):
            # element_type = list(bond.bond_order)[atom]
            # [t[0] for t in list(bond.bond_order) if t[atom] == 'O']
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

    def _compute_average(self, sub_pack):
        dist_arr = np.asarray([sg_bond.distance for sg_bond in sub_pack])
        return np.average(dist_arr)

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

def connectivity_helper(atoms: Atoms) -> dict[int, list]:
    """
    Return a dictionary whose key is the atom index and the value a list with the indices of the atoms connected to the key atom.

    Parameters
    ----------
    atoms : Atoms
        ASE atoms object.

    Returns
    -------
    dict[int, list]
        Dictionary with the connectivity information.
    """
    connections = {}
    if len(atoms) == 1: # C, H, O single atoms
        return connections
    if 'conn_pairs' not in atoms.arrays: 
        conn_matrix = get_voronoi_neighbourlist(atoms, 0.25, 1.0, ['C', 'H', 'O'])
        atoms.arrays['conn_pairs'] = conn_matrix
    conn_matrix = atoms.arrays['conn_pairs']
    for index, _ in enumerate(atoms):
        selected_rows = conn_matrix[np.logical_or(conn_matrix[:, 0] == index, conn_matrix[:, 1] == index)]
        connections[index] = [element for row in selected_rows for element in row if element != index]
    return connections


def bond_analysis(mol: Atoms, comp_dist: bool=False) -> BondPackage:
    """
    Returns a dictionary of frozen tuples containing information about
    the bond distances values between the different bonds of the molecule.

    Args:
        comp_dist (bool, optional): If True, the voronoi method will be
            used to compute the bonds before the bond analysis.
        bond_num (bool, optional): Take into account the number of bonds
            of every atom and separate the different distances.

    Returns:
        dict of frozentuples containing the minimum distances between
        different atoms.
    """
    if comp_dist:
        mol.arrays['conn_pairs'] = get_voronoi_neighbourlist(mol, 0.25, 1.0, ['C', 'H', 'O'])
    package = BondPackage()
    for row in range(mol.arrays['conn_pairs'].shape[0]):
        package.bond_add(Bond(mol, mol.arrays['conn_pairs'][row, 0], mol.arrays['conn_pairs'][row, 1]))
    return package

#TODO: done, but double-check later
def insaturation_matrix(mol: Atoms, voronoi: bool=False) -> np.ndarray:
    """Generate the insaturation bond matrix for an organic molecule.

    Args:
        mol (:obj`ase.atoms.Atoms): Organic molecule that will be analyzed to
            create the bond matrix.
        voronoi (bool, optional): If True, a voronoi connectivity analysis
            will be performed before creating the bond matrix. Defaults to
            False.

    Returns:
        :obj`np.ndarray` of size (n_noHatoms, n_noHatoms) containing the bonds
        between different atoms and the insaturations in the diagonal.

    Notes:
        At the moment only works with C and O, discarding the H.
        If Voronoi is True, all the previous bonds will be deleted from the
        molecule.
    """
    if voronoi:
        del mol.arrays['conn_pairs']
        mol.array['conn_pairs'] = get_voronoi_neighbourlist(mol, 0.25, 1, ['C', 'H', 'O'])

    not_h = [atom for atom in mol if atom.symbol != 'H']
    bond_mat = np.zeros((len(not_h), len(not_h)), dtype=int)

    for index, atom in enumerate(not_h):
        avail_con = BOND_ORDER[atom.symbol]
        condition = (mol.array['conn_pairs'][:, 0] == atom.index) or (mol.array['conn_pairs'][:, 1] == atom.index)
        indices = np.where(condition)[0]
        for connection in indices:
            avail_con -= 1
            if not 'H' in [mol[index].symbol for index in mol.arrays['conn_pairs'][connection, :]]:
                bond_mat[index, not_h.index(connection)] = 1
        bond_mat[index, index] = avail_con
    return bond_mat

def insaturation_solver(bond_mat: np.ndarray) -> np.ndarray:
    """Solve the insaturation matrix distributing all the avaliable bonds
    between the different atoms of the matrix.

    bond_mat (:obj`np.ndarray`): Bond matrix containing the information of the
        bonds and insaturations. See insaturation_matrix()  function for
        further information.

    Returns:
        :obj`np.ndarray` containing the insaturation matrix.
    """
    ori_mat = bond_mat.copy()
    new_mat = bond_mat.copy()
    while True:
        for row, value in enumerate(new_mat):
            if [new_mat[row] == 0][0].all():
                continue
            elif new_mat[row, row] == 0:
                new_mat[row] = 0
                new_mat[:, row] = 0
                break
            elif new_mat[row, row] < 0:
                continue
            bool_mat = [value != 0][0]
            bool_mat[row] = False
            if np.sum(bool_mat) == 1:
                col = np.argwhere(bool_mat)[0]
                new_mat[row, row] -= 1
                new_mat[col, col] -= 1
                new_mat[row, col] += 1
                new_mat[col, row] += 1

                ori_mat[row, row] -= 1
                ori_mat[col, col] -= 1
                ori_mat[row, col] += 1
                ori_mat[col, row] += 1
                break
        else:
            break
    return ori_mat

def insaturation_check(mol: Atoms) -> bool:
    """Check if all the bonds from an organic molecule are fullfilled.

    Returns:
        True if all the electrons are correctly distributed and False
        if the molecule is a radical.
    """
    bond_mat = insaturation_matrix(mol)
    bond_mat = insaturation_solver(bond_mat)
    return not np.diag(bond_mat).any()


def elem_inf(graph: nx.Graph) -> dict:
    """
    Given a molecular nx graph, return a dictionary with the number of atoms
    of each element in the molecule.

    Args:
        graph (:obj:`nx.Graph`): Molecular graph.

    Returns:
        :obj:`dict` with the number of atoms of each element in the molecule. (e.g. {'C': 3, 'H': 8})
    """
    elem_lst = [atom[1]['elem'] for atom in graph.nodes(data=True)]  # e.g. ['C', 'C', 'C']
    elem_uniq = [elem for numb, elem in enumerate(elem_lst)
                 if elem_lst.index(elem) == numb] # e.g. ['C']
    elem_count = []
    for elem in elem_uniq:
        elem_count.append(elem_lst.count(elem))
    return dict(zip(elem_uniq, elem_count))

def mkm_rxn_file(network, gb_rxn, filename):
    t_state_array=[]
    with open(filename, 'w') as outfile:
        inter_str = '{:20}\n'
        for gb_label in gb_rxn:
            outfile.write((inter_str.format(gb_label)))
        outfile.write('\n\n\n')
        for label_g, inter in network.gasses.items():
            label_gw=label_g.replace('g', '')
            #print(label_g)
            for label_i, inter_i in network.intermediates.items():
                #if label_gw==label_i:
                #if label_gw==label_i and (label_gw=='102101' or  label_gw=='101101'):
                if label_gw==label_i and (label_gw!='021101'):
                    label_g=inter.molecule.get_chemical_formula()
                    rxn_label = label_g + '(g) + i000000 -> i' + label_i  
                    #print(rxn_label)
                    outfile.write((inter_str.format(rxn_label)))
        for t_state in network.t_states:
            pintuit = 'single'
            label = t_state.code
            initial = [inter.code for inter in list(t_state.components[0])]
            final = [inter.code for inter in list(t_state.components[1])]
            if len(initial) == 1:
                initial *= 2
                pintuit = 'in_double'
            if len(final) == 1:
                final *= 2
                pintuit = 'fin_double'
            #inter = label.split('i')
            #print(t_state.r_type)
            intuit = 'add'
            for i in initial:
                for j in final:
                    if (i[2]=='2' or j[2]=='2') and (int(i[1])>1 or int(j[1])>1 ):#dont add glyoxal pathways
                    #if (i[0]=='2' or i[1]=='2' or i=='011101' or i=='111101' or j=='001101'):#dont add C2 pathways
                        #print(i,j)
                        intuit='dont add'
                    #if i=='121101' or i=='021101' or j=='001101' or j=='250101' or i=='250101'  or i=='212101' or j=='111101' or i=='111101':
                    #if  i=='021101' or j=='001101' or j=='250101' or i=='250101':  #or i=='212101' or j=='111101' or i=='111101':
                    if i=='021101' or j=='001101' or j=='250101' or i=='250101' or j=='141101' or i=='141101' or j=='131101' or i=='131101' or j=='121101' or i=='121101': 
                        intuit='dont add'
                    #if int(i[1])>4 or int(j[1])>4: #dont add ethanol pathways
                    #    counter='dont add'
            if intuit=='add':
                for i in range(len(initial)): #converting all i010101 to H(e) for py-mkm
                    if '(g)' not in initial[i]:
                        initial[i] = 'i' + initial[i]
                    if '(g)' not in final[i]:
                        final[i] = 'i' + final[i]
                if t_state.r_type=='C-OH': #Converting all OHs in final state to H2O and i00000 to H(e)
                    for i in range(len(final)):
                        if final[i] == 'i011101':
                            final[i] = 'H2O1(g)'
                            for j in range(len(initial)):
                                if initial[j] == 'i000000':
                                    initial[j] = 'i010101' 
                for i in range(len(initial)):   
                    if initial[i] == 'i010101':
                        initial[i] = 'H(e)'
                        for j in range(len(final)):
                            if final[j]=='i000000':
                                final.remove('i000000')
                                rxn_label_H = initial[0] + ' + ' + initial[1] + ' -> ' + final[0]
                                #print(rxn_label_H) 
                                intuit='add_electro'
                                break
                        break
                    if final[i] == 'i010101':
                        final[i] = 'H(e)'
                        for j in range(len(initial)):
                            if initial[j]=='i000000':
                                initial.remove('i000000')
                                rxn_label_H = initial[0] + ' -> ' + final[0] + ' + ' + final[1]
                                intuit='add_electro'
                                #print(rxn_label_H) 
                                break
                        break
            if intuit=='add' or intuit=='add_electro':
                t_state_array.append(t_state.code)
            if intuit=='add' and pintuit=='single':
                rxn_label = initial[0] + ' + ' + initial[1] + ' -> ' + final[0] + ' + ' + final[1]
                #except:
                #print(rxn_label,t_state.r_type)
                outfile.write((inter_str.format(rxn_label)))
            elif intuit=='add' and pintuit=='in_double':
                rxn_label = '2' + initial[0]  + ' -> ' + final[0] + ' + ' + final[1]
                #except:
                #print(rxn_label,t_state.r_type)
                outfile.write((inter_str.format(rxn_label)))
            elif intuit=='add' and pintuit=='fin_double':
                rxn_label = initial[0] + ' + ' + initial[1] + ' -> ' +'2' + final[0] 
                #except:
                #print(rxn_label,t_state.r_type)
                outfile.write((inter_str.format(rxn_label)))             
            elif intuit=='add_electro':
                outfile.write((inter_str.format(rxn_label_H)))
    return t_state_array

                
def mkm_g_file(network, filename='g.mkm'):
    with open(filename, 'w') as outfile:
        rxn_str = '{} {:.3f} {:.3f} {:.2f}\n'
        counter=0
        for label_g, inter in network.gasses.items():
            label_gw=label_g.replace('g', '')
            #print(label_g)
            for label_i, inter in network.intermediates.items():
                #if label_gw==label_i:
                #if label_gw==label_i and (label_gw=='102101' or  label_gw=='101101'):
                if label_gw==label_i and (label_gw!='021101'):
                    counter+=1
                    ener = 0
                    entropy = 0
                    coeff = 0
                    g_line = 'R' + str(counter) + ':'
                    outfile.write((rxn_str.format(g_line,ener,entropy,coeff)))
                    #print(g_line)
        for t_state in network.t_states:
            label = t_state.code
            initial = [inter.code for inter in list(t_state.components[0])]
            final = [inter.code for inter in list(t_state.components[1])]
            if len(initial) == 1:
                initial *= 2
            if len(final) == 1:
                final *= 2
            #inter = label.split('i')
            #print(t_state.r_type)
            intuit = 'add'
            
            for i in initial:
                for j in final:
                    if (i[2]=='2' or j[2]=='2') and (int(i[1])>1 or int(j[1])>1 ): #dont add glyoxal pathways
                    #if (i[0]=='2' or i[1]=='2' or i=='011101' or i=='111101' or j=='001101'):#dont add C2 pathways
                        #print(i,j)
                        intuit='dont add'
                    #if i=='121101' or i=='021101' or j=='001101' or j=='250101' or i=='250101' or i=='212101' or j=='111101' or i=='111101':
                    if i=='021101' or j=='001101' or j=='250101' or i=='250101': 
                        intuit='dont add'
                    #if int(i[1])>4 or int(j[1])>4: #dont add ethanol pathways
                    #    counter='dont add'
            if intuit=='add':
                if t_state.r_type=='C-OH': #Converting all OHs in final state to H2O and i00000 to H(e)
                    for i in range(len(final)):
                        if final[i] == '011101':
                            final[i] = 'g021101'
                            for j in range(len(initial)):
                                if initial[j] == '000000':
                                    initial[j] = '010101'
                #print(initial,final)
                for i in range(len(initial)):   
                    if initial[i] == '010101':
                        for j in range(len(final)):
                            if final[j]=='000000':
                                final.remove('000000')
                                break
                        intuit='add_electro'
                        #print(intuit)
                        coeff = 0.5
                        ener = 0.5 + network.intermediates[initial[0]].energy + network.intermediates[initial[1]].energy
                        entropy = network.intermediates[initial[0]].entropy + network.intermediates[initial[1]].entropy
                        break
                    if final[i] == '010101':
                        for j in range(len(initial)):
                            if initial[j]=='000000':
                                initial.remove('000000')
                                break
                        intuit='add_electro'
                        #print(intuit)
                        coeff = 0.5
                        ener = 0.5 + network.intermediates[final[0]].energy + network.intermediates[final[1]].energy
                        entropy = network.intermediates[final[0]].entropy + network.intermediates[final[1]].entropy
                        break
            if intuit=='add':
                counter+=1
                entropy = network.intermediates[final[0]].entropy + network.intermediates[final[1]].entropy
                #ener = t_state.energys
                coeff = 0.0
                if t_state.r_type=='C-O':
                    ener = 1 + network.intermediates[final[0]].energy + network.intermediates[final[1]].energy
                else:
                    ener = 0.7 + network.intermediates[final[0]].energy + network.intermediates[final[1]].energy
                g_line = 'R' + str(counter) + ':' 
                #print(g_line)
                entropy = float(298*entropy*1e-3)
                t_state.energy=ener
                t_state.entropy=entropy
                outfile.write((rxn_str.format(g_line,ener,entropy,coeff)))
            elif intuit=='add_electro':
                counter+=1
                g_line = 'R' + str(counter) + ':' 
                #print(g_line)
                entropy = float(298*entropy*1e-3)
                t_state.energy=ener
                t_state.entropy=entropy
                outfile.write((rxn_str.format(g_line,ener,entropy,coeff)))
        outfile.write('\n\n\n')
        inter_str = '{} {:.3f} {:.3f}\n'
        for label, inter in network.intermediates.items():
            if label=='010101':
                label = 'H(e)'
                g_line = label + ': ' 
            else:
                g_line = 'i' + label + ': ' 
            ener = inter.energy
            entropy = float(298*inter.entropy*1e-3)
            #print(g_line)
            outfile.write(inter_str.format(g_line, ener, entropy))
        outfile.write('\n\n\n')
        #gas_str = '{} {:.3f} \n'
        for label, inter in network.gasses.items():
            ener = inter.energy
            entropy = float(298*inter.entropy*1e-3)
            #print(label,entropy)
            label= inter.molecule.get_chemical_formula()
            print(label)
            g_line = label + '(g): ' 
            #print(g_line)
            try:
                outfile.write(inter_str.format(g_line,ener,entropy))
            except:
                'do nothing'

def mkm_g_file_TS(network, filename='g.mkm'):
    with open(filename, 'w') as outfile:
        rxn_str = '{} {:.3f} {:.3f} {:.2f}\n'
        counter=0
        for label_g, inter in network.gasses.items():
            label_gw=label_g.replace('g', '')
            #print(label_g)
            for label_i, inter in network.intermediates.items():
                #if label_gw==label_i:
                #if label_gw==label_i and (label_gw=='102101' or  label_gw=='101101'):
                if label_gw==label_i and (label_gw!='021101'):
                    counter+=1
                    ener = 0
                    entropy = 0
                    coeff = 0
                    g_line = 'R' + str(counter) + ':'
                    outfile.write((rxn_str.format(g_line,ener,entropy,coeff)))
                    #print(g_line)
        for t_state in network.t_states:
            label = t_state.code
            initial = [inter.code for inter in list(t_state.components[0])]
            final = [inter.code for inter in list(t_state.components[1])]
            if len(initial) == 1:
                initial *= 2
            if len(final) == 1:
                final *= 2
            #inter = label.split('i')
            #print(t_state.r_type)
            intuit = 'add'
            
            for i in initial:
                for j in final:
                    if (i[2]=='2' or j[2]=='2') and (int(i[1])>1 or int(j[1])>1 ): #dont add glyoxal pathways
                    #if (i[0]=='2' or i[1]=='2' or i=='011101' or i=='111101' or j=='001101'):#dont add C2 pathways
                        #print(i,j)
                        intuit='dont add'
                    #if i=='121101' or i=='021101' or j=='001101' or j=='250101' or i=='250101'  or i=='212101' or j=='111101' or i=='111101': 
                    #if i=='021101' or j=='001101' or j=='250101' or i=='250101': 
                    if i=='021101' or j=='001101' or j=='250101' or i=='250101' or j=='141101' or i=='141101' or j=='131101' or i=='131101' or j=='121101' or i=='121101': 
                        intuit='dont add'
                    #if int(i[1])>4 or int(j[1])>4: #dont add ethanol pathways
                    #    counter='dont add'
            if intuit=='add':
                if t_state.r_type=='C-OH': #Converting all OHs in final state to H2O and i00000 to H(e)
                    for i in range(len(final)):
                        if final[i] == '011101':
                            final[i] = 'g021101'
                            for j in range(len(initial)):
                                if initial[j] == '000000':
                                    initial[j] = '010101'
                #print(initial,final)
                for i in range(len(initial)):   
                    if initial[i] == '010101':
                        for j in range(len(final)):
                            if final[j]=='000000':
                                final.remove('000000')
                                break
                        intuit='add_electro'
                        #print(intuit)
                        coeff = 0.5
                        ener = 0.7 + network.intermediates[initial[0]].energy + network.intermediates[initial[1]].energy
                        entropy = network.intermediates[initial[0]].entropy + network.intermediates[initial[1]].entropy
                        break
                    if final[i] == '010101':
                        for j in range(len(initial)):
                            if initial[j]=='000000':
                                initial.remove('000000')
                                break
                        intuit='add_electro'
                        #print(intuit)
                        coeff = 0.5
                        ener = 0.7 + network.intermediates[final[0]].energy + network.intermediates[final[1]].energy
                        entropy = network.intermediates[final[0]].entropy + network.intermediates[final[1]].entropy
                        break
            if intuit=='add':
                counter+=1
                if t_state.energy == 0:
                    entropy = network.intermediates[final[0]].entropy + network.intermediates[final[1]].entropy
                    #ener = t_state.energys                
                    ener = 1 + network.intermediates[final[0]].energy + network.intermediates[final[1]].energy
                    # if counter==306:
                    #     # print(ener,entropy)
                else:
                    entropy = t_state.entropy
                    ener = t_state.energy               
                coeff = 0.0
                g_line = 'R' + str(counter) + ':' 
                #print(g_line)
                entropy = float(298*entropy*1e-3)
                t_state.energy=ener
                t_state.entropy=entropy
                outfile.write((rxn_str.format(g_line,ener,entropy,coeff)))
            elif intuit=='add_electro':
                counter+=1
                g_line = 'R' + str(counter) + ':' 
                #print(g_line)
                entropy = float(298*entropy*1e-3)
                t_state.energy=ener
                t_state.entropy=entropy
                outfile.write((rxn_str.format(g_line,ener,entropy,coeff)))
        outfile.write('\n\n\n')
        inter_str = '{} {:.3f} {:.3f}\n'
        for label, inter in network.intermediates.items():
            if label=='010101':
                label = 'H(e)'
                g_line = label + ': ' 
            else:
                g_line = 'i' + label + ': ' 
            ener = inter.energy
            entropy = float(298*inter.entropy*1e-3)
            #print(g_line)
            outfile.write(inter_str.format(g_line, ener, entropy))
        outfile.write('\n\n\n')
        #gas_str = '{} {:.3f} \n'
        for label, inter in network.gasses.items():
            ener = inter.energy
            entropy = float(298*inter.entropy*1e-3)
            #print(label,entropy)
            label= inter.molecule.get_chemical_formula()
            # print(label)
            g_line = label + '(g): ' 
            #print(g_line)
            try:
                outfile.write(inter_str.format(g_line,ener,entropy))
            except:
                'do nothing'

def calculate_weigth(elems):
    return sum([ELEM_WEIGTHS[key] * value for key, value in elems.items()])

def break_bonds(molecule: Atoms) -> dict[str, list[list[nx.Graph]]]:
    """
    Where the magic happens.
    Generate all possible C-O, C-C and C-OH bonds for the given molecule.
    Returns a dictionary whose keys are the different types of bond breakages
    and the values are lists of obj:`nx.Graph` containing the generated
    molecules after the breakage.
    Keys: 'C-O', 'C-C', 'C-OH', 'O-O', 'H-H'

    Args:
        molecule (obj:`ase.atoms.Atoms`): Molecule that will be used
            to detect the breakeages.

    Returns:
        dict with the keys being the types of bond breakings containing the 
        obj:`nx.DiGraph` with the generated molecules after the breakage.

    Note:
        The C-OH bond is considered for electrochemical reactions purposes.
    """
    connections = connectivity_helper(molecule)
    bonds = {'C-O': [], 'C-C': [], 'C-OH': [], 'O-O': [], 'H-H': [], 'O-OH': []}  # C-H and O-H already considered
    bond_pack = bond_analysis(molecule)
    mol_graph = ase_coord_2_graph(molecule, coords=False)
    for bond_type in (('O', 'C'), ('C', 'O'), ('C', 'C'), ('O', 'O'), ('H', 'H')):
        for pair in bond_pack.bond_search(bond_type):
            tmp_graph = mol_graph.copy()
            tmp_graph.remove_edge(pair.atom_1.index, pair.atom_2.index)
            sub_graphs = [tmp_graph.subgraph(comp).copy() for comp in
                        nx.connected_components(tmp_graph)]
            if bond_type in (('O', 'C'), ('C', 'O')):
                oh_bond = False
                for atom in pair.atoms: # check if O is connected to H
                    if atom.symbol == 'O':
                        if 'H' in [molecule[con].symbol for con in connections[atom.index]]:
                            oh_bond = True
                    else:
                        continue
                if oh_bond:
                    bonds['C-OH'].append(sub_graphs)
                else:
                    bonds['C-O'].append(sub_graphs)
            elif ('O', 'O') == bond_type:
                oh_bond = False
                for atom in pair.atoms: # check if O is connected to H
                    if atom.symbol == 'O':
                        if 'H' in [molecule[con].symbol for con in connections[atom.index]]:
                            oh_bond = True
                    else:
                        continue
                if oh_bond:
                    bonds['O-OH'].append(sub_graphs)
                else:
                    bonds['O-O'].append(sub_graphs)
            elif ('H', 'H') == bond_type:
                bonds['H-H'].append(sub_graphs)
            else:
                bonds['C-C'].append(sub_graphs)
    return bonds

# def break_and_connect(intermediates_dict: dict[str, Intermediate], surface: Atoms) -> list[ElementaryReaction]:
#     """
#     For an entire network, perform a breakage search (see `break_bonds`) for
#     all the intermediates, search if the generated submolecules belong to the
#     network and if affirmative, create an obj:`networks.ElementaryReaction` object
#     that connect all the species.

#     Args:
#         network (obj:`networks.ReactionNetwork`): Network in which the function
#             will be performed.
#         surface (obj:`networks.Intermediate`, optional): Surface intermediate.
#             defaults to '000000'.

#     Returns:
#         list of obj:`networks.ElementaryReaction` with all the generated
#         transition states.
#     """
#     cate = iso.categorical_node_match(['elem', 'elem', 'elem'], ['H', 'O', 'C'])
#     reaction_list = []
#     for intermediate in intermediates_dict.values():
#         sub_graphs = break_bonds(intermediate.molecule)  # generate all possible bond-breakages
#         for bond_breaking_type, graph_pairs in sub_graphs.items():
#             for graph_pair in graph_pairs:
#                 in_comp = [[surface, intermediate], []]
#                 for graph in graph_pair:
#                     for loop_inter in intermediates_dict.values():
#                         if len(loop_inter.graph.to_undirected()) != len(graph):
#                             continue
#                         if elem_inf(loop_inter.graph.to_undirected()) != elem_inf(graph):
#                             continue
#                         if nx.is_isomorphic(loop_inter.graph.to_undirected(), graph,
#                                             node_match=cate):
#                             in_comp[1].append(loop_inter)
#                             if len(in_comp[1]) == 2:
#                                 break
#                 chk_lst = [0, 0]
#                 for index, item in enumerate(in_comp):
#                     for mol in item:
#                         if mol.is_surface:
#                             continue
#                         chk_lst[index] += len(mol.molecule)

#                 if chk_lst[0] != chk_lst[1] and chk_lst[0] != chk_lst[1]/2:
#                     continue

#                 reaction = ElementaryReaction(r_type=bond_breaking_type, components=in_comp)
#                 for item in reaction_list:
#                     if item.components == reaction.components:
#                         break
#                 else:
#                     reaction_list.append(reaction)
#     return reaction_list

def find_matching_intermediates(graph, cate, cached_graphs):
    """
    Finds intermediates in the network that match the given graph.
    
    Args:
        graph (NetworkX Graph): The graph to match.
        cate (categorical_node_match): Node matching function for isomorphism check.
        cached_graphs (dict): Cached graph and element information for intermediates.
        
    Returns:
        list: Matching intermediates.
    """
    matching_intermediates = []
    graph_info = (len(graph), elem_inf(graph))
    
    for loop_inter, (cached_graph, cached_elem_inf) in cached_graphs.items():
        graph_len, elem_info = graph_info
        if cached_elem_inf != elem_info or len(cached_graph) != graph_len:
            continue

        if nx.is_isomorphic(cached_graph, graph, node_match=cate):
            matching_intermediates.append(loop_inter)
            if len(matching_intermediates) == 2:
                break
    return matching_intermediates

def validate_components(in_comp):
    """
    Validates the components based on the length of the molecule.
    
    Args:
        in_comp (list): List of components.
        
    Returns:
        bool: True if components are valid, False otherwise.
    """
    chk_lst = [0, 0]
    for index, item in enumerate(in_comp):
        for mol in item:
            if mol.is_surface:
                continue
            chk_lst[index] += len(mol.molecule)
    return chk_lst[0] == chk_lst[1] or chk_lst[0] == chk_lst[1]/2

def create_or_append_reaction(reaction, reaction_set, reaction_list):
    """
    Adds a new reaction to the list or updates an existing one.
    
    Args:
        reaction (ElementaryReaction): The reaction to add or update.
        reaction_set (set): Set of existing reactions for quick lookup.
        reaction_list (list): List of reactions.
    """
    if reaction.components not in reaction_set:
        reaction_set.add(reaction.components)
        reaction_list.append(reaction)

def break_and_connect(intermediates_dict: dict[str, Intermediate], surface: Atoms) -> list[ElementaryReaction]:
    """
    For an entire network, perform a breakage search for all the intermediates,
    search if the generated submolecules belong to the network and if so, create
    an ElementaryReaction object that connects all the species.
    
    Args:
        intermediates_dict (dict): Dictionary of intermediates in the network.
        surface (Atoms): Surface intermediate.
        
    Returns:
        list: List of ElementaryReaction objects.
    """
    cate = iso.categorical_node_match(['elem', 'elem', 'elem'], ['H', 'O', 'C'])
    reaction_set = set()
    reaction_list = []

    intermediates_values = list(intermediates_dict.values())
    cached_graphs = {inter: (inter.graph.to_undirected(), elem_inf(inter.graph)) for inter in intermediates_values}

    for intermediate in intermediates_values:
        sub_graphs = break_bonds(intermediate.molecule)
        
        for bond_breaking_type, graph_pairs in sub_graphs.items():
            for graph_pair in graph_pairs:
                in_comp = [[surface, intermediate], []]

                for graph in graph_pair:
                    matching_intermediates = find_matching_intermediates(graph, intermediates_dict, cate, cached_graphs)
                    in_comp[1].extend(matching_intermediates)

                if validate_components(in_comp):
                    reaction = ElementaryReaction(r_type=bond_breaking_type, components=in_comp)
                    create_or_append_reaction(reaction, reaction_set, reaction_list)

    return reaction_list

def change_r_type(network):
    """Given a network, search if the C-H breakages are correct, and if not
    correct them.

    Args:
        network (obj:`networks.ReactionNetwork`): Network in which the function
            will be performed.
    """
    for trans in network.t_states:
        if trans.r_type not in ['H-C', 'C-H']:
            continue
        flatten = [list(comp) for comp in trans.components]
        flatten_tmp = []
        for comp in flatten:
            flatten_tmp += comp
        flatten = flatten_tmp
        flatten.sort(key=lambda x: len(x.molecule.get_chemical_symbols()), reverse=True)
        flatten = flatten[1:3]  # With this step we will skip
        bonds = [bond_analysis(comp.molecule) for comp in flatten]
        bond_len = [item.bond_search(('C', 'O')) for item in bonds]
        if bond_len[0] == bond_len[1]:
            trans.r_type = 'C-H'
        else:
            trans.r_type = 'O-H'

def calculate_ts_energy(t_state, bader=False):
    """Calculate the ts energy of a transition state using the interpolation
     formula

    Args:
        t_state (obj:`networks.ElementaryReaction`): TS that will be evaluated.

    Returns:
        max value between the computed energy and the initial and final state
        energies.
    """
    components = [list(comp) for comp in t_state.bb_order()]
    alpha = INTERPOL[t_state.r_type]['alpha']
    beta = INTERPOL[t_state.r_type]['beta']
    if bader == False:
        e_is = [comp.energy for comp in components[0]]
        e_fs = [comp.energy for comp in components[1]]
    else:
        e_is = [comp.bader_energy for comp in components[0]]
        e_fs = [comp.bader_energy for comp in components[1]]
    tmp_vals = []
    for item in [e_is, e_fs]:
        if len(item) == 1:
            item *= 2
        item = sum(item)
        tmp_vals.append(item)
    e_is, e_fs = tmp_vals
    if t_state.is_electro:
        ts_ener = max(e_fs, e_is) + 0.05
    else:
        ts_ener = alpha * e_fs + (1. - alpha) * e_is + beta
    return max(e_is, e_fs, ts_ener)

def calc_TS_energy(t_state, bader=False):
    """Calculate the ts energy of a transition state using the interpolation
     formula

    Args:
        t_state (obj:`networks.ElementaryReaction`): TS that will be evaluated.

    Returns:
        max value between the computed energy and the initial and final state
        energies.
    """
    components = [list(comp) for comp in t_state.bb_order()]
    alpha = INTERPOL[t_state.r_type]['alpha']
    beta = INTERPOL[t_state.r_type]['beta']
    if bader == False:
        e_is = [comp.energy for comp in components[0]]
        e_fs = [comp.energy for comp in components[1]]
    else:
        e_is = [comp.bader_energy for comp in components[0]]
        e_fs = [comp.bader_energy for comp in components[1]]
    tmp_vals = []
    for item in [e_is, e_fs]:
        if len(item) == 1:
            item *= 2
        item = sum(item)
        tmp_vals.append(item)
    e_is, e_fs = tmp_vals
    if t_state.is_electro:
        ts_ener = max(e_fs, e_is) + 0.05
    else:
        ts_ener = alpha * e_fs + (1. - alpha) * e_is + beta
    return max(e_is, e_fs, ts_ener)

def generate_electron(t_state, electron='e-', proton='H+', def_h='000000', ener_gap=0.):
    new_ts = ElementaryReaction(r_type='C-H', is_electro=True)
    new_components = []
    for comp in t_state.components:
        tmp_comp = []
        for ind in comp:
            if ind.is_surface:
                tmp_comp.append(electron)
            elif ind.code == def_h:
                tmp_comp.append(proton)
            else:
                tmp_comp.append(ind)
        new_components.append(frozenset(tmp_comp))
    new_ts.components = new_components
    if t_state.energy is not None:
        new_ts.energy += ener_gap
    return new_ts

def search_electro(network, electron='e-', proton='H+', def_h='000000', ener_gap=0.):
    electro_states = []
    for t_state in network.t_states:
        if t_state.r_type not in ['C-H', 'H-C']:
            continue
        tmp_el = generate_electron(t_state, electron=electron,
                                   proton=proton, def_h=def_h, ener_gap=ener_gap)
        electro_states.append(tmp_el)
    return electro_states

def generate_colors(inter, colormap, norm, bader=False, custom_energy=None):
    """Given an intermediate with associated transition states, a colormap and
    a norm, return the colors of the different transition states depending of
    their energy.

    Args:
        inter (obj:`networks.Intermediate`): Intermediate that will be
            evaluated.
        colormap (obj:`matplotlib.cm`): Colormap to extract the colors.
        norm (obj:`matplotlib.colors.Normalize`): Norm to convert the energy
            value into a number between 0 and 1.

    Returns:
        2 lists of str both with a len of 7. The first containing the hex
        values of the colors and the second one containing the codes of the
        another part of the reaction.

    Notes:
        Both lists contain the colors and the intermediates taking into account
        the bond breaking type in this order:

        [Intermediate, C-OH, C-O, C-C, C-OH, C-O, C-C]
    """
    keys = ['C-OH', 'C-O', 'C-C']
    white = '#ffffff'
    full_colors = [white] * 9
    full_codes = [''] * 9
    for index, brk in enumerate(keys):
        try:
            colors = []
            codes = []
            if index == 2:
                state = 0
            else:
                state = 1
            for t_state in inter.t_states[state][brk]:
                if custom_energy is not None:
                    act_norm = norm(custom_energy)
                elif bader:
                    act_norm = norm(t_state.bader_energy)
                else:
                    act_norm = norm(t_state.energy)
                color = to_hex(colormap(act_norm))
                colors.append(color)
                for comp in t_state.components:
                    comp_lst = [mol.code for mol in list(comp) if not
                                mol.is_surface and not len(mol.molecule) == 1]
                    if inter in comp_lst:
                        continue
                    if len(comp_lst) == 2:
                        comp_lst = '{}<br/>{}'.format(*comp_lst)
                    else:
                        try:
                            comp_lst = comp_lst[0]
                        except IndexError:
                            pass
                    if act_norm > 0.5:
                        temp = '<FONT COLOR="#ffffff" POINT-SIZE="10">{}</FONT>'
                    else:
                        temp = '<FONT POINT-SIZE="10">{}</FONT>'
                    codes.append(temp.format(comp_lst))
            if len(colors) == 1:
                colors.append(white)
                colors.append(white)
                codes.append('')
                codes.append('')
            elif len(colors) == 2:
                colors.append(white)
                codes.append('')
            full_colors[index] = colors[0]
            full_colors[index + 3] = colors[1]
            full_colors[index + 6] = colors[2]
            full_codes[index] = codes[0]
            full_codes[index + 3] = codes[1]
            full_codes[index + 6] = codes[2]
        except KeyError:
            continue
    if custom_energy is not None:
        color = norm(custom_energy)
    elif bader:
        color = norm(inter.bader_energy)
    else:
        color = norm(inter.energy)
    color = to_hex(colormap(color))
    full_colors.insert(0, color)
    return full_colors, full_codes

def generate_label(formula, colors, codes, html_template=None):
    """Generate a html table with the colors and the codes generted with the
    generate_colorn function.

    Args:
        formula (str): Formula of the intermediate. Will be used as the title
            of the table.
        colors (list of str): List that contains the colors of the transition
            states associated to an intermediate.
        codes (list of str): List that contains the codes of the intermediates
            associated with the other part of the reaction.

    Returns:
        str with the generated html table compatible with dot language to use
        it as a label of a node.
    # """
    term = colors[0]
    rest = colors[1:]
    mix = [item for sublist in zip(rest, codes) for item in sublist]
    if html_template:
        label = html_template.format(term, formula, *mix)
    else:
        if len(term) > 6:
            label = BOX_TMP_3.format(term, formula, *mix)
        else:
            label = BOX_TMP.format(term, formula, *mix)
    return label

#TODO
def code_mol_graph(mol_graph, elems=['O', 'C']):
    """Given a molecule graph generated with the lib:`pyRDTP.operation.graph`
    node, return an str with the formula of the molecule.

    Args:
        mol_graph (obj:nx.Graph): Graph of the molecule.
        elems (list of objects, optional): List containing the elements that
            will be taken into account to walkt through the molecule. Defaults
            to ['O', 'C'].

    Retrns:
        str with the formula of the molecule with the format:
        CH-CO-CH3
    """
    new_graph = nx.DiGraph()
    for node in list(mol_graph.nodes()):
        if node.symbol in elems:
            new_graph.add_node(node)
    for edge in list(mol_graph.edges()):
        if edge[0].symbol in elems and edge[1].symbol in elems:
            new_graph.add_edge(*edge)
            new_graph.add_edge(*edge[::-1])

    max_val = 0
    for pair in combinations(list(new_graph.nodes()), 2):
        try:
            path = nx.shortest_path(new_graph, source=pair[0], target=pair[1])
        except nx.NetworkXNoPath:
            return ''
        path_len = len(path)
        if path_len > max_val:
            max_val = path_len
            longest_path = path

    if max_val == 0:
        longest_path = list(new_graph.nodes())

    path = ''
    connections = connectivity_helper(longest_path)
    for item in longest_path:
        if item.symbol == 'H':
            continue
        path += item.symbol
        count_H = 0
        H_lst = []
        oh_numb = []
        for hydro in item.connections:
            if hydro.symbol == 'H' and hydro not in H_lst:
                H_lst.append(hydro)
                count_H += 1
            if hydro.symbol == 'O':
                oh_numb.append(len([atom for atom in hydro.connections if
                                   atom.symbol == 'H']))
        if count_H > 1:
            path += '{}{}'.format('H', count_H)
        elif count_H == 1:
            path += '{}'.format('H')
        for numb in oh_numb:
            if numb == 0:
                path += '(O)'
            elif numb == 1:
                path += '(OH)'
            else:
                path += '(O{})'.format(numb)
        path += '-'
    path = path[:-1]
    return path

def radical_calc(inter):
    """Check if an intermediate is a radical.

    Args:
        inter (obj:`networks.Intermediate`): Intermediate to be tested.
    
    Returns:
        bool with the results of the test.
    """
    new_mol = inter.molecule.copy()
    del new_mol.arrays['conn_pairs']
    new_mol.arrays['conn_pairs'] = get_voronoi_neighbourlist(new_mol, 0,25, 1.0, ['C', 'H', 'O'])
    return insaturation_check(new_mol)

def underline_label(label):
    """Add the needed marks to a str to underline it in dot.

    Args:
       label (str): String that will be underlined.

    Returns:
       str with the underline marks.
    """
    temp = '<u>{}</u>'
    new_label = temp.format(label)
    return new_label

def change_color_label(label, color):
    """Add the needed marks to an str to change the font color on dat.

    Args:
        label (str): String to change the font color.
        color (str): Dot compatible color.

    Returns:
        str with the font color marks.
    """
    temp = '<FONT COLOR="{}">{}</FONT>'
    new_label = temp.format(color, label)
    return new_label

def adjust_co2(ase_atoms_obj):
    """Given a dict with elements, calculate the reference energy for
    the compound.

    Args:
        elements (dict): C, O, H as key and the number of atoms for every
            element as value.
    """
    GASES_ENER = {'CH4': -24.05681734,
                  'H2O': -14.51367559, # Water with solvent correction
                  'H'  : -3.383197435,
                  'CO2' : -22.96215586}
    
    elements_dict = {'C': sum([1 for atom in ase_atoms_obj if atom.symbol == 'C']),
                'H': sum([1 for atom in ase_atoms_obj if atom.symbol == 'H']),
                'O': sum([1 for atom in ase_atoms_obj if atom.symbol == 'O'])}
    
    pivot_dict = elements_dict.copy()
    
    for elem in ['O', 'C', 'H']:
        if elem not in pivot_dict:
            pivot_dict[elem] = 0
    

    energy = GASES_ENER['CO2'] * pivot_dict['C']
    energy += GASES_ENER['H2O'] * (pivot_dict['O'] - 2 * pivot_dict['C'])
    energy += GASES_ENER['H'] * (4 * pivot_dict['C'] + pivot_dict['H']
                                 - 2 * pivot_dict['O'])
    return energy

def read_object(filename):
    """Read a pickle object from the specified file

    Args:
        filename (str): Location of the file.

    Returns:
        Object readed from the pickle file.
    """
    with open(filename, 'rb') as obj_file:
        new_obj = pickle.load(obj_file)
    return new_obj

def write_object(obj, filename):
    """Write the given object to the specified file.

    Args:
        obj (obj): Object that will be written.
        filename (str): Name of the file where the
            object will be stored.
    """
    with open(filename, 'wb') as obj_file:
        new_file = pickle.dump(obj, obj_file)

def clear_graph(graph):
    """Generate a copy of the graph only using the edges and clearing the
    attributes of both nodes and edges.

    Args:
        graph (obj:`nx.DiGraph`): Base graph to clear.

    Returns:
        obj:`nx.Graph` that is a copy of the original without attributes.
    """
    new_graph = nx.DiGraph()
    for edge in graph.edges():
        new_graph.add_edge(*edge)
    return new_graph

def inverse_edges(graph):
    """Generate a copy of the graph and add additional edges that connect
    the nodes in the inverse direction while keeping the originals.

    Args:
        graph (obj:`nx.Graph`): Base graph to add the inverse edges.

    Returns:
        obj:`nx.Graph` that is a copy of the original but with the addition of
        the inverse edges
    """
    new_graph = graph.copy()
    for edge in graph.edges():
        new_graph.add_edge(edge[1], edge[0])
    return new_graph

def search_species(vertex_map, code):
    for index, species in enumerate(vertex_map):
        if code == species:
            return str(index)

def calc_path_energy(graph, path):
    accumulator = 0
    for index in range(1, len(path) - 1, 2):
        t_state_ener = graph.nodes[path[index]]['energy']
        try:
            inter_ener = graph.nodes[path[index - 1]]['energy']
        except KeyError:
            print(path[index - 1])
        accumulator += t_state_ener - inter_ener
    return accumulator        

def calc_all_energies(graph, path_lst):
    energy_lst = []
    for path in path_lst:
        energy = calc_path_energy(graph, path)
        energy_lst.append(energy)
    return tuple(energy_lst)

def calc_hydrogens_energy(inter, h_ener, s_ener, min_hydrogens=3, max_hydrogens=8, max_oxygens=1):
    if 'H' not in inter.molecule.get_chemical_symbols(): 
        mol_h_numb = 0
    else:
        mol_h_numb = sum(1 for atom in inter.molecule if atom.symbol == 'H')
        
    if 'O' not in inter.molecule.get_chemical_symbols():
        mol_h_numb += 1 * max_oxygens
    else:
        mol_h_numb += 1 * (max_oxygens - sum(1 for atom in inter.molecule if atom.symbol == 'O'))
    h_numb = max_hydrogens - mol_h_numb 
    s_numb =  mol_h_numb - min_hydrogens
    return (h_ener * h_numb + s_ener * s_numb)

def calc_ts_hydrogens(t_state, h_ener, s_ener, min_hydrogens=3, max_hydrogens=8, max_oxygens=1):
    comps = t_state.bb_order()[0]
    elem_info = [inter.molecule.get_chemical_symbols() for inter in comps if 'H' in inter.molecule.get_chemical_symbols()]
    if 'H' not in elem_info[0]:
        mol_h_numb = 0
    else:
        elem_info = sorted(elem_info, key=lambda x: x['H'], reverse=True)
        elem_info = elem_info[0]
        mol_h_numb = elem_info['H']
    if 'O' not in elem_info:
        mol_h_numb += 1 * max_oxygens
    else:
        mol_h_numb += 1 * (max_oxygens - elem_info['O'])
    h_numb = max_hydrogens - mol_h_numb 
    s_numb =  mol_h_numb - min_hydrogens
    return (h_ener * h_numb + s_ener * s_numb)  

def norm_ts_thickness(energy, boundaries=(0.5, 0.6, 0.7, 0.8, 0.9, 1.0), delta=1, offset=1, reverse=True):
    for index, value in enumerate(boundaries):
        if energy <= value:
            index_val = index
            break
    else:
        index_val = len(boundaries)
    if reverse:
        index_val = len(boundaries) - index_val
    return_val = (index_val * delta) + offset
    return return_val

def norm_ts_thickness_rate(rate, boundaries=(0.5, 0.6, 0.7, 0.8, 0.9, 1.0), delta=1, offset=0.1, reverse=True):
    for index, value in enumerate(boundaries):
        #print(rate,value,index)
        if rate >= value:
            index_val = index
            break
    else:
        index_val = len(boundaries)
    if reverse:
        index_val = len(boundaries) - index_val
    return_val = (index_val * delta) + offset
    return return_val

def print_intermediates(network, filename='inter.dat'):
    """Print a text file containing the information of the intermediates of a
    network.

    Args:
        network (obj:`ReactionNetwork`): Network containing the intermediates.
        filename (str, optional): Location where the file will be writed.
            Defaults to 'inter.dat'
    """
    header = 'Label          iO    Formula\n'
    inter_str = '{:6} {:.16f} {:.20}\n'
    with open(filename, 'w') as outfile:
        outfile.write(header)
        for label, inter in network.intermediates.items():
            formula = code_mol_graph(inter.graph)
            ener = inter.energy
            out_str = inter_str.format(label, ener, formula)
            outfile.write(out_str)

def print_intermediates_kinetics(network, filename='inter.dat'):
    """Print a text file containing the information of the intermediates of a
    network.

    Args:
        network (obj:`ReactionNetwork`): Network containing the intermediates.
        filename (str, optional): Location where the file will be writed.
            Defaults to 'inter.dat'
    """
    header = 'Label       iO       e   frq\n'
    inter_str = 'i{:6} {: .8f} {:2d} {:.20}\n'
    tmp_frq = '[0,0,0]'
    with open(filename, 'w') as outfile:
        outfile.write(header)
        for label, inter in network.intermediates.items():
            ener = inter.energy
            electrons = inter.electrons
            out_str = inter_str.format(label, ener, electrons, tmp_frq)
            outfile.write(out_str)

def print_t_states_kinetics(network, filename='ts.dat'):
    """Print a text file containing the information of the transition states
    of a network.

    Args:
        network (obj:`ReactionNetwork`): Network containing the transition
            states.
        filename (str, optional): Location where the file will be writed.
            Defaults to 'ts.dat'
    """
    header = '          Label                is1     is2     fs1     fs2        iO      e alpha beta   frq\n'
    tmp_frq = '[0,0,0]'
    inter_str = '{:28} i{:6} i{:6} i{:6} i{:6} {: .8f} {: 2d} {:3.2f} {:2.2f} {:.20}\n'
    with open(filename, 'w') as outfile:
        outfile.write(header)
        for t_state in network.t_states:
            label = t_state.code
            initial = [inter.code for inter in list(t_state.components[0])]
            final = [inter.code for inter in list(t_state.components[1])]

            electrons_fin = sum([inter.electrons for inter in
                                 list(t_state.components[1])])
            electrons_in = sum([inter.electrons for inter in
                                list(t_state.components[0])])
            if len(initial) == 1:
                initial *= 2
                electrons_in *= 2
            if len(final) == 1:
                final *= 2
                electrons_fin *= 2
            order = []
            for item in [initial, final]:
                mols = [network.intermediates[item[0]],
                        network.intermediates[item[1]]]
                if mols[0].is_surface:
                    order.append(item[::-1])
                elif not mols[1].is_surface and (len(mols[0].molecule) <
                                                 len(mols[1].molecule)):
                    order.append(item[::-1])
                else:
                    order.append(item)

            initial, final = order
            electrons = electrons_fin - electrons_in
            ener = t_state.energy
            alpha = INTERPOL[t_state.r_type]['alpha']
            beta = INTERPOL[t_state.r_type]['beta']
            label = ''
            for item in initial + final:
                label += 'i' + item
            outfile.write(inter_str.format(label, *initial, *final,
                                           ener, electrons, alpha,
                                           beta, tmp_frq))

def print_t_states(network, filename='ts.dat'):
    """Print a text file containing the information of the transition states
    of a network.

    Args:
        network (obj:`ReactionNetwork`): Network containing the transition
            states.
        filename (str, optional): Location where the file will be writed.
            Defaults to 'ts.dat'
    """
    header = '          Label               is1    is2    fs1    fs2           iO\n'
    inter_str = '{:20} {:6} {:6} {:6} {:6} {:.16f}\n'
    with open(filename, 'w') as outfile:
        outfile.write(header)
        for t_state in network.t_states:
            label = t_state.code
            initial = [inter.code for inter in list(t_state.components[0])]
            final = [inter.code for inter in list(t_state.components[1])]
            if len(initial) == 1:
                initial *= 2
            if len(final) == 1:
                final *= 2
            ener = t_state.energy
            entr = t_state.entropy
            # if ener == None:
            #     ener = 0.0
            # if entr == None:
            #     entr = 0.0
            # print(label,*initial,*final,ener, entr)
            outfile.write(inter_str.format(label, *initial, *final, ener, entr))

def print_gasses_kinetics(gas_dict, filename='gas.dat'):
    """Print a text file containing the information of the intermediates of a
    network.

    Args:
        network (obj:`ReactionNetwork`): Network containing the intermediates.
        filename (str, optional): Location where the file will be writed.
            Defaults to 'inter.dat'
    """
    header = 'Label    Formula              ads         gas      iO         e   mw   frq\n'
    inter_str = '{:6} {:20} i{:7} {: .8f} {: .8f} {:4.2f} {:2d} {:.20}\n'
    tmp_frq = '[0,0,0]'
    with open(filename, 'w') as outfile:
        outfile.write(header)
        for label, gas in gas_dict.items():
            ener = gas['energy']
            electrons = gas['electrons']
            try:
                gas['mol'].arrays['conn_pairs'] = get_voronoi_neighbourlist(gas['mol'], 0.25, 1.0, ['C', 'H', 'O'])
                formula = code_mol_graph(ase_coord_2_graph(gas['mol'], coords=False), ['C'])
            except nx.NetworkXNoPath:
                formula = gas['mol'].get_chemical_formula()
            if not formula:
                formula = gas['mol'].get_chemical_formula()
            weight = calculate_weigth(dict(Counter(gas['mol'].get_chemical_symbols())))
            out_str = inter_str.format(label, formula, label, ener,
                                       ener, weight, electrons, tmp_frq)
            outfile.write(out_str)

def read_TS_energies(filename):
    """Reads an energy filename and convert it to a dictionary.

    Args:
        filename (str): Location of the file containing the energies.

    Returns:
        Two dictionaris in which keys are the code values of the
        intermediates and the associated values are the energy of the
        intermediates.
        The first dictionary contains the values that do not contain failed,
        warning or WARNING at the end of their code. Moreover, if multiple
        codes with the same 6 characters start are found, only the one with
        the lesser energy is selected.
    """
    ener_dict = {}
    discard = {}
    repeated = {}
    entropy_dict = {}

    with open(filename, 'r') as infile:
        lines = infile.readlines()

    for sg_line in lines:
        code_array=[]
        try:
            code, energy, entropy = sg_line.split(' ')
            for i in code.split('-'):
                for j in i.split('+'):
                    code_array.append(j[1:])
            code_array.sort()
            #print(code_array)
            code = ""
            # traverse in the string 
            for j in code_array:
                code += j 
            # print(code)
        except ValueError:
            discard[code[:6]] = 0
        energy = float(energy)  # Energy needs to be a float
        entropy = float(entropy)
        
        #if code[0] == 'i':
        #    code = code[1:]

        #if len(code) > 6:
        #    continue
            # init_code = code[:6]
            # if code.endswith(('WARNING', 'warning', 'failed')):
            #     discard[init_code] = energy
            #     continue
            # if init_code not in repeated:
            #     repeated[init_code] = []
            # repeated[init_code].append(energy)
            # continue
        ener_dict[code] = energy
        entropy_dict[code] = entropy
#    for code, energies in repeated.items():
#        ener_dict[code] = min(energies)

    return ener_dict, entropy_dict, discard

def read_energies(filename):
    """Reads an energy filename and convert it to a dictionary.

    Args:
        filename (str): Location of the file containing the energies.

    Returns:
        Two dictionaris in which keys are the code values of the
        intermediates and the associated values are the energy of the
        intermediates.
        The first dictionary contains the values that do not contain failed,
        warning or WARNING at the end of their code. Moreover, if multiple
        codes with the same 6 characters start are found, only the one with
        the lesser energy is selected.
    """
    ener_dict = {}
    discard = {}
    repeated = {}
    entropy_dict = {}

    with open(filename, 'r') as infile:
        lines = infile.readlines()

    for sg_line in lines:
        try:
            code, energy, entropy = sg_line.split()
        except ValueError:
            discard[code[:6]] = 0
        energy = float(energy)  # Energy needs to be a float
        entropy = float(entropy)
        
        if code[0] == 'i':
            code = code[1:]

        if len(code) > 6:
            continue
            # init_code = code[:6]
            # if code.endswith(('WARNING', 'warning', 'failed')):
            #     discard[init_code] = energy
            #     continue
            # if init_code not in repeated:
            #     repeated[init_code] = []
            # repeated[init_code].append(energy)
            # continue
        ener_dict[code] = energy
        entropy_dict[code] = entropy
#    for code, energies in repeated.items():
#        ener_dict[code] = min(energies)

    return ener_dict, entropy_dict, discard

def adjust_electrons(ase_atoms_obj: Atoms):
    """Given a dict with elements, calculate the reference energy for
    the compound.

    Args:
        elements_dict (dict): C, O, H as key and the number of atoms for every
            element as value.
    """
    elements_dict = {'C': ase_atoms_obj.get_chemical_symbols().count('C'),
                'H': ase_atoms_obj.get_chemical_symbols().count('H'),
                'O': ase_atoms_obj.get_chemical_symbols().count('O')}

    n_electrons = (4 * elements_dict['C'] + elements_dict['H']
                 - 2 * elements_dict['O'])
    return n_electrons

def adjust_electrons_H(molecule: Atoms):
    elements = dict(Counter(molecule.get_chemical_symbols()))
    pivot_dict = elements.copy()
    if 'H' not in pivot_dict:
        pivot_dict['H'] = 0

    electrons = pivot_dict['H'] # + search_alcoxy(molecule)
    return electrons

def select_larger_inter(inter_lst):
    """Given a list of Intermediates, select the intermediate with the
    highest number of atoms.

    Args:
        inter_lst(list of obj:`networks.Intermediate`): List that will be
            evaluated.
    Returns:
        obj:`networks.Intermediate` with the bigger molecule.
    """
    atoms = [len(inter.mol) for inter in inter_lst]
    inter_max = [0, 0]
    for size, inter in zip(atoms, inter_lst):
        if size > inter_max[0]:
            inter_max[0] = size
            inter_max[1] = inter
    return inter_max[1]

#TODO
def search_electro_ts(network, electron='e-', proton='H', water='H2O', ener_up=0.05):
    """Search for all the possible electronic transition states of a network.

    Args:
        network (obj:`networks.ReactionNetwork`): Network in wich the electronic
            states will be generated.
        electron (any object): Object that represents an electron.
        proton (any object): Object that represents a proton.
        water (any object): Object that represents a water molecule.
    """
    cate = iso.categorical_node_match(['elem', 'elem'], ['H', 'O'])
    electro_ts = []
    for inter in network.intermediates.values():
        if 'O' not in dict(Counter(inter.molecule.get_chemical_symbols())):
            continue
        tmp_oxy = inter.molecule['O']
        for index, _ in enumerate(tmp_oxy):
            tmp_mol = inter.molecule.copy()
            oxygen = [atom for atom in tmp_mol if atom.symbol == "O"][index]

            hydrogen = [conect for conect in oxygen.connections if
                        conect.symbol == 'H']
            if not hydrogen:
                continue

            del tmp_mol[hydrogen[0]]
            del tmp_mol.arrays['conn_pairs']
            tmp_mol.arrays['conn_pairs'] = get_voronoi_neighbourlist(tmp_mol, 0.25, 1.0, ['C', 'H', 'O'])
            tmp_graph = ase_coord_2_graph(tmp_mol, coords=False)


            candidates = network.search_graph(tmp_graph, cate=cate)
            for new_inter in candidates:
                oh_comp = [[new_inter, proton], [inter, electron]]
                oh_ts = ElementaryReaction(components=oh_comp,
                                            r_type='O-H', is_electro=True)
                inter_energy = inter.energy + electron.energy
                new_inter_energy = new_inter.energy + proton.energy
                inter_energy_bader = inter.bader_energy + electron.energy
                new_inter_energy_bader = new_inter.bader_energy + proton.energy
                oh_ts.energy = max((inter_energy, new_inter_energy)) + ener_up
                oh_ts.bader_energy = max((inter_energy_bader, new_inter_energy_bader)) + ener_up
                electro_ts.append(oh_ts)

            del tmp_mol[oxygen]
            del tmp_mol.arrays['conn_pairs']
            try:
                tmp_mol.arrays['conn_pairs'] = get_voronoi_neighbourlist(tmp_mol, 0.25, 1.0, ['C', 'H', 'O'])
            except ValueError:
                continue
            tmp_graph = ase_coord_2_graph(tmp_mol, coords=False)

            
            candidates = network.search_graph(tmp_graph, cate=cate)
            for new_inter in candidates:
                wa_comp = [[inter, proton], [new_inter, water]]            
                wa_ts = ElementaryReaction(components=wa_comp,
                        r_type='C-OH', is_electro=True)
                inter_energy = inter.energy + electron.energy
                new_inter_energy = new_inter.energy + proton.energy
                inter_energy_bader = inter.bader_energy + electron.energy
                new_inter_energy_bader = new_inter.bader_energy + proton.energy
                wa_ts.energy = max((inter_energy, new_inter_energy)) + ener_up
                wa_ts.bader_energy = max((inter_energy_bader, new_inter_energy_bader)) + ener_up
                electro_ts.append(wa_ts)

    return electro_ts

def print_electro_kinetics(ts_lst, filename='elec.dat'):
    """Print a text file containing the information of the transition states
    of a network.

    Args:
        network (obj:`ReactionNetwork`): Network containing the transition
            states.
        filename (str, optional): Location where the file will be writed.
            Defaults to 'ts.dat'
    """
    header = '              Label                is1     is2     fs1     fs2        iO      e    frq\n'
    tmp_frq = '[0,0,0]'
    inter_str = '{:32} {:7} {:7} {:7} {:7} {: .8f} {: 2d} {:.20}\n'
    with open(filename, 'w') as outfile:
        outfile.write(header)
        for t_state in ts_lst:
            ordered = t_state.components
            initial = [inter.code for inter in list(ordered[0])]
            final = [inter.code for inter in list(ordered[1])]

            electrons_fin = sum([inter.electrons for inter in
                                 list(ordered[1])])
            electrons_in = sum([inter.electrons for inter in
                                list(ordered[0])])

            for item in [initial, final]:
                for index, _ in enumerate(item):
                    if not item[index]:
                        item[index] = 'None'
                    elif item[index][0] != 'g':
                        item[index] = 'i' + item[index]

            if len(initial) == 1:
                initial *= 2
            if len(final) == 1:
                final *= 2
            order = []
            for item in [initial, final]:
                if item[0] == 'None' or item[0].startswith('g'):
                    order.append(item[::-1])
                else:
                    order.append(item)
            initial, final = order
            label = ''
            for item in initial + final:
                if item == 'None':
                    label += 'g000000'
                else:
                    label += item
            electrons = electrons_fin - electrons_in
            ener = t_state.energy
            outfile.write(inter_str.format(label, *initial, *final,
                                           ener, electrons, tmp_frq))

#TODO
# def search_alcoxy(molecule: Atoms) -> int:
#     mol = molecule.copy()
#     # mol.connectivity_search_voronoi()
#     mol.arrays['conn_pairs'] = get_voronoi_neighbourlist(mol, 0.25, 1, ['C', 'H', 'O'])
#     # oxy = mol['O']
#     oxy_list = [atom for atom in mol if atom.symbol == "O"]
#     alco_numb = 0
#     for atom in oxy_list:
#         if np.count_nonzero(mol.arrays['conn_pairs'] == atom.index) > 1 or np.count_nonzero(mol.arrays['conn_pairs'] == atom.index) == 0:
#             continue
#         index = np.where(mol.arrays['conn_pairs'] == atom.index)[0]
#         index_2 = 0 if index[1]==atom.index else 1
#         elif mol[index_2].symbol != 'C':
#             continue
#         elif len(atom.connections[0].connections) != 4:
#             continue
#         else:
#             alco_numb += 1
#     return alco_numb
