import networkx as nx
import numpy as np
import networkx.algorithms.isomorphism as iso
from care.rnet.networks.elementary_reaction import ElementaryReaction
from care.rnet.networks.intermediate import Intermediate
from ase import Atoms
from care.rnet.utilities.functions import get_voronoi_neighbourlist
from care.rnet.graphs.graph_fn import ase_coord_2_graph
import multiprocessing as mp
import os
from collections import defaultdict

from care.rnet.utilities.bond import Bond, BondPackage

INTERPOL = {'O-H' : {'alpha': 0.39, 'beta': 0.89}, 
            'C-H' : {'alpha': 0.63, 'beta': 0.81},
            'H-C' : {'alpha': 0.63, 'beta': 0.81},
            'C-C' : {'alpha': 1.00, 'beta': 0.64},
            'C-O' : {'alpha': 1.00, 'beta': 1.24},
            'C-OH': {'alpha': 1.00, 'beta': 1.48}, 
            'O-O' : {'alpha': 1.00, 'beta': 1.00},  # added by me
            'default': {'alpha': 0.00, 'beta': 0.00}} 

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
    if len(atoms) == 1: 
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
    bonds = [Bond(mol, mol.arrays['conn_pairs'][row, 0], mol.arrays['conn_pairs'][row, 1]) for row in range(mol.arrays['conn_pairs'].shape[0])]
    package.bond_add(bonds)
    # for row in range(mol.arrays['conn_pairs'].shape[0]):
    #     package.bond_add(Bond(mol, mol.arrays['conn_pairs'][row, 0], mol.arrays['conn_pairs'][row, 1]))
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


def find_matching_intermediates(graph: nx.Graph, cate, cached_graphs: dict[str, tuple[nx.Graph, dict]]) -> list[str]:
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
    
    for inter_code, (cached_graph, cached_elem_inf) in cached_graphs.items():
        graph_len, elem_info = graph_info
        if cached_elem_inf != elem_info or len(cached_graph) != graph_len:
            continue

        if nx.is_isomorphic(cached_graph, graph, node_match=cate):
            matching_intermediates.append(inter_code)
            if len(matching_intermediates) == 2:
                break
    return matching_intermediates

def char_to_int(c):
    if c.isdigit():
        return int(c)
    elif 'a' <= c <= 'z':
        return ord(c) - ord('a') + 10
    elif 'A' <= c <= 'Z':
        return ord(c) - ord('A') + 36
    else:
        raise ValueError(f"Invalid character: {c}")

def int_to_char(i):
    if 0 <= i <= 9:
        return str(i)
    elif 10 <= i <= 35:
        return chr(i - 10 + ord('a'))
    elif 36 <= i <= 61:
        return chr(i - 36 + ord('A'))
    else:
        raise ValueError(f"Invalid integer: {i}")

def validate_components(in_comp):
    """
    Validates the components based on the element material balance.
    
    Args:
        in_comp (list): List of components.
        
    Returns:
        bool: True if components are valid, False otherwise.
    """
    chk_lst = [0, 0]
    for index, item in enumerate(in_comp):
        for mol in item:
            if mol == '0000000000*':
                continue
            n_C = sum(char_to_int(c) for c in mol[:2])
            n_H = sum(char_to_int(c) for c in mol[2:4])
            n_O = sum(char_to_int(c) for c in mol[4:6])
            chk_lst[index] += sum([n_C, n_H, n_O])
    return chk_lst[0] == chk_lst[1] or chk_lst[0] == chk_lst[1]*2

def create_or_append_reaction(reaction_info, reaction_set: set, reaction_list: list):
    """
    Adds a new reaction to the list or updates an existing one.
    
    Args:
        reaction (ElementaryReaction): The reaction to add or update.
        reaction_set (set): Set of existing reactions for quick lookup.
        reaction_list (list): List of reactions.

    Returns:
        bool: True if the reaction was added, False otherwise.
    """
    s_keys = [inter.code for inter in reaction_info[0][0]] + [inter.code for inter in reaction_info[0][1]]
    s_keys = list(set(s_keys))
    s_dict = {key: 0 for key in s_keys}
    for inter in reaction_info[0][0]:
        s_dict[inter.code] -= 1
    for inter in reaction_info[0][1]:
        s_dict[inter.code] += 1
    reaction = ElementaryReaction(r_type=reaction_info[1], components=reaction_info[0], stoic=s_dict)
    if reaction.components not in reaction_set:
        reaction_set.add(reaction.components)
        reaction_list.append(reaction)
        return True
    return False

def find_and_cache_matching_intermediates(cate, cached_graphs, graph_pair):
    return [
        find_matching_intermediates(graph, cate, cached_graphs)
        for graph in graph_pair
    ]

def process_graph_pair(args) -> tuple[list[list[str]], str]:
    cached_graphs, surface_code, intermediate_code, bond_breaking_type, graph_pair = args
    cate = iso.categorical_node_match(['elem', 'elem', 'elem'], ['H', 'O', 'C'])

    if len(graph_pair) ==1:
        rxn_components = [[intermediate_code], []]
    else:
        rxn_components = [[surface_code, intermediate_code], []]
    matching_intermediates = find_and_cache_matching_intermediates(cate, cached_graphs, graph_pair)
    
    # Flatten the list of lists
    flat_matching_intermediates = [item for sublist in matching_intermediates for item in sublist]
    rxn_components[1].extend(list(set(flat_matching_intermediates)))

    if not validate_components(rxn_components):
        return None
    
    # reaction = ElementaryReaction(r_type=bond_breaking_type, components=rxn_components)
    return rxn_components, bond_breaking_type

def break_bonds(molecule: Atoms) -> dict[str, list[list[nx.Graph]]]:
    connections = connectivity_helper(molecule)
    bonds = defaultdict(list)
    bond_pack = bond_analysis(molecule)
    mol_graph = ase_coord_2_graph(molecule, coords=False)

    def check_oh_bond(atom):
        return 'H' in [molecule[con].symbol for con in connections[atom.index]]

    # for bond_type in (('O', 'C'), ('C', 'O'), ('C', 'C'), ('O', 'O'), ('H', 'H')):
    for pair in bond_pack:
        tmp_graph = mol_graph.copy()
        tmp_graph.remove_edge(pair.atom_1.index, pair.atom_2.index)
        bond = sorted([pair.atom_1.symbol, pair.atom_2.symbol])
        bond = '-'.join(bond)
        sub_graphs = [tmp_graph.subgraph(comp).copy() for comp in nx.connected_components(tmp_graph)]
        if bond in ('C-O', 'O-O'):
            if any(check_oh_bond(atom) for atom in pair.atoms):
                bonds[bond+'H'].append(sub_graphs)
            else:
                bonds[bond].append(sub_graphs)
        else:
            bonds[bond].append(sub_graphs)

            # if bond_type in (('O', 'C'), ('C', 'O')):
            #     oh_bond = any(check_oh_bond(atom) for atom in pair.atoms)
            #     if oh_bond:
            #         bonds['C-OH'].append(sub_graphs)
            #     else:
            #         bonds['C-O'].append(sub_graphs)
            # elif bond_type == ('O', 'O'):
            #     oh_bond = any(check_oh_bond(atom) for atom in pair.atoms)
            #     if oh_bond:
            #         bonds['O-OH'].append(sub_graphs)
            #     else:
            #         bonds['O-O'].append(sub_graphs)
            # elif bond_type == ('H', 'H'):
            #     bonds['H-H'].append(sub_graphs)
            # else:
            #     bonds['C-C'].append(sub_graphs)

    return bonds

def break_and_connect(intermediates_dict: dict[str, Intermediate]) -> list[ElementaryReaction]:
    """
    Given a dictionary of Intermediates (closed-shell and dehydrogenated) and a surface, find all possible
    bond breaking reactions and return them as a list of ElementaryReactions.
    """
    reaction_set = set()
    reaction_list = []

    intermediates_values = list(intermediates_dict.values())
    cached_graphs = {inter.code[:-1]: (inter.graph.to_undirected(), elem_inf(inter.graph)) for inter in intermediates_values}

    args_list = []
    for intermediate in intermediates_values:
        if intermediate.is_surface:
            continue
        sub_graphs = break_bonds(intermediate.molecule)
        print('sub_graphs: ', sub_graphs)
        if not sub_graphs:
            continue

        for bond_breaking_type, graph_pairs in sub_graphs.items():
            for graph_pair in graph_pairs:
                args_list.append((cached_graphs, '0000000000', intermediate.code[:-1], bond_breaking_type, graph_pair))

    with mp.Pool(os.cpu_count()//2) as pool:
        results = pool.map(process_graph_pair, args_list)

    # post-process results
    for reaction in results:
        if reaction is not None:
            components_lhs = [intermediates_dict[code] for code in reaction[0][0]]
            components_rhs = [intermediates_dict[code] for code in reaction[0][1]]
            reaction_info = [components_lhs, components_rhs], reaction[1]
            create_or_append_reaction(reaction_info, reaction_set, reaction_list)

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

# def generate_electron(t_state, electron='e-', proton='H+', def_h='000000', ener_gap=0.):
#     new_ts = ElementaryReaction(r_type='C-H', is_electro=True)
#     new_components = []
#     for comp in t_state.components:
#         tmp_comp = []
#         for ind in comp:
#             if ind.is_surface:
#                 tmp_comp.append(electron)
#             elif ind.code == def_h:
#                 tmp_comp.append(proton)
#             else:
#                 tmp_comp.append(ind)
#         new_components.append(frozenset(tmp_comp))
#     new_ts.components = new_components
#     if t_state.energy is not None:
#         new_ts.energy += ener_gap
#     return new_ts

# def search_electro(network, electron='e-', proton='H+', def_h='000000', ener_gap=0.):
#     electro_states = []
#     for t_state in network.t_states:
#         if t_state.r_type not in ['C-H', 'H-C']:
#             continue
#         tmp_el = generate_electron(t_state, electron=electron,
#                                    proton=proton, def_h=def_h, ener_gap=ener_gap)
#         electro_states.append(tmp_el)
#     return electro_states

# def generate_colors(inter, colormap, norm, bader=False, custom_energy=None):
#     """Given an intermediate with associated transition states, a colormap and
#     a norm, return the colors of the different transition states depending of
#     their energy.

#     Args:
#         inter (obj:`networks.Intermediate`): Intermediate that will be
#             evaluated.
#         colormap (obj:`matplotlib.cm`): Colormap to extract the colors.
#         norm (obj:`matplotlib.colors.Normalize`): Norm to convert the energy
#             value into a number between 0 and 1.

#     Returns:
#         2 lists of str both with a len of 7. The first containing the hex
#         values of the colors and the second one containing the codes of the
#         another part of the reaction.

#     Notes:
#         Both lists contain the colors and the intermediates taking into account
#         the bond breaking type in this order:

#         [Intermediate, C-OH, C-O, C-C, C-OH, C-O, C-C]
#     """
#     keys = ['C-OH', 'C-O', 'C-C']
#     white = '#ffffff'
#     full_colors = [white] * 9
#     full_codes = [''] * 9
#     for index, brk in enumerate(keys):
#         try:
#             colors = []
#             codes = []
#             if index == 2:
#                 state = 0
#             else:
#                 state = 1
#             for t_state in inter.t_states[state][brk]:
#                 if custom_energy is not None:
#                     act_norm = norm(custom_energy)
#                 elif bader:
#                     act_norm = norm(t_state.bader_energy)
#                 else:
#                     act_norm = norm(t_state.energy)
#                 color = to_hex(colormap(act_norm))
#                 colors.append(color)
#                 for comp in t_state.components:
#                     comp_lst = [mol.code for mol in list(comp) if not
#                                 mol.is_surface and not len(mol.molecule) == 1]
#                     if inter in comp_lst:
#                         continue
#                     if len(comp_lst) == 2:
#                         comp_lst = '{}<br/>{}'.format(*comp_lst)
#                     else:
#                         try:
#                             comp_lst = comp_lst[0]
#                         except IndexError:
#                             pass
#                     if act_norm > 0.5:
#                         temp = '<FONT COLOR="#ffffff" POINT-SIZE="10">{}</FONT>'
#                     else:
#                         temp = '<FONT POINT-SIZE="10">{}</FONT>'
#                     codes.append(temp.format(comp_lst))
#             if len(colors) == 1:
#                 colors.append(white)
#                 colors.append(white)
#                 codes.append('')
#                 codes.append('')
#             elif len(colors) == 2:
#                 colors.append(white)
#                 codes.append('')
#             full_colors[index] = colors[0]
#             full_colors[index + 3] = colors[1]
#             full_colors[index + 6] = colors[2]
#             full_codes[index] = codes[0]
#             full_codes[index + 3] = codes[1]
#             full_codes[index + 6] = codes[2]
#         except KeyError:
#             continue
#     if custom_energy is not None:
#         color = norm(custom_energy)
#     elif bader:
#         color = norm(inter.bader_energy)
#     else:
#         color = norm(inter.energy)
#     color = to_hex(colormap(color))
#     full_colors.insert(0, color)
#     return full_colors, full_codes

# def generate_label(formula, colors, codes, html_template=None):
#     """Generate a html table with the colors and the codes generted with the
#     generate_colorn function.

#     Args:
#         formula (str): Formula of the intermediate. Will be used as the title
#             of the table.
#         colors (list of str): List that contains the colors of the transition
#             states associated to an intermediate.
#         codes (list of str): List that contains the codes of the intermediates
#             associated with the other part of the reaction.

#     Returns:
#         str with the generated html table compatible with dot language to use
#         it as a label of a node.
#     # """
#     term = colors[0]
#     rest = colors[1:]
#     mix = [item for sublist in zip(rest, codes) for item in sublist]
#     if html_template:
#         label = html_template.format(term, formula, *mix)
#     else:
#         if len(term) > 6:
#             label = BOX_TMP_3.format(term, formula, *mix)
#         else:
#             label = BOX_TMP.format(term, formula, *mix)
#     return label

# #TODO
# def code_mol_graph(mol_graph, elems=['O', 'C']):
#     """Given a molecule graph generated with the lib:`pyRDTP.operation.graph`
#     node, return an str with the formula of the molecule.

#     Args:
#         mol_graph (obj:nx.Graph): Graph of the molecule.
#         elems (list of objects, optional): List containing the elements that
#             will be taken into account to walkt through the molecule. Defaults
#             to ['O', 'C'].

#     Retrns:
#         str with the formula of the molecule with the format:
#         CH-CO-CH3
#     """
#     new_graph = nx.DiGraph()
#     for node in list(mol_graph.nodes()):
#         if node.symbol in elems:
#             new_graph.add_node(node)
#     for edge in list(mol_graph.edges()):
#         if edge[0].symbol in elems and edge[1].symbol in elems:
#             new_graph.add_edge(*edge)
#             new_graph.add_edge(*edge[::-1])

#     max_val = 0
#     for pair in combinations(list(new_graph.nodes()), 2):
#         try:
#             path = nx.shortest_path(new_graph, source=pair[0], target=pair[1])
#         except nx.NetworkXNoPath:
#             return ''
#         path_len = len(path)
#         if path_len > max_val:
#             max_val = path_len
#             longest_path = path

#     if max_val == 0:
#         longest_path = list(new_graph.nodes())

#     path = ''
#     connections = connectivity_helper(longest_path)
#     for item in longest_path:
#         if item.symbol == 'H':
#             continue
#         path += item.symbol
#         count_H = 0
#         H_lst = []
#         oh_numb = []
#         for hydro in item.connections:
#             if hydro.symbol == 'H' and hydro not in H_lst:
#                 H_lst.append(hydro)
#                 count_H += 1
#             if hydro.symbol == 'O':
#                 oh_numb.append(len([atom for atom in hydro.connections if
#                                    atom.symbol == 'H']))
#         if count_H > 1:
#             path += '{}{}'.format('H', count_H)
#         elif count_H == 1:
#             path += '{}'.format('H')
#         for numb in oh_numb:
#             if numb == 0:
#                 path += '(O)'
#             elif numb == 1:
#                 path += '(OH)'
#             else:
#                 path += '(O{})'.format(numb)
#         path += '-'
#     path = path[:-1]
#     return path

# def radical_calc(inter):
#     """Check if an intermediate is a radical.

#     Args:
#         inter (obj:`networks.Intermediate`): Intermediate to be tested.
    
#     Returns:
#         bool with the results of the test.
#     """
#     new_mol = inter.molecule.copy()
#     del new_mol.arrays['conn_pairs']
#     new_mol.arrays['conn_pairs'] = get_voronoi_neighbourlist(new_mol, 0,25, 1.0, ['C', 'H', 'O'])
#     return insaturation_check(new_mol)

# def underline_label(label):
#     """Add the needed marks to a str to underline it in dot.

#     Args:
#        label (str): String that will be underlined.

#     Returns:
#        str with the underline marks.
#     """
#     temp = '<u>{}</u>'
#     new_label = temp.format(label)
#     return new_label

# def change_color_label(label, color):
#     """Add the needed marks to an str to change the font color on dat.

#     Args:
#         label (str): String to change the font color.
#         color (str): Dot compatible color.

#     Returns:
#         str with the font color marks.
#     """
#     temp = '<FONT COLOR="{}">{}</FONT>'
#     new_label = temp.format(color, label)
#     return new_label

# def adjust_co2(ase_atoms_obj):
#     """Given a dict with elements, calculate the reference energy for
#     the compound.

#     Args:
#         elements (dict): C, O, H as key and the number of atoms for every
#             element as value.
#     """
#     GASES_ENER = {'CH4': -24.05681734,
#                   'H2O': -14.51367559, # Water with solvent correction
#                   'H'  : -3.383197435,
#                   'CO2' : -22.96215586}
    
#     elements_dict = {'C': sum([1 for atom in ase_atoms_obj if atom.symbol == 'C']),
#                 'H': sum([1 for atom in ase_atoms_obj if atom.symbol == 'H']),
#                 'O': sum([1 for atom in ase_atoms_obj if atom.symbol == 'O'])}
    
#     pivot_dict = elements_dict.copy()
    
#     for elem in ['O', 'C', 'H']:
#         if elem not in pivot_dict:
#             pivot_dict[elem] = 0
    

#     energy = GASES_ENER['CO2'] * pivot_dict['C']
#     energy += GASES_ENER['H2O'] * (pivot_dict['O'] - 2 * pivot_dict['C'])
#     energy += GASES_ENER['H'] * (4 * pivot_dict['C'] + pivot_dict['H']
#                                  - 2 * pivot_dict['O'])
#     return energy

# def read_object(filename):
#     """Read a pickle object from the specified file

#     Args:
#         filename (str): Location of the file.

#     Returns:
#         Object readed from the pickle file.
#     """
#     with open(filename, 'rb') as obj_file:
#         new_obj = pickle.load(obj_file)
#     return new_obj

# def write_object(obj, filename):
#     """Write the given object to the specified file.

#     Args:
#         obj (obj): Object that will be written.
#         filename (str): Name of the file where the
#             object will be stored.
#     """
#     with open(filename, 'wb') as obj_file:
#         new_file = pickle.dump(obj, obj_file)

# def clear_graph(graph):
#     """Generate a copy of the graph only using the edges and clearing the
#     attributes of both nodes and edges.

#     Args:
#         graph (obj:`nx.DiGraph`): Base graph to clear.

#     Returns:
#         obj:`nx.Graph` that is a copy of the original without attributes.
#     """
#     new_graph = nx.DiGraph()
#     for edge in graph.edges():
#         new_graph.add_edge(*edge)
#     return new_graph

# def inverse_edges(graph):
#     """Generate a copy of the graph and add additional edges that connect
#     the nodes in the inverse direction while keeping the originals.

#     Args:
#         graph (obj:`nx.Graph`): Base graph to add the inverse edges.

#     Returns:
#         obj:`nx.Graph` that is a copy of the original but with the addition of
#         the inverse edges
#     """
#     new_graph = graph.copy()
#     for edge in graph.edges():
#         new_graph.add_edge(edge[1], edge[0])
#     return new_graph

# def search_species(vertex_map, code):
#     for index, species in enumerate(vertex_map):
#         if code == species:
#             return str(index)

# def calc_path_energy(graph, path):
#     accumulator = 0
#     for index in range(1, len(path) - 1, 2):
#         t_state_ener = graph.nodes[path[index]]['energy']
#         try:
#             inter_ener = graph.nodes[path[index - 1]]['energy']
#         except KeyError:
#             print(path[index - 1])
#         accumulator += t_state_ener - inter_ener
#     return accumulator        

# def calc_all_energies(graph, path_lst):
#     energy_lst = []
#     for path in path_lst:
#         energy = calc_path_energy(graph, path)
#         energy_lst.append(energy)
#     return tuple(energy_lst)

# def calc_hydrogens_energy(inter, h_ener, s_ener, min_hydrogens=3, max_hydrogens=8, max_oxygens=1):
#     if 'H' not in inter.molecule.get_chemical_symbols(): 
#         mol_h_numb = 0
#     else:
#         mol_h_numb = sum(1 for atom in inter.molecule if atom.symbol == 'H')
        
#     if 'O' not in inter.molecule.get_chemical_symbols():
#         mol_h_numb += 1 * max_oxygens
#     else:
#         mol_h_numb += 1 * (max_oxygens - sum(1 for atom in inter.molecule if atom.symbol == 'O'))
#     h_numb = max_hydrogens - mol_h_numb 
#     s_numb =  mol_h_numb - min_hydrogens
#     return (h_ener * h_numb + s_ener * s_numb)

# def calc_ts_hydrogens(t_state, h_ener, s_ener, min_hydrogens=3, max_hydrogens=8, max_oxygens=1):
#     comps = t_state.bb_order()[0]
#     elem_info = [inter.molecule.get_chemical_symbols() for inter in comps if 'H' in inter.molecule.get_chemical_symbols()]
#     if 'H' not in elem_info[0]:
#         mol_h_numb = 0
#     else:
#         elem_info = sorted(elem_info, key=lambda x: x['H'], reverse=True)
#         elem_info = elem_info[0]
#         mol_h_numb = elem_info['H']
#     if 'O' not in elem_info:
#         mol_h_numb += 1 * max_oxygens
#     else:
#         mol_h_numb += 1 * (max_oxygens - elem_info['O'])
#     h_numb = max_hydrogens - mol_h_numb 
#     s_numb =  mol_h_numb - min_hydrogens
#     return (h_ener * h_numb + s_ener * s_numb)  

# def norm_ts_thickness(energy, boundaries=(0.5, 0.6, 0.7, 0.8, 0.9, 1.0), delta=1, offset=1, reverse=True):
#     for index, value in enumerate(boundaries):
#         if energy <= value:
#             index_val = index
#             break
#     else:
#         index_val = len(boundaries)
#     if reverse:
#         index_val = len(boundaries) - index_val
#     return_val = (index_val * delta) + offset
#     return return_val

# def norm_ts_thickness_rate(rate, boundaries=(0.5, 0.6, 0.7, 0.8, 0.9, 1.0), delta=1, offset=0.1, reverse=True):
#     for index, value in enumerate(boundaries):
#         #print(rate,value,index)
#         if rate >= value:
#             index_val = index
#             break
#     else:
#         index_val = len(boundaries)
#     if reverse:
#         index_val = len(boundaries) - index_val
#     return_val = (index_val * delta) + offset
#     return return_val

# def print_intermediates(network, filename='inter.dat'):
#     """Print a text file containing the information of the intermediates of a
#     network.

#     Args:
#         network (obj:`ReactionNetwork`): Network containing the intermediates.
#         filename (str, optional): Location where the file will be writed.
#             Defaults to 'inter.dat'
#     """
#     header = 'Label          iO    Formula\n'
#     inter_str = '{:6} {:.16f} {:.20}\n'
#     with open(filename, 'w') as outfile:
#         outfile.write(header)
#         for label, inter in network.intermediates.items():
#             formula = code_mol_graph(inter.graph)
#             ener = inter.energy
#             out_str = inter_str.format(label, ener, formula)
#             outfile.write(out_str)

# def print_intermediates_kinetics(network, filename='inter.dat'):
#     """Print a text file containing the information of the intermediates of a
#     network.

#     Args:
#         network (obj:`ReactionNetwork`): Network containing the intermediates.
#         filename (str, optional): Location where the file will be writed.
#             Defaults to 'inter.dat'
#     """
#     header = 'Label       iO       e   frq\n'
#     inter_str = 'i{:6} {: .8f} {:2d} {:.20}\n'
#     tmp_frq = '[0,0,0]'
#     with open(filename, 'w') as outfile:
#         outfile.write(header)
#         for label, inter in network.intermediates.items():
#             ener = inter.energy
#             electrons = inter.electrons
#             out_str = inter_str.format(label, ener, electrons, tmp_frq)
#             outfile.write(out_str)

# def print_t_states_kinetics(network, filename='ts.dat'):
#     """Print a text file containing the information of the transition states
#     of a network.

#     Args:
#         network (obj:`ReactionNetwork`): Network containing the transition
#             states.
#         filename (str, optional): Location where the file will be writed.
#             Defaults to 'ts.dat'
#     """
#     header = '          Label                is1     is2     fs1     fs2        iO      e alpha beta   frq\n'
#     tmp_frq = '[0,0,0]'
#     inter_str = '{:28} i{:6} i{:6} i{:6} i{:6} {: .8f} {: 2d} {:3.2f} {:2.2f} {:.20}\n'
#     with open(filename, 'w') as outfile:
#         outfile.write(header)
#         for t_state in network.t_states:
#             label = t_state.code
#             initial = [inter.code for inter in list(t_state.components[0])]
#             final = [inter.code for inter in list(t_state.components[1])]

#             electrons_fin = sum([inter.electrons for inter in
#                                  list(t_state.components[1])])
#             electrons_in = sum([inter.electrons for inter in
#                                 list(t_state.components[0])])
#             if len(initial) == 1:
#                 initial *= 2
#                 electrons_in *= 2
#             if len(final) == 1:
#                 final *= 2
#                 electrons_fin *= 2
#             order = []
#             for item in [initial, final]:
#                 mols = [network.intermediates[item[0]],
#                         network.intermediates[item[1]]]
#                 if mols[0].is_surface:
#                     order.append(item[::-1])
#                 elif not mols[1].is_surface and (len(mols[0].molecule) <
#                                                  len(mols[1].molecule)):
#                     order.append(item[::-1])
#                 else:
#                     order.append(item)

#             initial, final = order
#             electrons = electrons_fin - electrons_in
#             ener = t_state.energy
#             alpha = INTERPOL[t_state.r_type]['alpha']
#             beta = INTERPOL[t_state.r_type]['beta']
#             label = ''
#             for item in initial + final:
#                 label += 'i' + item
#             outfile.write(inter_str.format(label, *initial, *final,
#                                            ener, electrons, alpha,
#                                            beta, tmp_frq))

# def print_t_states(network, filename='ts.dat'):
#     """Print a text file containing the information of the transition states
#     of a network.

#     Args:
#         network (obj:`ReactionNetwork`): Network containing the transition
#             states.
#         filename (str, optional): Location where the file will be writed.
#             Defaults to 'ts.dat'
#     """
#     header = '          Label               is1    is2    fs1    fs2           iO\n'
#     inter_str = '{:20} {:6} {:6} {:6} {:6} {:.16f}\n'
#     with open(filename, 'w') as outfile:
#         outfile.write(header)
#         for t_state in network.t_states:
#             label = t_state.code
#             initial = [inter.code for inter in list(t_state.components[0])]
#             final = [inter.code for inter in list(t_state.components[1])]
#             if len(initial) == 1:
#                 initial *= 2
#             if len(final) == 1:
#                 final *= 2
#             ener = t_state.energy
#             entr = t_state.entropy
#             # if ener == None:
#             #     ener = 0.0
#             # if entr == None:
#             #     entr = 0.0
#             # print(label,*initial,*final,ener, entr)
#             outfile.write(inter_str.format(label, *initial, *final, ener, entr))

# def print_gasses_kinetics(gas_dict, filename='gas.dat'):
#     """Print a text file containing the information of the intermediates of a
#     network.

#     Args:
#         network (obj:`ReactionNetwork`): Network containing the intermediates.
#         filename (str, optional): Location where the file will be writed.
#             Defaults to 'inter.dat'
#     """
#     header = 'Label    Formula              ads         gas      iO         e   mw   frq\n'
#     inter_str = '{:6} {:20} i{:7} {: .8f} {: .8f} {:4.2f} {:2d} {:.20}\n'
#     tmp_frq = '[0,0,0]'
#     with open(filename, 'w') as outfile:
#         outfile.write(header)
#         for label, gas in gas_dict.items():
#             ener = gas['energy']
#             electrons = gas['electrons']
#             try:
#                 gas['mol'].arrays['conn_pairs'] = get_voronoi_neighbourlist(gas['mol'], 0.25, 1.0, ['C', 'H', 'O'])
#                 formula = code_mol_graph(ase_coord_2_graph(gas['mol'], coords=False), ['C'])
#             except nx.NetworkXNoPath:
#                 formula = gas['mol'].get_chemical_formula()
#             if not formula:
#                 formula = gas['mol'].get_chemical_formula()
#             weight = calculate_weigth(dict(Counter(gas['mol'].get_chemical_symbols())))
#             out_str = inter_str.format(label, formula, label, ener,
#                                        ener, weight, electrons, tmp_frq)
#             outfile.write(out_str)

# def read_TS_energies(filename):
#     """Reads an energy filename and convert it to a dictionary.

#     Args:
#         filename (str): Location of the file containing the energies.

#     Returns:
#         Two dictionaris in which keys are the code values of the
#         intermediates and the associated values are the energy of the
#         intermediates.
#         The first dictionary contains the values that do not contain failed,
#         warning or WARNING at the end of their code. Moreover, if multiple
#         codes with the same 6 characters start are found, only the one with
#         the lesser energy is selected.
#     """
#     ener_dict = {}
#     discard = {}
#     repeated = {}
#     entropy_dict = {}

#     with open(filename, 'r') as infile:
#         lines = infile.readlines()

#     for sg_line in lines:
#         code_array=[]
#         try:
#             code, energy, entropy = sg_line.split(' ')
#             for i in code.split('-'):
#                 for j in i.split('+'):
#                     code_array.append(j[1:])
#             code_array.sort()
#             #print(code_array)
#             code = ""
#             # traverse in the string 
#             for j in code_array:
#                 code += j 
#             # print(code)
#         except ValueError:
#             discard[code[:6]] = 0
#         energy = float(energy)  # Energy needs to be a float
#         entropy = float(entropy)
        
#         #if code[0] == 'i':
#         #    code = code[1:]

#         #if len(code) > 6:
#         #    continue
#             # init_code = code[:6]
#             # if code.endswith(('WARNING', 'warning', 'failed')):
#             #     discard[init_code] = energy
#             #     continue
#             # if init_code not in repeated:
#             #     repeated[init_code] = []
#             # repeated[init_code].append(energy)
#             # continue
#         ener_dict[code] = energy
#         entropy_dict[code] = entropy
# #    for code, energies in repeated.items():
# #        ener_dict[code] = min(energies)

#     return ener_dict, entropy_dict, discard

# def read_energies(filename):
#     """Reads an energy filename and convert it to a dictionary.

#     Args:
#         filename (str): Location of the file containing the energies.

#     Returns:
#         Two dictionaris in which keys are the code values of the
#         intermediates and the associated values are the energy of the
#         intermediates.
#         The first dictionary contains the values that do not contain failed,
#         warning or WARNING at the end of their code. Moreover, if multiple
#         codes with the same 6 characters start are found, only the one with
#         the lesser energy is selected.
#     """
#     ener_dict = {}
#     discard = {}
#     repeated = {}
#     entropy_dict = {}

#     with open(filename, 'r') as infile:
#         lines = infile.readlines()

#     for sg_line in lines:
#         try:
#             code, energy, entropy = sg_line.split()
#         except ValueError:
#             discard[code[:6]] = 0
#         energy = float(energy)  # Energy needs to be a float
#         entropy = float(entropy)
        
#         if code[0] == 'i':
#             code = code[1:]

#         if len(code) > 6:
#             continue
#             # init_code = code[:6]
#             # if code.endswith(('WARNING', 'warning', 'failed')):
#             #     discard[init_code] = energy
#             #     continue
#             # if init_code not in repeated:
#             #     repeated[init_code] = []
#             # repeated[init_code].append(energy)
#             # continue
#         ener_dict[code] = energy
#         entropy_dict[code] = entropy
# #    for code, energies in repeated.items():
# #        ener_dict[code] = min(energies)

#     return ener_dict, entropy_dict, discard

# def adjust_electrons(ase_atoms_obj: Atoms):
#     """Given a dict with elements, calculate the reference energy for
#     the compound.

#     Args:
#         elements_dict (dict): C, O, H as key and the number of atoms for every
#             element as value.
#     """
#     elements_dict = {'C': ase_atoms_obj.get_chemical_symbols().count('C'),
#                 'H': ase_atoms_obj.get_chemical_symbols().count('H'),
#                 'O': ase_atoms_obj.get_chemical_symbols().count('O')}

#     n_electrons = (4 * elements_dict['C'] + elements_dict['H']
#                  - 2 * elements_dict['O'])
#     return n_electrons

# def adjust_electrons_H(molecule: Atoms):
#     elements = dict(Counter(molecule.get_chemical_symbols()))
#     pivot_dict = elements.copy()
#     if 'H' not in pivot_dict:
#         pivot_dict['H'] = 0

#     electrons = pivot_dict['H'] # + search_alcoxy(molecule)
#     return electrons

# def select_larger_inter(inter_lst):
#     """Given a list of Intermediates, select the intermediate with the
#     highest number of atoms.

#     Args:
#         inter_lst(list of obj:`networks.Intermediate`): List that will be
#             evaluated.
#     Returns:
#         obj:`networks.Intermediate` with the bigger molecule.
#     """
#     atoms = [len(inter.mol) for inter in inter_lst]
#     inter_max = [0, 0]
#     for size, inter in zip(atoms, inter_lst):
#         if size > inter_max[0]:
#             inter_max[0] = size
#             inter_max[1] = inter
#     return inter_max[1]

# #TODO
# def search_electro_ts(network, electron='e-', proton='H', water='H2O', ener_up=0.05):
#     """Search for all the possible electronic transition states of a network.

#     Args:
#         network (obj:`networks.ReactionNetwork`): Network in wich the electronic
#             states will be generated.
#         electron (any object): Object that represents an electron.
#         proton (any object): Object that represents a proton.
#         water (any object): Object that represents a water molecule.
#     """
#     cate = iso.categorical_node_match(['elem', 'elem'], ['H', 'O'])
#     electro_ts = []
#     for inter in network.intermediates.values():
#         if 'O' not in dict(Counter(inter.molecule.get_chemical_symbols())):
#             continue
#         tmp_oxy = inter.molecule['O']
#         for index, _ in enumerate(tmp_oxy):
#             tmp_mol = inter.molecule.copy()
#             oxygen = [atom for atom in tmp_mol if atom.symbol == "O"][index]

#             hydrogen = [conect for conect in oxygen.connections if
#                         conect.symbol == 'H']
#             if not hydrogen:
#                 continue

#             del tmp_mol[hydrogen[0]]
#             del tmp_mol.arrays['conn_pairs']
#             tmp_mol.arrays['conn_pairs'] = get_voronoi_neighbourlist(tmp_mol, 0.25, 1.0, ['C', 'H', 'O'])
#             tmp_graph = ase_coord_2_graph(tmp_mol, coords=False)


#             candidates = network.search_graph(tmp_graph, cate=cate)
#             for new_inter in candidates:
#                 oh_comp = [[new_inter, proton], [inter, electron]]
#                 oh_ts = ElementaryReaction(components=oh_comp,
#                                             r_type='O-H', is_electro=True)
#                 inter_energy = inter.energy + electron.energy
#                 new_inter_energy = new_inter.energy + proton.energy
#                 inter_energy_bader = inter.bader_energy + electron.energy
#                 new_inter_energy_bader = new_inter.bader_energy + proton.energy
#                 oh_ts.energy = max((inter_energy, new_inter_energy)) + ener_up
#                 oh_ts.bader_energy = max((inter_energy_bader, new_inter_energy_bader)) + ener_up
#                 electro_ts.append(oh_ts)

#             del tmp_mol[oxygen]
#             del tmp_mol.arrays['conn_pairs']
#             try:
#                 tmp_mol.arrays['conn_pairs'] = get_voronoi_neighbourlist(tmp_mol, 0.25, 1.0, ['C', 'H', 'O'])
#             except ValueError:
#                 continue
#             tmp_graph = ase_coord_2_graph(tmp_mol, coords=False)

            
#             candidates = network.search_graph(tmp_graph, cate=cate)
#             for new_inter in candidates:
#                 wa_comp = [[inter, proton], [new_inter, water]]            
#                 wa_ts = ElementaryReaction(components=wa_comp,
#                         r_type='C-OH', is_electro=True)
#                 inter_energy = inter.energy + electron.energy
#                 new_inter_energy = new_inter.energy + proton.energy
#                 inter_energy_bader = inter.bader_energy + electron.energy
#                 new_inter_energy_bader = new_inter.bader_energy + proton.energy
#                 wa_ts.energy = max((inter_energy, new_inter_energy)) + ener_up
#                 wa_ts.bader_energy = max((inter_energy_bader, new_inter_energy_bader)) + ener_up
#                 electro_ts.append(wa_ts)

#     return electro_ts

# def print_electro_kinetics(ts_lst, filename='elec.dat'):
#     """Print a text file containing the information of the transition states
#     of a network.

#     Args:
#         network (obj:`ReactionNetwork`): Network containing the transition
#             states.
#         filename (str, optional): Location where the file will be writed.
#             Defaults to 'ts.dat'
#     """
#     header = '              Label                is1     is2     fs1     fs2        iO      e    frq\n'
#     tmp_frq = '[0,0,0]'
#     inter_str = '{:32} {:7} {:7} {:7} {:7} {: .8f} {: 2d} {:.20}\n'
#     with open(filename, 'w') as outfile:
#         outfile.write(header)
#         for t_state in ts_lst:
#             ordered = t_state.components
#             initial = [inter.code for inter in list(ordered[0])]
#             final = [inter.code for inter in list(ordered[1])]

#             electrons_fin = sum([inter.electrons for inter in
#                                  list(ordered[1])])
#             electrons_in = sum([inter.electrons for inter in
#                                 list(ordered[0])])

#             for item in [initial, final]:
#                 for index, _ in enumerate(item):
#                     if not item[index]:
#                         item[index] = 'None'
#                     elif item[index][0] != 'g':
#                         item[index] = 'i' + item[index]

#             if len(initial) == 1:
#                 initial *= 2
#             if len(final) == 1:
#                 final *= 2
#             order = []
#             for item in [initial, final]:
#                 if item[0] == 'None' or item[0].startswith('g'):
#                     order.append(item[::-1])
#                 else:
#                     order.append(item)
#             initial, final = order
#             label = ''
#             for item in initial + final:
#                 if item == 'None':
#                     label += 'g000000'
#                 else:
#                     label += item
#             electrons = electrons_fin - electrons_in
#             ener = t_state.energy
#             outfile.write(inter_str.format(label, *initial, *final,
#                                            ener, electrons, tmp_frq))

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
