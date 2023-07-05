from collections import namedtuple
from pyRDTP import geomio, vaspio
from pyRDTP.molecule import CatalyticSystem
import networkx as nx
from itertools import combinations, product
import networkx.algorithms.isomorphism as iso

MolPack = namedtuple('MolPack',['code', 'mol','graph', 'subs'])

def digraph(mol):
    graph_now = nx.DiGraph()
    for atom in mol:
        graph_now.add_node(atom, elem=atom.element)
        for connection in atom.connections:
            graph_now.add_edge(atom, connection)
    return graph_now

def get_all_subs(molecule, n_sub, element): #function to what point you want to remove hydrogens
    sel_atoms = molecule.atom_element_filter(element)
    list(combinations(sel_atoms, n_sub))
    mol_pack = []
    graph_pack = []
    for comb in combinations(sel_atoms, n_sub):
        new_mol = molecule.copy()
        int_list = []
        for atom in comb:
            int_list.append(atom.index)
        new_mol.atom_remove_list(int_list)
        new_mol.connectivity_search_voronoi(tolerance=0.25)
        mol_pack.append(new_mol)
        graph_pack.append(digraph(new_mol))
    return mol_pack, graph_pack

def get_unique(mol_pack, graph_pack, element): #function to get unqiue configs
    accepted = []

    em = iso.categorical_node_match('elem', element)
    for index, graph1 in enumerate(graph_pack):
        for graph2 in accepted:
            if nx.is_isomorphic(graph1, graph_pack[graph2], node_match=em):
                break
        else:
            accepted.append(index)
    return accepted

def code_name(molecule, group, index): #name of the molecule
    name_elem = ('C', 'H', 'O')
    name_str = ''
    infor = molecule.elem_inf()
    for item in name_elem:
        try:
            name_str += '{:01x}'.format(infor[item])
        except KeyError:
            name_str += str('0')
    name_str += '{:01x}'.format(group)
    name_str += '{:02x}'.format(index)
    return name_str

def decode(code): #storing of data?
    decoded = {'C': int(code[0], 16),
               'H': int(code[1], 16),
               'O': int(code[2], 16),
               'grp': int(code[3], 16),
               'iso': int(code[4:6], 16)}
    return decoded

def generate_range(molecule, element, subs, group): #generating all the possibilities
    mg_pack = {0: [MolPack(code_name(molecule, group, 1), molecule, digraph(molecule), {})]}
    for index in range(subs):
        tmp_pack = []
        pack = get_all_subs(molecule, index + 1, element)
        unique = get_unique(*pack, element)
        for i_code, struct in enumerate(unique):
            mol_sel = pack[0][struct]
            tmp_pack.append(MolPack(code_name(mol_sel, group, i_code+1), mol_sel, pack[1][struct], {'H': index+1}))
        mg_pack[index+1] = tmp_pack
    return mg_pack

def search_code(mg_pack, code):
    for _, pack in mg_pack.items():
         for obj in pack:
            if str(obj.code) == str(code):
                return obj
    else:
        return False

def generate_map(packing, elem):
    g = nx.DiGraph()
    em = iso.categorical_node_match('elem', elem)
    for index in range(len(packing) - 1):
        for father, son in product(packing[index], packing[index+1]):
            xx = nx.isomorphism.DiGraphMatcher(father.graph, son.graph, node_match=em)
            if xx.subgraph_is_isomorphic():
                g.add_edge(father.code, son.code)
    return g
    

def draw_dot(graph, filename):
    p=nx.drawing.nx_pydot.to_pydot(graph)
    p.write_png(filename)

def print_pack(pack, surface, poscar='poscar', distance=None, atom=None, at_lst=None, point=None, origin=None, vector=None):
    for _, item in pack.items():
        for obj in item:
            caty = CatalyticSystem('x')
            caty.surface_set(surface)
            caty.molecule_add(obj.mol)
            if distance is not None:
                if origin is None:
                    lowest = obj.mol.atom_obtain_lowest().coords
                else:
                    lowest = origin 
                if atom is not None:
                    caty.move_over_atom(obj.mol, atom, distance, origin=lowest)
                elif at_lst is not None:
                    caty.move_over_multiple_atoms(obj.mol, at_lst, distance, origin=lowest)
                elif point is not None:
                    obj.mol.move_to(point, origin='centroid')
                    obj.mol.move_vector([0., 0., distance])
                else:
                    caty.move_over_surface_center(obj.mol, distance=distance, origin=lowest)
            if vector is not None:
                caty.molecules[0].move_vector(vector)  
                        
            vaspio.print_vasp(caty, './{}/POSCAR.{}'.format(poscar, obj.code)) # storing the poscars

def save_poscars(pack,folder): #write a function to save gas phase POSCARs
    for _, item in pack.items():
        for obj in item:
            geomio.mol_to_file(obj.mol,'./{}/POSCAR_{}'.format(folder,obj.code),'contcar')

def generate_pack(molecule, niter, group):
    molecule.connectivity_search_voronoi()
    molecule.atom_index()
    return generate_range(molecule, 'H', niter, group)

