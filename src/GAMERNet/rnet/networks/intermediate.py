from ase import Atoms
from networkx import Graph, draw
from GAMERNet.rnet.utilities import functions as fn
from GAMERNet.rnet.graphs.graph_fn import ase_coord_2_graph
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds

PHASES = ['gas', 'ads', 'surf']
ELEMENTS = ['C', 'H', 'O', '*']

class Intermediate:
    """Intermediate class that defines the intermediate species of the network.

    Attributes:
        code (str): Code of the intermediate. 6 digits.
        molecule (obj:`ase.atoms.Atoms`): Associated molecule.
        graph (obj:`nx.graph`): Associated molecule graph.
        energy (float): DFT energy of the intermediate.
        entropy (float): Entropy of the intermediate
        formula (str): Formula of the intermediate.
    """

    phases = PHASES
    elements = ELEMENTS
    
    def __init__(self, 
                 code: str=None, 
                 molecule: Atoms=None,
                 graph: Graph=None, 
                 ads_configs: dict[str, dict]={},
                 is_surface: bool=False,
                 phase: str=None):
        
        self.code = code
        self.molecule = molecule
        self.is_surface = is_surface
        if not all(elem in ELEMENTS for elem in self.molecule.get_chemical_symbols()) and not self.is_surface:
            raise ValueError(f"Molecule {self.molecule} contains elements other than C, H, O")
        self.formula = molecule.get_chemical_formula()
        self._graph = graph
        self.ads_configs = ads_configs
        self.electrons = self.get_num_electrons()
        self.bader = None
        self.voltage = None
        self.smiles = self.get_smiles()
        
        if self.is_surface:
            self.phase = 'surf'
            self.closed_shell = None
        else:
            self.closed_shell = self.is_closed_shell()
            if phase not in PHASES:
                raise ValueError(f"Phase must be one of {PHASES}")
            self.phase = phase
        self.t_states = [{}, {}]

    def __getitem__(self, key: str):
        if key not in self.elements:
            raise ValueError(f"Element {key} not in {self.elements}")
        if key == '*':
            if self.phase == 'surf':
                return 1
            else:
                return 0
        return self.molecule.get_chemical_symbols().count(key)

    def __hash__(self):
        return hash(self.code)

    def __eq__(self, other):
        if isinstance(other, str):
            return self.code == other
        if isinstance(other, Intermediate):
            return self.code == other.code
        raise NotImplementedError

    def __repr__(self):        
        if self.phase in ('surf', 'ads'):
            txt = self.code + '({}*)'.format(self.formula)
        else:
            txt = self.code + '({}(g))'.format(self.formula)
        return txt
    
    def __str__(self):
        return self.__repr__()

    @property
    def bader_energy(self):
        if self.bader is None or self.voltage is None:
            return self.energy
        # -1 is the electron charge
        return self.energy - ((self.bader + self.electrons) * (-1.) * self.voltage)

    @classmethod
    def from_molecule(cls, ase_atoms_obj: Atoms, code=None, energy=None, std_energy=None,entropy=None, is_surface=False, phase=None):
        """Create an Intermediate using a molecule obj.

        Args:
            ase_atoms_obj (obj:`ase.atoms.Atoms`): ase.Atoms object from which the
                intermediate will be created.
            code (str, optional): Code of the intermediate. Defaults to None.
            energy (float, optional): Energy of the intermediate. Defaults to
                None.
            is_surface (bool, optional): Defines if the intermediate is the
                surface.

        Returns:
            obj:`Intermediate` with the given values.
        """
        new_mol = ase_atoms_obj.copy()
        new_mol.arrays['conn_pairs'] = fn.get_voronoi_neighbourlist(new_mol, 0.25, 1, ['C', 'H', 'O'])
        new_graph = ase_coord_2_graph(new_mol, coords=False)
        return cls(code=code, molecule=new_mol, graph=new_graph,
                        is_surface=is_surface, phase=phase)    

    @property
    def graph(self):
        if self._graph is None:
            self._graph = self.gen_graph()
        return self._graph

    @graph.setter
    def graph(self, other):
        self._graph = other

    def gen_graph(self):
        """Generate a graph of the molecule.

        Returns:
            obj:`nx.DiGraph` Of the associated molecule.
        """
        return fn.digraph(self.molecule, coords=False)
    
    def is_closed_shell(self):
        """
        Check if a molecule CxHyOz is closed-shell or not.
        """
        graph = self.graph
        molecule = self.molecule
        valence_electrons = {'C': 4, 'H': 1, 'O': 2}
        graph = graph.to_undirected()
        mol_composition = molecule.get_chemical_symbols()
        mol = {'C': mol_composition.count('C'), 'H': mol_composition.count('H'), 'O': mol_composition.count('O')} # CxHyOz

        if mol['C'] != 0 and mol['H'] == 0 and mol['O'] == 0: # Cx
                return False
        elif mol['C'] == 0 and mol['H'] != 0 and mol['O'] == 0: # Hy
                return True if mol['H'] == 2 else False
        elif mol['C'] == 0 and mol['H'] == 0 and mol['O'] != 0: # Oz
                return True if mol['O'] == 2 else False
        elif mol['C'] != 0 and mol['H'] == 0 and mol['O'] != 0: # CxOz
                return True if mol['C'] == 1 and mol['O'] in (1,2) else False
        elif mol['C'] == 0 and mol['H'] != 0 and mol['O'] != 0: # HyOz
                return True if mol['H'] == 2 and mol['O'] in (1,2) else False
        elif mol['C'] != 0 and mol['H'] != 0: # CxHyOz (z can be zero)
            node_val = lambda graph: {node: [graph.degree(node), 
                                        valence_electrons.get(graph.nodes[node]["elem"], 0)] for node in graph.nodes()}
            num_unsaturated_nodes = lambda dict: len([node for node in dict.keys() if dict[node][0] < dict[node][1]])
            node_valence_dict = node_val(graph)
            if num_unsaturated_nodes(node_valence_dict) == 0: # all atoms are saturated
                return True
            elif num_unsaturated_nodes(node_valence_dict) == 1: # only one unsaturated atom
                return False
            else:  # more than one unsaturated atom
                saturation_condition = lambda dict: all(dict[node][0] == dict[node][1] for node in dict.keys())
                while not saturation_condition(node_valence_dict):
                    unsat_nodes = [node for node in node_valence_dict.keys() if node_valence_dict[node][0] < node_valence_dict[node][1]]
                    O_unsat_nodes = [node for node in unsat_nodes if graph.nodes[node]["elem"] == 'O']  # all oxygens unsaturated
                    if len(O_unsat_nodes) != 0: # unsaturated oxygen atoms
                        for oxygen in O_unsat_nodes:
                            node_valence_dict[oxygen][0] += 1
                            # increase the valence of the oxygen neighbour by 1
                            for neighbour in graph.neighbors(oxygen): # only one neighbour
                                if node_valence_dict[neighbour][0] < node_valence_dict[neighbour][1]:
                                    node_valence_dict[neighbour][0] += 1
                                else:
                                    return False # O neighbour is saturated already
                    else: # CxHy
                         # select node with the highest degree
                        max_degree = max([node_valence_dict[node][0] for node in unsat_nodes])
                        max_degree_node = [node for node in unsat_nodes if node_valence_dict[node][0] == max_degree][0]
                        max_degree_node_unsat_neighbours = [neighbour for neighbour in graph.neighbors(max_degree_node) if neighbour in unsat_nodes]
                        if len(max_degree_node_unsat_neighbours) == 0: # all neighbours are saturated
                            return False
                        else:
                            node_valence_dict[max_degree_node][0] += 1
                            node_valence_dict[max_degree_node_unsat_neighbours[0]][0] += 1
                return True                
    
    def get_smiles(self):
        """
        Get the SMILES string of a molecule.
        """
        symbols = self.molecule.get_chemical_symbols()
        coords = self.molecule.get_positions()
        for i in range(len(coords)):  # needed for RDKit to read properly the coordinates
            for j in range(len(coords[i])):
                if abs(coords[i][j]) < 1.0e-3:
                    coords[i][j] = 0.0
        xyz = '\n'.join(f'{symbol} {x} {y} {z}' for symbol, (x, y, z) in zip(symbols, coords))
        xyz = "{}\n\n{}".format(len(self.molecule), xyz)
        rdkit_mol = Chem.MolFromXYZBlock(xyz)
        conn_mol = Chem.Mol(rdkit_mol)
        rdDetermineBonds.DetermineConnectivity(conn_mol)
        Chem.SanitizeMol(conn_mol, Chem.SANITIZE_SETHYBRIDIZATION)
        smiles = Chem.MolToSmiles(conn_mol)
        return smiles
    
    def get_num_electrons(self):
        """
        Get the number of electrons of the intermediate.
        """
        return 4 * self.molecule.get_chemical_symbols().count('C') + self.molecule.get_chemical_symbols().count('H') - 2 * self.molecule.get_chemical_symbols().count('O')