from ase import Atoms
from networkx import Graph
from GAMERNet.rnet.utilities import functions as fn
from GAMERNet.rnet.graphs.graph_fn import ase_coord_2_graph
# from GAMERNet.rnet.networks.utils import get_smiles, ase_to_rdkit
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds

class Intermediate:
    """Intermediate class that defines the intermediate species of the network.

    Attributes:
        code (str): Code of the intermediate.
        molecule (obj:`ase.atoms.Atoms`): Associated molecule.
        graph (obj:`nx.graph`): Associated molecule graph.
        energy (float): DFT energy of the intermediate.
        entropy (float): Entropy of the intermediate
        formula (str): Formula of the intermediate.
    """
    def __init__(self, 
                 code: str=None, 
                 molecule: Atoms=None,
                 adsorbate: Atoms=None, 
                 graph: Graph=None, 
                 energy: float=None,
                 std_energy: float=None, 
                 entropy: float=None,
                 formula: str=None, 
                 electrons: int=None,
                 is_surface: bool=False,
                 phase: str=None):
        
        self.code = code
        self.molecule = molecule
        self.adsorbate = adsorbate
        self._graph = graph
        self.energy = energy
        self.std_energy = std_energy
        self.entropy = entropy
        self.formula = formula
        self.electrons = electrons
        self.is_surface = is_surface
        self.bader = None
        self.voltage = None
        self.smiles = self.get_smiles()
        
        if self.is_surface:
            self.phase = 'surface'
            self.closed_shell = None
        else:
            try:
                self.closed_shell = self.is_closed_shell()
            except:
                self.closed_shell = None
            self.phase = phase
        self.t_states = [{}, {}]

    def __hash__(self):
        return hash(self.code)

    def __eq__(self, other):
        if isinstance(other, str):
            return self.code == other
        if isinstance(other, Intermediate):
            return self.code == other.code
        raise NotImplementedError

    def __repr__(self):
        string = (self.code + '({})'.format(self.molecule.get_chemical_formula()))
        return string

    # TODO: Use Santi function to generate the code
    # def draft(self):
    #     """Draft of the intermediate generated using the associated graph.

    #     Returns:
    #         obj:`matplotlib.pyplot.Figure` with the image of the draft.
    #     """
    #     color_map, node_size = [], []
    #     for node in self.graph.nodes():
    #         color_map.append(RGB_COLORS[node.element])
    #         node_size.append(CORDERO[node.element] * 5000)
    #     return nx.draw(self.graph, node_color=color_map,
    #                    node_size=node_size, width=15)

    @property
    def bader_energy(self):
        if self.bader is None or self.voltage is None:
            return self.energy
        # -1 is the electron charge
        return self.energy - ((self.bader + self.electrons) * (-1.) * self.voltage)

    @classmethod
    def from_molecule(cls, ase_atoms_obj, code=None, energy=None, std_energy=None,entropy=None, is_surface=False, phase=None):
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
        new_formula = new_mol.get_chemical_formula()
        return cls(code=code, molecule=new_mol, graph=new_graph,
                        formula=new_formula, energy=energy, std_energy = std_energy, entropy=entropy,
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
        Check if a molecule is closed-shell or not.
        IN PROGRESS
        """

        symbols = self.molecule.get_chemical_symbols()
        coords = self.molecule.get_positions()
        for i in range(len(coords)):  # needed for RDKit to read properly the coordinates
            for j in range(len(coords[i])):
                if abs(coords[i][j]) < 1.0e-6:
                    coords[i][j] = 0.0
        xyz = '\n'.join(f'{symbol} {x} {y} {z}' for symbol, (x, y, z) in zip(symbols, coords))
        xyz = "{}\n\n{}".format(len(self.molecule), xyz)
        rdkit_mol = Chem.MolFromXYZBlock(xyz)
        conn_mol = Chem.Mol(rdkit_mol)
        rdDetermineBonds.DetermineConnectivity(conn_mol)
        Chem.SanitizeMol(conn_mol, Chem.SANITIZE_SETHYBRIDIZATION)
        #TODO: Improve function as it is not working properly
        # unpaired_electrons = 0
        # for atom in conn_mol.GetAtoms():
        #     unpaired_electrons += atom.GetNumRadicalElectrons()

        # if unpaired_electrons == 0:
        #     return True
        # else:
        #     return False
        num_electrons = sum([Chem.GetPeriodicTable().GetNOuterElecs(atom.GetAtomicNum()) for atom in conn_mol.GetAtoms()])
        is_open_shell = num_electrons % 2 == 1
        return not is_open_shell
    
    def get_smiles(self):
        """
        Get the SMILES string of a molecule.
        """
        symbols = self.molecule.get_chemical_symbols()
        coords = self.molecule.get_positions()
        for i in range(len(coords)):  # needed for RDKit to read properly the coordinates
            for j in range(len(coords[i])):
                if abs(coords[i][j]) < 1.0e-6:
                    coords[i][j] = 0.0
        xyz = '\n'.join(f'{symbol} {x} {y} {z}' for symbol, (x, y, z) in zip(symbols, coords))
        xyz = "{}\n\n{}".format(len(self.molecule), xyz)
        rdkit_mol = Chem.MolFromXYZBlock(xyz)
        conn_mol = Chem.Mol(rdkit_mol)
        rdDetermineBonds.DetermineConnectivity(conn_mol)
        Chem.SanitizeMol(conn_mol, Chem.SANITIZE_SETHYBRIDIZATION)
        smiles = Chem.MolToSmiles(conn_mol)
        return smiles