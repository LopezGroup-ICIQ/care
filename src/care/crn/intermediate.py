from typing import Union
from io import StringIO

from ase import Atom, Atoms
from ase.io import write
from networkx import Graph, cycle_basis
from rdkit import Chem
from rdkit.Chem import AllChem

from care.constants import INTER_ELEMS, INTER_PHASES
from care.crn.utilities.species import (atoms_to_graph,
                                        get_voronoi_neighbourlist, 
                                        get_fragment_energy)


class Intermediate:
    """Intermediate class.

    Attributes:
        code (str): Code of the intermediate. InChiKey of the molecule.
        molecule (Union[obj:`ase.Atoms`, obj:`rdkit.Chem.rdchem.Mol`]): Associated molecule.
        graph (obj:`nx.graph`): Associated molecule graph.
        ads_configs (dict): Adsorption configurations of the intermediate.
        is_surface (bool): Defines if the intermediate corresponds to the empty surface.
        phase (str): Phase of the intermediate.
    """

    def __init__(
        self,
        code: str = None,
        molecule: Union[Atoms, Chem.rdchem.Mol] = None,
        graph: Graph = None,
        ads_configs: dict[str, dict] = {},
        is_surface: bool = False,
        phase: str = None,
    ):
        self.phases = INTER_PHASES
        self.elements = INTER_ELEMS
        self.code = code
        self.molecule = molecule

        if isinstance(self.molecule, Chem.rdchem.Mol):
            self.rdkit = molecule
        # If the molecule is an ASE Atoms object and its not empty, convert it to an RDKit molecule, else set the RDKit molecule to None
        elif isinstance(self.molecule, Atoms) and len(self.molecule) != 0:
            self.rdkit = self.ase_to_rdkit()
        else:
            self.rdkit = None

        if isinstance(self.molecule, Chem.Mol):
            self.molecule = self.rdkit_to_ase()

        self.is_surface = is_surface
        if (
            not all(
                elem in self.elements for elem in self.molecule.get_chemical_symbols()
            )
            and not self.is_surface
        ):
            raise ValueError(
                f"Molecule {self.molecule} contains elements other than {self.elements}"
            )
        self.formula = (
            self.molecule.get_chemical_formula() if not self.is_surface else "surface"
        )
        self._graph = graph
        self.ads_configs = ads_configs
        self.electrons = self.get_num_electrons()
        self.charge = 0
        self.mass = self.molecule.get_masses().sum()

        if not self.is_surface and len(self.molecule) != 0:
            self.smiles = self.get_smiles()
            self.cyclic = self.is_cyclic()
        else:
            self.smiles = None
            self.cyclic = None

        if self.is_surface:
            self.phase = "surf"
            self.closed_shell = None
        else:
            self.closed_shell = self.is_closed_shell()
            if phase not in self.phases:
                raise ValueError(f"Phase must be one of {self.phases}")
            self.phase = phase

    def __getitem__(self, key: str):
        if key not in self.elements:
            raise ValueError(f"Element {key} not in {self.elements}")
        if key == "*":
            if self.phase in ("surf", "ads"):
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
        if self.phase in ("surf", "ads"):
            txt = self.code + "({}*)".format(self.formula)
        elif self.phase == "solv":
            txt = self.code + "({}(aq))".format(self.formula)
        else:
            txt = self.code + "({}(g))".format(self.formula)
        return txt

    def __str__(self):
        return self.__repr__()

    @classmethod
    def from_molecule(
        cls,
        ase_atoms_obj: Atoms,
        code=None,
        energy=None,
        std_energy=None,
        entropy=None,
        is_surface=False,
        phase=None,
    ):
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
        if len(ase_atoms_obj) != 0:
            new_mol = ase_atoms_obj.copy()
            new_mol.arrays["conn_pairs"] = get_voronoi_neighbourlist(
                new_mol, 0.25, 1, ["C", "H", "O"]
            )
            new_graph = atoms_to_graph(new_mol, coords=False)
            return cls(
                code=code,
                molecule=new_mol,
                graph=new_graph,
                is_surface=is_surface,
                phase=phase,
            )
        else:
            return cls(
                code=code, molecule=ase_atoms_obj, is_surface=is_surface, phase=phase
            )

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
        return atoms_to_graph(self.molecule, coords=False)

    def is_cyclic(self):
        """
        Check if a molecule is cyclic or not.
        """
        cycles = list(cycle_basis(self.graph))
        return True if len(cycles) != 0 else False

    def is_closed_shell(self):
        """
        Check if a molecule is closed-shell or not.
        """
        graph = self.graph
        molecule = self.molecule
        valence_electrons = {"C": 4, "H": 1, "O": 2}
        mol_composition = molecule.get_chemical_symbols()
        mol = {
            "C": mol_composition.count("C"),
            "H": mol_composition.count("H"),
            "O": mol_composition.count("O"),
        }  # CxHyOz

        if mol["C"] != 0 and mol["H"] == 0 and mol["O"] == 0:  # Cx
            return False
        elif mol["C"] == 0 and mol["H"] != 0 and mol["O"] == 0:  # Hy
            return True if mol["H"] == 2 else False
        elif mol["C"] == 0 and mol["H"] == 0 and mol["O"] != 0:  # Oz
            return True if mol["O"] == 2 else False
        elif mol["C"] != 0 and mol["H"] == 0 and mol["O"] != 0:  # CxOz
            return True if mol["C"] == 1 and mol["O"] in (1, 2) else False
        elif mol["C"] == 0 and mol["H"] != 0 and mol["O"] != 0:  # HyOz
            return True if mol["H"] == 2 and mol["O"] in (1, 2) else False
        elif mol["C"] != 0 and mol["H"] != 0:  # CxHyOz (z can be zero)
            node_val = lambda graph: {
                node: [
                    graph.degree(node),
                    valence_electrons.get(graph.nodes[node]["elem"], 0),
                ]
                for node in graph.nodes()
            }
            num_unsaturated_nodes = lambda dict: len(
                [node for node in dict.keys() if dict[node][0] < dict[node][1]]
            )
            node_valence_dict = node_val(graph)
            if num_unsaturated_nodes(node_valence_dict) == 0:  # all atoms are saturated
                return True
            elif (
                num_unsaturated_nodes(node_valence_dict) == 1
            ):  # only one unsaturated atom
                return False
            else:  # more than one unsaturated atom
                saturation_condition = lambda dict: all(
                    dict[node][0] == dict[node][1] for node in dict.keys()
                )
                while not saturation_condition(node_valence_dict):
                    unsat_nodes = [
                        node
                        for node in node_valence_dict.keys()
                        if node_valence_dict[node][0] < node_valence_dict[node][1]
                    ]
                    O_unsat_nodes = [
                        node for node in unsat_nodes if graph.nodes[node]["elem"] == "O"
                    ]  # all oxygens unsaturated
                    if len(O_unsat_nodes) != 0:  # unsaturated oxygen atoms
                        for oxygen in O_unsat_nodes:
                            node_valence_dict[oxygen][0] += 1
                            # increase the valence of the oxygen neighbour by 1
                            for neighbour in graph.neighbors(
                                oxygen
                            ):  # only one neighbour
                                if (
                                    node_valence_dict[neighbour][0]
                                    < node_valence_dict[neighbour][1]
                                ):
                                    node_valence_dict[neighbour][0] += 1
                                else:
                                    return False  # O neighbour is saturated already
                    else:  # CxHy
                        # select node with the highest degree
                        max_degree = max(
                            [node_valence_dict[node][0] for node in unsat_nodes]
                        )
                        max_degree_node = [
                            node
                            for node in unsat_nodes
                            if node_valence_dict[node][0] == max_degree
                        ][0]
                        max_degree_node_unsat_neighbours = [
                            neighbour
                            for neighbour in graph.neighbors(max_degree_node)
                            if neighbour in unsat_nodes
                        ]
                        if (
                            len(max_degree_node_unsat_neighbours) == 0
                        ):  # all neighbours are saturated
                            return False
                        else:
                            node_valence_dict[max_degree_node][0] += 1
                            node_valence_dict[max_degree_node_unsat_neighbours[0]][
                                0
                            ] += 1
                return True

    def rdkit_to_ase(self) -> Atoms:
        """
        Generate an ASE Atoms object from an RDKit molecule.

        """
        rdkit_molecule = self.molecule

        # If there are no atoms in the molecule, return an empty ASE Atoms object (Surface)
        if rdkit_molecule.GetNumAtoms() == 0:
            return Atoms()

        # Generate 3D coordinates for the molecule
        rdkit_molecule = Chem.AddHs(
            rdkit_molecule
        )  # Add hydrogens if not already added
        AllChem.EmbedMolecule(rdkit_molecule, AllChem.ETKDG())

        # Get the number of atoms in the molecule
        num_atoms = rdkit_molecule.GetNumAtoms()

        # Initialize lists to store positions and symbols
        positions = []
        symbols = []

        # Extract atomic positions and symbols
        for atom_idx in range(num_atoms):
            atom_position = rdkit_molecule.GetConformer().GetAtomPosition(atom_idx)
            atom_symbol = rdkit_molecule.GetAtomWithIdx(atom_idx).GetSymbol()
            positions.append(atom_position)
            symbols.append(atom_symbol)

        # Create an ASE Atoms object
        ase_atoms = Atoms(
            [
                Atom(symbol=symbol, position=position)
                for symbol, position in zip(symbols, positions)
            ]
        )
        # Setting pbc to True
        ase_atoms.set_pbc(True)

        return ase_atoms

    def ase_to_rdkit(self):
        """
        Convert an ASE Atoms object to an RDKit molecule.
        """
        buffer = StringIO()

        # Write the ASE Atoms object to the buffer in PDB format
        write(buffer, self.molecule, format='proteindatabank')

        # The buffer's content is the PDB string, so reset the buffer's position to the start
        buffer.seek(0)

        # Read the content of the buffer
        pdb_string = buffer.read()

        rdkit_mol = Chem.MolFromPDBBlock(pdb_string, removeHs=False)
        return rdkit_mol

    def get_smiles(self):
        """
        Get the SMILES string of a molecule.
        """
        return Chem.MolToSmiles(self.rdkit)

    def get_num_electrons(self):
        """
        Get the number of valence electrons of the intermediate.
        """
        return 4 * self["C"] + 1 * self["H"] - 6 * self["O"]

    def ref_energy(self):
        """
        Get the reference energy of the intermediate.
        """
        return get_fragment_energy(self.molecule)
