from typing import Union
from io import StringIO
import numpy as np

from ase import Atom, Atoms
from ase.io import read, write
from networkx import cycle_basis
from rdkit import Chem
from rdkit.Chem import AllChem

from care.constants import INTER_PHASES, BOND_ORDER
from care.crn.utils.species import atoms_to_graph


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
    __slots__ = [
        "code",
    "molecule",
    "_graph",
    "_formula",
    "_electrons",
    "_charge",
    "_mass",
    "_smiles",
    "_cyclic",
    "_rdkit",
    "ads_configs",
    "is_surface",
    "phase",
    "charge",
    "closed_shell",
    "_gas_configs",
    ]
    phases = INTER_PHASES

    def __init__(
        self,
        code: str = None,
        molecule: Union[Atoms, Chem.rdchem.Mol] = None,
        is_surface: bool = False,
        phase: str = None,
    ):
        self.code = code
        self.is_surface = is_surface
        self._graph = None
        self._formula = None
        self._electrons = None
        self._charge = None
        self._mass = None
        self._smiles = None
        self._cyclic = None
        if isinstance(molecule, Chem.rdchem.Mol):
            self._rdkit = molecule
            self.molecule = self.rdkit_to_ase(molecule)
        else:
            self._rdkit = None
            self.molecule = molecule

        self.ads_configs = {}
        self.charge = 0

        if self.is_surface:
            self.phase = "surf"
            self.closed_shell = None
        else:
            self.closed_shell = self.is_closed_shell()
            if phase not in self.phases:
                raise ValueError(f"Phase must be one of {self.phases}")
            self.phase = phase
        self._gas_configs = None

    @property
    def formula(self):
        if self._formula is None:
            if len(self.molecule) == 0 and self.is_surface:
                self._formula = "surface"
            else:
                self._formula = self.molecule.get_chemical_formula()
        return self._formula
    
    @property
    def mass(self):
        if self._mass is None:
            self._mass = self.molecule.get_masses().sum()
        return self._mass

    def __getitem__(self, key: str):
        if key == "*":
            if self.phase in ("surf", "ads"):
                return 1
            else:
                return 0
        elif key == "q":
            return self.charge
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
        elif self.phase == "electro":
            txt = self.code + "({}(electro))".format(self.formula)
        else:
            txt = self.code + "({}(g))".format(self.formula)
        return txt

    def __str__(self):
        return self.__repr__()

    @classmethod
    def from_molecule(
        cls,
        ase_atoms_obj: Union[Atoms, str],
        code: str = None,
        is_surface: bool = False,
        phase: str = None,
    ) -> "Intermediate":
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
        if isinstance(ase_atoms_obj, str):
            ase_atoms_obj = read(ase_atoms_obj, format="vasp")
        elif not isinstance(ase_atoms_obj, Atoms):
            raise ValueError("ase_atoms_obj must be an ASE Atoms object or a string path to a POSCAR file.")
        if phase not in ["gas", "ads", "surf"]:
            raise ValueError("phase must be either 'gas' or 'ads'")
        return cls(
            code=code, molecule=ase_atoms_obj, is_surface=is_surface, phase=phase
        )
        
    @classmethod
    def from_smiles(cls, 
                smiles: str,
                phase: str = "gas") -> "Intermediate":
        """Create an Intermediate using a SMILES string.
        Args:
            smiles (str): SMILES string of the molecule.
        Returns:
            obj:`Intermediate` of the given SMILES.
        """
        phase_id = "g" if phase=="gas" else "*"    
        rdkit_mol = Chem.MolFromSmiles(smiles)
        inchikey = Chem.inchi.MolToInchiKey(rdkit_mol)        
        return cls(code=inchikey+phase_id, molecule=rdkit_mol, is_surface=False, phase="gas")
    
    @property
    def graph(self):
        if self._graph is None:
            self._graph = atoms_to_graph(self.molecule)
        return self._graph
    
    @property
    def rdkit(self):
        if self._rdkit is None and len(self.molecule) != 0:
            x = self.ase_to_rdkit()
            self._rdkit = x
        return self._rdkit

    @graph.setter
    def graph(self, other):
        self._graph = other

    @property
    def cyclic(self) -> bool:
        """
        Check if a molecule is cyclic or not.
        """
        if self.is_surface:
            self._cyclic = None
        if self._cyclic is None:            
            cycles = list(cycle_basis(self.graph))
            self._cyclic = True if len(cycles) != 0 else False
        return self._cyclic

    @property
    def gas_configs(self) -> list[Atoms]:
        if self._gas_configs is None:
            self._gas_configs = self.gen_gas_configs()
        return self._gas_configs
    
    def is_closed_shell(self) -> bool:
        """
        Check if molecule is a neutral closed-shell species using RDKit.
        Returns True if neutral and no unpaired electrons, False otherwise.
        """
        use_old = True if self["N"] == 0 else False
        if use_old:
            graph = self.graph

            if self["C"] != 0 and self["H"] == 0 and self["O"] == 0:  # Cx
                return False
            elif self["C"] == 0 and self["H"] != 0 and self["O"] == 0:  # Hy
                return True if self["H"] == 2 else False
            elif self["C"] == 0 and self["H"] == 0 and self["O"] != 0:  # Oz
                return True if self["O"] == 2 else False
            elif self["C"] != 0 and self["H"] == 0 and self["O"] != 0:  # CxOz
                return True if self["C"] == 1 and self["O"] in (1, 2) else False
            elif self["C"] == 0 and self["H"] != 0 and self["O"] != 0:  # HyOz
                return True if self["H"] == 2 and self["O"] in (1, 2) else False
            elif self["C"] != 0 and self["H"] != 0:  # CxHyOz (z can be zero)
                node_val = lambda graph: {
                    node: [
                        graph.degree(node),
                        BOND_ORDER.get(graph.nodes[node]["elem"], 0),
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
        else:
            if self.is_surface or len(self.molecule) == 0:
                return None 
            def get_valence_corrected_mol(smiles: str):
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    return None
                inchi = Chem.MolToInchi(mol)
                return Chem.MolFromInchi(inchi)
            
            mol = get_valence_corrected_mol(self.smiles)
            if mol is None:
                return False

            try:
                Chem.SanitizeMol(mol)
            except Exception:
                return False
            n_radicals = sum(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms())
            if n_radicals != 0:
                return False
            if Chem.GetFormalCharge(mol) != 0:
                return False

            return True

    def gen_gas_configs(self) -> list[Atoms]:
        """
        Generate a list of gas-phase ASE Atoms object from an RDKit molecule.
        Needed for adsorbate placement (if a better scan is preferred)

        """
        rdkit_molecule = self.rdkit
        if rdkit_molecule is None:
            return Atoms()

        rdkit_molecule = Chem.AddHs(
            rdkit_molecule
        ) 
        
        n = {k: self[k] for k in BOND_ORDER.keys()}

        def conformer_to_ase(mol, confId):
            conf = mol.GetConformer(confId)
            symbols = [a.GetSymbol() for a in mol.GetAtoms()]
            positions = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
            ase_atoms = Atoms(symbols=symbols, positions=positions)
            ase_atoms.set_cell([20, 20, 20])
            ase_atoms.set_pbc(True)
            return ase_atoms

        num_conformers = 50 * (1+n["C"]) + 10 * n["O"] + 2 * n["H"]
        randomseed = 42
        if rdkit_molecule.GetNumAtoms() > 2:
            AllChem.EmbedMultipleConfs(rdkit_molecule, numConfs=num_conformers, randomSeed=randomseed)
            results = AllChem.MMFFOptimizeMoleculeConfs(rdkit_molecule, numThreads=1)
            sorted_confs = sorted(enumerate(results), key=lambda x: x[1][1])
            lowest_confIds = [confId for confId, _ in sorted_confs[:3]]
            return [conformer_to_ase(rdkit_molecule, confId) for confId in lowest_confIds]
        else:
            AllChem.EmbedMolecule(rdkit_molecule, randomSeed=randomseed)
            return [conformer_to_ase(rdkit_molecule, 0)]

    def rdkit_to_ase(self, rdkit_molecule) -> Atoms:
        """
        Generate an ASE Atoms object from an RDKit molecule.

        """
        if rdkit_molecule.GetNumAtoms() == 0:
            return Atoms()

        rdkit_molecule = Chem.AddHs(
            rdkit_molecule
        )  # Add hydrogens if not already added

        num_C = sum(
            [1 for atom in rdkit_molecule.GetAtoms() if atom.GetSymbol() == "C"]
        )

        num_O = sum(
            [1 for atom in rdkit_molecule.GetAtoms() if atom.GetSymbol() == "O"]
        )

        num_H = sum(
            [1 for atom in rdkit_molecule.GetAtoms() if atom.GetSymbol() == "H"]
        )

        num_conformers = 100 * (1+num_C) + 10 * num_O + 2 * num_H
        randomseed = 42

        if rdkit_molecule.GetNumAtoms() > 2:
            AllChem.EmbedMultipleConfs(rdkit_molecule, numConfs=num_conformers, randomSeed=randomseed)
            confs = AllChem.MMFFOptimizeMoleculeConfs(rdkit_molecule)
            conf_energies = [item[1] for item in confs]
            lowest_conf = int(np.argmin(conf_energies))
            lowest_conf = AllChem.EmbedMolecule(rdkit_molecule, randomSeed=randomseed)
            xyz_coordinates = AllChem.MolToXYZBlock(rdkit_molecule, confId=lowest_conf)
            ase_atoms = read(StringIO(xyz_coordinates), format="xyz")
        else:
            AllChem.EmbedMolecule(rdkit_molecule, randomSeed=randomseed)
            num_atoms = rdkit_molecule.GetNumAtoms()
            positions = []
            symbols = []

            for atom_idx in range(num_atoms):
                atom_position = rdkit_molecule.GetConformer().GetAtomPosition(atom_idx)
                atom_symbol = rdkit_molecule.GetAtomWithIdx(atom_idx).GetSymbol()
                positions.append(atom_position)
                symbols.append(atom_symbol)

            ase_atoms = Atoms(
                [
                    Atom(symbol=symbol, position=position)
                    for symbol, position in zip(symbols, positions)
                ]
            )
        ase_atoms.set_cell([20, 20, 20])
        ase_atoms.set_pbc(True)
        ase_atoms.new_array("atom_tags", [1] * len(ase_atoms), dtype=int)
        return ase_atoms

    def ase_to_rdkit(self) -> Chem.rdchem.Mol:
        """
        Convert an ASE Atoms object to an RDKit molecule.
        """
        buffer = StringIO()
        curr_mol = self.molecule.copy()
        curr_mol.set_cell([20, 20, 20])

        write(buffer, curr_mol, format="proteindatabank")

        buffer.seek(0)

        pdb_string = buffer.read()

        rdkit_mol = Chem.MolFromPDBBlock(pdb_string, removeHs=False)
        if rdkit_mol is None:
            raise ValueError(f"RDKit failed to convert from ASE {self.code}({self.formula})")
        return rdkit_mol

    @property
    def smiles(self) -> str:
        if not self.is_surface and len(self.molecule) != 0:
            self._smiles = Chem.MolToSmiles(self.rdkit, allHsExplicit=False)
        else:
            self._smiles = None
        return self._smiles

    def get_smiles(self, allHsExplicit=False):
        return Chem.MolToSmiles(self.rdkit, allHsExplicit=allHsExplicit)

    @property
    def electrons(self) -> int:
        if self._electrons is None:
            self._electrons = sum([BOND_ORDER[elem]*self[elem] for elem in set(self.molecule.get_chemical_symbols())])
        return self._electrons
