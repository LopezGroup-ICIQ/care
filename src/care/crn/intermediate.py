from abc import ABC, abstractmethod
from typing import Union
from io import StringIO
import numpy as np

from ase import Atoms
from ase.io import read, write
from networkx import cycle_basis
from rdkit import Chem
from rdkit.Chem import AllChem

from care.constants import INTER_PHASES, BOND_ORDER
from care.crn.utils.species import atoms_to_graph


class Intermediate(ABC):
    """Abstract base class for all network species."""
    __slots__ = ("code", "phase", "charge")
    phases = INTER_PHASES

    def __init__(self, code: str, phase: str):
        self.code = code
        if phase not in self.phases:
            raise ValueError(f"Phase must be one of {self.phases}")
        self.phase = phase
        self.charge = 0

    @property
    @abstractmethod
    def formula(self) -> str:
        pass

    @property
    @abstractmethod
    def mass(self) -> float:
        pass

    def __hash__(self):
        return hash(self.code)

    def __eq__(self, other):
        if isinstance(other, str):
            return self.code == other
        if isinstance(other, Intermediate):
            return self.code == other.code
        return False

    def __str__(self):
        return self.__repr__()

    def __getitem__(self, key: str):
        if key == "*":
            return 1 if self.phase in ("surf", "ads") else 0
        elif key == "q":
            return self.charge
        return self._get_elem_count(key)

    @abstractmethod
    def _get_elem_count(self, key: str) -> int:
        pass

    @classmethod
    def from_molecule(
        cls,
        ase_atoms_obj: Union[Atoms, str],
        code: str = None,
        phase: str = None,
    ) -> "Intermediate":
        if isinstance(ase_atoms_obj, str):
            ase_atoms_obj = read(ase_atoms_obj, format="vasp")
        elif not isinstance(ase_atoms_obj, Atoms):
            raise ValueError("ase_atoms_obj must be an ASE Atoms object or a string path to a POSCAR file.")
        
        if phase == "gas":
            return GasSpecies(code=code, molecule=ase_atoms_obj)
        elif phase == "ads":
            return AdsorbedSpecies(code=code, molecule=ase_atoms_obj)
        elif phase == "surf":
            return SurfaceSite(code=code)
        else:
            raise ValueError("phase must be either 'gas', 'ads', or 'surf'")

    @classmethod
    def from_smiles(cls, smiles: str, phase: str = "gas") -> "Intermediate":
        phase_id = "g" if phase == "gas" else "*"    
        rdkit_mol = Chem.MolFromSmiles(smiles)
        inchikey = Chem.inchi.MolToInchiKey(rdkit_mol)        
        
        if phase == "gas":
            return GasSpecies(code=inchikey+phase_id, molecule=rdkit_mol)
        elif phase == "ads":
            return AdsorbedSpecies(code=inchikey+phase_id, molecule=rdkit_mol)
        else:
            raise ValueError("from_smiles only supports 'gas' or 'ads' phases")


class SurfaceSite(Intermediate):
    """Lightweight dummy intermediate for the empty catalyst surface."""
    __slots__ = ("_E")

    def __init__(self, code: str = "*"):
        super().__init__(code=code, phase="surf")
        self._E = 0.0

    @property
    def formula(self) -> str:
        return "surface"

    @property
    def mass(self) -> float:
        return 0.0

    def _get_elem_count(self, key: str) -> int:
        return 0

    @property
    def closed_shell(self):
        return None

    @property
    def cyclic(self):
        return None

    def __repr__(self):
        return f"{self.code}(*)"
    
    @property
    def E(self):
        return self._E
    
    @E.setter
    def E(self, value: float):
        self._E = value


class MolecularSpecies(Intermediate):
    """Base class for physical molecules requiring graphs and atomic data."""
    __slots__ = (
        "_ase_molecule", "_rdkit_molecule", "_graph", "_formula",
        "_electrons", "_mass", "_smiles", "_cyclic", "_closed_shell"
    )

    def __init__(self, code: str, phase: str, molecule: Union[Atoms, Chem.rdchem.Mol] = None):
        super().__init__(code, phase)
        self._graph = None
        self._formula = None
        self._electrons = None
        self._mass = None
        self._smiles = None
        self._cyclic = None
        self._closed_shell = None

        if isinstance(molecule, Chem.rdchem.Mol):
            self._rdkit_molecule = molecule
            self._ase_molecule = None
        else:
            self._rdkit_molecule = None
            self._ase_molecule = molecule
    
    @property
    def molecule(self) -> Atoms:
        if self._ase_molecule is None and self._rdkit_molecule is not None:
            self._ase_molecule = self.rdkit_to_ase(self._rdkit_molecule)
        return self._ase_molecule

    @molecule.setter
    def molecule(self, new_atoms: Atoms):
        """
        Allow overwriting the initial guess with a relaxed structure, 
        strictly enforcing that the atomic composition remains identical.
        """
        if self.molecule is not None and len(self.molecule) > 0:
            orig_symbols = sorted(self.molecule.get_chemical_symbols())
            new_symbols = sorted(new_atoms.get_chemical_symbols())
            
            if orig_symbols != new_symbols:
                raise ValueError(
                    f"Composition mismatch for species '{self.code}'.\n"
                    f"Original composition: {self.molecule.get_chemical_formula()}\n"
                    f"Attempted overwrite:  {new_atoms.get_chemical_formula()}\n"
                    f"Overwriting with a different composition is strictly forbidden."
                )
        
        self._ase_molecule = new_atoms
        self._graph = None

    @property
    def rdkit(self) -> Chem.rdchem.Mol:
        if self._rdkit_molecule is None and self.molecule is not None and len(self.molecule) != 0:
            self._rdkit_molecule = self.ase_to_rdkit()
        return self._rdkit_molecule

    @property
    def formula(self) -> str:
        if self._formula is None:
            if self.molecule is not None and len(self.molecule) > 0:
                self._formula = self.molecule.get_chemical_formula()
            else:
                self._formula = ""
        return self._formula

    @property
    def mass(self) -> float:
        if self._mass is None:
            if self.molecule is not None:
                self._mass = self.molecule.get_masses().sum()
            else:
                self._mass = 0.0
        return self._mass

    def _get_elem_count(self, key: str) -> int:
        if self.molecule is not None:
            return self.molecule.get_chemical_symbols().count(key)
        return 0

    @property
    def graph(self):
        if self._graph is None and self.molecule is not None:
            self._graph = atoms_to_graph(self.molecule)
        return self._graph

    @graph.setter
    def graph(self, other):
        self._graph = other

    @property
    def cyclic(self) -> bool:
        if self._cyclic is None:
            if self.graph is not None:
                cycles = list(cycle_basis(self.graph))
                self._cyclic = True if len(cycles) != 0 else False
            else:
                self._cyclic = False
        return self._cyclic

    @property
    def electrons(self) -> int:
        if self._electrons is None and self.molecule is not None:
            self._electrons = sum([BOND_ORDER.get(elem, 0) * self._get_elem_count(elem) for elem in set(self.molecule.get_chemical_symbols())])
        return self._electrons

    @property
    def smiles(self) -> str:
        if self._smiles is None and self.rdkit is not None:
            self._smiles = Chem.MolToSmiles(self.rdkit, allHsExplicit=False)
        return self._smiles

    def get_smiles(self, allHsExplicit=False):
        if self.rdkit is not None:
            return Chem.MolToSmiles(self.rdkit, allHsExplicit=allHsExplicit)
        return None

    @property
    def closed_shell(self):
        if self._closed_shell is None:
            self._closed_shell = self.is_closed_shell()
        return self._closed_shell

    def is_closed_shell(self) -> bool:
        """
        Check if molecule is a stable, neutral species capable of desorbing using RDKit.
        """
        if self.molecule is None or len(self.molecule) == 0:
            return False

        smiles = self.smiles
        if not smiles:
            return False

        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False

            inchi = Chem.MolToInchi(mol)
            if not inchi:
                return False
                
            corrected_mol = Chem.MolFromInchi(inchi)
            if corrected_mol is None:
                return False

            Chem.SanitizeMol(corrected_mol)

            if Chem.GetFormalCharge(corrected_mol) != 0:
                return False

            n_radicals = sum(atom.GetNumRadicalElectrons() for atom in corrected_mol.GetAtoms())
            
            if n_radicals == 0:
                return True
                
            # --- STABLE RADICAL WHITELIST ---
            stable_radicals = ["O2", "NO", "NO2"]
            if self.formula in stable_radicals:
                return True

            return False

        except Exception:
            return False
        
    def rdkit_to_ase(self, rdkit_molecule: Chem.rdchem.Mol) -> Atoms:
        """
        Generate an ASE Atoms object from an RDKit molecule with dynamic cell sizing.
        """
        if rdkit_molecule.GetNumAtoms() == 0:
            return Atoms(cell=[20, 20, 20], pbc=True)

        mol = Chem.AddHs(rdkit_molecule)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        
        if mol.GetNumAtoms() > 1:
            try:
                AllChem.MMFFOptimizeMolecule(mol)
            except Exception:
                pass

        num_atoms = mol.GetNumAtoms()
        positions = np.zeros((num_atoms, 3))
        symbols = []

        conf = mol.GetConformer()
        for i in range(num_atoms):
            pos = conf.GetAtomPosition(i)
            positions[i] = [pos.x, pos.y, pos.z]
            symbols.append(mol.GetAtomWithIdx(i).GetSymbol())

        ase_atoms = Atoms(symbols=symbols, positions=positions)
        extents = positions.ptp(axis=0)  # Calculate the bounding box of the molecule (max - min in X, Y, Z)
        cell_dims = np.maximum(20.0, extents + 10.0)
        
        ase_atoms.set_cell(cell_dims)
        ase_atoms.center()
        ase_atoms.set_pbc(True)
        ase_atoms.new_array("atom_tags", np.ones(num_atoms, dtype=int))
        
        return ase_atoms

    def ase_to_rdkit(self) -> Chem.rdchem.Mol:
        """
        Convert an ASE Atoms object to an RDKit molecule.
        """
        if self.molecule is None or len(self.molecule) == 0:
            return Chem.Mol()

        buffer = StringIO()
        
        write(buffer, self.molecule, format="proteindatabank")
        
        pdb_string = buffer.getvalue()

        # Pass sanitize=False to stop RDKit from rejecting non-standard valences 
        # or auto-injecting implicit hydrogens before we can disable them.
        rdkit_mol = Chem.MolFromPDBBlock(pdb_string, removeHs=False, sanitize=False)
        
        if rdkit_mol is None:
            raise ValueError(f"RDKit failed to convert from ASE {self.code}({self.formula})")
            
        # STRICT INTERVENTION: The ASE object is the single source of truth.
        # Force RDKit to never hallucinate implicit hydrogens to satisfy valences.
        for atom in rdkit_mol.GetAtoms():
            atom.SetNoImplicit(True)
            
        # Update topological properties without triggering standard valence checks
        rdkit_mol.UpdatePropertyCache(strict=False)
            
        return rdkit_mol


class GasSpecies(MolecularSpecies):
    """Gas phase intermediates."""
    __slots__ = ("_E",)

    def __init__(self, code: str, molecule: Union[Atoms, Chem.rdchem.Mol] = None):
        super().__init__(code=code, phase="gas", molecule=molecule)
        self._E = None

    def __repr__(self):
        if self.phase == "solv":
            return f"{self.code}({self.formula}(aq))"
        return f"{self.code}({self.formula}(g))"
    
    @property
    def E(self):
        return self._E
    
    @E.setter
    def E(self, value: float):
        self._E = value

    @property    
    def is_evaluated(self) -> bool:
        return self._E is not None


class AdsorbedSpecies(MolecularSpecies):
    """Surface-bound intermediates."""
    __slots__ = ("_catalyst", "ads_configs")

    def __init__(self, code: str, molecule: Union[Atoms, Chem.rdchem.Mol] = None):
        super().__init__(code=code, phase="ads", molecule=molecule)
        self.ads_configs = {}
        self._catalyst = None

    @property
    def catalyst(self):
        return self._catalyst

    @catalyst.setter
    def catalyst(self, catalyst):
        self._catalyst = catalyst

    @property    
    def is_evaluated(self) -> bool:
        return len(self.ads_configs) > 0

    def __repr__(self):
        return f"{self.code}({self.formula}*)"
    
    @property
    def lowest_energy_config(self) -> dict:
        """
        Retrieves the relaxed configuration with the lowest energy 
        from the evaluated adsorption configurations.
        """
        if not self.is_evaluated:
            raise ValueError(f"Species {self.code} has not been evaluated yet.")
        return min(self.ads_configs.values(), key=lambda conf: conf['mu'])
    
    @property
    def E(self):
        """
        Returns the energy (mu) of the lowest energy configuration.
        Returns None if not evaluated, maintaining a consistent API with GasSpecies.
        """
        if not self.is_evaluated:
            return None
        return self.lowest_energy_config["mu"]