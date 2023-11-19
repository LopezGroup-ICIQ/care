import os
import multiprocessing as mp
from rdkit.Chem import rdMolDescriptors
from rdkit import Chem

from care.rnet.intermediates_funcs import gen_alkanes_smiles, add_oxygens_to_molecule, gen_ether_smiles, rdkit_2_graph, id_group_dict, process_molecule
from care.rnet.utilities.functions import MolPack

import warnings
from rdkit import RDLogger

warnings.filterwarnings("ignore")
RDLogger.DisableLog('rdApp.*')


def is_desired_bond(bond: Chem.rdchem.Bond, atom_num1: int, atom_num2: int) -> bool:
    """
    Check if the bond is between the desired atom types
    
    Parameters
    ----------
    bond : rdkit.Chem.rdchem.Bond
        The bond to check
    atom_num1 : int
        The atomic number of the first atom
    atom_num2 : int
        The atomic number of the second atom
        
    Returns
    -------
    bool
        True if the bond is between the desired atom types, False otherwise
    """
    return ((bond.GetBeginAtom().GetAtomicNum() == atom_num1 and bond.GetEndAtom().GetAtomicNum() == atom_num2) or
            (bond.GetBeginAtom().GetAtomicNum() == atom_num2 and bond.GetEndAtom().GetAtomicNum() == atom_num1))

def get_chemical_formula(smiles: str) -> str:
    """
    Get the chemical formula of a molecule

    Parameters
    ----------
    smiles : str
        The SMILES string of the molecule

    Returns
    -------
    str
        The chemical formula of the molecule
    """
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    return rdMolDescriptors.CalcMolFormula(mol)

def find_unique_bonds(mol):
    # Aromaticity detection
    Chem.SanitizeMol(mol)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True, flagPossibleStereoCenters=True)

    # Assign symmetry classes to atoms
    symmetry_classes = Chem.CanonicalRankAtoms(mol, breakTies=False)

    # Iterate over bonds and find unique representative bonds
    unique_bonds = {}
    for bond in mol.GetBonds():
        atom1_sym_class = symmetry_classes[bond.GetBeginAtomIdx()]
        atom2_sym_class = symmetry_classes[bond.GetEndAtomIdx()]
        bond_key = tuple(sorted([atom1_sym_class, atom2_sym_class]))

        # Store only the first bond for each equivalent class
        if bond_key not in unique_bonds:
            unique_bonds[bond_key] = bond

    return list(unique_bonds.values())

# Initialize a set for unique reactions
unique_reactions = set()
def break_bonds(molecule: Chem.rdchem.Mol, bond_types: list[tuple[int, int]], processed_fragments: dict, original_smiles: str):
    """
    Break the bonds of a molecule and recursively call itself on the fragments
    
    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to break the bonds of
    bond_types : list
        A list of tuples of the desired bond types
    processed_fragments : dict
        A dictionary of processed fragments
    original_smiles : str
        The original SMILES string of the molecule
        
    Returns
    -------
        None
    """

    # Convert the current molecule to SMILES at this depth
    current_smiles = Chem.MolToSmiles(molecule, isomericSmiles=True, allHsExplicit=True)

    # Finding unique bonds
    unique_bonds = find_unique_bonds(molecule)

    total_bond_counter = 0
    for bond in unique_bonds:
        for atom_num1, atom_num2 in bond_types:
            if is_desired_bond(bond, atom_num1, atom_num2):
                mol_copy = Chem.RWMol(molecule)
                mol_copy.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

                frags = Chem.GetMolFrags(mol_copy, asMols=True, sanitizeFrags=False)
                frag_smiles_list = []
                for frag in frags:
                    frag_smiles = Chem.MolToSmiles(frag, isomericSmiles=True, allHsExplicit=True)
                    frag_smiles_list.append(frag_smiles)
                    
                    if frag_smiles not in processed_fragments[original_smiles]:
                        processed_fragments[original_smiles].append(frag_smiles)
                    
                    # Recursive call with the fragment as the new molecule
                    frag_mol = Chem.MolFromSmiles(frag_smiles, sanitize=False)
                    break_bonds(frag_mol, bond_types, processed_fragments, original_smiles)

                # Create a tuple for the reaction
                reaction_tuple = (current_smiles, tuple(sorted(frag_smiles_list)))
                
                # Check if the reaction tuple is unique
                if reaction_tuple not in unique_reactions:
                    unique_reactions.add(reaction_tuple)
                    # Convert smiles to formula
                    current_formula = get_chemical_formula(current_smiles)

                    # If one of the smiles in the fragment is H2 moecule, convert it to [H]
                    frag_smiles_list = [smiles if smiles != '[H][H]' else '[H]' for smiles in frag_smiles_list]
                    frag_formula_list = [get_chemical_formula(frag_smiles) for frag_smiles in frag_smiles_list]

                    reaction_str = f"{current_formula} -> {' + '.join(frag_formula_list)}, : {bond.GetBeginAtom().GetSymbol()}-{bond.GetEndAtom().GetSymbol()} bond"
                    print(reaction_str)
                    total_bond_counter += 1

    return total_bond_counter

def process_molecule(smiles, bond_types, processed_fragments):
    molecule = Chem.MolFromSmiles(smiles)
    molecule_with_H = Chem.AddHs(molecule)

    original_smiles = Chem.MolToSmiles(molecule_with_H, isomericSmiles=True, allHsExplicit=True)
    if original_smiles not in processed_fragments:
        processed_fragments[original_smiles] = []

    n_react = break_bonds(molecule_with_H, bond_types, processed_fragments, original_smiles)
    return n_react


def generate_intermediates(ncc: int, noc: int,) -> dict[str, dict[int, list[MolPack]]]:
    """
    Generates all the intermediates of the reaction network.

    Parameters
    ----------
    ncc : int
        Maximum number of carbon atoms in the intermediates.
    noc : int
        Maximum number of oxygen atoms in the intermediates.

    Returns
    -------
    dict[str, dict[int, list[MolPack]]]
        Dictionary containing the intermediates of the reaction network.
        each key is the smiles of a saturated CHO molecule, and each value is a dictionary   
        containing the intermediates of the reaction network for that molecule.
        Each key of the subdictionary is the number of hydrogen removed from the molecule, 
        and each value is a list of MolPack objects, each one representing a specific isomer.
    """
    
    # 1) Generate all closed-shell satuarated CHO molecules
    print("Generating saturated CHO molecules...")
    alkanes_smiles = gen_alkanes_smiles(ncc)    
    mol_alkanes = [Chem.MolFromSmiles(smiles) for smiles in list(alkanes_smiles)]
    print("Done generating saturated CHO molecules.")
    # 2) Add oxygens to each molecule
    print("Adding oxygens to molecules...")
    cho_smiles = [add_oxygens_to_molecule(mol, noc) for mol in mol_alkanes] 
    cho_smiles = [smiles for smiles_set in cho_smiles for smiles in smiles_set]  # flatten list of lists
    #epoxides_smiles = gen_epoxides_smiles(mol_alkanes, noc)
    print("Done adding oxygens to molecules.")
    # 3) Add ethers to each molecule
    print("Adding ethers to molecules...")
    ethers_smiles = gen_ether_smiles(mol_alkanes, noc)
    #cho_smiles += epoxides_smiles
    cho_smiles += ethers_smiles
    print("Done adding ethers to molecules.")
    cho_smiles += ['CO','C(O)O','O', 'OO', '[H][H]']
    all_smiles = cho_smiles + alkanes_smiles
    # Define bond types
    bond_types = [(6, 6), (6, 1), (6, 8), (8, 8), (1, 1), (8, 1)]
    
    # # Dictionary to keep track of processed fragments
    processed_fragments = {}
    # Process each molecule in the list
    for smiles in all_smiles:
        print(smiles)
        n_react = process_molecule(smiles, bond_types, processed_fragments)
    print(len(unique_reactions))
    

generate_intermediates(ncc=3, noc=-1)

# Saving the unique reactions in a text file
with open('unique_reactions.txt', 'w') as f:
    for reaction in unique_reactions:
        f.write(f'{reaction[0]} -> {reaction[1]}\n')
