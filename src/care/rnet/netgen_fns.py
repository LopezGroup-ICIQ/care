import multiprocessing as mp
from collections import defaultdict
from itertools import combinations
import re 
import warnings
from rdkit import RDLogger
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Draw
from ase import Atoms
from ase.neighborlist import neighbor_list
import networkx as nx
import matplotlib.pyplot as plt

from care.rnet.networks.utils import Intermediate, ElementaryReaction
from care.rnet.intermediates_funcs import gen_alkanes_smiles, add_oxygens_to_molecule, gen_ether_smiles, gen_epoxides_smiles
from care.rnet.networks.utils import gen_adsorption_reactions, gen_rearrangement_reactions

warnings.filterwarnings("ignore")
RDLogger.DisableLog('rdApp.*')

CORDERO = {'Ac': 2.15, 'Al': 1.21, 'Am': 1.80, 'Sb': 1.39, 'Ar': 1.06,
           'As': 1.19, 'At': 1.50, 'Ba': 2.15, 'Be': 0.96, 'Bi': 1.48,
           'B' : 0.84, 'Br': 1.20, 'Cd': 1.44, 'Ca': 1.76, 'C' : 0.76,
           'Ce': 2.04, 'Cs': 2.44, 'Cl': 1.02, 'Cr': 1.39, 'Co': 1.50,
           'Cu': 1.32, 'Cm': 1.69, 'Dy': 1.92, 'Er': 1.89, 'Eu': 1.98,
           'F' : 0.57, 'Fr': 2.60, 'Gd': 1.96, 'Ga': 1.22, 'Ge': 1.20,
           'Au': 1.36, 'Hf': 1.75, 'He': 0.28, 'Ho': 1.92, 'H' : 0.31,
           'In': 1.42, 'I' : 1.39, 'Ir': 1.41, 'Fe': 1.52, 'Kr': 1.16,
           'La': 2.07, 'Pb': 1.46, 'Li': 1.28, 'Lu': 1.87, 'Mg': 1.41,
           'Mn': 1.61, 'Hg': 1.32, 'Mo': 1.54, 'Ne': 0.58, 'Np': 1.90,
           'Ni': 1.24, 'Nb': 1.64, 'N' : 0.71, 'Os': 1.44, 'O' : 0.66,
           'Pd': 1.39, 'P' : 1.07, 'Pt': 1.36, 'Pu': 1.87, 'Po': 1.40,
           'K' : 2.03, 'Pr': 2.03, 'Pm': 1.99, 'Pa': 2.00, 'Ra': 2.21,
           'Rn': 1.50, 'Re': 1.51, 'Rh': 1.42, 'Rb': 2.20, 'Ru': 1.46,
           'Sm': 1.98, 'Sc': 1.70, 'Se': 1.20, 'Si': 1.11, 'Ag': 1.45,
           'Na': 1.66, 'Sr': 1.95, 'S' : 1.05, 'Ta': 1.70, 'Tc': 1.47,
           'Te': 1.38, 'Tb': 1.94, 'Tl': 1.45, 'Th': 2.06, 'Tm': 1.90,
           'Sn': 1.39, 'Ti': 1.60, 'Wf': 1.62, 'U' : 1.96, 'V' : 1.53,
           'Xe': 1.40, 'Yb': 1.87, 'Y' : 1.90, 'Zn': 1.22, 'Zr': 1.75}  # Atomic radii from Cordero et al. 


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

def find_unique_bonds(mol: Chem.rdchem.Mol) -> list[Chem.rdchem.Bond]:
    """
    Find the unique bonds in a molecule

    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
        The molecule to find the unique bonds of

    Returns
    -------
    list[rdkit.Chem.rdchem.Bond]
        The unique bonds in the molecule
    """

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
processed_molecules = set()
def break_bonds(molecule: Chem.rdchem.Mol, bond_types: list[tuple[int, int]], processed_fragments: dict, original_smiles: str)  -> None:
    """
    Recursively break bonds in a molecule and adds the reactions to the unique reactions set and the processed fragments to the processed fragments dictionary
    The function is recursive, and will break all the bonds of the desired types in the molecule, and then break all the bonds in the fragments, etc.

    Parameters
    ----------
    molecule : rdkit.Chem.rdchem.Mol
        The molecule to break bonds in (will be modified in place)
    bond_types : list[tuple[int, int]]
        The types of bonds to break, as tuples of atomic numbers
    processed_fragments : dict
        Dictionary to keep track of processed fragments
    original_smiles : str
        The original SMILES string of the molecule

    Returns
    -------
    None
        All the reactions are added to the unique reactions set, and all the processed fragments are added to the processed fragments dictionary
    """

    current_smiles = Chem.MolToSmiles(molecule, isomericSmiles=True, allHsExplicit=True)

    # Check if this molecule's reactions have already been processed
    if current_smiles in processed_molecules:
        return 0

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

                if len(frag_smiles_list) == 2:
                    # Correction for [HH] in the fragment smiles list
                    if frag_smiles_list[0] == '[HH]':
                        # Modify the fragment smiles list to have [H] instead of [HH]
                        frag_smiles_list[0] = '[H]'
                    
                    if frag_smiles_list[1] == '[HH]':
                        # Modify the fragment smiles list to have [H] instead of [HH]
                        frag_smiles_list[1] = '[H]'

                reaction_tuple = (current_smiles, tuple(sorted(frag_smiles_list)))
                
                # Check if the reaction tuple is unique
                if reaction_tuple not in unique_reactions:
                    r_type_atoms = sorted([Chem.Atom(atom_num1).GetSymbol(), Chem.Atom(atom_num2).GetSymbol()])
                    # If there is an [O][H] in the fragment, convert the O r_type to HO
                    if len(frag_smiles_list) == 2:
                        if reaction_tuple[1][0] == '[H][O]':
                            r_type_atoms[0] = 'OH'
                        elif reaction_tuple[1][1] == '[H][O]':
                            r_type_atoms[1] = 'OH'

                    r_type = f"{r_type_atoms[0]}-{r_type_atoms[1]}"

                    if r_type == 'OH-O':
                        r_type = 'O-OH'
                    if r_type == 'H-OH':
                        r_type = 'H-O'

                    # Extending the reaction tuple with the bond type
                    reaction_type_tuple = (reaction_tuple[0], reaction_tuple[1], r_type)
                    unique_reactions.add(reaction_type_tuple)
                    total_bond_counter += 1
        
        processed_molecules.add(current_smiles)


def process_molecule(smiles: str, bond_types: list[tuple[int, int]], processed_fragments: dict) -> None:
    """
    Process a molecule by breaking all the bonds of the desired types in the molecule, and then break all the bonds in the fragments, etc.

    Parameters
    ----------
    smiles : str
        The SMILES string of the molecule
    bond_types : list[tuple[int, int]]
        The types of bonds to break, as tuples of atomic numbers
    processed_fragments : dict
        Dictionary to keep track of processed fragments

    Returns
    -------
    None
        All the reactions are added to the unique reactions set, and all the processed fragments are added to the processed fragments dictionary
    """

    molecule = Chem.MolFromSmiles(smiles)
    molecule_with_H = Chem.AddHs(molecule)

    original_smiles = Chem.MolToSmiles(molecule_with_H, isomericSmiles=True, allHsExplicit=True)
    if original_smiles not in processed_fragments:
        processed_fragments[original_smiles] = []

    break_bonds(molecule_with_H, bond_types, processed_fragments, original_smiles)

def gen_inter_objs(inter_dict: dict[str, Chem.rdchem.Mol]) -> dict[str, list[Intermediate]]:
    """
    Generate the Intermediate objects for all the chemical species of the reaction network as a dictionary.
    For closed-shell species, gas and adsorbed phases are generated. For open-shell species, only the adsorbed phase is generated.

    Parameters
    ----------
    inter_dict : dict[str, Chem.rdchem.Mol]
        Dictionary containing the Chem.rdchem.Mol instances of all the chemical species of the reaction network.
        Each key is the InChIKey of a molecule, and each value is the corresponding Chem.rdchem.Mol instance.

    Returns
    -------
    intermediate_class_dict : dict[str, Intermediate]
        Dictionary containing the Intermediate instances of all the chemical species of the reaction network.
        Each key is the InChIKey of a molecule plus '*' or 'g' defining if its adsorber or in gas-phase, and each value the Intermediate instance for that molecule. 
    """

    inter_class_dict = {}
    for key, value in inter_dict.items():
        new_inter_ads = Intermediate(code=key+'*',
                                     molecule=value,
                                     phase='ads')
        inter_class_dict[key+'*'] = new_inter_ads

        if new_inter_ads.closed_shell:
            new_inter_gas = Intermediate(code=key+'g',
                                         molecule=value,
                                         phase='gas')
            inter_class_dict[key+'g']=new_inter_gas
    return inter_class_dict

def gen_adsorption_reactions(intermediates: dict[str, Intermediate], surf_inter: Intermediate) -> list[ElementaryReaction]:
    """
    Generate all Intermediates that can desorb from the surface and the corresponding desorption reactions, including
    the dissociative adsorption of H2 and O2.

    Parameters
    ----------
    intermediates : dict[str, Intermediate]
        Dictionary containing the Intermediate instances of all the chemical species of the reaction network.

    Returns
    -------
    adsorption_steps : list[ElementaryReaction]
        List of all adsorption reactions.
    """

    adsorption_steps = []
    for inter in intermediates.values():
        if inter.phase == 'gas':
            ads_inter = intermediates[inter.code[:-1] + '*']
            adsorption_steps.append(ElementaryReaction(components=(frozenset([surf_inter, inter]), frozenset([ads_inter])), r_type='adsorption'))

    # dissociative adsorptions for H2 and O2
    for molecule in ['UFHFLCQGNIYNRP-UHFFFAOYSA-N', 'MYMOFIZGZYHOMD-UHFFFAOYSA-N']:
        gas_code = molecule + 'g'
        if molecule == 'UFHFLCQGNIYNRP-UHFFFAOYSA-N': #H2
            ads_code = 'YZCKVEUIGOORGS-UHFFFAOYSA-N*'
        else: #O2
            ads_code = 'QVGXLLKOCUKJST-UHFFFAOYSA-N*'
        
        adsorption_steps.append(ElementaryReaction(components=(frozenset([surf_inter, intermediates[gas_code]]), frozenset([intermediates[ads_code]])), r_type='adsorption'))
    return adsorption_steps

def generate_inters_and_rxns(ncc: int, noc: int, ncores: int=mp.cpu_count()) -> tuple[dict[str, Intermediate], list[ElementaryReaction]]:
    """
    Generates all the intermediates and reactions of the reaction network.

    Parameters
    ----------
    ncc : int
        Maximum number of carbon atoms in the intermediates.
    noc : int
        Maximum number of oxygen atoms in the intermediates.

    Returns
    -------
    intermediates_dict : dict[str, Intermediate]
        Dictionary containing the Intermediate instances of all the chemical species of the reaction network.
        Each key is the InChIKey of a molecule, and each value is a list of Intermediate instances for that molecule.
    rxns_list : list[ElementaryReaction]
        List of all the reactions of the reaction network as ElementaryReaction instances.
    """
    
    # 1) Generate all closed-shell satuarated CHO molecules
    print("Generating saturated CHO molecules...")
    alkanes_smiles = gen_alkanes_smiles(ncc)    
    mol_alkanes = [Chem.MolFromSmiles(smiles) for smiles in list(alkanes_smiles)]
    print("Done generating saturated CHO molecules.")
    ethers_smiles = gen_ether_smiles(mol_alkanes, noc)
    mol_ethers = [Chem.MolFromSmiles(smiles) for smiles in ethers_smiles]
    mol_alkanes_ethers  = mol_alkanes + mol_ethers
    # 2) Add oxygens to each molecule
    print("Adding oxygens to molecules...")
    cho_smiles = [add_oxygens_to_molecule(mol, noc) for mol in mol_alkanes_ethers] 
    cho_smiles = [smiles for smiles_set in cho_smiles for smiles in smiles_set]  # flatten list of lists
    print("Done adding oxygens to molecules.")
    print("Adding ethers and epoxides to molecules...")
    epoxides_smiles = gen_epoxides_smiles(mol_alkanes, noc)
    cho_smiles += epoxides_smiles + ethers_smiles
    print("Done adding ethers and epoxides to molecules.")
    # 3) Add ethers to each molecule
    relev_species = ['CO', 'C(O)O','O', 'OO', '[H][H]']
    all_cho_smiles = cho_smiles + relev_species
    all_smiles = all_cho_smiles + alkanes_smiles
    # Define bond types
    bond_types = [(6, 6), (6, 1), (6, 8), (8, 8), (1, 1), (8, 1)]
    
    # # Dictionary to keep track of processed fragments
    processed_fragments = {}
    # Process each molecule in the list
    for smiles in all_smiles:
        process_molecule(smiles, bond_types, processed_fragments)
    
    # Converting the dictionary to a list
    frag_list = []
    for value in processed_fragments.values():
        frag_list += value
    
    # Adding explicit Hs to the smiles of relevant species
    frag_list = list(set(frag_list + cho_smiles + alkanes_smiles))
    frag_list = [Chem.MolFromSmiles(smiles) for smiles in frag_list]
    # Saving the intermediates in a dictionary, where the key is the smiles of the molecule and 
    # the values are the rdkit molecules
    relev_species_mol = [Chem.MolFromSmiles(smiles) for smiles in relev_species]
    cho_mol = [Chem.MolFromSmiles(smiles) for smiles in cho_smiles]
    all_mol_list = list(set(frag_list + cho_mol + mol_alkanes + relev_species_mol))

    # Generating a dictionary of intermediates: key is the InChIKey and value is the rdkit molecule
    intermediates_dict = {}
    for mol in all_mol_list:
        # Generating the InChIKey for the molecule
        inchikey = Chem.inchi.MolToInchiKey(mol)
        smiles = Chem.MolToSmiles(mol, isomericSmiles=True, allHsExplicit=True)
        intermediates_dict[inchikey] = mol
    print("Total number of intermediates:", len(frag_list))
    print("Total number of bond-breaking reactions:",len(unique_reactions))

    # Generate the Intermediate objects
    intermediates_class_dict = gen_inter_objs(intermediates_dict)

    surf_inter = Intermediate.from_molecule(Atoms(), code='*', is_surface=True, phase='surf')

    rxns_list = []
    for reaction in unique_reactions:
        # Converting the smiles to InChIKey
        reactant_inchikey = Chem.inchi.MolToInchiKey(Chem.MolFromSmiles(reaction[0]))
        reactant = intermediates_class_dict[reactant_inchikey + '*']
        if len(reaction[1]) == 2:
            product1_inchikey = Chem.inchi.MolToInchiKey(Chem.MolFromSmiles(reaction[1][0]))
            product2_inchikey = Chem.inchi.MolToInchiKey(Chem.MolFromSmiles(reaction[1][1]))
            # Getting the Intermediate objects from the dictionary
            product1 = intermediates_class_dict[product1_inchikey + '*']
            product2 = intermediates_class_dict[product2_inchikey + '*']
            reaction_components = [[surf_inter, reactant], [product1, product2]]
        else:
            product1_inchikey = Chem.inchi.MolToInchiKey(Chem.MolFromSmiles(reaction[1][0]))
            product1 = intermediates_class_dict[product1_inchikey + '*']
            reaction_components = [[reactant], [product1]]
        rxns_list.append(ElementaryReaction(components=reaction_components, r_type=reaction[2]))

    ads_steps = gen_adsorption_reactions(intermediates_class_dict, surf_inter)
    rxns_list.extend(ads_steps)
    print("Adsorption steps: {}".format(len(ads_steps)))
    rearr_steps = gen_rearrangement_reactions(intermediates_class_dict)
    print("Rearrangement steps: {}".format(len(rearr_steps)))
    rxns_list.extend(rearr_steps)

    return intermediates_class_dict, rxns_list

def are_same_isomer(mol1_smiles, mol2_smiles):

    # Saturating the molecules
    mol1_smiles = re.sub("[H]([0-9]){0,1}",'',mol1_smiles)
    mol2_smiles = re.sub("[H]([0-9]){0,1}",'',mol2_smiles)
    mol1_smiles = mol1_smiles.replace('[', '').replace(']', '')
    mol2_smiles = mol2_smiles.replace('[', '').replace(']', '')
    
    mol1_sat = Chem.MolFromSmiles(mol1_smiles)
    mol2_sat = Chem.MolFromSmiles(mol2_smiles)

    # Check if molecular formulas are the same
    formula1 = rdMolDescriptors.CalcMolFormula(mol1_sat)
    formula2 = rdMolDescriptors.CalcMolFormula(mol2_sat)

    if formula1 != formula2:
        return False
    
    # Generate canonical SMILES without explicit hydrogens
    smiles1 = Chem.MolToSmiles(mol1_sat, canonical=True)
    smiles2 = Chem.MolToSmiles(mol2_sat, canonical=True)

    if smiles1 != smiles2:
        return False
    elif smiles1 == smiles2 and formula1 == formula2:
        return True


def is_hydrogen_rearranged(smiles_1: str, smiles_2: str):
    """Check for two molecules if there are potential hydrogen rearrangements."""

    re_var = "(\(?\[?[C,O][H]?[0-9]{0,1}\]?\)?[0-9]{0,1})"
    chpped_smiles_1 = re.findall(re_var, smiles_1)
    chpped_smiles_2 = re.findall(re_var, smiles_2)

    check_list = []
    for idx1, block1 in enumerate(chpped_smiles_1):
        for idx2, block2 in enumerate(chpped_smiles_2):
            if idx1 == idx2:
                if block1 == block2:
                    check_list.append(True)
                else:
                    check_list.append(False)

    # If there are only two False and they are together, then it is a hydrogen rearrangement
    if check_list.count(False) == 2:
        # Checking if the False are neighbors
        for i in range(len(check_list) - 1):
            if check_list[i] == False and check_list[i + 1] == False:
                return True
            else:
                # If there is one block between the two False, check if it contains "(" character, if yes, then it is a hydrogen rearrangement
                if check_list[i] == False and check_list[i + 1] == True:
                    if '(' and ')' in chpped_smiles_1[i + 1] or '(' and ')' in chpped_smiles_2[i + 1]:
                        # Checking the next block
                        if i + 2 < len(check_list):
                            if "(" and ")" in chpped_smiles_1[i + 2] or "(" and ")" in chpped_smiles_2[i + 2]:
                                # Checking the next block
                                if i + 3 < len(check_list):
                                    if check_list[i + 3] == False:
                                        return True
                                    else:
                                        return False
                                else:
                                    return False
                            else:
                                #Check if that block is False
                                if check_list[i + 2] == False:
                                    return True
                                else:
                                    return False
                        else:
                            return False
                    else:
                        return False
    return False
    
def gen_rearrangement_reactions(intermediates: dict[str, Intermediate]) -> list[ElementaryReaction]:
    """
    Generate all 1,2-rearrangement reactions involving hydrogen atoms.

    Parameters
    ----------
    intermediates : dict[str, Intermediate]
        Dictionary containing the Intermediate instances of all the chemical species of the reaction network.

    Returns
    -------
    list[ElementaryReaction]
        List of all 1,2-rearrangement reactions involving hydrogen atoms.
    """
    from ase.visualize import view
    ads_inters = [inter for inter in intermediates.values() if inter.phase == 'ads']
    
    # Checking for constitutional isomers
    counter = 0
    rearrangement_rxns = []
    for inter1, inter2 in combinations(ads_inters, 2):
        smiles1 = Chem.MolToSmiles(inter1.rdkit)
        smiles2 = Chem.MolToSmiles(inter2.rdkit)
        if are_same_isomer(Chem.MolToSmiles(inter1.rdkit), Chem.MolToSmiles(inter2.rdkit)):
            # Checking if the formula of each molecule is the same
            formula1 = inter1.molecule.get_chemical_formula()
            formula2 = inter2.molecule.get_chemical_formula()
            if formula1 == formula2:
                if is_hydrogen_rearranged(smiles1, smiles2):
                    counter += 1
                    rearrangement_rxns.append(ElementaryReaction(components=(frozenset([inter1]), frozenset([inter2])), r_type='rearrangement'))
    return rearrangement_rxns