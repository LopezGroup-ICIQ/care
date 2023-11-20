
from rdkit.Chem import rdMolDescriptors
from rdkit import Chem
from ase import Atoms
from ase.neighborlist import neighbor_list
from itertools import combinations
from collections import defaultdict
import multiprocessing as mp

from care.rnet.networks.utils import Intermediate, ElementaryReaction
from care.rnet.intermediates_funcs import gen_alkanes_smiles, add_oxygens_to_molecule, gen_ether_smiles, process_molecule, gen_epoxides_smiles
from care.rnet.networks.utils import gen_adsorption_reactions, gen_rearrangement_reactions

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
processed_molecules = set()
def break_bonds(molecule: Chem.rdchem.Mol, bond_types: list[tuple[int, int]], processed_fragments: dict, original_smiles: str):
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

    return total_bond_counter

def process_molecule(smiles, bond_types, processed_fragments):
    molecule = Chem.MolFromSmiles(smiles)
    molecule_with_H = Chem.AddHs(molecule)

    original_smiles = Chem.MolToSmiles(molecule_with_H, isomericSmiles=True, allHsExplicit=True)
    if original_smiles not in processed_fragments:
        processed_fragments[original_smiles] = []

    n_react = break_bonds(molecule_with_H, bond_types, processed_fragments, original_smiles)
    return n_react

def gen_inter_objs(inter_dict: dict[str, Chem.rdchem.Mol]) -> dict:
    """
    Generates a dictionary where each key corresponds to an initial key from the input dictionary,
    and each value is a list of Intermediate instances (ads and gas) for that key.

    Parameters
    ----------
    inter_dict : dict
        Dictionary containing the intermediates of the reaction network.
        Each key is the SMILES of a saturated CHO molecule, and each value is a dictionary
        containing the intermediates of the reaction network for that molecule.

    Returns
    -------
    network_dict: dict
        Dictionary where each key maps to a list containing the Intermediate instances of the 
        chemical species of the reaction network.
    """
    network_dict = {}
    for key, value in inter_dict.items():
        intermediates_list = []

        new_inter_ads = Intermediate(code=key+'*',
                                     molecule=value,
                                     phase='ads')
        intermediates_list.append(new_inter_ads)

        if new_inter_ads.closed_shell:
            new_inter_gas = Intermediate(code=key+'g',
                                         molecule=value,
                                         phase='gas')
            intermediates_list.append(new_inter_gas)

        network_dict[key] = intermediates_list

    return network_dict

def is_hydrogen_rearranged(molecule_1: Atoms, molecule_2: Atoms):
    """
    Check for two molecules if there are potential hydrogen rearrangements.

    Parameters
    ----------
    molecule_1 : Atoms
        First molecule.
    molecule_2 : Atoms
        Second molecule.

    Returns
    -------
    bool
        True if there is a potential hydrogen rearrangement between the two molecules, False otherwise.
    """
    # Get indices of C, H, and O atoms
    c_indices = [atom.index for atom in molecule_1 if atom.symbol == 'C']
    h_indices = [atom.index for atom in molecule_1 if atom.symbol == 'H']
    o_indices = [atom.index for atom in molecule_1 if atom.symbol == 'O']
   
    # Defining cutoff for neighbor list
    cutoff = {('H', 'H'): 1.1, ('C', 'H'): 1.3, ('C', 'C'): 1.85, ('C', 'O'): 1.5, ('O', 'O'): 1.5, ('O', 'H'): 1.3}
    # Calculate neighbor lists for each Atoms object
    n_list_1 = neighbor_list('ijS', molecule_1, cutoff=cutoff)
    n_list_2 = neighbor_list('ijS', molecule_2, cutoff=cutoff)
   
    # Convert neighbor lists to sets of tuples (atom index, neighbor index)
    bonds_1 = set(zip(n_list_1[0], n_list_1[1]))
    bonds_2 = set(zip(n_list_2[0], n_list_2[1]))
   
    # Check if the connectivity for C and O atoms is the same
    if not all((c, o) in bonds_2 for c in c_indices for o in o_indices if (c, o) in bonds_1):
        return False
   
    # Check for rearrangement of H atoms
    rearranged_h = []
    for h_index in h_indices:
        # Neighbors in both Atoms objects
        neighbors_1 = {j for i, j in bonds_1 if i == h_index}
        neighbors_2 = {j for i, j in bonds_2 if i == h_index}
       
        # If the H atom has different neighbors, and the different neighbor is a C or O atom
        if neighbors_1 != neighbors_2 and any((n in c_indices or n in o_indices) for n in neighbors_1.symmetric_difference(neighbors_2)):
            rearranged_h.append(h_index)
   
    # Only one H atom should have different connectivity to be a rearrangement reaction
    return len(rearranged_h) == 1


def gen_adsorption_reactions(intermediates: dict[str, list[Intermediate]], surf_inter: Intermediate) -> list[ElementaryReaction]:
    """
    Generate all Intermediates that can desorb from the surface and the corresponding desorption reactions, including
    the dissociative adsorption of H2 and O2.

    Parameters
    ----------
    intermediates : dict[str, Intermediate]
        Dictionary containing the Intermediate instances of all the chemical species of the reaction network.

    Returns
    -------
    adsorption_steps: list[ElementaryReaction]
        List of all desorption reactions.
    """

    adsorption_steps = []
    for list_inters in intermediates.values():
        for inter in list_inters:
            if inter.phase == 'gas':
                ads_inter = intermediates[inter.code[:-1]][0]
                # stoic_dict = {surf_inter.code: -1, inter.code: -1, ads_inter.code: 1}     
                adsorption_steps.append(ElementaryReaction(components=(frozenset([surf_inter, inter]), frozenset([ads_inter])), r_type='adsorption'))

    # dissociative adsorptions for H2 and O2
    for molecule in ['UFHFLCQGNIYNRP-UHFFFAOYSA-N', 'MYMOFIZGZYHOMD-UHFFFAOYSA-N']:
        if molecule == 'UFHFLCQGNIYNRP-UHFFFAOYSA-N': #H2
            ads_code = 'YZCKVEUIGOORGS-UHFFFAOYSA-N'
        else: #O2
            ads_code = 'QVGXLLKOCUKJST-UHFFFAOYSA-N'
        # stoic_dict = {surf_inter.code: -2, intermediates[molecule][1].code: -1, intermediates[molecule.replace("2", "1")+'*'].code: 2}
        adsorption_steps.append(ElementaryReaction(components=(frozenset([surf_inter, intermediates[molecule][1]]), frozenset([intermediates[ads_code][0]])), r_type='adsorption'))
    return adsorption_steps

# def gen_rearrangement_reactions(intermediates: dict[str, Intermediate]) -> list[ElementaryReaction]:
#     """
#     Generate all 1,2-rearrangement reactions involving hydrogen atoms.

#     Parameters
#     ----------
#     intermediates : dict[str, Intermediate]
#         Dictionary containing the Intermediate instances of all the chemical species of the reaction network.

#     Returns
#     -------
#     rearrengement_steps: list[ElementaryReaction]
#         List of all 1,2-rearrangement reactions involving hydrogen atoms.
#     """
#     rearrangement_steps = []
#     inter_dict = defaultdict(list)
#     for list_inters in intermediates.values():
#         for inter in list_inters:
#             if inter.is_surface or inter.phase == 'gas':
#                 continue
#             inter_dict[inter.code[:8]].append(inter)
            
#     for value in inter_dict.values():
#         if len(value) == 1:
#             continue
#         rearrangement_pairs = [[inter_1, inter_2] for inter_1, inter_2 in combinations(value, 2) if is_hydrogen_rearranged(inter_1.molecule, inter_2.molecule)]
#         if rearrangement_pairs:
#             for pair in rearrangement_pairs:
#                 code_1 = pair[0].code
#                 code_2 = pair[1].code
#             stoic_dict = {code_1: -1, code_2: 1}            
#             rearrangement_steps.append(ElementaryReaction(components=(frozenset([intermediates[code_1]]), frozenset([intermediates[code_2]])), r_type='rearrangement', stoic=stoic_dict))
#     return rearrangement_steps

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
    ethers_smiles = gen_ether_smiles(mol_alkanes, noc)
    # epoxides_smiles = gen_epoxides_smiles(mol_alkanes, noc)
    # cho_smiles += epoxides_smiles + ethers_smiles
    cho_smiles += ethers_smiles
    print("Done adding oxygens to molecules.")
    # 3) Add ethers to each molecule
    print("Adding ethers to molecules...")
    # cho_smiles += epoxides_smiles
    # cho_smiles += ethers_smiles
    print("Done adding ethers to molecules.")
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
    for key, value in processed_fragments.items():
        frag_list += value
    
    # Adding explicit Hs to the smiles of relevant species
    frag_list = list(set(frag_list + cho_smiles + alkanes_smiles))
    frag_list = [Chem.MolFromSmiles(smiles) for smiles in frag_list]
    # Saving the intermediates in a dictionary, where the key is the smiles of the molecule and 
    # the values are the rdkit molecules
    relev_species_mol = [Chem.MolFromSmiles(smiles) for smiles in relev_species]
    cho_mol = [Chem.MolFromSmiles(smiles) for smiles in cho_smiles]
    all_mol_list = list(set(frag_list + cho_mol + mol_alkanes + relev_species_mol))

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
        product1_inchikey = Chem.inchi.MolToInchiKey(Chem.MolFromSmiles(reaction[1][0]))
        product2_inchikey = Chem.inchi.MolToInchiKey(Chem.MolFromSmiles(reaction[1][1]))
        # Getting the Intermediate objects from the dictionary
        reactant = intermediates_class_dict[reactant_inchikey][0]
        product1 = intermediates_class_dict[product1_inchikey][0]
        product2 = intermediates_class_dict[product2_inchikey][0]

        reaction_components = [[surf_inter, reactant], [product1, product2]]
        rxns_list.append(ElementaryReaction(components=reaction_components, r_type=reaction[2]))

    ads_steps = gen_adsorption_reactions(intermediates_class_dict, surf_inter)
    rxns_list.extend(ads_steps)
    print("Adsorption steps: {}".format(len(ads_steps)))
    # rearr_steps = gen_rearrangement_reactions(intermediates_class_dict)
    # rxns_list.extend(rearr_steps)

    return intermediates_class_dict, rxns_list
    



