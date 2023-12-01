import multiprocessing as mp
from itertools import combinations
import re 
import warnings
from rdkit import RDLogger
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from ase import Atoms
from collections import defaultdict
import numpy as np

from care.netgen.networks.intermediate import Intermediate
from care.netgen.networks.elementary_reaction import ElementaryReaction
from care.netgen.intermediates_funcs import gen_alkanes_smiles, add_oxygens_to_molecule, gen_ether_smiles, gen_epoxides_smiles

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

# def process_reaction(reaction, intermediates_class_dict, surf_inter):
#     # Converting the smiles to InChIKey
#     reactant_inchikey = Chem.inchi.MolToInchiKey(Chem.MolFromSmiles(reaction[0]))
#     reactant = intermediates_class_dict[reactant_inchikey + '*']

#     if len(reaction[1]) == 2:
#         # Handling the case where there are two products
#         product1_inchikey = Chem.inchi.MolToInchiKey(Chem.MolFromSmiles(reaction[1][0]))
#         product2_inchikey = Chem.inchi.MolToInchiKey(Chem.MolFromSmiles(reaction[1][1]))

#         product1 = intermediates_class_dict[product1_inchikey + '*']
#         product2 = intermediates_class_dict[product2_inchikey + '*']

#         reaction_components = [[surf_inter, reactant], [product1, product2]]
#     else:
#         # Handling the case where there is one product
#         product1_inchikey = Chem.inchi.MolToInchiKey(Chem.MolFromSmiles(reaction[1][0]))
#         product1 = intermediates_class_dict[product1_inchikey + '*']

#         reaction_components = [[reactant], [product1]]

#     return ElementaryReaction(components=reaction_components, r_type=reaction[2])

# def generate_rxns_list(unique_reactions, intermediates_class_dict, surf_inter, ncores):
#     with mp.Pool(ncores) as pool:
#         rxns_list = pool.starmap(process_reaction, [(reaction, intermediates_class_dict, surf_inter) for reaction in unique_reactions])
#     return rxns_list

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
    

    ############ Closed-shell species ############
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
    ############ End of closed-shell species ############
    



    # Define bond types (C-C, C-H, C-O, O-O, H-H, O-H)
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

    # rxns_list = generate_rxns_list(unique_reactions, intermediates_class_dict, surf_inter, ncores)

    print('Generating adsorption and rearrangement steps...')
    ads_steps = gen_adsorption_reactions(intermediates_class_dict, surf_inter)
    rxns_list.extend(ads_steps)
    print("Adsorption steps: {}".format(len(ads_steps)))
    print('Finished generating adsorption and rearrangement steps.')
    
    print('Generating rearrangement steps...')
    rearr_steps = gen_rearrangement_reactions(intermediates_class_dict)
    rxns_list.extend(rearr_steps)
    print("Rearrangement steps: {}".format(len(rearr_steps)))
    print('Finished generating rearrangement steps.')

    return intermediates_class_dict, rxns_list





##############################################################################################################
def process_ads_react_chunk(inter_chunk, intermediates, surf_inter):
    adsorption_steps = []
    for inter_code in inter_chunk:
        inter = intermediates[inter_code]
        if inter.phase == 'gas':
            ads_inter = intermediates[inter.code[:-1] + '*']
            adsorption_steps.append(ElementaryReaction(components=(frozenset([surf_inter, inter]), frozenset([ads_inter])), r_type='adsorption'))

        # Add the dissociative adsorptions for H2 and O2
        for molecule in ['UFHFLCQGNIYNRP-UHFFFAOYSA-N', 'MYMOFIZGZYHOMD-UHFFFAOYSA-N']:
            gas_code = molecule + 'g'
            if molecule == 'UFHFLCQGNIYNRP-UHFFFAOYSA-N': # H2
                ads_code = 'YZCKVEUIGOORGS-UHFFFAOYSA-N*'
            else: # O2
                ads_code = 'QVGXLLKOCUKJST-UHFFFAOYSA-N*'
            
            adsorption_steps.append(ElementaryReaction(components=(frozenset([surf_inter, intermediates[gas_code]]), frozenset([intermediates[ads_code]])), r_type='adsorption'))

    return adsorption_steps

def gen_adsorption_reactions(intermediates, surf_inter, num_processes=4):
    # Split intermediates into chunks
    inter_chunks = np.array_split(list(intermediates.keys()), num_processes)

    # Create a pool of workers
    with mp.Pool(processes=num_processes) as pool:
        # Map process_chunk function to each chunk
        results = pool.starmap(process_ads_react_chunk, [(chunk, intermediates, surf_inter) for chunk in inter_chunks])

    # Flatten the list of lists
    adsorption_steps = list(set([step for sublist in results for step in sublist])) 
    return adsorption_steps


##############################################################################################################
def are_same_isomer(mol1_smiles: str, mol2_smiles: str) -> bool:
    """
    Check if two molecules are the same constitutional isomers.
    
    Parameters
    ----------
    mol1_smiles : str
        The SMILES string of the first molecule
    mol2_smiles : str
        The SMILES string of the second molecule

    Returns
    -------
    bool
        True if the molecules are the same constitutional isomers, False otherwise
    """

    # Saturating the molecules
    mol1_smiles = re.sub(r"[H]([0-9]){0,1}",'',mol1_smiles)
    mol2_smiles = re.sub(r"[H]([0-9]){0,1}",'',mol2_smiles)
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

def is_hydrogen_rearranged(smiles_1: str, smiles_2: str) -> bool:
    """
    Check for two molecules if there are potential hydrogen rearrangements.
    
    Parameters
    ----------
    smiles_1 : str
        The SMILES string of the first molecule
    smiles_2 : str
        The SMILES string of the second molecule

    Returns
    -------
    bool
        True if there are potential hydrogen rearrangements, False otherwise
    """

    # Condition for splitting the smiles string
    re_var = r"(\(?\[?[C,O][H]?[0-9]{0,1}\]?\)?[0-9]{0,1})"

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
    
def check_rearrangement(pair):
    inter1, inter2 = pair
    smiles1 = Chem.MolToSmiles(inter1.rdkit)
    smiles2 = Chem.MolToSmiles(inter2.rdkit)
    # if are_same_isomer(smiles1, smiles2):
    if is_hydrogen_rearranged(smiles1, smiles2):
        return ElementaryReaction(components=(frozenset([inter1]), frozenset([inter2])), r_type='rearrangement')
    return None

# def gen_rearrangement_reactions(intermediates):
#     ads_inters = [inter for inter in intermediates.values() if inter.phase == 'ads']

#     # Group intermediates by chemical formula
#     formula_groups = defaultdict(list)
#     for inter in ads_inters:
#         formula_groups[inter.molecule.get_chemical_formula()].append(inter)

#     # Generate combinations for each formula group
#     pairs = []
#     for group in formula_groups.values():
#         pairs.extend(combinations(group, 2))

#     with mp.Pool() as pool:
#         results = pool.map(check_rearrangement, pairs)

#     # Filter out None values from the results
#     rearrangement_rxns = [result for result in results if result is not None]

#     return rearrangement_rxns

def group_by_formula(intermediates):
    formula_groups = defaultdict(list)
    for inter in intermediates:
        formula_groups[inter.molecule.get_chemical_formula()].append(inter)
    return formula_groups

def subgroup_by_isomers(intermediates):
    isomer_groups = []
    for inter in intermediates:
        found_group = False
        smiles_inter = Chem.MolToSmiles(inter.rdkit)

        for group in isomer_groups:
            representative = group[0]
            smiles_rep = Chem.MolToSmiles(representative.rdkit)
            if are_same_isomer(smiles_inter, smiles_rep):
                group.append(inter)
                found_group = True
                break

        if not found_group:
            isomer_groups.append([inter])

    return isomer_groups

def gen_rearrangement_reactions(intermediates):
    ads_inters = [inter for inter in intermediates.values() if inter.phase == 'ads']

    # Group intermediates by chemical formula
    formula_groups = group_by_formula(ads_inters)

    pairs = []
    for formula_group in formula_groups.values():
        # Subgroup each formula group by isomers
        isomer_subgroups = subgroup_by_isomers(formula_group)
        
        # Generate combinations within each isomer subgroup
        for subgroup in isomer_subgroups:
            pairs.extend(combinations(subgroup, 2))

    with mp.Pool() as pool:
        results = pool.map(check_rearrangement, pairs)

    # Filter out None values
    rearrangement_rxns = [result for result in results if result is not None]

    return rearrangement_rxns