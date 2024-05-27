import multiprocessing as mp
import time
import warnings
from rich.progress import Progress

from ase import Atoms
from prettytable import PrettyTable
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors

from care import ElementaryReaction, Intermediate
from care.crn.templates import adsorption, pcet, rearrengement, dissociation
from care.crn.utils.species import (add_oxygens, gen_alkanes, gen_epoxides,
                                        gen_ethers)


warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")


def format_description(description, width=45):
    """Format the progress bar description to a fixed width."""
    return description.ljust(width)[:width]


def is_desired_bond(bond: Chem.rdchem.Bond, z1: int, z2: int) -> bool:
    """
    Check if the bond is between the desired atom types

    Parameters
    ----------
    bond : rdkit.Chem.rdchem.Bond
        The bond to check
    z1 : int
        The atomic number of the first atom
    z2 : int
        The atomic number of the second atom

    Returns
    -------
    bool
        True if the bond is between the desired atom types, False otherwise
    """

    return (
        bond.GetBeginAtom().GetAtomicNum() == z1
        and bond.GetEndAtom().GetAtomicNum() == z2
    ) or (
        bond.GetBeginAtom().GetAtomicNum() == z2
        and bond.GetEndAtom().GetAtomicNum() == z1
    )


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
    Chem.AssignStereochemistry(
        mol, cleanIt=True, force=True, flagPossibleStereoCenters=True
    )

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


def break_bonds(
    molecule: Chem.rdchem.Mol,
    bond_types: list[tuple[int, int]],
    processed_fragments: dict,
    original_smiles: str,
    unique_reactions,
    processed_molecules,
) -> None:
    """
    Recursively break bonds in a molecule and filter unique reactions
    and fragments.
    The function is recursive, and will break all the bonds of the 
    desired types in the molecule, and then break all the bonds in the fragments, etc.

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

    current_smiles = Chem.MolToSmiles(molecule,
                                      isomericSmiles=True,
                                      allHsExplicit=True)

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
                    frag_smiles = Chem.MolToSmiles(
                        frag, isomericSmiles=True, allHsExplicit=True
                    )
                    frag_smiles_list.append(frag_smiles)

                    if frag_smiles not in processed_fragments[original_smiles]:
                        processed_fragments[original_smiles].append(frag_smiles)

                    # Recursive call with the fragment as the new molecule
                    frag_mol = Chem.MolFromSmiles(frag_smiles, sanitize=False)
                    break_bonds(
                        frag_mol,
                        bond_types,
                        processed_fragments,
                        original_smiles,
                        unique_reactions,
                        processed_molecules,
                    )

                if len(frag_smiles_list) == 2:
                    # Correction for [HH] in the fragment smiles list
                    if frag_smiles_list[0] == "[HH]":
                        # Modify the fragment smiles list to have [H] instead of [HH]
                        frag_smiles_list[0] = "[H]"

                    if frag_smiles_list[1] == "[HH]":
                        # Modify the fragment smiles list to have [H] instead of [HH]
                        frag_smiles_list[1] = "[H]"

                reaction_tuple = (current_smiles, tuple(sorted(frag_smiles_list)))

                # Check if the reaction tuple is unique
                if reaction_tuple not in unique_reactions:
                    r_type_atoms = sorted(
                        [
                            Chem.Atom(atom_num1).GetSymbol(),
                            Chem.Atom(atom_num2).GetSymbol(),
                        ]
                    )

                    r_type = f"{r_type_atoms[0]}-{r_type_atoms[1]}"

                    # Extending the reaction tuple with the bond type
                    reaction_type_tuple = (reaction_tuple[0], reaction_tuple[1], r_type)
                    unique_reactions.add(reaction_type_tuple)
                    total_bond_counter += 1

        processed_molecules.add(current_smiles)


def process_molecule(
    smiles: str,
    bond_types: list[tuple[int, int]],
    processed_fragments: dict,
    unique_reactions,
    processed_molecules,
) -> None:
    """
    Process a molecule by breaking all the bonds 
    of the desired type recursively.

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

    original_smiles = Chem.MolToSmiles(
        molecule_with_H, isomericSmiles=True, allHsExplicit=True
    )
    if original_smiles not in processed_fragments:
        processed_fragments[original_smiles] = []

    break_bonds(
        molecule_with_H,
        bond_types,
        processed_fragments,
        original_smiles,
        unique_reactions,
        processed_molecules,
    )


def process_inter_objs_chunk(chunk, progress_queue):
    """
    Process a chunk of the inter_dict dictionary.

    Parameters
    ----------
    chunk : dict
        A subset of inter_dict with key-value pairs to process.

    Returns
    -------
    dict
        A dictionary with the generated Intermediate objects for the given chunk.
    """

    inter_class_dict_chunk = {}
    for key, value in chunk.items():
        code = key + "*"
        new_inter_ads = Intermediate(code=code, molecule=value, phase="ads")
        inter_class_dict_chunk[code] = new_inter_ads

        if new_inter_ads.closed_shell:  # closed-shell also appear in gas phase
            code = key + "g"
            new_inter_gas = Intermediate(code=code, molecule=value, phase="gas")
            inter_class_dict_chunk[code] = new_inter_gas
    progress_queue.put(1)

    return inter_class_dict_chunk


def gen_intermediates_dict(
    inter_dict: dict[str, Chem.rdchem.Mol]
) -> dict[str, Intermediate]:
    """
    Generate the Intermediate objects for all the chemical species of the reaction network as a dictionary.

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

    # Number of chunks equals the number of available CPU cores
    n_cores = mp.cpu_count()

    # Splitting the dictionary into chunks
    keys = list(inter_dict.keys())
    if len(keys) < n_cores:
        chunk_size = 1
    else:
        chunk_size = len(keys) // n_cores
    chunks = [
        dict(
            zip(
                keys[i : i + chunk_size],
                [inter_dict[key] for key in keys[i : i + chunk_size]],
            )
        )
        for i in range(0, len(keys), chunk_size)
    ]

    manager_inter_obj = mp.Manager()
    progress_queue_inter = manager_inter_obj.Queue()

    tasks = [(chunk, progress_queue_inter) for chunk in chunks]
    with mp.Pool(mp.cpu_count()) as pool:
        result_async = pool.starmap_async(process_inter_objs_chunk, tasks)
        with Progress() as progress:
            task_desc = format_description("[green]Processing Intermediate objects...")
            task = progress.add_task(task_desc, total=len(tasks))
            processed_items = 0

            while not result_async.ready():
                while not progress_queue_inter.empty():
                    progress_queue_inter.get()
                    processed_items += 1
                    progress.update(task, advance=1)


    # Combine the results from all chunks
    combined_result = {}
    for result in result_async.get():
        combined_result.update(result)

    return combined_result


def gen_chemical_space(
    ncc: int, 
    noc: int, 
    cyclic: bool, 
    additional_rxns: bool, 
    electro: bool) -> tuple[dict[str, Intermediate], list[ElementaryReaction]]:
    """
    Generate the CRN blueprint by applying 
    reaction templates to the chemical space defined by 
    the input parameters.

    Parameters
    ----------
    ncc : int
        Network Carbon Cutoff, maximum number of C atoms in the intermediates
    noc : int
        Network Oxygen Cutoff, Maximum number of O atoms in the intermediates.
        if is a negative number, then the noc is set to the max number of O atoms in the intermediates.
    cyclic : bool
        If True, generates cyclic compounds (epoxides).
    additional_rxns : bool
        If True, additional reactions are generated (rearrangement reactions).
    electro : bool
        If True, proton-coupled electron transfer reactions are generated.

    Returns
    -------
    intermediates_dict : dict[str, Intermediate]
        Dictionary of the Intermediate instances of all the chemical species of the reaction network.
        Each key is the InChIKey of a molecule, and each value is a list of Intermediate instances for that molecule.
    rxns_list : list[ElementaryReaction]
        List of all the reactions of the reaction network as ElementaryReaction instances.
    """
    noc = noc if noc >= 0 else ncc * 2 + 2
    if noc > ncc * 2 + 2:
        raise ValueError("The noc value cannot be greater than ncc * 2 + 2.")
    t00 = time.time()
    with Progress() as progress:
        task_desc = format_description("[green]Generating Chemical Space...")
        task = progress.add_task(task_desc, total=6)

        # Step 0: Generate relevant species
        relev_species = ["[C-]#[O+]", "C(=O)=O", "O", "O=O", "[H][H]"]
        progress.update(task, advance=1)

        # Step 1: Generate Alkanes
        alkanes_smiles, mol_alkanes = gen_alkanes(ncc)
        progress.update(task, advance=1)

        if noc > 0:
            relev_species =  ["C(=O)=O", "O", "O=O", "[H][H]"] if noc == 1 else ["O", "O=O", "[H][H]"]
            # Step 2: Generate Ethers
            ethers_smiles, mol_ethers = gen_ethers(mol_alkanes, noc)
            progress.update(task, advance=1)

            if cyclic:
                # Step 3: Generate Epoxides
                epox_smiles, mol_epox = gen_epoxides(mol_alkanes, noc)
                progress.update(task, advance=1)
            else:
                epox_smiles, mol_epox = [], []
                progress.update(task, advance=1)


            # Step 4: Add Oxygens
            alkanes_oxy_smiles = [add_oxygens(mol, noc) for mol in mol_alkanes]
            ethers_epox_oxy_smiles = [add_oxygens(mol, noc-1) for mol in mol_ethers+mol_epox]
            cho_smiles = alkanes_oxy_smiles + ethers_epox_oxy_smiles
            cho_smiles = [smiles for smiles_set in cho_smiles for smiles in smiles_set]
            progress.update(task, advance=1)

            # Step 5: Finalize the Species List
            cho_smiles += epox_smiles + ethers_smiles
            saturated_species_smiles = list(set(alkanes_smiles + cho_smiles + relev_species))
            progress.update(task, advance=1)

        else:
            # Step 2: Finalize the Species List
            saturated_species_smiles = alkanes_smiles + relev_species
            progress.update(task, advance=1)
    t0 = time.time() - t00

    t01 = time.time()
    # Define bond types (C-C, C-H, C-O, O-O, H-H, O-H)
    bond_types = [(6, 6), (6, 1), (6, 8), (8, 8), (1, 1), (8, 1)]

    # Dictionary to keep track of processed fragments
    processed_fragments, unique_reactions, processed_molecules = {}, set(), set()
    # Process each molecule in the list
    with Progress() as progress:
        task_desc = format_description("[green]Generating extended Chemical Space...")
        task = progress.add_task(task_desc, total=len(saturated_species_smiles))
        for smiles in saturated_species_smiles:
            process_molecule(
                smiles,
                bond_types,
                processed_fragments,
                unique_reactions,
                processed_molecules,
            )
            progress.update(task, advance=1)

    # Converting the dictionary to a list
    frag_list = []
    for value in processed_fragments.values():
        frag_list += value

    frag_list = list(set(frag_list))
    all_mol_list = [
        Chem.MolFromSmiles(smiles)
        for smiles in list(set(frag_list + saturated_species_smiles + relev_species))
    ]

    # Generating a dictionary where keys:InChIKeys and values: Chem.Mol molecule
    rdkit_inters_dict = {}
    for mol in all_mol_list:
        # Generating the InChIKey for the molecule
        inchikey = Chem.inchi.MolToInchiKey(mol)
        rdkit_inters_dict[inchikey] = mol
    t1 = time.time() - t01

    # Generate the Intermediate objects
    t02 = time.time()
    intermediates_dict = gen_intermediates_dict(rdkit_inters_dict)
    t2 = time.time() - t02
    surf_inter = Intermediate.from_molecule(
        Atoms(), code="*", is_surface=True, phase="surf"
    )

    t03 = time.time()
    rxns_list = []
    with Progress() as progress:
        task_desc = format_description("[green]Processing ElementaryReactions...")
        task = progress.add_task(task_desc, total=len(unique_reactions))

        for reaction in unique_reactions:
            reactant = intermediates_dict[
                Chem.inchi.MolToInchiKey(Chem.MolFromSmiles(reaction[0])) + "*"
            ]
            product1 = intermediates_dict[
                Chem.inchi.MolToInchiKey(Chem.MolFromSmiles(reaction[1][0])) + "*"
            ]

            if len(reaction[1]) == 2:
                product2 = intermediates_dict[
                    Chem.inchi.MolToInchiKey(Chem.MolFromSmiles(reaction[1][1])) + "*"
                ]
                reaction_components = [[surf_inter, reactant], [product1, product2]]
            else:
                reaction_components = [[reactant], [product1]]

            rxns_list.append(
                ElementaryReaction(components=reaction_components, r_type=reaction[2])
            )
            progress.update(task, advance=1)
    rxns_list = list(set(rxns_list)) # Temporary because of CO addition via SMILES
    t3 = time.time() - t03
    
    # Bond-breaking reactions
    # t03 = time.time()
    # dissociation_reactions = gen_dissociation_reactions(intermediates_dict)
    # reactions.extend(dissociation_reactions)
    # t3 = time.time() - t03
    
    # Adsorption/Desorption reactions
    t04 = time.time()
    ads_steps = adsorption.gen_adsorption_reactions(intermediates_dict)
    rxns_list.extend(ads_steps)
    t4 = time.time() - t04
    
    # (1,2)-H shift rearrangement reactions
    if additional_rxns:
        t05 = time.time()
        rearr_steps = rearrengement.gen_rearrangement_reactions(intermediates_dict)
        rxns_list.extend(rearr_steps)
        t5 = time.time() - t05
        
    # Proton-coupled electron transfer reactions        
    if electro:
        t06 = time.time()
        pcets = pcet.gen_pcet_reactions(intermediates_dict, rxns_list)
        rxns_list.extend(pcets)
        t6 = time.time() - t06

    # Create a table object
    table = PrettyTable()

    table.field_names = ["Category", "Number of Items", "Time (s)"]
    table.add_row(["Saturated molecules", len(saturated_species_smiles), f"{t0:.2f}"])
    table.add_row(["Fragments and unsaturated molecules", len(frag_list), f"{t1:.2f}"], divider=True)
    table.add_row(["Bond-breaking reactions", len(unique_reactions), f"{t1:.2f}"])
    table.add_row(["Adsorption reactions", len(ads_steps), f"{t4:.2f}"])
        
    if additional_rxns and electro:
        table.add_row(["Rearrangement reactions", len(rearr_steps), f"{t5:.2f}"])
        table.add_row(["PCET reactions", len(pcets), f"{t6:.2f}"], divider=True)
        table.add_row(
        ["Total number of species", len(intermediates_dict), f"{t0 + t1 + t2:.2f}"])
        table.add_row(["Total number of reactions", len(rxns_list), f"{t1 + t3 + t4 + t5+ t6:.2f}"])
    elif additional_rxns and not electro:
        table.add_row(["Rearrangement reactions", len(rearr_steps), f"{t5:.2f}"], divider=True)
        table.add_row(
        ["Total number of species", len(intermediates_dict), f"{t0 + t1 + t2:.2f}"])
        table.add_row(
            ["Total number of reactions", len(rxns_list), f"{t1 + t3 + t4 + t5:.2f}"]
        )
    elif not additional_rxns and electro:
        table.add_row(["PCET reactions", len(pcets), f"{t6:.2f}"], divider=True)
        table.add_row(
        ["Total number of species", len(intermediates_dict), f"{t0 + t1 + t2:.2f}"])
        table.add_row(
            ["Total number of reactions", len(rxns_list), f"{t1 + t3 + t4 + t6:.2f}"]
        )
    else:
        table.add_row(
        ["Total number of species", len(intermediates_dict), f"{t0 + t1 + t2:.2f}"])
        table.add_row(
            ["Total number of reactions", len(rxns_list), f"{t1 + t3 + t4:.2f}"]
        )

    print(f"\n{table}")
    
    return intermediates_dict, rxns_list
