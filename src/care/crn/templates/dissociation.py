"""Bond-breaking template"""

import multiprocessing as mp

from ase import Atoms
from rich.progress import Progress
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, MolFromSmiles, MolToSmiles
from rdkit.Chem.inchi import MolToInchiKey

from care import ElementaryReaction, Intermediate
from care.constants import BOND_TYPES


class BondBreaking(ElementaryReaction):
    """Class for bond-breaking reactions."""

    def __init__(self, components, r_type):
        super().__init__(components=components, r_type=r_type)

    def reverse(self):
        self.__class__ = BondFormation
        self.components = self.components[::-1]
        for k, v in self.stoic.items():
            self.stoic[k] = -v
        if self.e_rxn:
            self.e_rxn = -self.e_rxn[0], self.e_rxn[1]
            self.e_is, self.e_fs = self.e_fs, self.e_is

        if self.e_act:
            self.e_act = (
                self.e_act[0] + self.e_rxn[0],
                (self.e_act[1] ** 2 + self.e_rxn[1] ** 2) ** 0.5,
            )

    def bb_order(self):
        """
        Set the elementary reaction in the bond-breaking direction, e.g.:
        CH4 + * -> CH3 + H*
        """
        pass


class BondFormation(ElementaryReaction):
    """Class for bond-formation reactions."""

    def __init__(self, components, r_type):
        super().__init__(components=components, r_type=r_type)

    def reverse(self):
        self.__class__ = BondBreaking
        self.components = self.components[::-1]
        for k, v in self.stoic.items():
            self.stoic[k] = -v
        if self.e_rxn:
            self.e_rxn = -self.e_rxn[0], self.e_rxn[1]
            self.e_is, self.e_fs = self.e_fs, self.e_is

        if self.e_act:
            self.e_act = (
                self.e_act[0] + self.e_rxn[0],
                (self.e_act[1] ** 2 + self.e_rxn[1] ** 2) ** 0.5,
            )

    def bb_order(self):
        """
        Set the elementary reaction in the bond-breaking direction, e.g.:
        CH4 + * -> CH3 + H*
        """
        self.reverse()


def dissociate(
    chemical_space: list[str], ncpus: int = mp.cpu_count(), show_progress: bool = False
) -> tuple[dict[str, Intermediate], list[BondBreaking]]:
    """
    Generate all potential dissociation reactions and fragments given an initial set of molecules.

    Parameters
    ----------
    chemical_space : list[str]
        List of the SMILES of the molecules in the chemical space.
    ncpus : int
        Number of CPUs to use for the generation of the Intermediate objects.
        Default is the number of CPUs available.
    show_progress : bool
        Whether to show the progress bar, useful when running the function in a script
        or for huge chemical spaces. Default is False.

    Returns:
    --------
    inters : dict[str, Intermediate]
        Dictionary with Intermediate instances produced by the bond-breaking template.
            Key: InChIKey of the gas molecule plus '*' or 'g' defining if its phase (adsorbed or gas-phase).
    rxns : list[ElementaryReaction]
        List of the dissociation reactions of the reaction network as ElementaryReaction instances.
    """

    processed_fragments, unique_reactions, processed_molecules = {}, set(), set()

    if show_progress:
        with Progress() as progress:
            task_desc = format_description("[green]Generating extended Chemical Space...")
            task = progress.add_task(task_desc, total=len(chemical_space))
            for smiles in chemical_space:
                process_molecule(
                    smiles,
                    processed_fragments,
                    unique_reactions,
                    processed_molecules,
                )
                progress.update(task, advance=1)
    else:
        for smiles in chemical_space:
            process_molecule(
                smiles, processed_fragments, unique_reactions, processed_molecules
            )

    # Converting the dictionary to a list
    frag_list = []
    for value in processed_fragments.values():
        frag_list += value

    frag_list = list(set(frag_list))
    all_mol_list = [
        MolFromSmiles(smiles) for smiles in list(set(frag_list + chemical_space))
    ]

    # Generate the Intermediate objects
    rdkit_inters = {MolToInchiKey(mol): mol for mol in all_mol_list}
    inters = gen_intermediates_dict(rdkit_inters, ncpus, show_progress)
    active_site = Intermediate(code="*", molecule=Atoms(), is_surface=True, phase="surf")
    rxns = []

    if show_progress:
        with Progress() as progress:
            task_desc = format_description("[green]Processing ElementaryReactions...")
            task = progress.add_task(task_desc, total=len(unique_reactions))

            for reaction in unique_reactions:
                reactant = inters[MolToInchiKey(MolFromSmiles(reaction[0])) + "*"]
                product1 = inters[MolToInchiKey(MolFromSmiles(reaction[1][0])) + "*"]

                if len(reaction[1]) == 2:
                    product2 = inters[MolToInchiKey(MolFromSmiles(reaction[1][1])) + "*"]
                    reaction_components = [[active_site, reactant], [product1, product2]]
                else:
                    reaction_components = [[reactant], [product1]]

                rxns.append(
                    BondBreaking(components=reaction_components, r_type=reaction[2])
                )
                progress.update(task, advance=1)
    else:
        for reaction in unique_reactions:
            reactant = inters[MolToInchiKey(MolFromSmiles(reaction[0])) + "*"]
            product1 = inters[MolToInchiKey(MolFromSmiles(reaction[1][0])) + "*"]

            if len(reaction[1]) == 2:
                product2 = inters[MolToInchiKey(MolFromSmiles(reaction[1][1])) + "*"]
                reaction_components = [[active_site, reactant], [product1, product2]]
            else:
                reaction_components = [[reactant], [product1]]

            rxns.append(
                BondBreaking(components=reaction_components, r_type=reaction[2])
            )

    return inters, rxns


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


def smiles2formula(smiles: str) -> str:
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

    mol = MolFromSmiles(smiles, sanitize=False)
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


def process_molecule(
    smiles: str,
    processed_fragments: dict[str, list[list[str]]],
    unique_reactions: set[tuple[str, tuple[str], str]],
    processed_molecules: set[str],
) -> None:
    """
    Process a molecule by breaking all the bonds
    of the desired type recursively.

    Parameters
    ----------
    smiles : str
        The SMILES string of the molecule
    processed_fragments : dict
        Dictionary to keep track of processed fragments

    Returns
    -------
    None
        All the reactions are added to the unique reactions set
        , and all the processed fragments are added to the
        processed fragments dictionary
    """

    molecule = MolFromSmiles(smiles)
    molecule_with_H = Chem.AddHs(molecule)

    original_smiles = MolToSmiles(
        molecule_with_H, isomericSmiles=True, allHsExplicit=True
    )
    if original_smiles not in processed_fragments:
        processed_fragments[original_smiles] = []

    break_bonds(
        molecule_with_H,
        processed_fragments,
        original_smiles,
        unique_reactions,
        processed_molecules,
    )


def break_bonds(
    molecule: Chem.rdchem.Mol,
    processed_fragments: dict[str, list[list[str]]],
    original_smiles: str,
    unique_reactions: set[tuple[str, tuple[str], str]],
    processed_molecules: set[str],
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
        All the reactions are added to the unique reactions set,
        and all the processed fragments are added to the processed fragments dictionary
    """

    current_smiles = MolToSmiles(molecule, isomericSmiles=True, allHsExplicit=True)

    if current_smiles in processed_molecules:
        return 0

    unique_bonds = find_unique_bonds(molecule)

    total_bond_counter = 0
    for bond in unique_bonds:
        for Z_atom1, Z_atom2 in BOND_TYPES:
            if is_desired_bond(bond, Z_atom1, Z_atom2):
                mol_copy = Chem.RWMol(molecule)
                mol_copy.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

                frags = Chem.GetMolFrags(mol_copy, asMols=True, sanitizeFrags=False)
                frag_smiles_list = []
                for frag in frags:
                    frag_smiles = MolToSmiles(
                        frag, isomericSmiles=True, allHsExplicit=True
                    )
                    frag_smiles_list.append(frag_smiles)

                    if frag_smiles not in processed_fragments[original_smiles]:
                        processed_fragments[original_smiles].append(frag_smiles)

                    # Recursive call with the fragment as the new molecule
                    frag_mol = MolFromSmiles(frag_smiles, sanitize=False)
                    break_bonds(
                        frag_mol,
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

                rxn_tuple = (current_smiles, tuple(sorted(frag_smiles_list)))

                # Check if the reaction tuple is unique
                if rxn_tuple not in unique_reactions:
                    bbtype = sorted(
                        [
                            Chem.Atom(Z_atom1).GetSymbol(),
                            Chem.Atom(Z_atom2).GetSymbol(),
                        ]
                    )

                    unique_reactions.add(
                        (rxn_tuple[0], rxn_tuple[1], f"{bbtype[0]}-{bbtype[1]}")
                    )
                    total_bond_counter += 1

        processed_molecules.add(current_smiles)


def gen_intermediates_dict(
    inter_dict: dict[str, Chem.rdchem.Mol],
    ncpu: int=mp.cpu_count(),
    show_progress: bool=False
) -> dict[str, Intermediate]:
    """
    Generate the Intermediate objects for all the chemical species of the reaction network as a dictionary.

    Parameters
    ----------
    inter_dict : dict[str, Chem.rdchem.Mol]
        Dictionary containing the Chem.rdchem.Mol instances
        of all the chemical species of the reaction network.
        Each key is the InChIKey of a molecule,
        and each value is the corresponding Chem.rdchem.Mol instance.
    ncpu : int
        Number of CPUs to use for the generation of the Intermediate objects.
        Default is the number of CPUs available.
    show_progress : bool
        Whether to show the progress bar, useful when running the function in a script
        or for huge chemical spaces. Default is False.

    Returns
    -------
    intermediate_class_dict : dict[str, Intermediate]
        Dictionary containing the Intermediate instances
        of all the chemical species of the reaction network.
        Each key is the InChIKey of the molecule plus '*' or 'g'
        defining if its adsorbed or in gas-phase,
        and each value the Intermediate instance.
    """
    keys = list(inter_dict.keys())
    chunk_size = 1 if len(keys) < ncpu else len(keys) // ncpu
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
    with mp.Pool(ncpu) as pool:
        result_async = pool.starmap_async(process_inter_objs_chunk, tasks)
        if show_progress:
            with Progress() as progress:
                task_desc = format_description("[green]Processing Intermediate objects...")
                task = progress.add_task(task_desc, total=len(tasks))
                processed_items = 0

                while not result_async.ready():
                    while not progress_queue_inter.empty():
                        progress_queue_inter.get()
                        processed_items += 1
                        progress.update(task, advance=1)
        else:
            result_async.wait()

    combined_result = {}
    for result in result_async.get():
        combined_result.update(result)

    return combined_result


def process_inter_objs_chunk(chunk, progress_queue) -> dict[str, Intermediate]:
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
    inter_dict_chunk = {}
    for key, value in chunk.items():
        inter_dict_chunk[key + "*"] = Intermediate(code=key+"*", molecule=value, phase="ads")
        if inter_dict_chunk[key + "*"].closed_shell:
            inter_dict_chunk[key + "g"] = Intermediate(
                code=key + "g", molecule=value, phase="gas"
            )
    progress_queue.put(1)
    return inter_dict_chunk
