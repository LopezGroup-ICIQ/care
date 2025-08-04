"""Module containing molecular templates for constructing the
Chemical Space (CS) of the CRN."""

from itertools import product

from rich.progress import Progress
from rdkit.Chem import MolToSmiles, MolFromSmiles
from rdkit import Chem


def format_description(description, width=45):
    """Format the progress bar description to a fixed width."""
    return description.ljust(width)[:width]


def gen_chemical_space(ncc: int, noc: int, cyclic: bool, show_progress: bool=False) -> list[str]:
    """Generate Chemical Space of the CRN.
    Chemical Space is the set of fully saturated hydrocarbons, ethers, epoxides, and alcohol species
    up to a given number of carbon and oxygen atoms.

    Args:
        ncc (int): Network Carbon Cutoff, maximum number of C atoms in the intermediates
        noc (int): Network Oxygen Cutoff, maximum number of O atoms in the intermediates.
        cyclic (bool): If True, generates cyclic compounds (epoxides).
        show_progress (bool, optional): If True, a progress bar is shown for each step of the blueprint generation. Defaults to False.

    Returns:
        list[str]: SMILES representation of the chemical space.
    """

    noc = noc if noc >= 0 else ncc * 2 + 2
    if noc > ncc * 2 + 2:
        raise ValueError(
            "The number of oxygen atoms must be smaller or equal than 2 * ncc + 2."
        )
    relev_species = ["O", "O=O", "[H][H]"]

    if show_progress:
        with Progress() as progress:
            task_desc = format_description("[green]Generating Chemical Space...")
            task = progress.add_task(task_desc, total=5)

            # Step 1: Generate Alkanes
            alkanes_smiles, mol_alkanes = gen_alkanes(ncc)
            progress.update(task, advance=1)
            
            if noc > 0:
                # Generate Ethers
                ethers_smiles, mol_ethers = gen_ethers(mol_alkanes, noc)
                progress.update(task, advance=1)

                if cyclic:
                    # Generate Epoxides
                    epox_smiles, mol_epox = gen_epoxides(mol_alkanes, noc)
                    progress.update(task, advance=1)
                else:
                    epox_smiles, mol_epox = [], []
                    progress.update(task, advance=1)

                # Add Oxygens
                alkanes_oxy_smiles = [add_oxygens(mol, noc) for mol in mol_alkanes]
                ethers_epox_oxy_smiles = [
                    add_oxygens(mol, noc - 1) for mol in mol_ethers + mol_epox
                ]
                cho_smiles = alkanes_oxy_smiles + ethers_epox_oxy_smiles
                cho_smiles = [smiles for smiles_set in cho_smiles for smiles in smiles_set]
                progress.update(task, advance=1)

                cho_smiles += epox_smiles + ethers_smiles
                progress.update(task, advance=1)
                return list(set(alkanes_smiles + cho_smiles + relev_species))
            else: 
                progress.update(task, advance=1)
                return alkanes_smiles + ["[H][H]"]
    else:        
        alkanes_smiles, mol_alkanes = gen_alkanes(ncc)
        if noc > 0:
            ethers_smiles, mol_ethers = gen_ethers(mol_alkanes, noc)

            if cyclic:
                epox_smiles, mol_epox = gen_epoxides(mol_alkanes, noc)
            else:
                epox_smiles, mol_epox = [], []

            alkanes_oxy_smiles = [add_oxygens(mol, noc) for mol in mol_alkanes]
            ethers_epox_oxy_smiles = [
                add_oxygens(mol, noc - 1) for mol in mol_ethers + mol_epox
            ]
            cho_smiles = alkanes_oxy_smiles + ethers_epox_oxy_smiles
            cho_smiles = [smiles for smiles_set in cho_smiles for smiles in smiles_set]
            cho_smiles += epox_smiles + ethers_smiles
            return list(set(alkanes_smiles + cho_smiles + relev_species))
        else:
            return alkanes_smiles + ["[H][H]"]


def generate_alkanes_recursive(n_carbon: int, main_chain="") -> list[str]:
    """
    Generate all possible linear and branched alkanes with a given number of carbon atoms.

    Parameters
    ----------
    n_carbon : int
        Maximum number of carbon atoms in the alkanes.
    main_chain : str, optional
        String where SMILES will be generated, by default ""

    Returns
    -------
    unique_alkanes : list[str]
        List of unique alkanes in SMILES format.
    """
    if n_carbon == 0:
        return [main_chain]
    if n_carbon < 0:
        return []

    alkanes = []

    # Continue the main chain
    new_chain = main_chain + "C"
    alkanes += generate_alkanes_recursive(n_carbon - 1, new_chain)

    # Add branches
    if len(main_chain) > 0:
        for i in range(len(main_chain)):
            if main_chain[i] == "C":
                new_chain = main_chain[:i] + "C(C)" + main_chain[i + 1 :]
                alkanes += generate_alkanes_recursive(n_carbon - 1, new_chain)

    # Remove duplicates
    unique_alkanes = list(set(alkanes))
    return unique_alkanes


def canonicalize_smiles(smiles_list: list[str], rmv_quatr_C: bool) -> list[str]:
    """
    Canonicalize a list of SMILES strings.

    Parameters
    ----------
    smiles_list : list[str]
        List of SMILES strings.

    Returns
    -------
    list[str]
        List of unique SMILES strings.
    """
    canonical_smiles_set = set()
    for smiles in smiles_list:
        mol = MolFromSmiles(smiles)
        if mol is not None:
            # If the molecule contains a quaternary carbon, do not add it to the list
            if rmv_quatr_C:
                if any(atom.GetDegree() == 4 for atom in mol.GetAtoms()):
                    continue
            canonical_smiles = MolToSmiles(mol, isomericSmiles=False)
            canonical_smiles_set.add(canonical_smiles)
    return list(canonical_smiles_set)


def gen_alkanes(ncc: int) -> tuple[list[str], list[Chem.Mol]]:
    """
    Generate all possible alkanes from 1 Carbon atom to the given number of carbon atoms.

    Parameters
    ----------
    ncc : int
        Maximum number of carbon atoms in the alkanes.

    Returns
    -------
    unique_alkanes : list[str]
        List of unique alkanes in SMILES format.
    mol_alkanes : list[Chem.Mol]
        List of RDKit molecules of alkanes.
    """
    all_alkanes = []
    for i in range(1, ncc + 1):
        all_alkanes += generate_alkanes_recursive(i)
    unique_alkanes = canonicalize_smiles(all_alkanes, rmv_quatr_C=False)
    mol_alkanes = [MolFromSmiles(smiles) for smiles in unique_alkanes]
    return unique_alkanes, mol_alkanes


def gen_epoxides(mol_alkanes: list, noc: int) -> list[str]:
    """
    Generate all possible epoxides smiles based on the given number of carbon and oxygen atoms.

    Parameters
    ----------
    mol_alkanes : list
        List of RDKit molecules of alkanes.
    no : int
        Maximum number of oxygen atoms in the epoxides.

    Returns
    -------
    list[str]
        List of epoxides SMILES strings.
    """

    epoxides = []
    cond1 = lambda atoms: atoms[0].GetDegree() < 4 and atoms[1].GetDegree() < 4
    if noc == 0:
        return epoxides
    for mol in mol_alkanes:
        # generate list of tuple of adjacent atoms satisfying the cond1
        c_pairs = [
            (atom, nbr)
            for atom in mol.GetAtoms()
            for nbr in atom.GetNeighbors()
            if cond1((atom, nbr))
        ]
        for c_pair in c_pairs:
            tmp_mol = Chem.RWMol(mol)
            oxygen_idx = tmp_mol.AddAtom(Chem.Atom("O"))
            tmp_mol.AddBond(
                c_pair[0].GetIdx(), oxygen_idx, order=Chem.rdchem.BondType.SINGLE
            )
            tmp_mol.AddBond(
                c_pair[1].GetIdx(), oxygen_idx, order=Chem.rdchem.BondType.SINGLE
            )
            tmp_mol.UpdatePropertyCache()
            smiles = MolToSmiles(tmp_mol, canonical=True)
            epoxides.append(smiles)
    smiles_epoxides = list(set(epoxides))

    # Generating mol objects
    mol_epoxides = [MolFromSmiles(smiles) for smiles in epoxides]
    return smiles_epoxides, mol_epoxides


def gen_ethers(mol_alkanes: list, noc: int) -> tuple[list[str], list[Chem.Mol]]:
    """
    Add an oxygen atom to an alkane to generate an ether.

    Parameters
    ----------
    alkane_smiles_list : list[str]
        List of alkane SMILES strings.

    Returns
    -------
    unique_ethers : list[str]
        List of ether SMILES strings.
    mol_ethers : list[Chem.Mol]
        List of RDKit molecules of ethers.
    """

    ethers = []
    if noc == 0:
        return ethers, []

    for mol in mol_alkanes:
        # generate list of tuple of adjacent atoms satisfying the cond1
        c_pairs = [
            (atom, nbr) for atom in mol.GetAtoms() for nbr in atom.GetNeighbors()
        ]
        for c_pair in c_pairs:
            # Insert oxygen atom between the two carbon atoms
            tmp_mol = Chem.RWMol(mol)
            oxygen_idx = tmp_mol.AddAtom(Chem.Atom("O"))
            tmp_mol.AddBond(
                c_pair[0].GetIdx(), oxygen_idx, order=Chem.rdchem.BondType.SINGLE
            )
            tmp_mol.AddBond(
                c_pair[1].GetIdx(), oxygen_idx, order=Chem.rdchem.BondType.SINGLE
            )
            # Delete the bond between the two carbon atoms
            tmp_mol.RemoveBond(c_pair[0].GetIdx(), c_pair[1].GetIdx())
            tmp_mol.UpdatePropertyCache()
            smiles = MolToSmiles(tmp_mol, canonical=True)
            ethers.append(smiles)
    unique_ethers = list(set(ethers))
    mol_ethers = [MolFromSmiles(smiles) for smiles in unique_ethers]
    return unique_ethers, mol_ethers


def add_oxygens(mol: Chem.Mol, noc: int) -> set[str]:
    """
    Add up to 'noc' oxygen atoms to suitable carbon atoms in the molecule.

    Parameters
    ----------
    mol : Chem.Mol
        RDKit molecule.
    noc : int
        Maximum number of oxygens to add. If noc < 0, add as many oxygens as possible.

    Returns
    -------
    unique_molecules : set[str]
        Set of all possible molecules with added oxygens (SMILES format).
    """

    if noc == 0:
        return set()

    unique_molecules = set()

    # Filter out carbon atoms with no free sites
    suitable_carbons = [
        (atom.GetIdx(), 4 - atom.GetDegree())
        for atom in mol.GetAtoms()
        if atom.GetSymbol() == "C" and atom.GetDegree() < 4
    ]
    if not suitable_carbons:
        return unique_molecules

    # Generate combinations
    combos = [list(range(num + 1)) for _, num in suitable_carbons]

    for combo in product(*combos):
        total_oxygens = sum(combo)
        if total_oxygens == 0 or (noc >= 0 and total_oxygens > noc):
            continue

        tmp_mol = Chem.RWMol(mol)
        for num_oxygens, (carbon_idx, _) in zip(combo, suitable_carbons):
            for _ in range(num_oxygens):
                oxygen_idx = tmp_mol.AddAtom(Chem.Atom("O"))
                tmp_mol.AddBond(
                    carbon_idx, oxygen_idx, order=Chem.rdchem.BondType.SINGLE
                )

        tmp_mol.UpdatePropertyCache()
        smiles = MolToSmiles(tmp_mol, canonical=True)
        unique_molecules.add(smiles)

    return unique_molecules
