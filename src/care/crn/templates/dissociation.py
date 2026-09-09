"""Template for bond-breaking and bond-formation surface reactions."""

import multiprocessing as mp

from rich.progress import Progress
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, MolFromSmiles, MolToSmiles
from rdkit.Chem.inchi import MolToInchiKey

from care import ElementaryReaction, Intermediate
from care.crn.intermediate import GasSpecies, AdsorbedSpecies, SurfaceSite
from care.constants import BOND_TYPES

from copy import deepcopy

from ase.optimize import BFGS
import networkx as nx
import numpy as np
from torch.cuda import empty_cache

from care.crn.utils.graph import atoms_to_graph, extract_adsorbate, is_adsorbate_fragmented, connectivity_signature
from care.constants import CORDERO


class BondBreaking(ElementaryReaction):
    """Class for bond-breaking reactions."""

    def __init__(self, components, r_type=None, stoic=None):
        super().__init__(components=components, r_type=r_type, stoic=stoic)
        self.requires_neb = True

    def reverse(self):
        super().reverse()
        self.__class__ = BondFormation

    def bb_order(self):
        pass

    def get_is(self) -> None:
        """Fetch the joined molecule geometry (Reactant)."""
        joined_species = [inter for inter in list(self.reactants) if isinstance(inter, AdsorbedSpecies)][0]
        idx = min(joined_species.ads_configs, key=lambda x: joined_species.ads_configs[x]['mu'])
        
        self.is_atoms = joined_species.ads_configs[idx]["ase"].copy()
        self.is_graph = atoms_to_graph(self.is_atoms, self.is_atoms.get_array("atom_tags"), surface_order=-1, filter=True)

    def get_fs(self, evaluator) -> None:
        """Split the joined molecule A* into fragments B* + C*."""
        joined_species = [inter for inter in list(self.reactants) if isinstance(inter, AdsorbedSpecies)][0]
        target_nx = _build_fragment_nx(self.stoic, list(self.products))
        
        self.fs_atoms, self.fs_graph = _generate_split_state(
            joined_species=joined_species,
            target_nx_signature=connectivity_signature(target_nx),
            bond=tuple(self.r_type.split("-")),
            evaluator=evaluator,
            repr_hr=self.repr_hr
        )

    def get_states(self, evaluator) -> None:
        self.get_is()
        self.get_fs(evaluator)


class BondFormation(ElementaryReaction):
    """Class for bond-formation reactions of the type A* + B* -> C*."""

    def __init__(self, components, r_type=None, stoic=None):
        super().__init__(components=components, r_type=r_type, stoic=stoic)
        self.requires_neb = True

    def reverse(self):
        super().reverse()
        self.__class__ = BondBreaking

    def bb_order(self):
        self.reverse()

    def get_fs(self) -> None:
        """Fetch C*."""
        joined_species = [inter for inter in list(self.products) if isinstance(inter, AdsorbedSpecies)][0]
        idx = min(joined_species.ads_configs, key=lambda x: joined_species.ads_configs[x]['mu'])
        
        self.fs_atoms = joined_species.ads_configs[idx]["ase"].copy()
        self.fs_graph = atoms_to_graph(self.fs_atoms, self.fs_atoms.get_array("atom_tags"), surface_order=-1, filter=True)

    def get_is(self, evaluator) -> None:
        """Generate the initial state geometry A* + B* (Reactants)."""
        joined_species = [inter for inter in list(self.products) if isinstance(inter, AdsorbedSpecies)][0]
        target_nx = _build_fragment_nx(self.stoic, list(self.reactants))
        
        self.is_atoms, self.is_graph = _generate_split_state(
            joined_species=joined_species,
            target_nx_signature=connectivity_signature(target_nx),
            bond=tuple(self.r_type.split("-")),
            evaluator=evaluator,
            repr_hr=self.repr_hr
        )

    def get_states(self, evaluator) -> None:
        self.get_fs()
        self.get_is(evaluator)

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
    active_site = SurfaceSite()
    rxns = []

    if show_progress:
        with Progress() as progress:
            task_desc = format_description("[green]Processing ElementaryReactions...")
            task = progress.add_task(task_desc, total=len(unique_reactions))

            for reaction in unique_reactions:
                reactant = inters[reaction[0] + "*"]
                product1 = inters[reaction[1][0] + "*"]

                if len(reaction[1]) == 2:
                    product2 = inters[reaction[1][1] + "*"]
                    reaction_components = [[active_site, reactant], [product1, product2]]
                else:
                    reaction_components = [[reactant], [product1]]

                rxns.append(
                    BondBreaking(components=reaction_components, r_type=reaction[2])
                )
                progress.update(task, advance=1)
    else:
        for reaction in unique_reactions:
            reactant = inters[reaction[0] + "*"]
            product1 = inters[reaction[1][0] + "*"]

            if len(reaction[1]) == 2:
                product2 = inters[reaction[1][1] + "*"]
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
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        pass

    Chem.AssignStereochemistry(
        mol, cleanIt=True, force=True, flagPossibleStereoCenters=True
    )
    symmetry_classes = Chem.CanonicalRankAtoms(mol, breakTies=False)

    unique_bonds = {}
    for bond in mol.GetBonds():
        atom1_sym_class = symmetry_classes[bond.GetBeginAtomIdx()]
        atom2_sym_class = symmetry_classes[bond.GetEndAtomIdx()]
        bond_key = tuple(sorted([atom1_sym_class, atom2_sym_class]))

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
    try:
        Chem.SanitizeMol(molecule)
    except Exception:
        pass

    current_smiles = MolToSmiles(molecule, isomericSmiles=True, allHsExplicit=True)

    if original_smiles not in processed_fragments:
        processed_fragments[original_smiles] = []

    if current_smiles in processed_molecules:
        return

    try:
        unique_bonds = find_unique_bonds(molecule)
    except Exception as e:
        print(f"Warning: find_unique_bonds failed for {current_smiles}. Skipping.")
        processed_molecules.add(current_smiles)
        return

    total_bond_counter = 0
    for bond in unique_bonds:
        for Z_atom1, Z_atom2 in BOND_TYPES:
            if is_desired_bond(bond, Z_atom1, Z_atom2):
                mol_copy = Chem.RWMol(molecule)
                
                a1_idx = bond.GetBeginAtomIdx()
                a2_idx = bond.GetEndAtomIdx()
                
                mol_copy.RemoveBond(a1_idx, a2_idx)

                a1 = mol_copy.GetAtomWithIdx(a1_idx)
                a2 = mol_copy.GetAtomWithIdx(a2_idx)
                
                a1.SetNumRadicalElectrons(a1.GetNumRadicalElectrons() + 1)
                a2.SetNumRadicalElectrons(a2.GetNumRadicalElectrons() + 1)
                
                a1.SetNoImplicit(True)  
                a2.SetNoImplicit(True)

                frags = Chem.GetMolFrags(mol_copy, asMols=True, sanitizeFrags=False)
                
                frag_smiles_list = []
                frag_mols_list = [] 
                
                for frag in frags:
                    try:
                        Chem.SanitizeMol(frag)
                    except Exception as e:
                        break
                    
                    frag_smiles = MolToSmiles(
                        frag, isomericSmiles=True, allHsExplicit=True
                    )
                    frag_smiles_list.append(frag_smiles)
                    frag_mols_list.append(frag)

                else:  
                    for smi in frag_smiles_list:
                        if smi not in processed_fragments[original_smiles]:
                            processed_fragments[original_smiles].append(smi)

                    for frag_mol in frag_mols_list:
                        break_bonds(
                            frag_mol,
                            processed_fragments,
                            original_smiles,
                            unique_reactions,
                            processed_molecules,
                        )
                    
                    bbtype = sorted(
                        [
                            Chem.Atom(Z_atom1).GetSymbol(),
                            Chem.Atom(Z_atom2).GetSymbol(),
                        ]
                    )
                    bbtype_str = f"{bbtype[0]}-{bbtype[1]}"

                    if len(frag_smiles_list) >= 1:
                        current_inchi = MolToInchiKey(MolFromSmiles(current_smiles))
                        frag_inchis = tuple(sorted([MolToInchiKey(MolFromSmiles(s)) for s in frag_smiles_list]))
                        rxn_tuple = (current_inchi, frag_inchis, bbtype_str)
                        if rxn_tuple not in unique_reactions:
                            unique_reactions.add(rxn_tuple)
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
        inter_dict_chunk[key + "*"] = AdsorbedSpecies(code=key+"*", molecule=value)
        if inter_dict_chunk[key + "*"].closed_shell:
            inter_dict_chunk[key + "g"] = GasSpecies(code=key + "g", molecule=value)
    progress_queue.put(1)
    return inter_dict_chunk


def _build_fragment_nx(stoic: dict, fragment_species_list: list) -> nx.Graph:
    """Returns a NetworkX graph representing the disconnected fragments (e.g., B* + C*)."""
    competitors = [inter for inter in fragment_species_list if isinstance(inter, AdsorbedSpecies)]
    
    if len(competitors) == 1:
        if abs(stoic[competitors[0].code]) == 2:  # 2B*
            return nx.disjoint_union(competitors[0].graph, competitors[0].graph)
        elif abs(stoic[competitors[0].code]) == 1:  # Ring opening
            return competitors[0].graph.copy()
                    
    elif len(competitors) == 2:  # B* + C* 
        return nx.disjoint_union(competitors[0].graph, competitors[1].graph)
        
    raise ValueError(f"Fragment stoichiometry not supported: {[c.code for c in competitors]}")


def _generate_split_state(
    joined_species: AdsorbedSpecies, 
    target_nx_signature: tuple, 
    bond: tuple, 
    evaluator,
    repr_hr: str
):
    """
    Core engine to split a joined adsorbate into two fragments.
    Returns the optimized separated Atoms object and its corresponding Graph.
    """
    if not joined_species.ads_configs:
        raise ValueError(f"No adsorbed configurations found for {joined_species.code}.")
    
    idx = min(joined_species.ads_configs, key=lambda x: joined_species.ads_configs[x]['mu'])
    joined_atoms = joined_species.ads_configs[idx]["ase"]
    atom_tags_array = joined_atoms.get_array("atom_tags")
    joined_graph = atoms_to_graph(joined_atoms, atom_tags_array, surface_order=-1, filter=True)
    
    n_nodes = len(joined_graph)
    adsorbate_node_ids = [i for i in range(n_nodes) if atom_tags_array[i] == 1]
    slab_node_ids = [i for i in range(n_nodes) if atom_tags_array[i] == 0]

    elem1, elem2 = bond
    potential_edges = [(u, v) for u, v in joined_graph.edges() if (
        (joined_graph.nodes[u]['elem'] == elem1 and joined_graph.nodes[v]['elem'] == elem2) or 
        (joined_graph.nodes[u]['elem'] == elem2 and joined_graph.nodes[v]['elem'] == elem1)
    )]

    u, v = None, None
    if len(potential_edges) == 0 and len(target_nx_signature) == 2: 
        u, v = adsorbate_node_ids[0], adsorbate_node_ids[1]
    elif len(potential_edges) > 0:
        for potential_u, potential_v in potential_edges:
            data = joined_graph.copy()
            if data.has_edge(potential_u, potential_v):
                data.remove_edge(potential_u, potential_v)
            adsorbate = extract_adsorbate(data)
            if connectivity_signature(adsorbate) == target_nx_signature:
                u, v = potential_u, potential_v
                break
                
    if u is None or v is None:
        raise RuntimeError(f"Could not identify the bond {bond} to break.")

    node_indices_B, node_indices_C = {u}, {v}
    queue_B, queue_C = [u], [v]
    while queue_B or queue_C:
        new_queue_B, new_queue_C = [], []
        for node in queue_B:
            for nbr in joined_graph.neighbors(node):
                if nbr in adsorbate_node_ids and nbr not in node_indices_B and nbr not in node_indices_C:
                    node_indices_B.add(nbr)
                    new_queue_B.append(nbr)
        for node in queue_C:
            for nbr in joined_graph.neighbors(node):
                if nbr in adsorbate_node_ids and nbr not in node_indices_B and nbr not in node_indices_C:
                    node_indices_C.add(nbr)
                    new_queue_C.append(nbr)
        queue_B, queue_C = new_queue_B, new_queue_C
        
    node_indices_B = list(node_indices_B)
    node_indices_C = list(node_indices_C)

    slab_positions = joined_atoms.positions[slab_node_ids]
    z_max = np.max(slab_positions[:, 2])
    cm_B = joined_atoms.get_center_of_mass(indices=node_indices_B)
    cm_C = joined_atoms.get_center_of_mass(indices=node_indices_C)
    atoms_to_move = node_indices_C if abs(z_max - cm_B[2]) <= abs(z_max - cm_C[2]) else node_indices_B

    split_atoms = deepcopy(joined_atoms)
    min_z = min(joined_atoms.positions[atoms_to_move, 2])
    split_atoms.positions[atoms_to_move] += [0, 0, 2.0 + z_max - min_z]
    
    vector = joined_atoms.positions[u] - joined_atoms.positions[v] if atoms_to_move == node_indices_B else joined_atoms.positions[v] - joined_atoms.positions[u]
    vector[2] = 0.0
    
    if abs(joined_atoms.positions[v][0] - joined_atoms.positions[u][0]) < 1.0 and abs(joined_atoms.positions[v][1] - joined_atoms.positions[u][1]) < 1.0:
        delta = CORDERO[joined_atoms[u].symbol] + CORDERO[joined_atoms[v].symbol] + getattr(evaluator, "tol", 0.0)
        vector[0] += delta
        vector[1] += delta
        
    direction = vector / np.linalg.norm(vector)
    increment = 0
    
    opt_class = getattr(evaluator, 'optimizer_class', BFGS)
    while True:
        if increment >= 3.0:
            print(f"{repr_hr}: Maximum increment reached ({increment}); displacement vector: {direction}")
            return None, None
            
        split_atoms.positions[atoms_to_move] += (getattr(evaluator, "dx", 1.5) + increment) * direction
        split_atoms.wrap()
        
        evaluator._assign_calculator(split_atoms)
        opt = opt_class(split_atoms, logfile=getattr(evaluator, 'logfile', None))
        opt.run(fmax=getattr(evaluator, 'fmax', 0.05), steps=getattr(evaluator, 'max_steps', 100))
        empty_cache()
        
        split_graph = atoms_to_graph(split_atoms, atom_tags_array, surface_order=-1, filter=False)
        if is_adsorbate_fragmented(split_graph):
            split_atoms.calc = None
            return split_atoms, split_graph
            
        increment += 0.5
