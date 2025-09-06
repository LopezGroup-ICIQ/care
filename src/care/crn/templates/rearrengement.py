"""(1,2)-H shift rearrangement reaction template."""

from collections import defaultdict
from itertools import combinations
import multiprocessing as mp
import re

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rich.progress import Progress

from care import Intermediate, ElementaryReaction


class Rearrangement(ElementaryReaction):
    """Class for (1,2)-H shift rearrangement reactions."""

    def __init__(self, components, r_type):
        super().__init__(components=components, r_type=r_type)

    def reverse(self):
        self.components = self.components[::-1]
        for k, v in self.stoic.items():
            self.stoic[k] = -v
        if self.e_rxn != None:
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
        If is not in the bond-breaking direction, reverse it
        Adsorption steps are reversed to desorption steps, while desorption steps are preserved.

        Note: Rearrangement reactions do not have an intrinsic bond-breaking direction.
        """
        pass


def gen_rearrangement_reactions(
    intermediates: dict[str, Intermediate], num_cpu: int = mp.cpu_count(), show_progress: bool = False
) -> list[Rearrangement]:
    """
    Generate the (1,2)-H shift rearrangement reactions.

    Parameters
    ----------
    intermediates : dict[str, Intermediate]
        Dictionary containing the intermediates of the reaction network.
        Each key is the InChIKey of the molecule, and values are the corresponding Intermediate instances.
    num_cpu : int, optional
        Number of CPU cores to use for the generation, by default mp.cpu_count()
    show_progress : bool, optional
        If True, a progress bar is shown for the generation, by default False

    Returns
    -------
    list[Rearrangement]
        List of the rearrangement reactions of the reaction network.
    """

    ads_inters = [inter for inter in intermediates.values() if inter.phase == "ads"]

    formula_groups = group_by_formula(ads_inters)

    # Dictionary to store pairs for each subgroup
    subgroup_pairs_dict = {}
    index = 0
    for formula_group in formula_groups.values():
        isomer_subgroups = subgroup_by_isomers(formula_group)

        # Generate combinations within each isomer subgroup and store in dictionary
        for subgroup in isomer_subgroups:
            subgroup_pairs_dict[f"subgroup_{index}"] = list(combinations(subgroup, 2))
            index += 1

    # Splitting the dictionary into chunks
    keys = list(subgroup_pairs_dict.keys())
    chunk_size = len(keys) // num_cpu
    if chunk_size == 0:
        chunk_size = 1
    chunks = [
        dict(
            zip(
                keys[i : i + chunk_size],
                [subgroup_pairs_dict[key] for key in keys[i : i + chunk_size]],
            )
        )
        for i in range(0, len(keys), chunk_size)
    ]

    manager_rearr_rxn = mp.Manager()
    progress_queue_rxn = manager_rearr_rxn.Queue()

    tasks = [(chunk, progress_queue_rxn) for chunk in chunks]

    if show_progress:
        with mp.Pool(num_cpu) as pool:
            result_async = pool.starmap_async(process_subgroup, tasks)
            with Progress() as progress:
                task_desc = format_description(
                    "[green]Generating rearrangement reactions..."
                )
                task = progress.add_task(task_desc, total=len(tasks))
                processed_items = 0

                while not result_async.ready():
                    while not progress_queue_rxn.empty():
                        progress_queue_rxn.get()
                        processed_items += 1
                        progress.update(task, advance=1)
    else:
        with mp.Pool(num_cpu) as pool:
            result_async = pool.starmap_async(process_subgroup, tasks)
            while not result_async.ready():
                while not progress_queue_rxn.empty():
                    progress_queue_rxn.get()

    return [rxn for sublist in result_async.get() for rxn in sublist]


def check_rearrangement(pair: tuple[Intermediate, Intermediate]) -> Rearrangement:
    """
    Check if a pair of intermediates is a 1,2-rearrangement reaction.

    Parameters
    ----------
    pair : tuple[Intermediate, Intermediate]
        A tuple containing two Intermediate instances.

    Returns
    -------
    Rearrengement
        An ElementaryReaction instance if the pair is a 1,2-rearrangement reaction, None otherwise.
    """

    inter1, inter2 = pair
    smiles1 = Chem.MolToSmiles(inter1.rdkit)
    smiles2 = Chem.MolToSmiles(inter2.rdkit)
    if is_hydrogen_rearranged(smiles1, smiles2):
        return Rearrangement(
            components=(frozenset([inter1]), frozenset([inter2])),
            r_type="rearrangement",
        )
    return None


def group_by_formula(
    intermediates: list[Intermediate],
) -> dict[str, list[Intermediate]]:
    """
    Group a list of intermediates by chemical formula.

    Parameters
    ----------
    intermediates : list[str, Intermediate]
        List of intermediates to group.

    Returns
    -------
    dict[str, list[Intermediate]]
        A dictionary where each key is a chemical formula and each value is a list of intermediates with that chemical formula.
    """
    formula_groups = defaultdict(list)
    for inter in intermediates:
        formula_groups[inter.formula].append(inter)
    return formula_groups


def subgroup_by_isomers(intermediates: list[Intermediate]) -> list[list[Intermediate]]:
    """
    Subgroup a list of intermediates by isomers.

    Parameters
    ----------
    intermediates : list[str, Intermediate]

    Returns
    -------
    list[list[Intermediate]]
        A list of subgroups, where each subgroup is a list of intermediates.
    """
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


def process_subgroup(subgroup_pairs_dict, progress_queue):
    """
    Process a chunk of subgroups.

    Parameters
    ----------
    subgroup_pairs_dict : dict
        A dictionary where each key is a unique identifier for a subgroup and
        each value is a list of pairs from that subgroup.

    Returns
    -------
    list
        A list of results after processing all pairs in all subgroups.
    """
    results = []
    for pairs in subgroup_pairs_dict.values():
        # Process each pair in the subgroup
        for pair in pairs:
            result = check_rearrangement(pair)
            if result is not None:
                results.append(result)
    progress_queue.put(1)

    return results


def format_description(description, width=45):
    """Format the progress bar description to a fixed width."""
    return description.ljust(width)[:width]


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

    # Saturate the molecules
    mol1_smiles = re.sub(r"[H]([0-9]){0,1}", "", mol1_smiles)
    mol2_smiles = re.sub(r"[H]([0-9]){0,1}", "", mol2_smiles)
    mol1_smiles = mol1_smiles.replace("[", "").replace("]", "")
    mol2_smiles = mol2_smiles.replace("[", "").replace("]", "")

    mol1_sat = Chem.MolFromSmiles(mol1_smiles)
    mol2_sat = Chem.MolFromSmiles(mol2_smiles)

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
    Check if two molecules have potential hydrogen rearrangements.

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
                    if (
                        "("
                        and ")" in chpped_smiles_1[i + 1]
                        or "("
                        and ")" in chpped_smiles_2[i + 1]
                    ):
                        # Checking the next block
                        if i + 2 < len(check_list):
                            if (
                                "("
                                and ")" in chpped_smiles_1[i + 2]
                                or "("
                                and ")" in chpped_smiles_2[i + 2]
                            ):
                                # Checking the next block
                                if i + 3 < len(check_list):
                                    if check_list[i + 3] == False:
                                        return True
                            else:
                                # Check if that block is False
                                if check_list[i + 2] == False:
                                    return True
    return False
