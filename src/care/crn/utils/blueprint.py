import time
import warnings
import multiprocessing as mp

from prettytable import PrettyTable
from rdkit.Chem import MolFromSmiles, AddHs

from care import ReactionNetwork, INTER_ELEMS
from care.crn.templates import adsorption, pcet, rearrengement, dissociation, chemspace


def format_description(description, width=45):
    """Format the progress bar description to a fixed width."""
    return description.ljust(width)[:width]

def get_elements(mol_list):
            elements = set()
            for mol in mol_list:
                if mol:
                    for atom in mol.GetAtoms():
                        elements.add(atom.GetSymbol())
            return elements

def gen_blueprint(
    ncc: int = None,
    noc: int = None,
    cs: list[str] = None,
    reactants: list[str] = None,
    products: list[str] = None,
    cyclic: bool = None,
    additional_rxns: bool = None,
    electro: bool = None,
    num_cpu: int = mp.cpu_count(),
    show_progress: bool = False, 
    show_final_summary: bool = False,
) -> ReactionNetwork:
    """
    Generate the reaction network blueprint.

    Parameters
    ----------
    ncc : int
        Network Carbon Cutoff, maximum number of C atoms in the intermediates
    noc : int
        Network Oxygen Cutoff, Maximum number of O atoms in the intermediates.
        if is a negative number, then the noc is set to the max number of O atoms in the intermediates.
    cs : list[str]
        List of SMILES of the molecules defining the Chemical Space of the CRN.
        You can provide cs or ncc and noc. If both are provided, cs is used.
    reactants : list[str]
        List of SMILES of the reactant molecules. If provided, the function will attempt to
        orient the reactions such that these molecules are reactants. This is a heuristic and may not always be correct, especially if the same molecule appears in both reactants and products.
    products : list[str]
        List of SMILES of the product molecules. If provided, the function will attempt to
        orient the reactions such that these molecules are products. This is a heuristic and may not always be correct, especially if the same molecule appears in both reactants and products.
    cyclic : bool
        If True, generates cyclic compounds (epoxides). Only used with ncc/noc.
    additional_rxns : bool
        If True, 1-2-H shift rearrangement reactions are generated.
    electro : bool
        If True, proton-coupled electron transfer reactions are generated.
    num_cpu : int, optional
        Number of CPU cores to use for the CRN generation, by default mp.cpu_count()
    show_progress : bool, optional
        If True, a progress bar is shown for each step of the blueprint generation, by default False

    Returns
    -------
    ReactionNetwork
        The reaction network blueprint.

    Notes
    -----
    - The function accepts three main ways to build the blueprint:
        1. Provide a chemical space (cs) as a list of SMILES.
        2. Provide network carbon and oxygen cutoffs (ncc and noc) to generate a chemical space.
        3. Provide reactants and products as lists of SMILES, which will be used to infer the chemical space and orient the reactions.
    """
    intermediates, reactions = {}, []

    table = PrettyTable()
    table.field_names = ["Category", "Number of Items", "Time (s)"]

    # Generate the chemical space (CS)
    t0cs = time.time()
    if reactants and products:
        reactants_mols = [AddHs(MolFromSmiles(smiles, True), False) for smiles in reactants]
        products_mols = [AddHs(MolFromSmiles(smiles, True), False) for smiles in products]
        reactants_elements = get_elements(reactants_mols)
        products_elements = get_elements(products_mols)

        if reactants_elements != products_elements:
            missing_in_prod = reactants_elements - products_elements
            missing_in_react = products_elements - reactants_elements
            error_msg = "Element mismatch between reactants and products!\n"
            if missing_in_prod:
                error_msg += f"Elements in reactants but missing in products: {missing_in_prod}\n"
            if missing_in_react:
                error_msg += f"Elements in products but missing in reactants: {missing_in_react}"
            raise ValueError(error_msg)
        
        is_forming = max(m.GetNumAtoms() for m in products_mols) > max(m.GetNumAtoms() for m in reactants_mols)
        chemical_space = reactants + products
    elif cs:
        chemical_space = [s for s in cs if MolFromSmiles(s)]
        is_forming = False
        if len(chemical_space) < len(cs):
            warnings.warn("Some SMILES in 'cs' were invalid and removed.", UserWarning)
    elif ncc is not None and noc is not None:
        chemical_space = chemspace.gen_chemical_space(ncc, noc, cyclic, show_progress)
        is_forming = False
    else:
        raise ValueError("Insufficient parameters. Provide (reactants/products), (cs), or (ncc/noc).")
    tcs = time.time() - t0cs
    table.add_row(["Chemical Space", len(chemical_space), f"{tcs:.2f}"])

    cs_mols = [AddHs(MolFromSmiles(smiles, True), False) for smiles in chemical_space]
    elements = get_elements(cs_mols)
    unsupported_elements = set(elements) - set(INTER_ELEMS)
    
    if unsupported_elements:
        raise ValueError(
            f"Chemical space contains unsupported elements: {unsupported_elements}. "
            f"CARE currently supports only: {set(INTER_ELEMS)}"
        )

    # Extend CS with molecules originating from dissociation of CS species
    t0ecs = time.time()
    bb_inters, bb_steps = dissociation.dissociate(chemical_space, num_cpu, show_progress)
    nfrags = len(bb_inters) - len(chemical_space)
    nbbsteps = len(bb_steps)
    intermediates.update(bb_inters)
    reactions.extend(bb_steps)
    tecs = time.time() - t0ecs
    trxn = tecs
    table.add_row(
        ["Fragments and unsaturated molecules", nfrags, f"{tecs:.2f}"], divider=True
    )
    table.add_row(["Dissociation reactions", nbbsteps, f""])

    # Adsorption/Desorption reactions
    t02 = time.time()
    ads_steps = adsorption.gen_adsorption_reactions(intermediates, num_cpu, show_progress)
    nadssteps = len(ads_steps)
    reactions.extend(ads_steps)
    t2 = time.time() - t02
    trxn += t2
    table.add_row(["Adsorption reactions", nadssteps, f"{t2:.2f}"])

    # (1,2)-H shift rearrangement reactions
    if additional_rxns:
        t03 = time.time()
        rearr_steps = rearrengement.gen_rearrangement_reactions(intermediates, num_cpu, show_progress)
        nrearrsteps = len(rearr_steps)
        reactions.extend(rearr_steps)
        t3 = time.time() - t03
        trxn += t3
        table.add_row(["Rearrangement reactions", nrearrsteps, f"{t3:.2f}"])

    # Proton-coupled electron transfer (PCET) reactions
    if electro:
        t04 = time.time()
        pcets = pcet.gen_pcet_reactions(intermediates, reactions, show_progress)
        npcets = len(pcets)
        reactions.extend(pcets)
        t4 = time.time() - t04
        trxn += t4
        table.add_row(["PCET reactions", npcets, f"{t4:.2f}"])

    table.add_row(["", "", ""], divider=True)
    table.add_row(["Total number of species", len(intermediates), f"{tcs+tecs:.2f}"])
    table.add_row(["Total number of reactions", len(reactions), f"{trxn:.2f}"])

    if show_final_summary:
        print(f"\n{table}")

    if reactants and products:
        for rxn in reactions:
            if isinstance(rxn, dissociation.BondBreaking) and is_forming:
                rxn.reverse()
            if isinstance(rxn, adsorption.Adsorption):
                if rxn.adsorbate.get_smiles() in reactants:
                    pass
                else:
                    rxn.reverse()

    def rxn_sort_key(rxn):
        if isinstance(rxn, adsorption.Adsorption):
            category = 0
        elif isinstance(rxn, adsorption.Desorption):
            category = 2
        else:
            category = 1
        return (category, len(rxn.reactants), len(rxn.products))

    reactions.sort(key=rxn_sort_key)
    crn = ReactionNetwork(reactions)

    if reactants and products:
        rxns_to_reverse = set()
        for gr in crn.global_reactions:
            route = crn.get_route_stoichiometry(gr)
            for idx, sigma in route.items():
                if sigma < 0:
                    rxns_to_reverse.add(idx)
        for idx in rxns_to_reverse:
            crn.reverse_reaction(idx)
    return crn
