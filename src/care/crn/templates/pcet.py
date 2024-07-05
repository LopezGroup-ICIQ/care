"""Proton-coupled electron transfer (PCET) reaction template, implemented
according to the Computational Hydrogen Electrode (CHE) reference."""

from ase import Atoms
from rich.progress import Progress

from care import ElementaryReaction, Intermediate
from care.crn.utils.electro import Proton, Electron, Water


def gen_pcet_reactions(
    intermediates: dict[str, Intermediate], reactions: list[ElementaryReaction]
) -> None:
    """
    Generate the proton-coupled electron transfer reactions
    of the reaction network as ElementaryReaction instances.
    Computational Hydrogen Electrode (CHE) is used as the reference.

    Args:
    ----
    intermediates : dict[str, Intermediate]
        Dictionary containing the Intermediate instances of the chemical space of the reaction network.
        Each key is the InChIKey of the molecule, and values are the corresponding Intermediate instance.
    reactions : list[ElementaryReaction]
        List of the reactions of the reaction network as ElementaryReaction instances.

    Returns:
    -------
    pcets : list[ElementaryReaction]
        List of the proton-coupled electron transfer reactions of the reaction network as ElementaryReaction instances.
    """

    pcets = []
    rtype = "PCET"

    oh_code = "TUJKJAMUKRIRHC-UHFFFAOYSA-N*"
    h_ads = [
        inter
        for inter in intermediates.values()
        if inter.formula == "H" and inter.phase == "ads"
    ][0]
    active_site = Intermediate(
        code="*", molecule=Atoms(), phase="surf", is_surface=True
    )

    pcets.append(
        ElementaryReaction(
            components=[[Proton(), Electron(), active_site], [h_ads]], r_type=rtype
        )
    )  # H+ + e- + * -> H*

    with Progress() as progress:
        task_desc = format_description("[green]Generating PCET reactions...")
        task = progress.add_task(task_desc, total=len(reactions))

        for rxn in reactions:
            new_reactants, new_products = [], []
            if rxn.r_type in ("H-O", "C-H"):
                for reactant in rxn.reactants:
                    if reactant.formula == "H":
                        new_reactants.extend([Proton(), Electron()])
                    else:
                        if not reactant.is_surface:
                            new_reactants.append(reactant)
                for product in rxn.products:
                    if product.formula == "H":
                        new_products.extend([Proton(), Electron()])
                    else:
                        if not product.is_surface:
                            new_products.append(product)

                pcets.append(
                    ElementaryReaction(
                        components=[new_reactants, new_products], r_type=rtype
                    )
                )
            elif rxn.r_type in ("C-O", "O-O"):
                if oh_code in [inter.code for inter in rxn]:
                    for reactant in rxn.reactants:
                        if reactant.is_surface:
                            new_reactants.extend([Electron(), Proton()])
                        elif reactant.formula == "HO":
                            new_reactants.append(Water())
                        else:
                            new_reactants.append(reactant)
                    for product in rxn.products:
                        if product.is_surface:
                            new_products.extend([Electron(), Proton()])
                        elif product.formula == "HO":
                            new_products.append(Water())
                        else:
                            new_products.append(product)

                    pcets.append(
                        ElementaryReaction(
                            components=[new_reactants, new_products], r_type="PCET"
                        )
                    )
                else:
                    continue
            else:
                pass
            progress.update(task, advance=1)

    return pcets


def format_description(description, width=45):
    """Format the progress bar description to a fixed width."""
    return description.ljust(width)[:width]
