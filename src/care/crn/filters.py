from care import Intermediate


def is_alkane(inter: Intermediate) -> bool:
    """
    Check if the reactant is an alkane.

    Args:
        reactant (Intermediate): The reactant.

    Returns:
        bool: True if the reactant is an alkane, False otherwise.
    """
    if not inter.closed_shell:
        return False
    else:
        if inter["O"] > 0:
            return False
        else:
            nC, nH = inter["C"], inter["H"]
            return nH == 2 * nC + 2


def is_alkene(inter: Intermediate) -> bool:
    """
    Check if the reactant is an alkene.

    Args:
        reactant (Intermediate): The reactant.

    Returns:
        bool: True if the reactant is an alkane, False otherwise.
    """
    if not inter.closed_shell:
        return False
    else:
        if inter["O"] > 0:
            return False
        else:
            nC, nH = inter["C"], inter["H"]
            return nH == 2 * nC


def is_alkyne(inter: Intermediate) -> bool:
    """
    Check if the reactant is an alkyne.

    Args:
        reactant (Intermediate): The reactant.

    Returns:
        bool: True if the reactant is an alkane, False otherwise.
    """
    if not inter.closed_shell:
        return False
    else:
        if inter["O"] > 0:
            return False
        else:
            nC, nH = inter["C"], inter["H"]
            return nH == 2 * nC - 2


def is_alcohol(inter: Intermediate) -> bool:
    """
    Check if the reactant is an alcohol.

    Args:
        reactant (Intermediate): The reactant.

    Returns:
        bool: True if the reactant is an alcohol, False otherwise.
    """
    if not inter.closed_shell:
        return False
    else:
        if inter["O"] != 1:
            return False
        else:
            nC, nH = inter["C"], inter["H"]
            return nH == 2 * nC + 2
