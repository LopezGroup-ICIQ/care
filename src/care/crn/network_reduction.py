import numpy as np

from scipy.sparse import diags

from care import ReactionNetwork
from care.constants import INTER_ELEMS

def obtain_elemental_flux(mkm_data: dict):
    fluxes = {}
    def extend_vector(v):  # inters_info vectors miss "*" species at the end (TODO: fix upstream)
        return np.concatenate([v, [0]])

    Rmol = mkm_data["consumption_rate"] # (NC, NR) sparse matrix
    elements = [x for x in mkm_data["inters_info"].keys() if x in INTER_ELEMS]
    for elem in elements:
        elem_vec = extend_vector(mkm_data["inters_info"][elem])
        d = diags(elem_vec)
        r_elem = d @ Rmol  # (NC, NR) sparse matrix
        r_elem_abs = r_elem.copy()
        r_elem_abs.data = np.abs(r_elem_abs.data)
        f_elem_vector = 0.5 * np.sum(r_elem_abs, axis=0)
        fluxes[elem] = np.asarray(f_elem_vector).flatten()
    return fluxes

def prune_by_elemental_flux(
    crn: ReactionNetwork,
    element_flux_dict: dict[str, np.ndarray],
    relative_threshold: float = 0.001, 
    floor: float = 1e-30,
) -> ReactionNetwork:
    """
    Prunes a ReactionNetwork, keeping only reactions with an elemental
    flux above a given relative threshold for ANY of the provided elements.

    Args:
        crn: The original ReactionNetwork.
        element_flux_vectors: A list of 1D arrays (n_reactions), e.g.,
                              [F_C_vector, F_H_vector, F_O_vector].
        relative_threshold: The cutoff, relative to *each element's*
                            own max flux. (e.g., 0.001 keeps reactions
                            > 0.1% of max C-flux OR > 0.1% of max H-flux).
        floor: Minimum absolute threshold to avoid zero cutoffs.

    Returns:
        A new, pruned ReactionNetwork.

    Notes:
    - Ensure that the CRN is correctly rewired after the kinetic simulation
    """
    if not element_flux_dict:
        print("Warning: No elemental flux vectors provided. Returning empty network.")
        return ReactionNetwork([])

    # 1. Calculate the absolute threshold for each element
    absolute_thresholds = {}
    for elem, vec in element_flux_dict.items():
        max_flux = np.max(vec)
        if max_flux < floor:  # Avoid division by zero
            print(f"Warning: Max elemental flux for element {elem} is zero.")
            absolute_thresholds[elem] = floor
        else:
            absolute_thresholds[elem] = max_flux * relative_threshold

    # 2. Iterate through reactions and check all elemental fluxes
    reactions_to_keep = []
    for j, rxn in enumerate(crn.reactions):
        keep_this_reaction = False
        
        # Check if reaction 'j' is important for ANY element
        for elem, vec in element_flux_dict.items():
            if vec[j] > absolute_thresholds[elem]:
                keep_this_reaction = True
                break  # Found a reason to keep it, move to next reaction
        
        if keep_this_reaction:
            reactions_to_keep.append(rxn)

    print(f"Original network: {crn.num_reactions} reactions")
    print(f"Pruned network:   {len(reactions_to_keep)} reactions "
          f"(> {relative_threshold:.1e} relative flux for any element)")

    return ReactionNetwork(reactions_to_keep)
