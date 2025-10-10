from pickle import load
from typing import Union

import numpy as np
import pandas as pd
from numba import njit
from sklearn.linear_model import LinearRegression

from care.constants import R


@njit
def net_rate(y, kd, kr,
             sf_data, sf_indices, sf_indptr,
             sb_data, sb_indices, sb_indptr):
    rates = np.empty_like(kd)
    n_reactions = kd.shape[0]

    for i in range(n_reactions):  # loop over reactions
        forward_product = 1.0
        backward_product = 1.0

        # forward exponents (row i of sf)
        for idx in range(sf_indptr[i], sf_indptr[i+1]):
            j = sf_indices[idx]        # species index
            exp = sf_data[idx]         # exponent
            forward_product *= y[j] ** exp

        # backward exponents (row i of sb)
        for idx in range(sb_indptr[i], sb_indptr[i+1]):
            j = sb_indices[idx]
            exp = sb_data[idx]
            backward_product *= y[j] ** exp

        rates[i] = kd[i] * forward_product - kr[i] * backward_product

    return rates


def calc_eapp(t, r, gas_mask):
    """
    Evaluates the apparent activation energy for all the species whose formation rate is higher than zero.
    Args:
        temperature_vector(ndarray): Array containing the studied temperature range in Kelvin
        reaction_rate_vector(ndarray): Array containing the reaction rate at different temperatures
    Returns:
        Apparent reaction energy in kJ/mol at the specified temperature.
    """
    x = 1 / t
    eapp = np.zeros(len(gas_mask[:-1]))
    for i, inter in enumerate(gas_mask[:-1]):
        Eapp = -(R / 1000.0)
        if inter and np.all(r[:, i] > 0):
            lm = LinearRegression()
            reg = lm.fit(x.reshape(-1, 1), np.log(r[:, i]).reshape(-1, 1))
            Eapp *= reg.coef_[0, 0]  # kJ/mol
            eapp[i] = Eapp
        else:
            eapp[i] = None
    return eapp

@njit
def jacobian_fill_numba(y, kd, kr,
                        sf_data, sf_indices, sf_indptr,
                        sb_data, sb_indices, sb_indptr,
                        vT_data, vT_indices, vT_indptr,
                        rows, cols, values):
    """
    Fill preallocated triplet arrays (rows, cols, values).
    Returns number of entries written.
    All arrays are 0-based (SciPy's CSR layout).
    vT_* corresponds to v_sparse.T (shape: n_reactions x n_species).
    """
    nr = kd.shape[0]
    pos = 0

    for r in range(nr):
        # forward/backward index ranges for reaction r
        sf_start = sf_indptr[r]
        sf_stop  = sf_indptr[r+1]
        sb_start = sb_indptr[r]
        sb_stop  = sb_indptr[r+1]

        # compute forward product and backward product (full)
        fprod = 1.0
        for idx in range(sf_start, sf_stop):
            j = sf_indices[idx]
            e = sf_data[idx]
            if e != 0:
                fprod *= y[j] ** e

        bprod = 1.0
        for idx in range(sb_start, sb_stop):
            j = sb_indices[idx]
            e = sb_data[idx]
            if e != 0:
                bprod *= y[j] ** e

        # contributions from forward participants
        for idx in range(sf_start, sf_stop):
            s = sf_indices[idx]
            exp = sf_data[idx]
            if exp == 0:
                continue
            # product of forward terms excluding species s:
            prod_except_s = 1.0
            for k in range(sf_start, sf_stop):
                j = sf_indices[k]
                ej = sf_data[k]
                if j == s:
                    # multiply by y[s]^(ej-1) if ej>1, else multiply by 1 (ej==1)
                    if ej - 1 > 0:
                        prod_except_s *= y[j] ** (ej - 1)
                    else:
                        prod_except_s *= 1.0
                else:
                    prod_except_s *= y[j] ** ej
            dfr = kd[r] * exp * prod_except_s

            # distribute to stoichiometric rows (vT: reaction-row CSR)
            for jdx in range(vT_indptr[r], vT_indptr[r+1]):
                i = vT_indices[jdx]     # species row index
                coeff = vT_data[jdx]
                rows[pos] = i
                cols[pos] = s
                values[pos] = coeff * dfr
                pos += 1

        for idx in range(sb_start, sb_stop):
            s = sb_indices[idx]
            exp = sb_data[idx]
            if exp == 0:
                continue
            prod_except_s = 1.0
            for k in range(sb_start, sb_stop):
                j = sb_indices[k]
                ej = sb_data[k]
                if j == s:
                    if ej - 1 > 0:
                        prod_except_s *= y[j] ** (ej - 1)
                    else:
                        prod_except_s *= 1.0
                else:
                    prod_except_s *= y[j] ** ej
            dbr = -kr[r] * exp * prod_except_s

            for jdx in range(vT_indptr[r], vT_indptr[r+1]):
                i = vT_indices[jdx]
                coeff = vT_data[jdx]
                rows[pos] = i
                cols[pos] = s
                values[pos] = coeff * dbr
                pos += 1

    return pos

def analyze_elemental_balance(mkm_results: Union[dict, str], inters):
    """
    Analyze convergence of microkinetic simulation. To be converged, 
    the sum of coverages, and the ratio of in/out elemental flows to the surface
    must be 1.
    Args:
        mkm_results(dict): Output dict generated by kinetic simulation.
        inters(dict): dictionary of Intermediate objects.
    Returns:
        in_div_out(dict): if key is "*", the value is the sum of the coverages (must be 1), 
                            if key is an element ("C", "H", etc.), the value is the input/output
                            ratio of the elemental flows. When the kinetic simulation is converged, these values should
                            be around 1.
    """
    if isinstance(mkm_results, str):
        with open(mkm_results, "rb") as f:
            mkm_results = load(f)

    rows = []
    for k, inter in inters.items():
        if inter.phase == "gas" and k in mkm_results["inters"]:
            idx = mkm_results["inters"].index(k)
            rows.append({
                "formula": inter.formula,
                "code": inter.code,
                "consumption_rate": mkm_results["total_consumption_rate"][idx],
                "C": inter["C"],
                "H": inter["H"],
                "O": inter["O"],
                "N": inter["N"]
            })

    df = pd.DataFrame(rows)

    outflow = (df[["C", "H", "O"]].mul(df["consumption_rate"].clip(lower=0), axis=0)).sum()
    inflow = (df[["C", "H", "O"]].mul(df["consumption_rate"].clip(upper=0), axis=0)).sum()
    in_div_out = (inflow.abs() / outflow).round(2).astype(float).to_dict()

    for elem in ["C", "H", "O"]:
        if inflow[elem] == 0 and outflow[elem] == 0:
            in_div_out[elem] = 1.0
    gas_mask = mkm_results["gas_mask"]
    in_div_out["*"] =  sum(mkm_results["y"][~gas_mask])
    return in_div_out

