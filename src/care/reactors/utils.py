from pickle import load
from typing import Union, Dict, Any

import numpy as np
import pandas as pd
from numba import njit
from sklearn.linear_model import LinearRegression

from care.constants import R


@njit
def net_rate(y: np.ndarray, kd: np.ndarray, kr: np.ndarray,
             sf_data: np.ndarray, sf_indices: np.ndarray, sf_indptr: np.ndarray,
             sb_data: np.ndarray, sb_indices: np.ndarray, sb_indptr: np.ndarray) -> np.ndarray:
    rates = np.empty_like(kd)
    n_reactions = kd.shape[0]

    for i in range(n_reactions):
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


def generate_simulation_report(results_dict: Dict[str, Any], 
                               output_filename: str = "simulation_report.xlsx") -> None:
    """
    Generate a structured Excel report with four sheets: Species, Reactions, Settings, and Activity.

    Args:
        results_dict (Dict[str, Any]): Dictionary containing simulation results and metadata.
        output_filename (str): Name of the output Excel file.
    Returns:
        None
    """

    print(f"Generating report: {output_filename}...")
    species_df, reactions_df, performance_df = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    # Use ExcelWriter for multi-sheet output
    with pd.ExcelWriter(output_filename, engine='openpyxl', mode="w") as writer:
        species_info = results_dict["inters_info"]
        species_df["idx"] = list(range(len(species_info['codes'])))
        species_df["InChIKey"] = species_info['codes']
        species_df["formula"] = species_info['formulas']
        species_df["phase"] = ["gas" if x == 1 else "adsorbed" for x in results_dict['gas_mask']]
        for elem in species_info['elements']:
            species_df[f"n{elem}"] = species_info[elem] + [0]
        species_df["theta0"] = results_dict['y0']
        species_df["theta"] = results_dict['y']
        species_df["unit"] = ["Pa" if x == 1 else "-" for x in results_dict['gas_mask']]
        species_df["reactants"] = ["True" if i in results_dict['reactants_idxs'] else "False" for i, x in enumerate(species_info['codes'])]
        species_df["products"] = ["True" if i in results_dict['products_idxs'] else "False" for i, x in enumerate(species_info['codes'])]
        species_df["formation/consumption rate (1/s)"] = results_dict['total_consumption_rate']

        species_df.to_excel(writer, sheet_name='Species', index=True)


        # --- Sheet 2: Reaction Information (Simple Table) ---
        reactions_df["idx"] = list(range(len(results_dict["kf"])))
        reactions_df["reaction"] = results_dict["rxn_strings"]
        reactions_df["kdir"] = results_dict["kf"]
        reactions_df["krev"] = results_dict["kr"]
        reactions_df["forward rate (1/s)"] = results_dict["forward_rate"]
        reactions_df["backward rate (1/s)"] = results_dict["backward_rate"]
        reactions_df["net rate (1/s)"] = results_dict["net_rate"]
        reactions_df["reversibility"] = results_dict["reversibility"]
        reactions_df.to_excel(writer, sheet_name='Reactions', index=True)
        print("  - Sheet 'Reaction Info' created.")

        # --- Sheet 3: Summary of performance metrics ---
        SHEET = 'Activity'
        current_row = 0
        row_labels = [species_info["formulas"][i] for i in results_dict['reactants_idxs']]
        column_labels = [species_info["formulas"][i] for i in results_dict['products_idxs']]
        # conversion (vector)
        conversion_df = pd.DataFrame(results_dict["conversion"]*100.0, index=row_labels, columns=["Conversion (%)"])
        name = 'Conversion (%)'
        df = pd.DataFrame({0: [f"{name}"]})
        df.to_excel(writer, sheet_name=SHEET, index=False, header=False, startrow=current_row)
        current_row += 2
        conversion_df.to_excel(writer, sheet_name=SHEET, startrow=current_row)
        current_row += conversion_df.shape[0] + 3
        #selectivity
        for elem, matrix in results_dict["selectivity"].items():
            performance_df = pd.DataFrame(matrix*100.0, index=row_labels, columns=column_labels)
            name = f'Selectivity ({elem}-based, %)'
            df = pd.DataFrame({0: [f"{name}"]})
            df.to_excel(writer, sheet_name=SHEET, index=False, header=False, startrow=current_row)
            current_row += 2
            performance_df.to_excel(writer, sheet_name=SHEET, startrow=current_row)
            current_row += performance_df.shape[0] + 3
        # yield
        for elem, matrix in results_dict["yield"].items():
            performance_df = pd.DataFrame(matrix*100.0, index=row_labels, columns=column_labels)
            name = f'Yield ({elem}-based, %)'
            df = pd.DataFrame({0: [f"{name}"]})
            df.to_excel(writer, sheet_name=SHEET, index=False, header=False, startrow=current_row)
            current_row += 2
            performance_df.to_excel(writer, sheet_name=SHEET, startrow=current_row)
            current_row += performance_df.shape[0] + 3

        # --- Sheet 4: Simulation settings ---
        settings = {
            "Temperature (K)": results_dict['T'],
            "Pressure (Pa)": results_dict['P'],
            "Applied potential (V vs RHE)": results_dict.get('U', 'N/A'),
            "pH (-)": results_dict.get('pH', 'N/A'),
            "Number of species": len(species_info['codes']),
            "Number of reactions": len(results_dict["kf"]),
            "Reactor model": "Differential PFR", 
            "Material": results_dict.get("surface", "N/A"),
            "ODE backend": "Python Scipy" if results_dict.get("solver", None) == "Python" else "Julia DifferentialEquations.jl",
            "ODE solver": "BDF" if results_dict.get("solver", None) == "Python" else results_dict.get("jl_solver", "N/A"),
            "Simulation time (s)": results_dict.get("time", "N/A"),
        }
        for elem in species_info['elements']:
            settings[f"{elem} elemental balance (IN/OUT)"] = results_dict[f'in_div_out_{elem}']
        settings_df = pd.DataFrame(list(settings.items()), columns=["Setting", "Value"])
        settings_df.to_excel(writer, sheet_name='Settings', index=False)      
            
    print(f"Report successfully generated at: **{output_filename}**")
    return
