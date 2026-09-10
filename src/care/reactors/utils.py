from dataclasses import dataclass, field
from pickle import load
from typing import Tuple, Union, Dict, Any, List, Optional

import numpy as np
import pandas as pd
from numba import njit
from scipy.constants import physical_constants
from scipy.sparse import spmatrix

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


def calc_eapp(T: float, dT: float, rates_zero: np.ndarray, rates_minus: np.ndarray, rates_plus: np.ndarray, gas_mask: np.ndarray, products_idxs: list[int] = None) -> np.ndarray:
    """
    Evaluates the apparent activation energy for specified product species using central finite differences.
    
    Args:
        T (float): Base temperature in Kelvin.
        dT (float): Temperature delta used for the plus/minus runs.
        rates_zero (ndarray): Net rates at T.
        rates_minus (ndarray): Net rates at T - dT.
        rates_plus (ndarray): Net rates at T + dT.
        gas_mask (ndarray): Boolean mask indicating gas-phase species.
        products_idxs (list[int], optional): Indices of the product species to evaluate. 
                                             If None, evaluates all gas species with a positive rate.
                                             
    Returns:
        ndarray: Apparent activation energies in kJ/mol. Non-evaluated species are set to np.nan.
    """
    T_plus = T + dT
    T_minus = T - dT
    R_kJ = R / 1000.0  # kJ/(mol*K)
    tol = 1e-40
    eapp = np.full(len(gas_mask), np.nan)

    if products_idxs is None:
        products_idxs = np.where(gas_mask & (rates_zero > tol))[0]

    for i in products_idxs:
        r_plus = rates_plus[i]
        r_minus = rates_minus[i]
        
        if r_plus > tol and r_minus > tol:
            d_ln_r = np.log(r_plus) - np.log(r_minus)
            d_inv_T = (1.0 / T_plus) - (1.0 / T_minus)
            
            eapp[i] = -R_kJ * (d_ln_r / d_inv_T)
            
    return eapp

def calc_napp(y_minus: float, y_plus: float, rates_minus: np.ndarray, rates_plus: np.ndarray, products_idxs: list[int]) -> np.ndarray:
    """
    Evaluates the apparent reaction order for product species using finite differences.
    
    Args:
        y_minus (float): The perturbed initial partial pressure  at -dy.
        y_plus (float): The perturbed initial partial pressure at +dy.
        rates_minus (ndarray): Net rates resulting from y_minus.
        rates_plus (ndarray): Net rates resulting from y_plus.
        products_idxs (list[int]): Indices of the product species to evaluate.
        
    Returns:
        ndarray: Apparent reaction orders. Non-evaluated species are set to np.nan.
    """
    napp = np.full(len(rates_minus), np.nan)
    tol  = 1e-40
    
    for i in products_idxs:
        r_plus = rates_plus[i]
        r_minus = rates_minus[i]

        if r_plus > tol and r_minus > tol:
            d_ln_r = np.log(r_plus) - np.log(r_minus)
            d_ln_y = np.log(y_plus) - np.log(y_minus)
            
            napp[i] = d_ln_r / d_ln_y
            
    return napp

def calc_drc(
    T: float, 
    e_minus: float, 
    e_zero: float, 
    e_plus: float, 
    rates_minus: np.ndarray, 
    rates_zero: np.ndarray, 
    rates_plus: np.ndarray, 
    products_idxs: list[int]
) -> np.ndarray:
    """
    Evaluates Campbell's Degree of Rate Control (DRC) for product species.
    Handles barrierless reactions by dynamically falling back to forward finite 
    differences if the negative perturbation yields identical rates to the unperturbed state.
    """
    kB_eV = physical_constants['Boltzmann constant in eV/K'][0]
    
    drc = np.full(len(rates_zero), np.nan)
    tol = 1e-40
    
    for i in products_idxs:
        r_plus = rates_plus[i]
        r_zero = rates_zero[i]
        r_minus = rates_minus[i] if rates_minus is not None else None

        if r_plus > tol and r_zero > tol:
            if r_minus is None:
                d_ln_r = np.log(r_plus) - np.log(r_zero)
                d_e = e_plus - e_zero
                
            elif r_minus > tol:
                d_ln_r = np.log(r_plus) - np.log(r_minus)
                d_e = e_plus - e_minus 
                
            else:
                continue

            drc[i] = -kB_eV * T * (d_ln_r / d_e)
            
    return drc

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

def analyze_elemental_balance(mkm_results: Union[dict, str]) -> dict:
    """
    Analyze convergence of microkinetic simulation. To be converged, 
    the sum of coverages, and the ratio of in/out elemental flows to the surface
    must be 1.
    Args:
        mkm_results(dict): Output dict generated by kinetic simulation.
    Returns:
        in_div_out(dict): if key is "*", the value is the sum of the coverages (must be 1), 
                            if key is an element ("C", "H", etc.), the value is the input/output
                            ratio of the elemental flows. When the kinetic simulation is converged, these values should
                            be around 1.
    """
    if isinstance(mkm_results, str):
        with open(mkm_results, "rb") as f:
            mkm_results = load(f)
    elements = mkm_results["inters_info"]["elements"]
    codes = mkm_results["inters_info"]["codes"]
    formulas = mkm_results["inters_info"].get("formulas", codes)
    gas_mask = mkm_results["gas_mask"]
    inert_mask = mkm_results["inert_mask"]
    rows = []
    nc = len(mkm_results["y"])
    for i in range(nc):
        if gas_mask[i] and not inert_mask[i]:
            rows.append({
                "formula": formulas[i],
                "code": codes[i],
                "formation_rate": mkm_results["total_formation_rate"][i],
                **{elem: mkm_results["inters_info"][elem][i] for elem in elements}
            })  
    df = pd.DataFrame(rows)
    for col in elements + ["formation_rate"]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    outflow = (df[elements].mul(df["formation_rate"].clip(lower=0), axis=0)).sum()
    inflow = (df[elements].mul(df["formation_rate"].clip(upper=0), axis=0)).sum()
    in_div_out = (inflow.abs() / outflow).round(2).fillna(0).to_dict()

    for elem in elements:
        if inflow[elem] == 0 and outflow[elem] == 0:
            in_div_out[elem] = 1.0
    in_div_out["*"] =  sum(mkm_results["y"][~gas_mask])
    return in_div_out


def generate_simulation_report(results_dict: Dict[str, Any], 
                               output_filename: str = "simulation_report.xlsx") -> None:
    """
    Generate a structured Excel report with Sheets: Species, Reactions, Settings, Activity, 
    Apparent Kinetics, and Sensitivity Analysis.

    Args:
        results_dict (Dict[str, Any]): Dictionary containing simulation results and metadata.
        output_filename (str): Name of the output Excel file.
    Returns:
        None
    """

    print(f"Generating report: {output_filename}...")
    species_df, reactions_df = pd.DataFrame(), pd.DataFrame()

    with pd.ExcelWriter(output_filename, engine='openpyxl', mode="w") as writer:
        # --- Sheet 1: Species Information ---
        species_info = results_dict["inters_info"]
        species_df["idx"] = list(range(len(species_info['codes'])))
        species_df["InChIKey"] = species_info.get('codes', None)
        species_df["formula"] = species_info.get('formulas', species_info.get('codes', None))
        species_df["phase"] = ["gas" if x == 1 else "adsorbed" for x in results_dict['gas_mask']]
        for elem in species_info['elements']:
            species_df[f"n{elem}"] = species_info[elem]
        species_df["theta0"] = results_dict['y0']
        species_df["theta"] = results_dict['y']
        species_df["unit"] = ["Pa" if x == 1 else "-" for x in results_dict['gas_mask']]
        species_df["reactants"] = ["True" if i in results_dict['reactants_idxs'] else "False" for i, x in enumerate(species_info['codes'])]
        species_df["products"] = ["True" if i in results_dict['products_idxs'] else "False" for i, x in enumerate(species_info['codes'])]
        species_df["formation rate (1/s)"] = results_dict['total_formation_rate']
        species_df.to_excel(writer, sheet_name='Species', index=False)


        # --- Sheet 2: Reaction Information ---
        reactions_df["idx"] = list(range(len(results_dict["net_rate"])))
        reactions_df["reaction"] = results_dict.get("rxn_strings", None)
        reactions_df["kdir"] = results_dict.get("kf", None)
        reactions_df["krev"] = results_dict.get("kr", None)
        reactions_df["forward rate (1/s)"] = results_dict["forward_rate"]
        reactions_df["backward rate (1/s)"] = results_dict["backward_rate"]
        reactions_df["net rate (1/s)"] = results_dict["net_rate"]
        reactions_df["reversibility"] = results_dict["reversibility"]
        reactions_df.to_excel(writer, sheet_name='Reactions', index=False)


        # --- Sheet 3: Summary of performance metrics ---
        SHEET = 'Activity'
        current_row = 0
        x = species_info.get("formulas", species_info.get("codes", None))
        row_labels = [x[i] for i in results_dict['reactants_idxs']]
        column_labels = [x[i] for i in results_dict['products_idxs']]
        
        # conversion (vector)
        conversion_df = pd.DataFrame(results_dict["conversion"]*100.0, index=row_labels, columns=["Conversion (%)"])
        name = 'Conversion (%)'
        df = pd.DataFrame({0: [f"{name}"]})
        df.to_excel(writer, sheet_name=SHEET, index=False, header=False, startrow=current_row)
        current_row += 2
        conversion_df.to_excel(writer, sheet_name=SHEET, startrow=current_row)
        current_row += conversion_df.shape[0] + 3
        
        # selectivity
        for elem, matrix in results_dict["selectivity"].items():
            if np.isnan(matrix).all(): continue
            performance_df = pd.DataFrame(matrix*100.0, index=row_labels, columns=column_labels)
            name = f'Selectivity ({elem}-based, %)'
            df = pd.DataFrame({0: [f"{name}"]})
            df.to_excel(writer, sheet_name=SHEET, index=False, header=False, startrow=current_row)
            current_row += 2
            performance_df.to_excel(writer, sheet_name=SHEET, startrow=current_row)
            current_row += performance_df.shape[0] + 3
            
        # yield
        for elem, matrix in results_dict.get("yield", {}).items():
            if np.isnan(matrix).all(): continue
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
            "Number of reactions": len(results_dict["net_rate"]),
            "Reactor model": "Differential PFR", 
            "Material": results_dict.get("surface", "N/A"),
            "ODE backend": "Python Scipy" if results_dict.get("solver", None) == "Python" else "Julia DifferentialEquations.jl",
            "ODE solver": "BDF" if results_dict.get("solver", None) == "Python" else results_dict.get("jl_solver", "N/A"),
            "Simulation time (s)": results_dict.get("time", "N/A"),
        }
        for elem in species_info['elements']:
            key = f'in_div_out_{elem}'
            if key in results_dict:
                settings[f"{elem} elemental balance (IN/OUT)"] = results_dict[key]
        settings_df = pd.DataFrame(list(settings.items()), columns=["Setting", "Value"])
        settings_df.to_excel(writer, sheet_name='Settings', index=False)


        # --- Sheet 5: Apparent kinetics (Eapp and napp if available) ---
        if "eapp_dict" in results_dict or "napp_dict" in results_dict:
            SHEET_KIN = 'Apparent Kinetics'
            current_row = 0
            
            if "eapp_dict" in results_dict and results_dict["eapp_dict"]:
                df_title = pd.DataFrame({0: ["Apparent Activation Energy (kJ/mol)"]})
                df_title.to_excel(writer, sheet_name=SHEET_KIN, index=False, header=False, startrow=current_row)
                current_row += 2
                eapp_df = pd.DataFrame(list(results_dict["eapp_dict"].items()), columns=["Product", "Eapp (kJ/mol)"])
                eapp_df.set_index("Product", inplace=True)
                eapp_df.to_excel(writer, sheet_name=SHEET_KIN, startrow=current_row)
                current_row += eapp_df.shape[0] + 3

            if "napp_dict" in results_dict and results_dict["napp_dict"]:
                df_title = pd.DataFrame({0: ["Apparent Reaction Orders"]})
                df_title.to_excel(writer, sheet_name=SHEET_KIN, index=False, header=False, startrow=current_row)
                current_row += 2
                # Transpose gives rows = Reactants, columns = Products
                napp_df = pd.DataFrame(results_dict["napp_dict"]).T
                napp_df.to_excel(writer, sheet_name=SHEET_KIN, startrow=current_row)


        # --- Sheet 6: Sensitivity analysis (DRC/DSC if available) ---
        if "drc_dict" in results_dict or "dsc_dict" in results_dict:
            SHEET_SENS = 'Sensitivity Analysis'
            current_row = 0
            
            if "drc_dict" in results_dict and results_dict["drc_dict"]:
                df_title = pd.DataFrame({0: ["Degree of Rate Control (DRC)"]})
                df_title.to_excel(writer, sheet_name=SHEET_SENS, index=False, header=False, startrow=current_row)
                current_row += 2
                drc_df = pd.DataFrame(results_dict["drc_dict"]).T
                drc_df.to_excel(writer, sheet_name=SHEET_SENS, startrow=current_row)
                current_row += drc_df.shape[0] + 3

            if "dsc_dict" in results_dict and results_dict["dsc_dict"]:
                for ref_name, targets_dict in results_dict["dsc_dict"].items():
                    df_title = pd.DataFrame({0: [f"Degree of Selectivity Control (DSC) - Ref: {ref_name}"]})
                    df_title.to_excel(writer, sheet_name=SHEET_SENS, index=False, header=False, startrow=current_row)
                    current_row += 2
                    dsc_df = pd.DataFrame(targets_dict)
                    dsc_df.to_excel(writer, sheet_name=SHEET_SENS, startrow=current_row)
                    current_row += dsc_df.shape[0] + 3

    print(f"Report successfully generated at: {output_filename}")
    return

@dataclass
class MKMRun:
    """
    Data container for Microkinetic Modeling simulation results.
    """
    # ==========================================
    # FIELDS WITHOUT DEFAULTS
    # ==========================================
    
    # --- Operating Conditions ---
    T: float
    P: float
    
    # --- Integration Results ---
    t: float
    time: float
    status: int
    y0: np.ndarray
    y: np.ndarray
    
    # --- Kinetic Rates ---
    kf: np.ndarray
    kr: np.ndarray
    forward_rate: np.ndarray
    backward_rate: np.ndarray
    reversibility: np.ndarray
    net_rate: np.ndarray
    formation_rate: spmatrix
    total_formation_rate: np.ndarray
    
    # --- Network Config & Masks ---
    v: spmatrix
    formulas: List[str]
    inters: List[str]
    inters_info: Dict[str, Any]
    rxn_strings: List[str]
    gas_mask: np.ndarray
    inert_mask: np.ndarray
    reactants_idxs: np.ndarray
    products_idxs: np.ndarray
    
    # --- Reactor Performance Metrics ---
    conversion: np.ndarray
    selectivity: Dict[str, np.ndarray]
    yyield: Dict[str, np.ndarray]
    
    # --- Solver Metadata ---
    solver: str
    jl_solver: str
    precision: str
    rtol: float
    atol: float
    tfin: float
    nsims: int

    # ==========================================
    # FIELDS WITH DEFAULTS
    # ==========================================
    
    U: Optional[float] = None
    pH: Optional[float] = None
    maxiters: Optional[int] = None
    clip_eact: float = -1.0
    material: str = "N/A"
    catalyst_mass: float = 0.0
    balances: Dict[str, float] = field(default_factory=dict)
    raw_eapp: Optional[np.ndarray] = None
    raw_napp: Optional[Dict[int, np.ndarray]] = None
    raw_drc: Optional[Dict[int, np.ndarray]] = None 

    @classmethod
    def from_dict(cls, data: dict) -> "MKMRun":
        """
        Convenience constructor to map the raw dictionary output directly 
        into MKMRun object.
        """
        if "yield" in data:
            data["yyield"] = data.pop("yield")

        balances = {
            k.replace('in_div_out_', ''): v 
            for k, v in data.items() if k.startswith('in_div_out_')
        }

        if "eapp" in data:
            data["raw_eapp"] = data.pop("eapp")
        
        if "napp" in data:
            data["raw_napp"] = data.pop("napp")

        if "drc" in data:
            data["raw_drc"] = data.pop("drc")

        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}

        filtered_data["balances"] = balances
        
        return cls(**filtered_data)

    @property
    def coverages(self) -> Dict[str, float]:
        """Returns a clean dictionary of surface species and their final coverages."""
        return {
            self.formulas[i]: self.y[i]
            for i in range(len(self.y))
            if not self.gas_mask[i] and self.y[i] > 1e-10
        }

    def __str__(self) -> str:
        """Provides a beautiful, human-readable summary of the run."""
        output = [
            "==================================================",
            "                 MKM RUN SUMMARY                  ",
            "==================================================",
            f"Material:    {self.material}",
            f"Temperature: {self.T} K",
            f"Pressure:    {self.P / 1e5:.2f} bar",
            "",
            "--- Performance ---",
            f"Time Elapsed: {self.time:.4f} s",
            f"Simulated t:  {self.t:.2e} s",
            f"Status:       {'Success' if self.status == 1 else 'Failed'}",
            f"Solver:       {self.solver} ({self.jl_solver}, {self.precision})",
            "",
            "--- Top Surface Coverages ---"
        ]
        
        sorted_cov = sorted(self.coverages.items(), key=lambda item: item[1], reverse=True)
        for formula, cov in sorted_cov[:5]: 
            output.append(f"  {formula:<12} {cov:.4e}")
            
        output.append("==================================================")
        return "\n".join(output)
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def get_performance_summary(self) -> str:
        """Formats conversion, selectivity, yield, and apparent activation energies into readable tables."""
        reactants = [self.formulas[i] for i in self.reactants_idxs]
        products = [self.formulas[i] for i in self.products_idxs]

        lines = ["\n==================================================",
                 "              PERFORMANCE SUMMARY                 ",
                 "=================================================="]
                 
        # Conversion
        lines.append("\n--- Conversion ---")
        for r_name, conv in zip(reactants, self.conversion):
            lines.append(f"  {r_name:<10}: {conv * 100:>7.2f}%")

        # Formation/Consumption Rates
        lines.append("\n--- Reaction Rates (s⁻¹) ---")
        lines.append("  Reactants (Consumption):")
        for r_name, rate in self.reactant_consumption_rates.items():
            lines.append(f"    {r_name:<10}: {rate:>10.3e}")
            
        lines.append("  Products (Formation):")
        for p_name, rate in self.product_formation_rates.items():
            lines.append(f"    {p_name:<10}: {rate:>10.3e}")

        eapp_dict = self.eapp
        if eapp_dict:
            lines.append(f"\n--- Apparent Activation Energy (kJ/mol) at {self.T} K ---")
            for p_name, val in eapp_dict.items():
                lines.append(f"  {p_name:<10}: {val:>7.2f}")

        napp_dict = self.napp
        if napp_dict:
            lines.append("\n--- Apparent Reaction Orders ---")
            header = f"  {'Reactant':<10}" + "".join([f"{p:>10}" for p in products])
            lines.append(header)

            for r_name, p_orders in napp_dict.items():
                row_str = f"  {r_name:<10}"
                for p_name in products:
                    val = p_orders.get(p_name, np.nan)
                    if np.isnan(val):
                        row_str += f"{'N/A':>10}"
                    else:
                        row_str += f"{val:>10.2f}"
                lines.append(row_str)

        metrics = [("Selectivity", self.selectivity), ("Yield", self.yyield)]
        
        for metric_name, metric_dict in metrics:
            for elem, matrix in metric_dict.items():
                if np.isnan(matrix).all():
                    continue

                lines.append(f"\n--- {metric_name} ({elem}) ---")

                header = f"  {'':<10}" + "".join([f"{p:>10}" for p in products])
                lines.append(header)

                for i, r_name in enumerate(reactants):
                    row_str = f"  {r_name:<10}"
                    for j, p_name in enumerate(products):
                        val = matrix[i, j]
                        if np.isnan(val):
                            row_str += f"{'N/A':>10}"
                        else:
                            row_str += f"{val * 100:>9.2f}%"
                    lines.append(row_str)

        drc_dict = self.drc
        if drc_dict:
            lines.append("\n--- Degree of Rate Control (Threshold > 0.001) ---")
            header = f"  {'Reaction':<35}" + "".join([f"{p:>10}" for p in products])
            lines.append(header)
            
            for r_str, p_drcs in drc_dict.items():
                row_str = f"  {r_str:<35}"
                for p_name in products:
                    val = p_drcs.get(p_name, np.nan)
                    if np.isnan(val):
                        row_str += f"{'N/A':>10}"
                    else:
                        row_str += f"{val:>10.3f}"
                lines.append(row_str)

            lines.append("  " + "-" * (35 + 10 * len(products)))
            sum_str = f"  {'Sum':<35}"
            for p_idx in self.products_idxs:
                total_drc = sum(
                    float(drc_array[p_idx]) 
                    for drc_array in self.raw_drc.values() 
                    if drc_array[p_idx] is not None and not np.isnan(drc_array[p_idx])
                )
                sum_str += f"{total_drc:>10.3f}"
            lines.append(sum_str)

        dsc_dict = self.dsc
        if dsc_dict:
            for r_name, p_dscs in dsc_dict.items():
                lines.append(f"\n--- Degree of Selectivity Control (ref: {r_name}) ---")
                header = f"  {'Reaction':<35}" + "".join([f"{p:>10}" for p in products])
                lines.append(header)

                active_rxns = set()
                for rxn_dict in p_dscs.values():
                    active_rxns.update(rxn_dict.keys())

                for r_str in self.rxn_strings:
                    if r_str not in active_rxns:
                        continue
                        
                    row_str = f"  {r_str:<35}"
                    for p_name in products:
                        val = p_dscs.get(p_name, {}).get(r_str, np.nan)
                        if np.isnan(val):
                            row_str += f"{'N/A':>10}"
                        else:
                            row_str += f"{val:>10.3f}"
                    lines.append(row_str)

                lines.append("  " + "-" * (35 + 10 * len(products)))
                sum_str = f"  {'Sum (Expected ~0.0)':<35}"
                for p_name in products:
                    total_dsc = sum(p_dscs.get(p_name, {}).values())
                    sum_str += f"{total_dsc:>10.3f}"
                lines.append(sum_str)
                    
        lines.append("\n==================================================")
        return "\n".join(lines)
    
    def to_dict(self) -> dict:
        from dataclasses import asdict
        
        res = asdict(self)

        res["selectivity"] = self.selectivity
        
        res["yield"] = self.yyield
        if "yyield" in res:
            del res["yyield"]
            
        res["surface"] = self.material
        
        if hasattr(self, "balances") and self.balances:
            for elem, val in self.balances.items():
                res[f"in_div_out_{elem}"] = val

        res["coverages"] = self.coverages
        res["reactant_consumption_rates"] = self.reactant_consumption_rates
        res["product_formation_rates"] = self.product_formation_rates

        if self.raw_eapp is not None:
            res["eapp_dict"] = self.eapp
            
        if self.raw_napp is not None:
            res["napp_dict"] = self.napp

        if self.raw_drc is not None:
            res["drc_dict"] = self.drc
            res["rds_dict"] = self.rds
            res["dsc_dict"] = self.dsc
            
        return res
    
    def export(self, filename:str="mkm_report.xlsx") -> None:
        """
        Generate Excel file with simulation results.
        """
        if not filename.endswith(".xlsx"):
            filename += ".xlsx"
        generate_simulation_report(self.to_dict(), filename)
    
    @property
    def reactant_consumption_rates(self) -> Dict[str, float]:
        """Returns the consumption rate (s⁻¹) for each reactant."""
        return {
            self.formulas[i]: float(-self.total_formation_rate[i])
            for i in self.reactants_idxs
        }

    @property
    def product_formation_rates(self) -> Dict[str, float]:
        """Returns the formation rate (s⁻¹) for each product."""
        return {
            self.formulas[i]: float(self.total_formation_rate[i])
            for i in self.products_idxs
        }
    
    @property
    def eapp(self) -> Dict[str, float]:
        """
        Returns a dictionary mapping product formulas to their 
        apparent activation energies (kJ/mol).
        """
        if self.raw_eapp is None:
            return {}
            
        return {
            self.formulas[i]: self.raw_eapp[i]
            for i in self.products_idxs
            if not np.isnan(self.raw_eapp[i])
        }
    
    @property
    def napp(self) -> Dict[str, Dict[str, float]]:
        if self.raw_napp is None:
            return {}
            
        napp_dict = {}
        for r_idx, orders_array in self.raw_napp.items():
            r_idx_int = int(r_idx)
            r_formula = self.formulas[r_idx_int]
            
            product_orders = {}
            for p_idx in self.products_idxs:
                val = orders_array[p_idx]
                if val is not None and not np.isnan(val):
                    product_orders[self.formulas[p_idx]] = float(val)
                    
            if product_orders:
                napp_dict[r_formula] = product_orders
                
        return napp_dict
    
    @property
    def drc(self) -> Dict[str, Dict[str, float]]:
        if self.raw_drc is None:
            return {}
            
        drc_dict = {}
        for r_idx, drc_array in self.raw_drc.items():
            r_idx_int = int(r_idx)
            r_str = self.rxn_strings[r_idx_int]
            
            product_drcs = {}
            for p_idx in self.products_idxs:
                val = drc_array[p_idx]
                if val is not None and not np.isnan(val) and abs(val) > 1e-6:
                    product_drcs[self.formulas[p_idx]] = float(val)
                    
            if product_drcs:
                drc_dict[r_str] = product_drcs
                
        return drc_dict
    
    @property
    def rds(self) -> Dict[str, Tuple[int, str, float]]:
        """
        Returns a dictionary mapping each product formula to a tuple of:
        the RDS index, the string representation of the RDS and its Degree of Rate Control (DRC).
        """
        if self.raw_drc is None or not self.raw_drc:
            return {}
            
        rds_dict = {}
        for p_idx in self.products_idxs:
            p_name = self.formulas[p_idx]
            max_drc = -np.inf
            rds_rxn = None
            best_r_idx = None
            
            for r_idx, drc_array in self.raw_drc.items():
                r_idx_int = int(r_idx) 
                val = drc_array[p_idx]

                if val is not None and not np.isnan(val) and val > max_drc:
                    max_drc = float(val)
                    rds_rxn = self.rxn_strings[r_idx_int]
                    best_r_idx = r_idx_int

            if rds_rxn is not None and max_drc > 1e-6:
                rds_dict[p_name] = (best_r_idx, rds_rxn, max_drc)
                
        return rds_dict
    
    @property
    def dsc(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Degree of selectivity control.
        Format: {'Reference Product': {'Target Product': {'Reaction String': DSC_value}}}
        """
        if self.raw_drc is None or not self.raw_drc:
            return {}
            
        dsc_dict = {}

        for ref_idx in self.products_idxs:
            ref_name = self.formulas[ref_idx]
            target_dscs = {}
            
            for p_idx in self.products_idxs:
                if ref_idx == p_idx:
                    continue
                    
                p_name = self.formulas[p_idx]
                reaction_dscs = {}
                
                for rxn_idx, drc_array in self.raw_drc.items():
                    rxn_idx_int = int(rxn_idx) 
                    p_drc = drc_array[p_idx]
                    ref_drc = drc_array[ref_idx]
                    
                    if p_drc is not None and ref_drc is not None and not np.isnan(p_drc) and not np.isnan(ref_drc):
                        dsc_val = float(p_drc) - float(ref_drc)
                        
                        if abs(dsc_val) > 1e-6:
                            reaction_dscs[self.rxn_strings[rxn_idx_int]] = dsc_val
                            
                if reaction_dscs:
                    target_dscs[p_name] = reaction_dscs
                    
            if target_dscs:
                dsc_dict[ref_name] = target_dscs
                
        return dsc_dict


