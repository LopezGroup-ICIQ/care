"""
Differential Plug-Flow Reactor (PFR) model.

Being it a zero-conversion model, conversion (X) is zero by definition, 
consequently yields (Y = X*S) are also zero. TOF and selectivity
can be evaluated, as well as apparent activation energy and reaction orders.
"""

import os
from pathlib import Path
from pickle import load
from time import time
from typing import Optional, Union, Dict
import warnings

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import isspmatrix_csr, vstack, csr_matrix

from care import ReactionNetwork
from care.crn.intermediate import GasSpecies
from care.constants import INTER_ELEMS
from care.reactors.reactor import ReactorModel
from care.reactors.utils import (net_rate, 
                                 jacobian_fill_numba, 
                                 calc_eapp, 
                                 calc_napp,
                                 calc_drc, 
                                 analyze_elemental_balance, 
                                 MKMRun)


class DifferentialPFR(ReactorModel):
    _jl = None

    def __init__(
        self,
        crn: ReactionNetwork,
        P: float = 100000.0,
        T: float = 298.0,
        pH: float = 7.0, 
        U: float = 0.0,
        print_progress: bool = True,
    ):
        """
        Differential Plug-Flow Reactor (PFR) model.
        """

        self.crn = crn
        self.reactions = self.crn.reactions
        self.nr = len(self.reactions)
        
        self.intermediates = self.crn.intermediates

        self._kd = None  
        self._kr = None  
        self.clip_eact = -1.0

        self._reset_state()

        self.T = T
        self.P = P
        self.pH = pH
        self.U = U

        self.sstol = 0.01  
        self.sum_ddt, self.time = [], []
        self.print_progress = print_progress

    def _reset_state(self):
        if not isspmatrix_csr(self.crn.v):
            raise ValueError("Stoichiometric matrix v must be a SciPy sparse CSR matrix")
        
        self.v_sparse = self.crn.v.copy()
        self.v_forward_sparse = self.v_sparse.multiply(self.v_sparse < 0).multiply(-1).T.tocsr()
        self.v_backward_sparse = self.v_sparse.multiply(self.v_sparse > 0).T.tocsr()
        self.sparsity = (1 - self.v_sparse.nnz / (self.v_sparse.shape[0] * self.v_sparse.shape[1])) * 100

        sorted_inter_keys = sorted(list(self.intermediates.keys()))
        self.inters_code = sorted_inter_keys + ["*"]
        self.inters_formula = [self.intermediates[inter].formula for inter in sorted_inter_keys] + ["*"]
        self.gas_formulas = [
            self.intermediates[x].formula 
            for x in self.intermediates.keys() 
            if isinstance(self.intermediates[x], GasSpecies)
        ]
        self.gas_mask = np.array([self.intermediates[inter].phase == "gas" for inter in sorted_inter_keys] + [False])
        
        self.nc = self.v_sparse.shape[0]  # includes surface site *
        self.inert_mask = np.zeros(len(self.gas_mask), dtype=bool)
        self.elements = self.crn.elements
        
        inters_dict = {}
        inters_dict["codes"] = self.inters_code
        inters_dict["formulas"] = self.inters_formula
        for elem in self.elements:
            inters_dict[elem] = [self.intermediates[x][elem] for x in sorted_inter_keys] + [0]
        inters_dict["elements"] = self.elements
        self.inters_info = inters_dict

    def get_state_dict(self, y: np.ndarray) -> dict[str, float]:
        """
        Translates a raw state array into a human-readable dictionary 
        mapping species formulas to their molar fractions/coverages.
        """
        return dict(zip(self.inters_code, y))

    @property
    def T(self):
        "Reactor temperature in Kelvin."
        return self._T
    
    @T.setter
    def T(self, value: float):
        """Updating T invalidates the current kinetic constants."""
        if not (0 < value <= 10000):
            raise ValueError(f"Temperature must be strictly positive and <= 10000 K, got {value}.")
        self._T = value
        self._kd = None
        self._kr = None
    
    @property
    def P(self):
        "Reactor pressure in Pascal."
        return self._P
    
    @P.setter
    def P(self, value: float):
        if not (0 < value <= 1e10):
            raise ValueError(f"Pressure must be strictly positive and <= 1e10 Pa, got {value}.")
        self._P = value

    @property
    def pH(self):
        "Reactor pH."
        return self._pH
    
    @pH.setter
    def pH(self, value: float):
        if not (0 < value <= 14):
            raise ValueError(f"pH must be strictly positive and <= 14, got {value}.")
        self._pH = value

    @property
    def U(self):
        "Reactor applied potential in Volt."
        return self._U
    
    @U.setter
    def U(self, value: float):
        self._U = value

    @property
    def kd(self) -> np.ndarray:
        """Forward kinetic constants in s^-1."""
        if self._kd is None:
            self._update_kinetic_constants()
        return self._kd
    
    @property
    def kr(self) -> np.ndarray:
        """Backward kinetic constants in s^-1."""
        if self._kr is None:
            self._update_kinetic_constants()
        return self._kr
    
    @kd.setter
    def kd(self, value: np.ndarray):
        self._kd = value

    @kr.setter
    def kr(self, value: np.ndarray):
        self._kr = value
    
    def _update_kinetic_constants(self):
        """Re-evaluates kinetic constants based on current T and clip_eact."""
        self._kd = np.zeros(self.nr)
        self._kr = np.zeros(self.nr)
        
        for j, rxn in enumerate(self.reactions):
            self._kd[j], self._kr[j] = rxn.get_kinetic_constants(
                t=self.T, 
                clip_eact=self.clip_eact
            )

    def _resolve_species_index(self, identifier: str, is_gas: bool) -> int:
        """
        Resolves a user-provided string to a matrix index.
        Attempts an exact code/InChIKey match first, then falls back to a phase-filtered formula match.
        """
        if identifier in self.inters_code:
            idx = self.inters_code.index(identifier)
            expected_phase = "gas" if is_gas else "surface"
            if self.gas_mask[idx] != is_gas:
                raise ValueError(f"Species '{identifier}' found, but it is not a {expected_phase} species.")
            return idx

        matches = [
            i for i, f in enumerate(self.inters_formula) 
            if f == identifier and self.gas_mask[i] == is_gas
        ]

        if not matches:
            return -1
        if len(matches) > 1:
            clashing_keys = [self.inters_code[i] for i in matches]
            raise ValueError(
                f"Formula '{identifier}' is ambiguous (multiple isomers found). "
                f"Please use the exact InChIKey instead: {clashing_keys}"
            )
            
        return matches[0]

    def run(
        self,
        iv: Optional[Dict[str, float]] = None,
        cov0: Optional[Dict[str, float]] = None,
        mkm_checkpoint: Optional[Union[str, Path]] = None,
        solver: str = "Julia",
        tfin: float = 1e30,
        atol: float = 1e-15,
        rtol: float = 1e-12,
        clip_eact: float = -1.0,
        eapp: bool = False,
        dT: float = 1.0,
        napp: bool = False,
        dy: float = 0.01,
        drc: bool = False,
        de: float = 0.02, 
        rewire_network: bool = False,
        **kwargs
    ) -> MKMRun:
        """
        Run kinetic simulation up to steady-state.

        This method orchestrates the full integration workflow, including initialization
        of partial pressures, optional dynamic perturbations to estimate apparent activation 
        energies (E_app) and reaction orders (n_app), and final packaging of the results.

        Args:
            iv (Optional[Dict[str, float]]): Inlet molar fractions of gas-phase species.
                Keys are chemical formulas, values are molar fractions (must sum to 1.0).
            cov0 (Optional[Dict[str, float]]): Initial surface coverage dictionary. If None, 
                th surface is considered uncovered.
            mkm_checkpoint (Optional[Union[str, Path]]): Path to a saved MKM run (.pkl). 
                If provided, `iv` and `oc` are bypassed.
            solver (str): ODE solver backend ("Julia" or "Python"). Default is "Julia".
            tfin (float): Final integration time [s]. Default is 1e30.
            atol (float): Absolute tolerance for the ODE solver. Default is 1e-15.
            rtol (float): Relative tolerance for the ODE solver. Default is 1e-12.
            clip_eact (float): Threshold for barrierless clipping (eV). Negative disables clipping.
            eapp (bool): If True, computes apparent activation energies.
            dT (float): Temperature step (K) used for E_app estimation. Default is 1.0 K.
            napp (bool): If True, computes apparent reaction orders. 
                Requires at least one inert diluent in `iv`.
            dy (float): Molar fraction step used for n_app estimation. Default is 0.01.
            drc(bool): If True, computes degree of rate control for all elementary reactions.
            de(float): Energy step in eV for drc estimation. Default to 0.01.
            rewire_network (bool): If True, reverses elementary reactions in the ReactionNetwork
            according to the simulation output. Default is True.
            **kwargs: Additional parameters passed to the backend solver.

        Returns:
            MKMRun: Data container holding kinetics, rates, and performance metrics.
            
        Raises:
            FileNotFoundError: If the provided `mkm_checkpoint` path does not exist.
            ValueError: If neither a checkpoint nor valid `iv`/`oc` dictionaries are provided.

        Notes:
            - The degree of rate control analysis is extremely sensitive on the perturbation applied.
        """
        self._reset_state()
        self.clip_eact = clip_eact
        
        if mkm_checkpoint:
            mkm_checkpoint = Path(mkm_checkpoint)
            if mkm_checkpoint.is_file():
                with open(mkm_checkpoint, "rb") as f:
                    inputs = load(f)
                y0 = inputs["y0"]
                print(f"Starting integration from loaded MKM checkpoint: {mkm_checkpoint}")
            else:
                raise FileNotFoundError(f"Checkpoint not found at: {mkm_checkpoint}")
        elif iv is None:
            raise ValueError("Either mkm_checkpoint or reactant mixture composition `iv` must be provided.")
        else:
            if not np.isclose(sum(iv.values()), 1.0):
                raise ValueError("Sum of molar fractions in `iv` must equal 1.0.")

            inlet_reacting_codes = set()
            for identifier in iv.keys():
                idx = self._resolve_species_index(identifier, is_gas=True)
                if idx != -1:
                    inlet_reacting_codes.add(self.inters_code[idx])
                    
            if not inlet_reacting_codes:
                raise ValueError(
                    "The input dictionary 'iv' must contain at least one valid reacting gas species. "
                    "All provided species were evaluated as inerts."
                )

            adsorptions = self.crn.adsorptions
            desorptions = self.crn.desorptions
            
            for i, reaction in enumerate(self.crn.reactions):
                if reaction in adsorptions:
                    if not any(inter.code in inlet_reacting_codes for inter in reaction.reactants if inter.phase == "gas"):
                        self.crn.reverse_reaction(i)
                elif reaction in desorptions:
                    if any(inter.code in inlet_reacting_codes for inter in reaction.products if inter.phase == "gas"):
                        self.crn.reverse_reaction(i)

            self._kd = None
            self._kr = None
                    
            self.v_sparse = self.crn.v.copy()
            self.v_forward_sparse = self.v_sparse.multiply(self.v_sparse < 0).multiply(-1).T.tocsr()
            self.v_backward_sparse = self.v_sparse.multiply(self.v_sparse > 0).T.tocsr()

            y0 = np.zeros(self.nc, dtype=np.float64)
            if cov0:
                sum_cov = sum(cov0.values())
                if sum_cov > 1.0:
                    raise ValueError("Initial surface coverages cannot sum to more than 1.0.")
                
                for identifier, coverage in cov0.items():
                    idx = self._resolve_species_index(identifier, is_gas=False)
                    if idx == -1:
                        raise ValueError(f"Surface poison '{identifier}' not found in the network.")
                    y0[idx] = coverage
                    
                y0[-1] = 1.0 - sum_cov
            else:
                y0[-1] = 1.0  

            inerts, inert_idx, inert_y0 = [], [], []
            
            for identifier, molar_fraction in iv.items():
                idx = self._resolve_species_index(identifier, is_gas=True)
                
                if idx == -1:
                    warnings.warn(f"Gas species '{identifier}' not found. Treating as inert.", UserWarning)
                    inerts.append(identifier)
                    inert_idx.append(len(y0))
                    inert_y0.append(self.P * molar_fraction)
                else:
                    y0[idx] = self.P * molar_fraction

            if inerts:
                y0 = np.concatenate([y0, inert_y0])
                self.gas_mask = np.concatenate([self.gas_mask, np.ones(len(inerts), dtype=bool)])
                self.inters_code += inerts
                self.inters_formula += inerts
                self.v_sparse = vstack([self.v_sparse, csr_matrix((len(inerts), self.nr), dtype=np.int8)]).tocsr()
                self.v_forward_sparse = self.v_sparse.multiply(self.v_sparse < 0).multiply(-1).T.tocsr()
                self.v_backward_sparse = self.v_sparse.multiply(self.v_sparse > 0).T.tocsr()
                self.inert_mask = np.concatenate([self.inert_mask, np.ones(len(inerts), dtype=bool)])
                for elem in self.elements:
                    self.inters_info[elem].extend([0] * len(inerts))
                self.inters_info["codes"] = self.inters_code
                self.inters_info["formulas"] = self.inters_formula
            
            run_napp = napp if inerts else False
            if napp and not inerts:
                warnings.warn("Apparent reaction order estimation requires inert diluents in `iv`. Setting napp=False.")

        results = self.integrate(y0, solver, rtol, atol, tfin, **kwargs)
        results["T"] = self.T
        results["P"] = self.P
        if self.crn.crn_type == "electro":
            results["U"] = self.U
            results["pH"] = self.pH
        results["rxn_strings"] = [rxn.repr_hr for rxn in self.reactions]
        results["Material"] = self.crn.surface.slab.get_chemical_formula() if self.crn.surface else "N/A"
        results["clip_eact"] = clip_eact

        # Performance Metrics
        tcr_1d = np.asarray(results["total_formation_rate"]).flatten()
        reactants_idxs = np.where(self.gas_mask & (y0 > 0) & ~self.inert_mask & (tcr_1d <= 0))[0]
        products_idxs = np.where(self.gas_mask & ~self.inert_mask & (tcr_1d > 0))[0]
        results["reactants_idxs"] = reactants_idxs
        results["products_idxs"] = products_idxs
        
        results["conversion"] = np.zeros(len(reactants_idxs))
        selectivity_matrix = np.zeros((len(reactants_idxs), len(products_idxs), len(self.elements)))
        yield_matrix = np.zeros((len(reactants_idxs), len(products_idxs), len(self.elements)))
        
        n_info = self.inters_info
        for e, elem in enumerate(self.elements):
            for i, reactant in enumerate(reactants_idxs):
                n_elem_reactant = n_info[elem][reactant]
                for j, product in enumerate(products_idxs):
                    if n_elem_reactant == 0 or abs(tcr_1d[reactant]) < 1e-70:
                        selectivity_matrix[i, j, e] = np.nan
                    else:
                        selectivity_matrix[i, j, e] = (tcr_1d[product] * n_info[elem][product]) / (abs(tcr_1d[reactant]) * n_elem_reactant)

        results["selectivity"] = {elem: selectivity_matrix[:, :, e] for e, elem in enumerate(self.elements)}
        results["yield"] = {elem: yield_matrix[:, :, e] for e, elem in enumerate(self.elements)}
        
        balance_dict = analyze_elemental_balance(results)
        for k, v in balance_dict.items():
            results[f"in_div_out_{k}"] = v
            
        results["Catalyst mass (g)"] = 0.0
        results["formulas"] = self.inters_formula
        results["nsims"] = 1

        y_ss = results["y"]  # steady-state results

        if eapp:
            print("Estimating apparent activation energies for global reactions...")
            base_T = self.T
            self.T = base_T - dT
            rates_minus = np.asarray(self.integrate(y_ss, solver, rtol, atol, tfin, **kwargs)["total_formation_rate"]).flatten()
            self.T = base_T + dT
            rates_plus = np.asarray(self.integrate(y_ss, solver, rtol, atol, tfin, **kwargs)["total_formation_rate"]).flatten()
            self.T = base_T
            
            results["eapp"] = calc_eapp(self.T, dT, tcr_1d, rates_minus, rates_plus, self.gas_mask, products_idxs)
            results["nsims"] += 2

        if run_napp:
            results["napp"] = {}
            print("Estimating apparent reaction orders for reactants...")
            main_inert_idx = np.where(self.inert_mask)[0][0]
            
            for idx in reactants_idxs:
                y_minus = y_ss.copy()
                y_plus = y_ss.copy()
                dy_P = self.P * dy

                actual_dy_minus = y_ss[idx] - max(y_ss[idx] - dy_P, 1e-12)
                y_minus[idx] -= actual_dy_minus
                y_minus[main_inert_idx] += actual_dy_minus
                

                actual_dy_plus = y_ss[main_inert_idx] - max(y_ss[main_inert_idx] - dy_P, 1e-12)
                y_plus[idx] += actual_dy_plus
                y_plus[main_inert_idx] -= actual_dy_plus
                
                rates_minus = np.asarray(self.integrate(y_minus, solver, rtol, atol, tfin, **kwargs)["total_formation_rate"]).flatten()
                rates_plus = np.asarray(self.integrate(y_plus, solver, rtol, atol, tfin, **kwargs)["total_formation_rate"]).flatten()

                results["napp"][idx] = calc_napp(y_minus[idx], y_plus[idx], rates_minus, rates_plus, products_idxs)
                results["nsims"] += 2

        if drc:
            print("Estimating Degree of Rate Control (DRC)...")
            results["drc"] = {}
            kd_base = self.kd.copy()
            kr_base = self.kr.copy() 
            for i, reaction in enumerate(self.crn.reactions):
                original_e_ts = reaction.e_ts
                original_private_e_ts = reaction._e_ts
                
                # +de perturbation
                reaction.e_ts = original_e_ts + de
                self._kd[i], self._kr[i] = reaction.get_kinetic_constants(self.T, self.clip_eact)
                rates_plus = np.asarray(self.integrate(y_ss, solver, rtol, atol, tfin, **kwargs)["total_formation_rate"]).flatten()
                results["nsims"] += 1

                # -de perturbation
                if np.isclose(original_e_ts, reaction.e_is) or np.isclose(original_e_ts, reaction.e_fs):  
                    results["drc"][i] = calc_drc(
                        self.T, 
                        original_e_ts,
                        original_e_ts, 
                        original_e_ts + de, 
                        None,
                        tcr_1d, 
                        rates_plus, 
                        products_idxs
                    )
                else:               
                    actual_e_minus = max(original_e_ts - de, max(reaction.e_is, reaction.e_fs))
                    reaction.e_ts = actual_e_minus
                    self._kd[i], self._kr[i] = reaction.get_kinetic_constants(self.T, self.clip_eact)
                    rates_minus = np.asarray(self.integrate(y0, solver, rtol, atol, tfin, **kwargs)["total_formation_rate"]).flatten()
                    results["nsims"] += 1
                    
                    results["drc"][i] = calc_drc(
                        self.T, 
                        actual_e_minus,
                        original_e_ts, 
                        original_e_ts + de, 
                        rates_minus,
                        tcr_1d, 
                        rates_plus, 
                        products_idxs
                    )          

                reaction._e_ts = original_private_e_ts
                reaction._e_act = None
                self._kd[i] = kd_base[i]
                self._kr[i] = kr_base[i]

        if rewire_network:
            for i, _ in enumerate(self.reactions):
                if results["net_rate"][i] < 0:
                    self.crn.reverse_reaction(i)

        return MKMRun.from_dict(results)

    def __str__(self) -> str:
        y = f"Differential Plug-Flow Reactor (PFR) \n"
        y += f"Temperature = {self.T} K, Pressure = {self.P/1e5} bar"
        if self.crn.crn_type == "electro":
            y += f", pH = {self.pH}, U = {self.U}\n"
        else:
            y += "\n"
        y += f"Network with {self.nc} species and {self.nr} reactions on {self.crn.catalyst} catalyst"
        return y
    
    def forward_rate(self, y: np.ndarray) -> np.ndarray:
        rates = np.empty(self.nr, dtype=np.float64)
        v = self.v_forward_sparse.tocsr()

        for j in range(v.shape[0]):
            start, end = v.indptr[j], v.indptr[j+1]
            idx = v.indices[start:end]
            exps = v.data[start:end]
            rates[j] = self.kd[j] * np.prod(y[idx] ** exps)
        return rates

    def backward_rate(self, y: np.ndarray) -> np.ndarray:
        rates = np.empty(self.nr, dtype=np.float64)
        v = self.v_backward_sparse.tocsr()

        for j in range(v.shape[0]):
            start, end = v.indptr[j], v.indptr[j+1]
            idx = v.indices[start:end]
            exps = v.data[start:end]
            rates[j] = self.kr[j] * np.prod(y[idx] ** exps)

        return rates

    def net_rate(self, y: np.ndarray) -> np.ndarray:
        """
        Returns the net reaction rate for each elementary reaction.
        Args:
            y(ndarray): surface coverage + partial pressures array [-/Pa].
        Returns:
            (ndarray): Net reaction rate of the elementary reactions [1/s].
        """
        return self.forward_rate(y) - self.backward_rate(y)

    def ode(
        self,
        _: float,
        y: np.ndarray,
    ) -> np.ndarray:
        rates = net_rate(
            y,
            self.kd, self.kr,
            self.v_forward_sparse.data, self.v_forward_sparse.indices, self.v_forward_sparse.indptr,
            self.v_backward_sparse.data, self.v_backward_sparse.indices, self.v_backward_sparse.indptr,
        )
        dydt = self.v_sparse.dot(rates)
        dydt[self.gas_mask] = 0.0
        return dydt

    def jacobian(self, _, y: np.ndarray) -> csr_matrix:
        """
        Assemble sparse Jacobian as CSR. Returns scipy.sparse.csr_matrix (nc x nc).
        """
        # ensure reaction-row CSR (shape: n_reactions x n_species)
        vT = self.v_sparse.T.tocsr()

        # arrays for forward/backward (already reaction-row in your __init__)
        sf_data, sf_indices, sf_indptr = (
            self.v_forward_sparse.data,
            self.v_forward_sparse.indices,
            self.v_forward_sparse.indptr,
        )
        sb_data, sb_indices, sb_indptr = (
            self.v_backward_sparse.data,
            self.v_backward_sparse.indices,
            self.v_backward_sparse.indptr,
        )

        # vT arrays
        vT_data, vT_indices, vT_indptr = vT.data, vT.indices, vT.indptr

        # precompute conservative upper bound for triplets:
        nnz_max = 0
        for r in range(self.nr):
            n_v = vT_indptr[r+1] - vT_indptr[r]     # stoichiometry nonzeros for reaction r
            n_sf = sf_indptr[r+1] - sf_indptr[r]   # forward participants
            n_sb = sb_indptr[r+1] - sb_indptr[r]   # backward participants
            nnz_max += n_v * (n_sf + n_sb)

        if nnz_max == 0:
            # empty Jacobian (no reactions)
            return csr_matrix((self.nc, self.nc), dtype=np.float64)

        # preallocate triplet arrays
        rows = np.empty(nnz_max, dtype=np.int32)
        cols = np.empty(nnz_max, dtype=np.int32)
        vals = np.empty(nnz_max, dtype=np.float64)

        used = jacobian_fill_numba(
            y, self.kd, self.kr,
            sf_data, sf_indices, sf_indptr,
            sb_data, sb_indices, sb_indptr,
            vT_data, vT_indices, vT_indptr,
            rows, cols, vals,
        )

        if used == 0:
            J = csr_matrix((self.nc, self.nc), dtype=np.float64)
        else:
            # slice to used entries and build CSR (duplicates will be summed)
            J = csr_matrix((vals[:used], (rows[:used], cols[:used])), shape=(self.nc, self.nc))

        J = J.tolil()
        J[self.gas_mask, :] = 0.0
        J = J.tocsr()

        return J

    def steady_state(
        self,
        t: float,
        y: np.ndarray,
    ) -> float:
        """Steady state termination condition.
        It triggers when the sum of coverages is 1, and the elemental
        input and output flows are equal (in=out for C, H, etc.)
        """
        in_div_out = {"*": sum(y[self.gas_mask])}
        rates = net_rate(
            y,
            self.kd, self.kr,
            self.v_forward_sparse.data, self.v_forward_sparse.indices, self.v_forward_sparse.indptr,
            self.v_backward_sparse.data, self.v_backward_sparse.indices, self.v_backward_sparse.indptr,
        )
        dydt = self.v_sparse.dot(rates)
        inflow, outflow = {}, {}
        elem_dict = {k: v for k, v in self.inters_info.items() if k in INTER_ELEMS}
        for elem, counts in elem_dict.items():
            inflow[elem] = 0.0
            outflow[elem] = 0.0
            for i, coeff in enumerate(counts):
                if self.gas_mask[i]:
                    contrib = coeff * dydt[i]
                    if contrib > 0:
                        outflow[elem] += contrib
                    elif contrib < 0:
                        inflow[elem] += abs(contrib)
            if inflow[elem] == 0.0 and outflow[elem] == 0.0:
                in_div_out[elem] = 1
            else:
                in_div_out[elem] = inflow[elem] / (outflow[elem] + np.finfo(float).eps)
        self.time.append(t)
        sum_balances = sum(in_div_out.values())
        self.sum_ddt.append(sum_balances)
        if self.print_progress:
            print(
                f"t={t}s    sum_balances = {sum_balances}"
            )
        max_dev = max([abs(x - 1) for x in in_div_out.values()])
        if max_dev <= self.sstol:
            print("STEADY-STATE  REACHED!!!")
            return 0
        return 1

    steady_state.terminal = True
    steady_state.direction = 0

    def gas_change_event(
        self,
        _: float,
        y: np.ndarray,
    ) -> float:
        """
        Event function to detect when the gas phase changes.
        """
        Py_gas = np.sum(y[self.gas_mask])
        return 0 if Py_gas != self.P else 1

    gas_change_event.terminal = False
    gas_change_event.direction = 0

    def _get_julia_solver(self):
        """Lazy-load Julia and the .jl script"""
        if DifferentialPFR._jl is None:
            import juliacall
            mkm = juliacall.newmodule("mkm")
            script_dir = os.path.dirname(os.path.abspath(__file__))
            julia_solver_path = os.path.join(script_dir, "pfr_solver.jl")
            julia_solver_path = julia_solver_path.replace("\\", "/")
            mkm.seval(f'include("{julia_solver_path}")')
            DifferentialPFR._jl = mkm.SparsePFR
            
        return DifferentialPFR._jl
    
    def integrate(
        self,
        y0: np.ndarray,
        solver: str,
        rtol: float,
        atol: float,
        tfin: float,
        analytical_jacobian: bool = True,
        impose_nonnegativity: bool = True,
        log_transform: bool = False,
        precision: int = 64,
        jl_solver: str = "FBDF", 
        maxiters: int = 100_000,
        show_progress: bool = False,
        **kwargs,
    ) -> dict:
        """Integrate the ODE system up to steady-state."""
        if solver == "Julia":
            results = {}
            time0 = time()
            y, t = self.integrate_jl_cpu(
                y0, rtol=rtol, atol=atol, tfin=tfin,
                analytical_jacobian=analytical_jacobian, 
                impose_nonnegativity=impose_nonnegativity,
                log_transform=log_transform, 
                precision=precision, 
                jl_solver=jl_solver,
                maxiters=maxiters,
                show_progress=show_progress
            )
            results["y"] = y
            results["t"] = t
            results["time"] = time() - time0
            results["status"] = 1
        elif solver == "Python":
            self.sum_ddt = []
            ode_events = ([self.steady_state, self.gas_change_event])
            time0 = time()
            res_ivp = solve_ivp(
                self.ode,
                (0, tfin),
                y0,
                method="BDF",
                events=ode_events,
                jac=self.jacobian if analytical_jacobian else None,
                atol=atol,
                rtol=rtol,
                jac_sparsity=None,
            )
            results = {}
            results["time"] = time() - time0
            results["y"] = res_ivp.y[:, -1]
            results["t"] = res_ivp.t[-1]
            results["time_ss"] = self.time
            results["sum_ddt"] = self.sum_ddt
            results["status"] = res_ivp.status
        else:
            raise ValueError("Invalid solver. Choose between 'Python' or 'Julia'.")
            
        results["forward_rate"] = self.forward_rate(results["y"])
        results["backward_rate"] = self.backward_rate(results["y"])
        
        # Protect against ZeroDivisionError for reversibility
        with np.errstate(divide='ignore', invalid='ignore'):
            results["reversibility"] = np.where(results["backward_rate"] != 0, 
                                                results["forward_rate"] / results["backward_rate"], 
                                                np.inf)
            
        results["net_rate"] = self.net_rate(results["y"])
        results["formation_rate"] = self.v_sparse.multiply(results["net_rate"])
        results["total_formation_rate"] = results["formation_rate"].sum(axis=1)
        
        # Package metadata needed for performance metrics
        results["gas_mask"] = self.gas_mask
        results["inert_mask"] = self.inert_mask
        results["inters"] = self.inters_code
        results["inters_info"] = self.inters_info
        results["y0"] = y0
        results["T"] = self.T
        results["P"] = self.P
        results["rtol"] = rtol
        results["atol"] = atol
        results["tfin"] = tfin
        results["v"] = self.v_sparse
        results["solver"] = solver
        results["kf"] = self.kd
        results["kr"] = self.kr
        results["jl_solver"] = jl_solver if solver == "Julia" else "Python"
        results["precision"] = precision if solver == "Julia" else 64
        results["maxiters"] = maxiters if solver == "Julia" else None
        
        return results

    def reaction_rate(self, product_idx: int, consumption_rate: np.ndarray) -> float:
        return np.sum(consumption_rate[product_idx, :])
    
    def integrate_jl_cpu(
        self,
        y0: np.ndarray,
        rtol: float,
        atol: float,
        tfin: float,
        analytical_jacobian: bool = True,
        impose_nonnegativity: bool = True,
        log_transform: bool = False,
        precision: int = 64,
        jl_solver: str = "FBDF",
        maxiters: int = 1000000,
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Integrate the ODE system using the Julia-based solver on CPU, supporting Float64 or BigFloat.
        """
        sparse_pfr = self._get_julia_solver()
        import juliacall
        jl_base = juliacall.Main.Base 

        if precision > 64:
            y0_in = [str(x) for x in y0]
            kd_in = [str(x) for x in self.kd]
            kr_in = [str(x) for x in self.kr]
        else:
            y0_in = y0
            kd_in = self.kd
            kr_in = self.kr
            
        elem_dict = {k: v for k, v in self.inters_info.items() if k in INTER_ELEMS}
        jl_elem_dict = jl_base.Dict([(k, jl_base.Vector(v)) for k, v in elem_dict.items()])

        vT = self.v_sparse.T.tocsr()
        solution, time = sparse_pfr.setup_and_solve(
            y0_in, kd_in, kr_in, self.gas_mask,
            vT.data.astype(np.int8), vT.indices, vT.indptr,
            self.v_forward_sparse.data.astype(np.int8), self.v_forward_sparse.indices, self.v_forward_sparse.indptr,
            self.v_backward_sparse.data.astype(np.int8), self.v_backward_sparse.indices, self.v_backward_sparse.indptr,
            atol, rtol, self.sstol, tfin,
            analytical_jacobian, impose_nonnegativity, log_transform, precision, jl_solver, maxiters, show_progress, 
            jl_elem_dict
        )
        dtype = np.float64 if precision == 64 else np.float128
        return np.array(solution, dtype=dtype), float(time)
