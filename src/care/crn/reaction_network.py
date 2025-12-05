import os
from pickle import load
from typing import Union, Optional

import networkx as nx
import numpy as np
from scipy.sparse import vstack, coo_matrix

from care import ElementaryReaction, Intermediate, Surface
from care.constants import OC_KEYS, INTER_ELEMS
from care.crn.utils.electro import Electron

class ReactionNetwork(nx.DiGraph):
    """
    Base class for surface reaction networks.

    Attributes:
        reactions (list of obj:`ElementaryReaction`): List containing the
            elementary reactions of the network.
        surface (obj:`Surface`): Surface of the network.
        oc (dict of str: float): Dictionary containing the operating conditions
    """

    def __init__(
        self,
        reactions: Optional[list[ElementaryReaction]] = None,
        surface: Optional[Surface] = None,
        oc: Optional[dict[str, float]] = None,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.surface = surface
        if reactions is None:
            reactions = []

        if oc is not None:
            if all([key in OC_KEYS for key in oc.keys()]):
                self.oc = oc
            else:
                raise ValueError(f"Keys of oc must be in {OC_KEYS}")
        else:
            self.oc = {"T": 0, "P": 0, "U": 0, "pH": 0}
        intermediate_map = {}
        for rxn in reactions:
            for inter in list(rxn.reactants) + list(rxn.products):
                if inter.code not in intermediate_map:
                    self.add_node(
                        inter
                    )
                    intermediate_map[inter.code] = inter
        for rxn in reactions:
            reactant_codes = [inter.code for inter in rxn.reactants]
            product_codes = [inter.code for inter in rxn.products]
            rxn.components = ([intermediate_map[i] for i in reactant_codes],
                              [intermediate_map[i] for i in product_codes])
            self.add_node(rxn)
            for inter in list(rxn.reactants):
                self.add_edge(
                    inter,
                    rxn,
                )
            for inter in list(rxn.products):
                self.add_edge(
                    rxn,
                    inter,
                )
        self._intermediates = self.get_intermediates()
        self._reactions = self.get_reactions()
        self._v = self.build_stoichiometry()
        self._es = self.build_es_matrix()
        self._elements = self.get_elements()

    def get_intermediates(self):
        return {x.code: x for x in self.nodes if isinstance(x, Intermediate) and x.phase in ("ads", "gas")}

    def get_reactions(self):
        return [x for x in self.nodes if isinstance(x, ElementaryReaction)]

    @property
    def intermediates(self):
        if self._intermediates is None:
            self._intermediates = self.get_intermediates()
        return self._intermediates

    @property
    def reactions(self):
        if self._reactions is None:
            self._reactions = self.get_reactions()
        return self._reactions
    
    @property
    def adsorptions(self):
        return [x for x in self.reactions if isinstance(x, ElementaryReaction) and x.r_type == "adsorption"]
    
    @property
    def desorptions(self):
        return [x for x in self.reactions if isinstance(x, ElementaryReaction) and x.r_type == "desorption"]    

    @property
    def num_intermediates(self):
        return len(self.intermediates)
    
    @property
    def num_reactions(self):
        return len(self.reactions)

    @property
    def num_closed_shell_mols(self):
        return len([x for x in self.intermediates.values() if x.closed_shell and x.phase == "gas"])

    @property
    def temperature(self):
        return self.oc["T"]

    @temperature.setter
    def temperature(self, other: float):
        self.oc["T"] = other

    @property
    def pressure(self):
        return self.oc["P"]

    @pressure.setter
    def pressure(self, other: float):
        self.oc["P"] = other

    @property
    def overpotential(self):
        return self.oc["U"]

    @overpotential.setter
    def overpotential(self, other: float):
        self.oc["U"] = other

    @property
    def pH(self):
        return self.oc["pH"]

    @pH.setter
    def pH(self, other: float):
        self.oc["pH"] = other
    
    @property
    def crn_type(self):
        return "thermal" if Electron() not in self.intermediates else "electro"
    
    @property
    def elements(self):
        return self._elements
    
    @property
    def v(self):
        return self._v
    
    @property
    def es(self):
        return self._es

    def build_stoichiometry(self):
        inters = list(self.intermediates.keys()) + ["*"]
        index_map = {code: idx for idx, code in enumerate(inters)}

        max_edges = 8
        n_reactions = len(self.reactions)
        n_species = self.num_intermediates + 1

        rows = np.empty(max_edges * n_reactions, dtype=np.int32)
        cols = np.empty(max_edges * n_reactions, dtype=np.int32)
        data = np.empty(max_edges * n_reactions, dtype=np.int8)

        k = 0
        for i, reaction in enumerate(self.reactions):
            for reactant in self.predecessors(reaction):
                if reactant.phase in ("ads", "gas", "surf"):
                    rows[k] = index_map[reactant.code]
                    cols[k] = i
                    data[k] = reaction.stoic[reactant.code]
                    k += 1
            for product in self.successors(reaction):
                if product.phase in ("ads", "gas", "surf"):
                    rows[k] = index_map[product.code]
                    cols[k] = i
                    data[k] = reaction.stoic[product.code]
                    k += 1
        rows = rows[:k]
        cols = cols[:k]
        data = data[:k]

        v = coo_matrix((data, (rows, cols)), shape=(n_species, n_reactions)).tocsr()
        return v
    
    def build_es_matrix(self):
        """get element-species dense matrix"""
        m = np.zeros((len(INTER_ELEMS), self.num_intermediates+1), dtype=np.int8)
        for j, inter in enumerate(self.intermediates.values()):
            for i, elem in enumerate(INTER_ELEMS):
                m[i, j] = inter[elem]
        m[-2, -1] = 1  # surface site
        m = m[:-1, :]  # delete charge row
        return m
    
    def get_elements(self):
        elements = set()
        for inter in self.intermediates.values():
            elements.update(set(inter.molecule.get_chemical_symbols()))
        return sorted(list(elements))

    @property
    def ncc(self):
        return max([x["C"] for x in self.intermediates.values()], default=0)

    @property
    def noc(self):
        return max([x["O"] for x in self.intermediates.values() if x.formula != "O2"], default=0)
    
    def reverse_reaction(self, i):
        rxn = self.reactions[i]
        self.remove_edges_from(list(self.in_edges(rxn)) + list(self.out_edges(rxn)))
        rxn.reverse()
        self.add_node(rxn)
        for r in rxn.reactants:
            self.add_edge(r, rxn)
        for p in rxn.products:
            self.add_edge(rxn, p)
        self._v[:, i] *= -1

    def remove_intermediate(self, intermediate: Union[Intermediate, list[Intermediate]]):
        reactions_to_remove = []
        if isinstance(intermediate, Intermediate):
            reactions_to_remove += list(self.pred[intermediate]) 
            reactions_to_remove += list(self.succ[intermediate])
            self.remove_node(intermediate)
        elif isinstance(intermediate, list):
            for x in intermediate:
                reactions_to_remove += list(self.pred[x])
                reactions_to_remove += list(self.succ[x])
            self.remove_nodes_from(intermediate)        
        self.remove_reaction(reactions_to_remove)

    def remove_reaction(self, reaction: Union[ElementaryReaction, list[ElementaryReaction]]):
        """
        Removes reaction nodes and then any resulting isolated species nodes.

        Args:
            threshold (float): The value to compare against.
        """
        if isinstance(reaction, ElementaryReaction):
            self.remove_node(reaction)
        elif isinstance(reaction, list):
            self.remove_nodes_from(reaction)

        num_removed_in_pass = 1
        tot_gas_removed, tot_ads_removed, tot_rxns_removed = 0, 0, len(reaction)
        while num_removed_in_pass > 0:
            num_removed_in_pass = 0
            deg_dict = self.degree()
            isolated_species_gas, isolated_species_surf, isolated_rxns = [], [], []
            for x, deg in deg_dict:
                if isinstance(x, Intermediate):
                    if x.phase == "gas" and deg == 0:
                        isolated_species_gas.append(x)
                    if x.phase == "ads":
                        if deg < 2:
                            isolated_species_surf.append(x)
                else:
                    if deg < len(x.reactants) + len(x.products):
                        isolated_rxns.append(x)
            if isolated_species_gas:
                self.remove_nodes_from(isolated_species_gas)
                num_removed_in_pass += len(isolated_species_gas)
                tot_gas_removed += len(isolated_species_gas)
            if isolated_species_surf:
                self.remove_nodes_from(isolated_species_surf)
                num_removed_in_pass += len(isolated_species_surf)
                tot_ads_removed += len(isolated_species_surf)
            if isolated_rxns:
                self.remove_nodes_from(isolated_rxns)
                num_removed_in_pass += len(isolated_rxns)
                tot_rxns_removed += len(isolated_rxns)
                
        self._intermediates = self.get_intermediates()
        self._reactions = self.get_reactions()
        self._v = self.build_stoichiometry()
        self._es = self.build_es_matrix()
        print(f"Removed {tot_rxns_removed} reactions, {tot_gas_removed} gas species, and {tot_ads_removed} adsorbed intermediates")

    def __getitem__(self, other: Union[str, int]):
        if isinstance(other, str):
            return self.intermediates[other]
        elif isinstance(other, int):
            return self.reactions[other]
        else:
            raise TypeError("Index must be str or int")

    def __str__(self):
        string = "ReactionNetwork({} surface species, {} gas molecules, {} elementary reactions)\n".format(
            self.num_intermediates - self.num_closed_shell_mols,
            self.num_closed_shell_mols,
            self.num_reactions,
        )
        string += "Surface: {}\n".format(self.surface)
        string += "Network Carbon cutoff: {}\n".format(self.ncc)
        string += "Network Oxygen cutoff: {}\n".format(self.noc)
        string += "Type: {}\n".format(self.crn_type)
        return string

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return self.num_reactions

    def __iter__(self):
        return iter(x for x in self.nodes(data=True) if isinstance(x[0], ElementaryReaction))

    def __contains__(self, other: Union[str, Intermediate, ElementaryReaction]):
        if isinstance(other, str):
            return other in self.nodes(data=False)
        elif isinstance(other, Intermediate):
            return other in self.intermediates.values()
        elif isinstance(other, ElementaryReaction):
            return other in self.reactions
        else:
            raise TypeError("Index must be str, Intermediate or ElementaryReaction")

    def get_reaction_table(self, rxns: list = None) -> None:
        if rxns is None:
            rxns = self.reactions
        repr_hr_width = max(len(step.repr_hr) for step in rxns) + 2
        dhr_width = 10
        eact_width = 10
        class_width = 20
        index_width = 5
        r_type_width = 10
        header = "{:<{}} {:<{}} {:<{}} {:<{}} {:<{}} {}".format(
            "Idx", index_width, "Step", repr_hr_width, "r-type", r_type_width, "DHR (eV)", dhr_width, "Eact (eV)", eact_width, "Class"
        )
        print(header)
        print("=" * (index_width + repr_hr_width + r_type_width + dhr_width + eact_width + class_width))

        for idx, step in enumerate(rxns):
            index_str = str(idx).ljust(index_width)
            repr_hr_str = step.repr_hr.ljust(repr_hr_width)
            r_type_str = step.r_type.ljust(r_type_width) if "-" in step.r_type else "-".ljust(r_type_width)
            dhr_str = "{:+.2f}".format(step.e_rxn[0]).ljust(dhr_width)
            eact_str = "{:+.2f}".format(step.e_act[0]).ljust(eact_width)
            class_str = str(type(step)).split(".")[-1].strip("'>").ljust(class_width)
            print(f"{index_str}{repr_hr_str} {r_type_str} {dhr_str} {eact_str} {class_str}")

    def get_hubs(self, n: int = None) -> dict[str, int]:
        """
        Get hubs of the network.

        Returns:
            hubs (dict): Dictionary containing the intermediates and the number
                of reactions in which they are involved, sorted in descending
                order.
        """
        hubs ={node.formula + f"{"*" if node.phase == "ads" else ""}": self.degree(node) for node in self.nodes if isinstance(node, Intermediate)}
        if n is not None:
            return dict(sorted(hubs.items(), key=lambda item: item[1], reverse=True)[:n])
        else:   
            return hubs
        
    def run_microkinetic(
        self,
        iv: dict[str, float] = None,
        oc: dict[str, float] = None,
        mkm_path: Optional[str] = None,
        nruns: int = 1,
        solver: str = "Julia",
        tfin: float = 1e30,
        atol: float = 1e-15,
        rtol: float = 1e-12,
        clip_eact: float = -1.0,
        rewire_network: bool = True,
        **kwargs
    ) -> dict:
        """
        Run microkinetic simulation on the CRN.
        Args:
            iv (dict of str: float): Dictionary containing the inlet molar
                fractions of gas phase species. Keys are the chemical formulas
                of the species, values are the molar fractions. Sum of all
                values must be 1.0.
            oc (dict of str: float): Dictionary containing the operating
                conditions. Keys must be in OC_KEYS. e.g. {"T": 600,
                "P": 1.0, "U": 0.0, "pH": 0.0}. If the network is thermal,
                U and pH are ignored.
            mkm_path (str, optional): Path to checkpoint from previous
                MKM results stored as .pkl. If provided, iv and oc are ignored.
            nruns (int, optional): Number of runs for uncertainty quantification.
                If > 1, uncertainty quantification is performed. Default to 1 (no uq).
            solver (str, optional): Solver to use. Default is "Julia".
            tfin (float, optional): Final time for integration in seconds. Default is 1e30 [s].
            atol (float, optional): Absolute tolerance for ODE integration.
                Default is 1e-15.
            rtol (float, optional): Relative tolerance for ODE integration.
                Default is 1e-12.
            clip_eact (float, optional): If positive, reactions with activation barrier 
                eact > clip_eact in both directions will be clipped such that the smallest barrier
                between the two directions is equal to clip_eact. Useful to reduce stiffness of the ODEs.
                If set to zero, reaction will be assumed to be barrierless. Default is -1 (no clipping)
            rewire_network (bool, optional): If True, the network will be rewired based on the reaction rates   
                obtained from the kinetic simulation. Default is True.
            **kwargs: Additional keyword arguments to pass to the Reactor.integrate() method.
        Returns:
            results (dict): Dictionary containing the results of the
                microkinetic simulation.
        """
        from care.reactors import DifferentialPFR
        from care.reactors.utils import analyze_elemental_balance
        from scipy.sparse import csr_matrix
        
        reactions = self.reactions
        intermediates = self.intermediates
        
        if mkm_path is not None:
            if os.path.isfile(mkm_path):
                with open(mkm_path, "rb") as f:
                    inputs = load(f)
                v = inputs["v"]
                T = inputs["T"]
                P = inputs["P"]
                y0 = inputs["y"]
                gas_mask = inputs["gas_mask"]
                inters_info = inputs["inters_info"]
                inters_formula = inputs["formulas"]
                print(f"Starting integration from loaded MKM checkpoint {mkm_path}")
                uq = True if nruns > 1 else False
                n_reactions = v.shape[1]
            else:
                raise ValueError("mkm_path does not point to a valid file")
        elif oc is None:
            raise ValueError("Either mkm_path or both iv and oc must be provided")
        else:
            n_reactions = len(reactions)
            uq = True if nruns > 1 else False

            if not np.isclose(sum(iv.values()), 1.0):
                raise ValueError("Sum of molar fractions is not 1.0")

            T = oc.get("T", self.temperature)
            if T is None:
                raise ValueError("temperature not specified")

            P = oc.get("P", self.pressure)
            if P is None:
                raise ValueError("pressure not specified")

            if self.crn_type == "electro":
                U = oc.get("U")
                PH = oc.get("pH")
                if U is None or PH is None:
                    raise ValueError("electrochemical conditions require U and pH")
                
            inters = list(intermediates.keys())
            inters_formula = [intermediates[x].formula for x in inters] + ["*"]
            gas_mask = np.array([inter.phase == "gas" for inter in intermediates.values()] + [False])
            inters.append("*")
            inters_dict = {}
            inters_dict["formulas"] = inters_formula
            inters_dict["codes"] = inters
            for elem in self.elements:
                inters_dict[elem] = [x[elem] for x in intermediates.values()]
            inters_dict["elements"] = self.elements
            inters_info = inters_dict
            inlet_molecules = [inter for inter in iv.keys() if inter in inters_formula]            
            inlet_molecules = set(inlet_molecules)        

            for i, reaction in enumerate(self.adsorptions):
                if not any(inter.formula in inlet_molecules for inter in self.predecessors(reaction)):
                    self.reverse_reaction(i)
            for i, reaction in enumerate(self.desorptions):
                if any(inter.formula in inlet_molecules for inter in self.successors(reaction)):
                    self.reverse_reaction(i)

            v = self.v.copy()
            y0 = np.zeros(len(inters), dtype=np.float64)
            y0[-1] = 1.0

            inerts, inert_idx, inert_y0 = [], [], []
            formula_set = set(inters_formula)
            for k, val in iv.items():
                if k not in formula_set:
                    inerts.append(k)
                    inert_idx.append(len(y0))
                    inert_y0.append(P * val)
                else:
                    idx = next(i for i, (_, formula) in enumerate(zip(inters, inters_formula)) if formula == k and gas_mask[i])
                    y0[idx] = P * val

            if inerts:
                y0 = np.concatenate([y0, inert_y0])
                gas_mask = np.concatenate([gas_mask, np.ones(len(inerts), dtype=bool)])
                inters += inerts
                v = vstack([v, csr_matrix((len(inerts), n_reactions), dtype=np.int8)]).tocsr()

        if uq:
            kf = np.zeros((n_reactions, nruns))
            kr = np.zeros((n_reactions, nruns))
            for j, rxn in enumerate(reactions):
                for run in range(nruns):
                    kf[j, run], kr[j, run] = rxn.get_kinetic_constants(t=T, uq=True, clip_eact=clip_eact)
        else:
            kf = np.zeros(n_reactions)
            kr = np.zeros(n_reactions)
            for j, rxn in enumerate(reactions):
                kf[j], kr[j] = rxn.get_kinetic_constants(t=T, uq=False, clip_eact=clip_eact)

        reactor = DifferentialPFR(v=v, kd=kf, kr=kr, gas_mask=gas_mask,
                                inters=inters_info, pressure=P, temperature=T)
        print(reactor)
        RTOL, ATOL, TFIN = rtol, atol, tfin
        settings_str = f"rtol={RTOL}, atol={ATOL}, tfin={TFIN}s"
        if "precision" in kwargs:
            precision = kwargs["precision"]
            settings_str += f", prec={kwargs['precision']} bits"
        else: 
            precision = 64
        if not isinstance(clip_eact, dict):
            if clip_eact >= 0:
                if clip_eact == 0:
                    settings_str += f", barrierless reactions"
                else:
                    settings_str += f", clip_Eact={clip_eact} eV"
        else:
            settings_str += f", user-defined BEPs"
        if "jl_solver" in kwargs:
            settings_str += f", jl_solver={kwargs['jl_solver']}"

        print(f"ODE settings: {settings_str}")
        results = {}

        if uq:
            results_runs = []
            for run in range(nruns):
                reactor.kd = kf[:, run]
                reactor.kr = kr[:, run]
                results_runs.append(reactor.integrate(y0, solver, RTOL, ATOL, TFIN, **kwargs))
            keys = results_runs[0].keys()
            results = {k: np.mean([r[k] for r in results_runs], axis=0) for k in keys if isinstance(results_runs[0][k], np.ndarray)}
            results.update({k+"_std": np.std([r[k] for r in results_runs], axis=0) for k in keys if isinstance(results_runs[0][k], np.ndarray)})
            results["runs"] = results_runs
        else:
            results = reactor.integrate(y0, 
                                        solver, 
                                        RTOL, 
                                        ATOL, 
                                        TFIN,
                                        **kwargs)
        results["formulas"] = inters_formula
        results["U"] = oc.get("U", None)
        results["pH"] = oc.get("pH", None)
        results["kf"] = kf
        results["kr"] = kr
        results["rxn_strings"] = [rxn.repr_hr for rxn in reactions]
        results["Material"] = self.surface.slab.get_chemical_formula() if self.surface else "N/A"
        results["precision"] = f"float{precision}"
        results["Catalyst mass (g)"] = 0.0
        balance_dict = analyze_elemental_balance(results, intermediates)
        print(f"Elemental balances at t={results["t"]:.2e} s: C={balance_dict["C"]:.2e}, H={balance_dict["H"]:.2e}, O={balance_dict["O"]:.2e}, *={balance_dict["*"]:.2e}")
        for k, v in balance_dict.items():
            results[f"in_div_out_{k}"] = v
        results["clip_eact"] = clip_eact

        if rewire_network:
            r = results["net_rate"]
            for i, rxn in enumerate(self.reactions):
                if r[i] < 0:
                    self.reverse_reaction(i)
        return results
