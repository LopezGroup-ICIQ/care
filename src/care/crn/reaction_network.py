from typing import Union, Optional
from tqdm import tqdm

import networkx as nx
import numpy as np
from scipy.sparse import lil_matrix, vstack

from care import ElementaryReaction, Intermediate, Surface
from care.constants import OC_KEYS
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
                    v=abs(rxn.stoic[inter.code]),
                )
            for inter in list(rxn.products):
                self.add_edge(
                    rxn,
                    inter,
                    v=abs(rxn.stoic[inter.code]),
                )
            self._intermediates = self.get_intermediates()
            self._reactions = self.get_reactions()
            self._v = self.build_stoichiometry()

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
    def v(self):
        return self._v
    
    def build_stoichiometry(self):
        inters = list(self.intermediates.keys()) + ["*"]
        index_map = {code: idx for idx, code in enumerate(inters)}
        v = lil_matrix((self.num_intermediates + 1, self.num_reactions), dtype=np.int8)
        for i, reaction in enumerate(self.reactions):
            for reactant in self.predecessors(reaction):
                if reactant.phase in ("ads", "gas", "surf"):
                    v[index_map[reactant.code], i] = -self.edges[reactant, reaction]['v']
            for product in self.successors(reaction):
                if product.phase in ("ads", "gas", "surf"):
                    v[index_map[product.code], i] = self.edges[reaction, product]['v']
        v = v.tocsr()
        return v

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
            self.add_edge(r, rxn, v=abs(rxn.stoic[r.code]))
        for p in rxn.products:
            self.add_edge(rxn, p, v=abs(rxn.stoic[p.code]))
        self._v[:, i] *= -1

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

    def get_reaction_table(self) -> None:
        repr_hr_width = max(len(step.repr_hr) for step in self.reactions) + 2
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

        for idx, step in enumerate(self.reactions):
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
        iv: dict[str, float],
        oc: dict[str, float],
        uq: bool = False,
        nruns: int = 100,
        solver: str = "Julia",
        ss_tol: float = 1e-10,
        tfin: float = 1e6,
        gpu: bool = False,
        atol: float = 1e-20,
        rtol: float = 1e-8,
        **kwargs
    ) -> dict:
        """
        Optimized microkinetic simulation runner.
        """
        from care.reactors import DifferentialPFR
        from scipy.sparse import csr_matrix

        reactions = self.reactions
        intermediates = self.intermediates
        n_reactions = len(reactions)

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
            
        inlet_molecules = [inter for inter in iv.keys() if inter in inters_formula]            
        for i, reaction in tqdm(enumerate(self.reactions), desc="Reversing reactions"):
            if reaction.r_type == "adsorption":
                reactants_formulas = [inter.formula for inter in self.predecessors(reaction)]
                if not any(
                    inlet_molecule in reactants_formulas for inlet_molecule in inlet_molecules
                ):
                    self.reverse_reaction(i)
            elif reaction.r_type == "desorption":
                products_formulas = [inter.formula for inter in self.successors(reaction)]
                if any(
                    inlet_molecule in products_formulas for inlet_molecule in inlet_molecules
                ):
                    self.reverse_reaction(i)
            elif reaction.r_type == "PCET":
                if U < 0:
                    if (Electron(), reaction) not in self.edges:
                        self.reverse_reaction(i)
                else:
                    if (Electron(), reaction) not in self.edges:
                        self.reverse_reaction(i)
            else:
                if any(
                    [
                        inter.closed_shell and inter.formula not in inlet_molecules
                        for inter in self.predecessors(reaction)
                    ]
                ):
                    self.reverse_reaction(i)
                elif any(
                    [
                        inter.closed_shell and inter.formula in inlet_molecules
                        for inter in self.successors(reaction)
                    ]
                ):
                    self.reverse_reaction(i)
                else:
                    pass
        print("done")

        v = self.v.copy()
        y0 = np.zeros(len(inters), dtype=np.float64)
        y0[-1] = 1.0  # empty surface

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
                    kf[j, run], kr[j, run] = rxn.get_kinetic_constants(t=T, uq=True)
        else:
            kf = np.zeros(n_reactions)
            kr = np.zeros(n_reactions)
            for j, rxn in enumerate(reactions):
                kf[j], kr[j] = rxn.get_kinetic_constants(t=T, uq=False)

        reactor = DifferentialPFR(v=v, kd=kf, kr=kr, gas_mask=gas_mask,
                                inters=inters, pressure=P, temperature=T)

        if uq:
            results_runs = []
            for run in range(nruns):
                reactor.kd = kf[:, run]
                reactor.kr = kr[:, run]
                results_runs.append(reactor.integrate(y0, solver, rtol, atol, ss_tol, tfin, gpu))
            keys = results_runs[0].keys()
            results = {k: np.mean([r[k] for r in results_runs], axis=0) for k in keys if isinstance(results_runs[0][k], np.ndarray)}
            results.update({k+"_std": np.std([r[k] for r in results_runs], axis=0) for k in keys if isinstance(results_runs[0][k], np.ndarray)})
            results["runs"] = results_runs
            results["inters"] = inters
            results["formulas"] = inters_formula
            results["gas_mask"] = gas_mask
            results["y0"] = y0
            return results

        else:
            RTOL, ATOL = rtol, atol
            for _ in range(20):  # max attempts
                results = reactor.integrate(y0, solver, RTOL, ATOL, ss_tol, tfin, gpu)
                if results["status"] in (0, 1):
                    break
                ATOL /= 10
                if _ % 2 == 0:
                    RTOL /= 10
            else:
                raise RuntimeError("Failed to reach steady state")

            results["inters"] = inters
            results["formulas"] = inters_formula
            results["gas_mask"] = gas_mask
            results["y0"] = y0
            return results
