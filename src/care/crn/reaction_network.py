from typing import Union, Optional

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
import scipy.optimize

from care import ElementaryReaction, Intermediate, Surface
from care.crn.intermediate import SurfaceSite, AdsorbedSpecies, GasSpecies
from care.crn.templates.adsorption import Adsorption, Desorption
from care.constants import OC_KEYS, INTER_ELEMS
from care.crn.templates.pcet import PCET
from care.crn.global_reaction import GlobalReaction


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
            new_reactants = [intermediate_map[r.code] for r in rxn.reactants]
            new_products = [intermediate_map[p.code] for p in rxn.products]
            rxn.components = (new_reactants, new_products)
            self.add_node(rxn)
            for r in rxn.reactants:
                self.add_edge(r, rxn)
            for p in rxn.products:
                self.add_edge(rxn, p)

        self._reset_state()

    def _reset_state(self):
        self._intermediates = self.get_intermediates()
        self._reactions = self.get_reactions()
        self._v = self.build_stoichiometry()
        self._es = self.build_es_matrix()
        self._elements = self.get_elements()
        self._global_reactions_cache = None

    @classmethod
    def from_cutoffs(
        cls, 
        ncc: int, 
        noc: int, 
        cyclic: bool = False,
        additional_rxns: bool = False,
        electro: bool = False,
        **kwargs
    ) -> "ReactionNetwork":
        """Generate a network based on carbon and oxygen cutoffs."""
        from care.crn.utils.blueprint import gen_blueprint
        
        return gen_blueprint(
            ncc=ncc, 
            noc=noc, 
            cyclic=cyclic,
            additional_rxns=additional_rxns,
            electro=electro,
            **kwargs
        )

    @classmethod
    def from_species(
        cls, 
        reactants: list[str], 
        products: list[str],
        additional_rxns: bool = False,
        electro: bool = False,
        **kwargs
    ) -> "ReactionNetwork":
        """Generate a network to connect specific reactants and products."""
        from care.crn.utils.blueprint import gen_blueprint
        
        return gen_blueprint(
            reactants=reactants, 
            products=products,
            additional_rxns=additional_rxns,
            electro=electro,
            **kwargs
        )
        
    @classmethod
    def from_chemical_space(
        cls, 
        cs: list[str],
        additional_rxns: bool = False,
        electro: bool = False,
        **kwargs
    ) -> "ReactionNetwork":
        """Generate a network from an explicit list of SMILES."""
        from care.crn.utils.blueprint import gen_blueprint
        
        return gen_blueprint(
            cs=cs,
            additional_rxns=additional_rxns,
            electro=electro,
            **kwargs
        )
    
    def save_to(self, filepath: str, compress: bool = True) -> None:
        """
        Saves the ReactionNetwork to a JSON or compressed JSON file.
        
        Args:
            filepath (str): The path where the network will be saved.
            compress (bool): If True, compresses the file using gzip. Defaults to True.
        """
        from care.io import save_network
        save_network(self, filepath, compress=compress)

    @classmethod
    def load_from(cls, filepath: str) -> "ReactionNetwork":
        """
        Loads a ReactionNetwork from a JSON or compressed JSON file.
        
        Args:
            filepath (str): The path to the saved network file.
            
        Returns:
            ReactionNetwork: The loaded network instance.
        """
        from care.io import load_network
        return load_network(filepath)

    def get_intermediates(self):
        return {x.code: x for x in self.nodes if isinstance(x, (GasSpecies, AdsorbedSpecies))}

    def get_reactions(self):
        rxns = [x for x in self.nodes if isinstance(x, ElementaryReaction)]
        
        def sort_key(rxn):
            if isinstance(rxn, Adsorption):
                rank = 0
            elif isinstance(rxn, Desorption):
                rank = 2
            else:
                rank = 1
            class_name = rxn.__class__.__name__
            return (rank, class_name, rxn.code)
            
        return sorted(rxns, key=sort_key)  # keep deterministic order

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
        return [x for x in self.reactions if isinstance(x, Adsorption)]
    
    @property
    def desorptions(self):
        return [x for x in self.reactions if isinstance(x, Desorption)]    

    @property
    def num_intermediates(self):
        return len(self.intermediates)
    
    @property
    def num_reactions(self):
        return len(self.reactions)

    @property
    def num_closed_shell_mols(self):
        return len([x for x in self.intermediates.values() if isinstance(x, GasSpecies)])

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
        return "electro" if any(isinstance(x, PCET) for x in self.reactions) else "thermal"
    
    @property
    def elements(self):
        return self._elements
    
    @property
    def v(self):
        return self._v
    
    @property
    def es(self):
        return self._es

    @property
    def catalyst(self):
        return self.surface
    
    def add_catalyst(self, catalyst: Surface):
        self.surface = catalyst
        for inter in self.intermediates.values():
            if isinstance(inter, (AdsorbedSpecies, SurfaceSite)):
                inter.catalyst = catalyst
        for reaction in self.reactions:
            reaction.catalyst = catalyst

    @property
    def is_thermo_evaluated(self) -> bool:
        if not self.reactions:
            return False
        return all(rxn.is_thermo_evaluated for rxn in self.reactions)

    @property
    def is_kinetic_evaluated(self) -> bool:
        if not self.reactions:
            return False
        return all(rxn.is_kinetic_evaluated for rxn in self.reactions)
    
    @property
    def is_evaluated(self) -> bool:
        for rxn in self.reactions:
            if not rxn.is_evaluated:
                return False
        return True

    def build_stoichiometry(self):
        inters = sorted(list(self.intermediates.keys())) + ["*"]
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
                if isinstance(reactant, (GasSpecies, AdsorbedSpecies, SurfaceSite)):
                    rows[k] = index_map[reactant.code]
                    cols[k] = i
                    data[k] = reaction.stoic[reactant.code]
                    k += 1
            for product in self.successors(reaction):
                if isinstance(product, (GasSpecies, AdsorbedSpecies, SurfaceSite)):
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
    
    def reverse_reaction(self, i: int) -> None:
        rxn = self.reactions[i]
        self.remove_edges_from(list(self.in_edges(rxn)) + list(self.out_edges(rxn)))
        rxn.reverse()
        self.add_node(rxn)
        for r in rxn.reactants:
            self.add_edge(r, rxn)
        for p in rxn.products:
            self.add_edge(rxn, p)
        self._v[:, i] *= -1

    def remove_intermediate(self, intermediate: Union[Intermediate, list[Intermediate]]) -> None:
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

    def remove_reaction(self, reaction: Union[ElementaryReaction, list[ElementaryReaction]]) -> None:
        """
        Removes reaction nodes and then any resulting isolated species nodes.

        Args:
            reaction (ElementaryReaction or list of ElementaryReaction): The reaction(s) to remove from the network.
        """

        if isinstance(reaction, ElementaryReaction):
            self.remove_node(reaction)
            initial_removed_count = 1
        elif isinstance(reaction, list):
            self.remove_nodes_from(reaction)
            initial_removed_count = len(reaction)

        num_removed_in_pass = 1
        tot_gas_removed, tot_ads_removed = 0, 0
        tot_rxns_removed = initial_removed_count
        while num_removed_in_pass > 0:
            num_removed_in_pass = 0
            deg_dict = self.degree()
            isolated_species_gas, isolated_species_surf, isolated_rxns = [], [], []
            for x, deg in deg_dict:
                if isinstance(x, GasSpecies) and deg == 0:
                    isolated_species_gas.append(x)
                if isinstance(x, AdsorbedSpecies) and deg < 2:
                    isolated_species_surf.append(x)
                if isinstance(x, ElementaryReaction) and deg < len(x.reactants) + len(x.products):
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
                
        self._reset_state()

        print(f"Removed {tot_rxns_removed} reactions, {tot_gas_removed} gas species, and {tot_ads_removed} adsorbed intermediates")

    def __getitem__(self, other: Union[str, int]):
        if isinstance(other, str):
            return self.intermediates[other]
        elif isinstance(other, int):
            return self.reactions[other]
        else:
            raise TypeError("Index must be str or int")

    def __str__(self):
        string = "ReactionNetwork({} surface species, {} molecules, {} elementary reactions)\n".format(
            self.num_intermediates - self.num_closed_shell_mols,
            self.num_closed_shell_mols,
            self.num_reactions,
        )
        string += f"Elements: {', '.join(self.elements)}\n"
        string += "Catalyst: {}\n".format(self.surface)
        string += "Type: {}\n".format(self.crn_type)

        thermo_status = "Yes" if self.is_thermo_evaluated else "No"
        kinetic_status = "Yes" if self.is_kinetic_evaluated else "No"
        string += f"Evaluated: Thermodynamics ({thermo_status}) | Kinetics ({kinetic_status})\n"
        
        if self.global_reactions:
            string += "Global reactions:\n"
            for i, rxn in enumerate(self.global_reactions):
                string += f"  {i+1}: {rxn.repr_hr}\n"
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

    def get_reaction_table(self, rxns: list = None, return_df: bool = False):
        if rxns is None:
            rxns = self.reactions

        label_dh = "ΔE (eV)"
        label_ea = "Eₐ (eV)"

        def format_energy(val):
            if val is not None:
                return f"{val:+.2f}"
            return "N/A"

        data = []
        for idx, step in enumerate(rxns):
            row = {
                "Idx": idx,
                "Step": step.repr_hr,
                "r-type": step.r_type if "-" in getattr(step, 'r_type', "") else "-",
                label_dh: format_energy(getattr(step, 'e_rxn', None)),
                label_ea: format_energy(getattr(step, 'e_act', None)),
                "Class": type(step).__name__
            }
            data.append(row)

        gr_data = []
        global_rxns = getattr(self, "global_reactions", [])
        if global_rxns:
            for idx, step in enumerate(global_rxns, start=1):
                row = {
                    "Idx": f"GR{idx}",
                    "Step": step.repr_hr,
                    "r-type": getattr(step, 'r_type', "-") if "-" in getattr(step, 'r_type', "-") else "-",
                    label_dh: format_energy(getattr(step, 'e_rxn', None)),
                    label_ea: format_energy(getattr(step, 'e_act', None)),
                    "Class": type(step).__name__
                }
                gr_data.append(row)

        if return_df:
            combined_data = data + gr_data
            return pd.DataFrame(combined_data).set_index("Idx")

        all_steps = rxns + global_rxns
        repr_hr_width = max((len(step.repr_hr) for step in all_steps), default=10) + 2

        header = (
            f"{'Idx':<5} {'Reaction':<{repr_hr_width}} "
            f"{'r-type':<10} {label_dh:<10} "
            f"{label_ea:<10} Class"
        )
        
        print(header)
        print("-" * len(header))
        
        for r in data:
            print(
                f"{str(r['Idx']):<5} {r['Step']:<{repr_hr_width}} "
                f"{r['r-type']:<10} {r[label_dh]:<10} "
                f"{r[label_ea]:<10} {r['Class']}"
            )
            
        if gr_data:
            print("-" * len(header))
            for r in gr_data:
                print(
                    f"{str(r['Idx']):<5} {r['Step']:<{repr_hr_width}} "
                    f"{r['r-type']:<10} {r[label_dh]:<10} "
                    f"{r[label_ea]:<10} {r['Class']}"
                )

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
    
    def __eq__(self, other):
        # compare for structural equivalence, not state equivalence
        # a network with A+B->C and one with C->A+B would be considered equivalent
        if not isinstance(other, ReactionNetwork):
            return NotImplemented
        
        if self.surface != other.surface:
            return False
        
        my_inters = set(x.code for x in self.intermediates.values())
        other_inters = set(x.code for x in other.intermediates.values())
        
        if my_inters != other_inters:
            return False
              
        return set(self.reactions) == set(other.reactions)
    
    @property
    def global_reactions(self) -> list[GlobalReaction]:
        """
        Identifies and balances the global chemical reactions represented by the network.
        Strictly enforces that gas_reactants are on the left (reactants) and 
        gas_products are on the right (products).
        
        Returns:
            list[GlobalReaction]: A list of balanced global reactions.
        """
        gas_inters = [x for x in self.intermediates.values() if isinstance(x, GasSpecies)]
        if not gas_inters:
            return []

        ads_rxns = self.adsorptions
        gas_reactants = []
        gas_products = []

        for g in gas_inters:
            is_reactant = any(g in rxn.reactants for rxn in ads_rxns)
            if is_reactant:
                gas_reactants.append(g)
            else:
                gas_products.append(g)

        if not gas_reactants or not gas_products:
            return []

        elements = self.elements
        E_R = np.zeros((len(elements), len(gas_reactants)))
        for j, g in enumerate(gas_reactants):
            for i, elem in enumerate(elements):
                E_R[i, j] = g[elem]

        E_P = np.zeros((len(elements), len(gas_products)))
        for j, g in enumerate(gas_products):
            for i, elem in enumerate(elements):
                E_P[i, j] = g[elem]

        unique_reactions = []

        for t, _ in enumerate(gas_products):
            
            other_products = [g for i, g in enumerate(gas_products) if i != t]
            
            E_P_other = np.zeros((len(elements), len(other_products)))
            for j, g in enumerate(other_products):
                for i, elem in enumerate(elements):
                    E_P_other[i, j] = g[elem]

            A_eq = np.hstack([E_P_other, -E_R])
            b_eq = -E_P[:, t]
            c = np.ones(A_eq.shape[1])
            res = scipy.optimize.linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method='highs')

            if res.success:
                vec = np.zeros(len(gas_products) + len(gas_reactants))
                vec[t] = 1.0
                idx_other = 0
                for i in range(len(gas_products)):
                    if i != t:
                        vec[i] = res.x[idx_other]
                        idx_other += 1                        
                for j in range(len(gas_reactants)):
                    vec[len(gas_products) + j] = -res.x[len(other_products) + j]

                non_zero = vec[np.abs(vec) > 1e-5]
                if len(non_zero) == 0:
                    continue
                    
                vec = vec / np.min(np.abs(non_zero))
                for multiplier in range(1, 40):
                    test_vec = vec * multiplier
                    if np.allclose(test_vec, np.round(test_vec), atol=1e-2):
                        vec = np.round(test_vec)
                        break

                vec_tuple = tuple(np.round(vec, 2))
                if vec_tuple not in [tuple(np.round(v, 2)) for v in unique_reactions]:
                    unique_reactions.append(vec)

        global_rxns = []
        all_gas = gas_products + gas_reactants

        for vec in unique_reactions:
            left_side = []
            right_side = []
            stoic_dict = {}

            for idx, coeff in enumerate(vec):
                if np.abs(coeff) < 1e-2:
                    continue
                
                species = all_gas[idx]
                coeff_val = int(abs(coeff))
                
                if coeff < 0:
                    left_side.append(species)
                else:
                    right_side.append(species)
                stoic_dict[species.code] = coeff_val

            if left_side and right_side:
                global_rxns.append(GlobalReaction([left_side, right_side], stoic_dict))
        
        global_rxns = sorted(global_rxns)
        self._global_reactions_cache = global_rxns
        return self._global_reactions_cache
    
    def get_route_stoichiometry(self, global_rxn: GlobalReaction) -> dict[int, float]:
        """
        Calculates the stoichiometric number for each elementary reaction 
        required to complete one cycle of the given global reaction.
        Uses L1-norm minimization to find the sparsest valid pathway.
        """
        inters_keys = sorted(list(self.intermediates.keys())) + ["*"]
        nu_net = np.zeros(len(inters_keys))
        
        for r in global_rxn.reactants:
            if r.code in inters_keys:
                nu_net[inters_keys.index(r.code)] = -abs(global_rxn.stoic[r.code])
                
        for p in global_rxn.products:
            if p.code in inters_keys:
                nu_net[inters_keys.index(p.code)] = abs(global_rxn.stoic[p.code])
                
        # L1 minimization: Min sum(|σ|) subject to V * σ = nu_net
        # σ = u - w (u >= 0, w >= 0). Min sum(u + w)
        V = self.v.toarray()
        nr = self.num_reactions
        
        c = np.ones(2 * nr)
        A_eq = np.hstack([V, -V])
        b_eq = nu_net
        
        res = scipy.optimize.linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), method='highs')
        
        if not res.success:
            raise ValueError("Could not find a valid elementary route for the given global reaction.")
            
        # Reconstruct σ array and filter out numerical noise
        sigma = res.x[:nr] - res.x[nr:]
        sigma[np.abs(sigma) < 1e-5] = 0.0

        route = {}
        for i, _ in enumerate(self.reactions):
            if sigma[i] != 0:
                route[i] = round(sigma[i], 4)
                
        return route
    
    def plot(
        self,
        filename: str = None,
        figsize: tuple = (18, 15),
        rankdir: str = "TB",
        rank_sep: float = 0.3,
        node_sep: float = 0.15,
        fontsize: int = 50,
        legend_fontsize: int = 20,
        layout_engine: str = "dot",
        show_species_labels: bool = True,
        arrowhead: str = "normal",
        fontname: str = "Arial",
        dpi: int = 200
    ) -> None:
        from care.crn.visualize import plot_crn
        
        plot_crn(
            graph=self,
            filename=filename,
            figsize=figsize,
            rankdir=rankdir,
            rank_sep=rank_sep,
            node_sep=node_sep,
            fontsize=fontsize,
            legend_fontsize=legend_fontsize,
            layout_engine=layout_engine,
            show_species_labels=show_species_labels,
            arrowhead=arrowhead,
            fontname=fontname,
            dpi=dpi
        )
