"""
Interface to GAME-Net-UQ model.
"""

import os
from typing import Optional, Union

from ase import Atoms
from ase.db import connect
import networkx as nx
import numpy as np
import torch
torch.set_float32_matmul_precision('high')
from torch import no_grad, tensor, cat, compile
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from care import Intermediate, ElementaryReaction, Surface
from care.evaluators import IntermediateEnergyEstimator, ReactionEnergyEstimator
from care.evaluators.gamenet_uq import MODEL_PATH, ADSORBATE_ELEMS, METALS
from care.adsorption import place_adsorbate
from care.constants import INTER_ELEMS, K_B, METAL_STRUCT_DICT
from care.crn.utils.electro import Proton, Electron, Water
from care.evaluators.gamenet_uq.functions import load_model
from care.evaluators.gamenet_uq.graph import atoms_to_data
from care.evaluators.gamenet_uq.graph_filters import extract_adsorbate
from care.evaluators.gamenet_uq.graph_tools import pyg_to_nx
from care.evaluators.utils import connectivity_signature
from care.crn.templates import BondBreaking


class GameNetUQInter(IntermediateEnergyEstimator):
    def __init__(
        self,
        surface: Surface,
        device: str = "cpu",
        dft_db_path: Optional[str] = None,
        num_configs: int = 3,
        use_uq: bool = False,
        torch_compile: bool = False,
        **kwargs
    ):
        """Interface for GAME-Net-UQ for intermediates.

        Args:
            surface (Surface, optional): Surface of interest.
            device (str, optional): Device to use for evaluation. Defaults to "cpu".
            dft_db_path (Optional[str], optional): Path to ASE database for retrieving
                DFT data. Defaults to None.
            num_configs (int, optional): Number of configurations to consider for the adsorbed phase.
                Defaults to 3.
            use_uq (bool, optional): Whether to use uncertainty in the evaluation. Defaults to False.
                if True, the configurations will be sorted in ascending order of uncertainty in the ads_configs attribute.
            torch_compile (bool, optional): Whether to compile the model using Torch. Defaults to False.
        """

        self.model = load_model(MODEL_PATH)
        if torch_compile:
            self.model = compile(self.model, backend="inductor", fullgraph=False, mode="default")
        self.device = device
        self.num_params = sum(p.numel() for p in self.model.parameters())
        self.model.to(self.device)
        self.surface = surface
        self.num_configs = num_configs
        self.use_uq = use_uq

        if dft_db_path is not None and os.path.exists(dft_db_path):
            self.db = connect(dft_db_path)
        else:
            self.db = None
        if not all([elem in self.surface_domain for elem in surface.slab.get_chemical_symbols()]):
            raise ValueError(
                f'GAME-Net-UQ can only evaluate surfaces with {", ".join(self.surface_domain)} elements.'
            )

    def __call__(self,
                 intermediate: Intermediate,
                 **kwargs) -> None:
        if isinstance(intermediate, Intermediate):
            self.eval(intermediate, **kwargs)
        else:
            return NotImplementedError("Input must be an Intermediate object.")

    @property
    def adsorbate_domain(self):
        return ADSORBATE_ELEMS

    @property
    def surface_domain(self):
        return METALS

    def __repr__(self) -> str:
        return (
            f"GAME-Net-UQ ({int(self.num_params/1000)}K params, device={self.device})"
        )

    def retrieve_from_db(self, intermediate: Intermediate) -> bool:
        """
        Check if the intermediate is in the DFT database and in affirmative case, update the intermediate
        with the most stable configuration.

        Parameters
        ----------
        intermediate : Intermediate
            The intermediate to evaluate.

        Returns
        -------
        bool
            True if the intermediate is in the database, False otherwise.
        """
        if self.db is None:
            return False

        inchikey = intermediate.code[:-1]  # del phase-identifier
        phase = intermediate.phase
        metal = self.surface.metal if phase == "ads" else "N/A"
        hkl = self.surface.facet if phase == "ads" else "N/A"
        metal_struct = f"{METAL_STRUCT_DICT[metal]}({hkl})" if phase == "ads" else "N/A"

        if intermediate.formula == 'H2':
            inchikey = 'SMIUJKHFIOXZIP-UHFFFAOYSA-N'

        stable_conf, max = [], np.inf
        for row in self.db.select(
            f"calc_type=int,metal={metal},facet={metal_struct},inchikey={inchikey}"
        ):
            atoms_object = row.toatoms()

            if not atoms_object:
                return False

            adsorbate = Atoms(
                symbols=[
                    atom.symbol for atom in atoms_object if atom.symbol in INTER_ELEMS
                ],
                positions=[
                    atom.position for atom in atoms_object if atom.symbol in INTER_ELEMS
                ],
            )

            if not len(adsorbate):
                return False

            if row.get("scaled_energy") < max:
                stable_conf.append([atoms_object, row.get("scaled_energy")])
                max = row.get("scaled_energy")

        if len(stable_conf):
            intermediate.ads_configs = {
                f"dft": {
                    "ase": stable_conf[-1][0],
                    "pyg": atoms_to_data(stable_conf[-1][0]),
                    "mu": stable_conf[-1][1],
                    "s": 0,
                }
            }
            return True

        return False

    def eval(
        self,
        intermediate: Intermediate, **kwargs
    ) -> None:
        """
        Estimate the energy of a state.

        Parameters
        ----------
        intermediate : Intermediate
            The intermediate to evaluate.

        Returns
        -------
        None
            Updates the Intermediate object with the estimated energy.
            Multiple adsorption configurations are stored in the ads_configs attribute.
        """
        if not all([elem in self.adsorbate_domain for elem in intermediate.molecule.get_chemical_symbols()]):
            raise ValueError(
                f'GAME-Net-UQ can only evaluate adsorbates/molecules with {", ".join(self.adsorbate_domain)} elements.'
            )
        if intermediate.phase == "surf":  # active site
            intermediate.ads_configs = {
                "surf": {"ase": intermediate.molecule, "mu": 0.0, "s": 0.0}
            }
        elif intermediate.phase == "gas":  # gas phase
            if self.db is not None and self.retrieve_from_db(intermediate):
                return
            else:
                config = intermediate.molecule
                with no_grad():
                    pyg = atoms_to_data(config, filter=False)
                    pyg = pyg.to(self.device).to(self.device)
                    y = self.model(pyg)
                    intermediate.ads_configs = {
                        "gas": {
                            "ase": config,
                            "mu": (
                                y[0] * self.model.y_scale_params["std"]
                                + self.model.y_scale_params["mean"]
                            ).item(),  # eV
                            "s": (y[1] * self.model.y_scale_params["std"]).item(),  # eV
                        }
                    }

        elif intermediate.phase == "ads":  # adsorbed
            if self.db and self.retrieve_from_db(intermediate):
                return
            else:
                adsorptions = place_adsorbate(intermediate, self.surface, self.num_configs)
                graphs = [
                    atoms_to_data(adsorption, filter=False) for adsorption in adsorptions
                ]
                loader = DataLoader(
                    graphs, batch_size=len(graphs), shuffle=False
                )
                with no_grad():
                    for batch in loader:
                        batch = batch.to(self.device)
                        y = self.model(batch)
                ads_config_dict = {}
                for i, adsorption in enumerate(adsorptions):
                        ads_config_dict[f"{i}"] = {}
                        ads_config_dict[f"{i}"]["ase"] = adsorption
                        ads_config_dict[f"{i}"]["mu"] = (
                            y[0][i] * self.model.y_scale_params["std"]
                            + self.model.y_scale_params["mean"]
                        ).item()  # eV
                        ads_config_dict[f"{i}"]["s"] = (
                            y[1][i] * self.model.y_scale_params["std"]
                        ).item()  # eV
                criterion = 's' if self.use_uq else 'mu'
                ads_config_dict = dict(
                    sorted(ads_config_dict.items(), key=lambda item: item[1][criterion])
                )
                intermediate.ads_configs = ads_config_dict
        else:
            raise ValueError("Phase not supported by the current estimator.")


class GameNetUQRxn(ReactionEnergyEstimator):
    def __init__(
        self,
        device: str = "cpu",
        T: float = 298.0,
        ref_electrode: str = "SHE",
        pH: float = 7.0,
        U: float = 0.0,
        use_uq: bool = False,
        torch_compile: bool = False,
        **kwargs
    ):
        """
        Interface for evaluating reaction properties using GAME-Net-UQ.

        Properties evaluated are transition state energy, reaction energy, and activation energy in eV.
        
        Args:
                device (str): Device to use for evaluation. Defaults to "cpu".
                T (float): Temperature in Kelvin. Required for electrochemical reactions. Defaults to 298 K.
                ref_electrode (str): Reference electrode required for electrochemical reactions. It can be 
                                "SHE" (Standard Hydrogen Electrode) or "RHE" (Reversible Hydrogen Electrode).
                                Defaults to SHE. With RHE, T and pH are not required.
                pH (float): pH of the system. Required for electrochemical reactions. Defaults to 7.
                U (float): Potential of the system. Required for electrochemical reactions. Defaults to 0 V.
                           Negative values refer to reductive potential, positive values to oxidative potential.
                use_uq (bool): Whether to use uncertainty in the evaluation. Defaults to False.
                    If True, the configuration with the lowest uncertainty is selected for reaction energy evaluation.
                torch_compile (bool, optional): Whether to compile the model using Torch. Defaults to False.
        """
        super().__init__(
            T=T,
            ref_electrode=ref_electrode,
            pH=pH,
            U=U,
            **kwargs
        )
        self.model = load_model(MODEL_PATH)
        if torch_compile:
            self.model = compile(self.model, backend="inductor", fullgraph=False, mode="default")
        self.device = device
        self.model.to(self.device)
        self.num_params = sum(p.numel() for p in self.model.parameters())
        self.use_uq = use_uq
        self.is_mlp = False
        self.supports_batching = True

    def adsorbate_domain(self):
        return ADSORBATE_ELEMS

    def surface_domain(self):
        return METALS

    def __repr__(self) -> str:
        return (
            f"GAME-Net-UQ ({int(self.num_params/1000)}K params, device={self.device})"
        )

    def calc_reaction_energy(self, reaction: ElementaryReaction) -> None:
        """
        Get the reaction energy of the elementary reaction.

        Args:
            reaction (ElementaryReaction): Elementary reaction.
        """
        mu_is, var_is, mu_fs, var_fs = 0.0, 0.0, 0.0, 0.0
        for species in list(reaction.reactants) + list(reaction.products):
            if species.is_surface:
                continue
            elif isinstance(species, Electron):  # Electrochemical conditions
                mu_is += abs(min(0, reaction.stoic["e-"])) * (abs(reaction.stoic["e-"])*self.U + (1 if self.ref_electrode == "SHE" else 0) * 2.303 * K_B * self.T * self.pH)
                mu_fs += abs(max(0, reaction.stoic["e-"])) * (abs(reaction.stoic["e-"])*self.U + (1 if self.ref_electrode == "SHE" else 0) * 2.303 * K_B * self.T * self.pH)
                var_is += 0.0
                var_fs += 0.0
                continue
            elif isinstance(species, (Water, Proton)):  # Electrochemical conditions
                species_formula = "H2O" if isinstance(species, Water) else "H2"
                x = 0.5 if species_formula == "H2" else 1.0
                gas_inter = [
                    inter
                    for inter in reaction.extra_intermediates.values()
                    if inter.formula == species_formula and inter.phase == "gas"
                ][0]
                energy_list = [
                    config["mu"] * x for config in gas_inter.ads_configs.values()
                ]
                s_list = [
                    config["s"] * x
                    for config in gas_inter.ads_configs.values()
                ]
            else:
                energy_list = [
                    config["mu"]
                    for config in species.ads_configs.values()
                ]
                s_list = [
                    config["s"]
                    for config in species.ads_configs.values()
                ]
            if not self.use_uq:  # Select configuration with lowest energy
                e_min_config = min(energy_list)
                s_min_config = s_list[energy_list.index(e_min_config)]
            else:  # Select configuration with lowest uncertainty 
                s_min_config = min(s_list)
                e_min_config = energy_list[s_list.index(s_min_config)]
            mu_is += abs(min(0, reaction.stoic[species.code])) * e_min_config
            mu_fs += abs(max(0, reaction.stoic[species.code])) * e_min_config
            var_is += abs(min(0, reaction.stoic[species.code])) * s_min_config**2
            var_fs += abs(max(0, reaction.stoic[species.code])) * s_min_config**2
        reaction.e_is = mu_is, var_is ** 0.5
        reaction.e_fs = mu_fs, var_fs ** 0.5
        reaction.e_rxn = mu_fs - mu_is, (var_fs + var_is) ** 0.5

    def calc_reaction_barrier(self, reaction: ElementaryReaction) -> None:
        """
        Get activation energy of the elementary reaction.

        Args:
            reaction (ElementaryReaction): Elementary reaction.
        """
        e_act_mu = reaction.e_ts[0] - reaction.e_is[0]
        if e_act_mu == 0.0:  # barrierless exothermic
            e_act_var = 0.0
        elif e_act_mu == reaction.e_rxn[0]:  # barrierless endothermic
            e_act_var = reaction.e_rxn[1]
        else:  # with barrier
            e_act_var = (reaction.e_ts[1] ** 2 + reaction.e_is[1] ** 2) ** 0.5
        reaction.e_act = e_act_mu, e_act_var

    def _build_product_nx(self, step: ElementaryReaction) -> nx.Graph:
        """Return a NetworkX graph representing the product fragments B* + C*."""
        competitors = [
            inter
            for inter in list(step.products)
            if not inter.is_surface
        ]
        if len(competitors) == 1:
            if abs(step.stoic[competitors[0].code]) == 2:  # A* -> 2B*
                nx0 = competitors[0].graph.copy()
                nx1 = competitors[0].graph.copy()
                offset = nx0.number_of_nodes()
                mapping = {n: n + offset for n in nx1.nodes()}
                nx1 = nx.relabel_nodes(nx1, mapping)
                return nx.compose(nx0, nx1)
            elif abs(step.stoic[competitors[0].code]) == 1:  # A* -> B* (ring opening)
                return competitors[0].graph.copy()
            else:
                raise ValueError("Reaction stoichiometry not supported.")
        elif len(competitors) == 2:  # A* -> B* + C* (B and C different)
            nx0 = competitors[0].graph.copy()
            nx1 = competitors[1].graph.copy()
            offset = nx0.number_of_nodes()
            mapping = {n: n + offset for n in nx1.nodes()}
            nx1 = nx.relabel_nodes(nx1, mapping)
            return nx.compose(nx0, nx1)
        else:
            raise ValueError("Reaction stoichiometry not supported.")

    def _find_potential_edges(self, graph: Data, bond: tuple[str, str]) -> list[int]:
        potential_edges = []
        for i in range(graph.num_edges):
            edge_idxs = graph.edge_index[:, i]
            atom1, atom2 = graph.elem[edge_idxs[0]], graph.elem[edge_idxs[1]]
            if (atom1, atom2) == bond or (atom2, atom1) == bond:
                potential_edges.append(i)
        return potential_edges

    def ts_graph(self, step: ElementaryReaction) -> Data:
        """
        Generate transition state graph representing the surface bond-breaking
        elementary reaction A* + * -> B* + C*.

        Args:
            step (ElementaryReaction): Bond-breaking reaction

        Returns:
            Data: graph representing the TS graph
        """
        bond = tuple(step.r_type.split("-"))

        # 1) Select initial state A*, convert to full adsorption graph, and find potential edges
        A = [
            inter for inter in list(step.reactants) if not inter.is_surface
        ][0]

        idx = min(
            A.ads_configs,
            key=lambda x: A.ads_configs[x]['s' if self.use_uq else 'mu'],
        )
        ts_graph = atoms_to_data(A.ads_configs[idx]["ase"], 
                                 surface_order=-1, filter=False)
        if not isinstance(step, BondBreaking):
            return ts_graph  
        n_nodes = ts_graph.num_nodes
        n_edges = ts_graph.num_edges
        potential_edges = self._find_potential_edges(ts_graph, bond)
        if len(potential_edges) == 0:
            raise RuntimeError(f"No edges found matching bond {bond} in ts_graph.")

        # 2) Find TS edge to label via isomorphic comparison
        nx_bc = self._build_product_nx(step)
        nx_bc_signature = connectivity_signature(nx_bc)
        for _, e_idx in enumerate(potential_edges):
            u = ts_graph.edge_index[0, e_idx].item()
            v = ts_graph.edge_index[1, e_idx].item()
            mask = ~(
                ((ts_graph.edge_index[0] == u) & (ts_graph.edge_index[1] == v)) |
                ((ts_graph.edge_index[0] == v) & (ts_graph.edge_index[1] == u))
            )
            edge_index_new = ts_graph.edge_index[:, mask]
            edge_attr_new = ts_graph.edge_attr[mask]
            data = ts_graph.clone()
            data.edge_index = edge_index_new
            data.edge_attr = edge_attr_new
            adsorbate = extract_adsorbate(data, ["C", "H", "O", "N", "S"])
            nx_adsorbate = pyg_to_nx(adsorbate)

            if connectivity_signature(nx_adsorbate) == nx_bc_signature:
                ts_graph.edge_attr[e_idx] = 1
                idx = ((ts_graph.edge_index[0] == v) & (ts_graph.edge_index[1] == u)).nonzero(as_tuple=True)[0].item()
                ts_graph.edge_attr[idx] = 1
                break

        # 3) Assign each adsorbate node to one of the two fragments B* or C*
        adsorbate_node_indices = [i for i in range(n_nodes) if ts_graph.elem[i] in ADSORBATE_ELEMS]
        node_indices_B, node_indices_C = {u}, {v}
        neighbors = {i: set() for i in range(n_nodes)}
        for i in range(n_edges):
            a, b = ts_graph.edge_index[:, i].tolist()
            neighbors[a].add(b)
            neighbors[b].add(a)
        queue_B, queue_C = [u], [v]

        while queue_B or queue_C:
            new_queue_B, new_queue_C = [], []

            for node in queue_B:
                for nbr in neighbors[node]:
                    if nbr in adsorbate_node_indices and nbr not in node_indices_B and nbr not in node_indices_C:
                        node_indices_B.add(nbr)
                        new_queue_B.append(nbr)

            for node in queue_C:
                for nbr in neighbors[node]:
                    if nbr in adsorbate_node_indices and nbr not in node_indices_B and nbr not in node_indices_C:
                        node_indices_C.add(nbr)
                        new_queue_C.append(nbr)

            queue_B, queue_C = new_queue_B, new_queue_C

        node_indices_B = list(node_indices_B)
        node_indices_C = list(node_indices_C)

        # 4) Find which of the two fragments is not connected to the surface
        connected_to_B, connected_to_C = False, False
        for node in node_indices_B:
            if any(nbr not in adsorbate_node_indices for nbr in neighbors[node]):
                connected_to_B = True
                break 

        for node in node_indices_C:
            if any(nbr not in adsorbate_node_indices for nbr in neighbors[node]):
                connected_to_C = True
                break

        # 5) Find surface atom to connect to the unconnected fragment
        # Select the 2-hop surface atom with lowest coordination number
        # avoiding surface atoms already interacting with the adsorbate
        if not connected_to_B or not connected_to_C:
            min_gcn_idx = min(
                ts_graph.surf_hops[2],
                key=lambda idx: ts_graph.x[idx, -1].item()
            )

        # 6) Add undirected edge between unconnected fragment and surface atom
        for frag, connected in [(u, connected_to_B), (v, connected_to_C)]:
            if not connected:
                ts_graph.edge_index = cat(
                    (ts_graph.edge_index, tensor([[frag, min_gcn_idx], [min_gcn_idx, frag]])), dim=1
                )
                ts_graph.edge_attr = cat(
                    (ts_graph.edge_attr, tensor([[0], [0]])), dim=0
                )

        # 7) Remove from total graph the surface atoms which are not within the 2-hop neighborhood
        atoms_to_keep = set(ts_graph.surf_hops[0]) | set(ts_graph.surf_hops[1]) | set(ts_graph.surf_hops[2])
        g = ts_graph.subgraph(tensor(list(atoms_to_keep)))
        del g.surf_hops
        return g

    def eval(
        self,
        x: Union[ElementaryReaction, list[ElementaryReaction]] ,
    ) -> None:
        """
        Estimate the reaction and the activation energies of a reaction step.

        Args:
            reaction (ElementaryReaction): The elementary reaction.
        """
        if isinstance(x, ElementaryReaction):
            for species in list(x.reactants) + list(x.products):
                if not species.is_surface and species.ads_configs == {}:
                    raise ValueError(f"Species in {x.repr_hr} ElementaryReaction are not evaluated.")
            with no_grad():
                self.calc_reaction_energy(x)
                if isinstance(x, BondBreaking):  # GNN evaluates TS from bond-breaking direction
                    ts_graph = self.ts_graph(x).to(self.device)  # unscaled output
                    y = self.model(ts_graph)  # scaled output
                    y_ts = y[0].item() * self.model.y_scale_params["std"] + self.model.y_scale_params["mean"], y[1].item() * self.model.y_scale_params["std"]
                    if y_ts[0] > x.e_is[0] and y_ts[0] > x.e_fs[0]:  # correct predicted TS between IS and FS
                        x.e_ts = y_ts
                    else: # wrong predicted TS between IS and FS, collapse to barrierless
                        x.e_ts = x.e_is if x.e_is[0] > x.e_fs[0] else x.e_fs
                else:  # barrierless, e_ts collapses to the highest among e_is and e_ts
                    x.e_ts = x.e_is if x.e_is[0] > x.e_fs[0] else x.e_fs
                self.calc_reaction_barrier(x)
        elif isinstance(x, list):
            ts_graphs = []
            for rxn in x:
                self.calc_reaction_energy(rxn)
                ts_graph = self.ts_graph(rxn).to(self.device)
                ts_graphs.append(ts_graph)
            loader = DataLoader(ts_graphs, batch_size=len(ts_graphs), shuffle=False)
            with no_grad():
                for batch in loader:
                    batch = batch.to(self.device)
                    y = self.model(batch)
                    for i, rxn in enumerate(x):
                        if isinstance(rxn, BondBreaking):
                            y_ts = y[0][i].item() * self.model.y_scale_params["std"] + self.model.y_scale_params["mean"], y[1][i].item() * self.model.y_scale_params["std"]
                            if y_ts[0] > rxn.e_is[0] and y_ts[0] > rxn.e_fs[0]:  # correct predicted TS between IS and FS
                                rxn.e_ts = y_ts
                            else:  # wrong predicted TS between IS and FS, collapse to barrierless
                                rxn.e_ts = rxn.e_is if rxn.e_is[0] > rxn.e_fs[0] else rxn.e_fs
                        else:
                            rxn.e_ts = rxn.e_is if rxn.e_is[0] > rxn.e_fs[0] else rxn.e_fs
            for rxn in x:
                self.calc_reaction_barrier(rxn)
