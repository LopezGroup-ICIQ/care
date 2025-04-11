"""
Interface to GAME-Net-UQ model.
"""

import os
from typing import Optional

from ase import Atoms
from ase.db import connect
from copy import deepcopy
import networkx as nx
import numpy as np
from torch import no_grad, cuda, tensor, cat
from torch_geometric.data import Data

from care import Intermediate, ElementaryReaction, Surface
from care.evaluators import IntermediateEnergyEstimator, ReactionEnergyEstimator
from care.evaluators.gamenet_uq import MODEL_PATH, ADSORBATE_ELEMS, METALS, METAL_STRUCT_DICT
from care.adsorption import place_adsorbate
from care.constants import INTER_ELEMS, K_B
from care.crn.utils.electro import Proton, Electron, Water
from care.evaluators.gamenet_uq.functions import load_model
from care.evaluators.gamenet_uq.graph import atoms_to_data
from care.evaluators.gamenet_uq.graph_filters import extract_adsorbate
from care.evaluators.gamenet_uq.graph_tools import pyg_to_nx
from care.crn.templates import BondBreaking


class GameNetUQInter(IntermediateEnergyEstimator):
    def __init__(
        self,
        surface: Surface,
        dft_db_path: Optional[str] = None,
        num_configs: int = 3,
        use_uq: bool = False,
        **kwargs
    ):
        """Interface for GAME-Net-UQ for intermediates.

        Args:
            surface (Surface, optional): Surface of interest.
            dft_db_path (Optional[str], optional): Path to ASE database for retrieving
                DFT data. Defaults to None.
            num_configs (int, optional): Number of configurations to consider for the adsorbed phase.
                Defaults to 3.
            use_uq (bool, optional): Whether to use uncertainty in the evaluation. Defaults to False.
                if True, the configurations will be sorted in ascending order of uncertainty in the ads_configs attribute.
        """

        self.model = load_model(MODEL_PATH)
        self.device = "cpu"
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
                    pyg = atoms_to_data(config)
                    pyg = pyg.to(self.device)
                    y = self.model(pyg)
                    intermediate.ads_configs = {
                        "gas": {
                            "ase": config,
                            "pyg": pyg,
                            "mu": (
                                y.mean * self.model.y_scale_params["std"]
                                + self.model.y_scale_params["mean"]
                            ).item(),  # eV
                            "s": (y.scale * self.model.y_scale_params["std"]).item(),  # eV
                        }
                    }

        elif intermediate.phase == "ads":  # adsorbed
            if self.db and self.retrieve_from_db(intermediate):
                return
            else:
                adsorptions = place_adsorbate(intermediate, self.surface, self.num_configs)
                ads_config_dict = {}
                for i, adsorption in enumerate(adsorptions):
                    with no_grad():
                        ads_config_dict[f"{i}"] = {}
                        ads_config_dict[f"{i}"]["ase"] = adsorption
                        ads_config_dict[f"{i}"]["pyg"] = atoms_to_data(
                            adsorption
                        )
                        y = self.model(ads_config_dict[f"{i}"]["pyg"])
                        ads_config_dict[f"{i}"]["mu"] = (
                            y.mean * self.model.y_scale_params["std"]
                            + self.model.y_scale_params["mean"]
                        ).item()  # eV
                        ads_config_dict[f"{i}"]["s"] = (
                            y.scale * self.model.y_scale_params["std"]
                        ).item()  # eV

                # Select best configurations based on the mean (mu) or the uncertainty (s)
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
        intermediates: dict[str, Intermediate],
        T: float = 298.0,
        ref_electrode: str = "SHE",
        pH: float = 7.0,
        U: float = 0.0,
        use_uq: bool = False,
        **kwargs
    ):
        """
        Interface for evaluating reaction properties using GAME-Net-UQ.

        Properties evaluated are transition state energy, reaction energy, and activation energy in eV.
        
        Args:
                intermediates (dict): Dictionary of intermediates already evaluated.
                T (float): Temperature in Kelvin. Required for electrochemical reactions. Defaults to 298 K.
                ref_electrode (str): Reference electrode required for electrochemical reactions. It can be 
                                "SHE" (Standard Hydrogen Electrode) or "RHE" (Reversible Hydrogen Electrode).
                                Defaults to SHE. With RHE, T and pH are not required.
                pH (float): pH of the system. Required for electrochemical reactions. Defaults to 7.
                U (float): Potential of the system. Required for electrochemical reactions. Defaults to 0 V.
                           Negative values refer to reductive potential, positive values to oxidative potential.
                use_uq (bool): Whether to use uncertainty in the evaluation. Defaults to False.
                    If True, the configuration with the lowest uncertainty is selected for reaction energy evaluation.
        """
        self.model = load_model(MODEL_PATH)
        self.device = "cuda" if cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.num_params = sum(p.numel() for p in self.model.parameters())
        self.use_uq = use_uq
        self.intermediates = intermediates
        self.ref_electrode = ref_electrode
        if self.ref_electrode not in ["SHE", "RHE"]:
            raise ValueError(
                f"Electrode potential must be SHE or RHE. {self.ref_electrode} is not supported."
            )
        self.pH = pH
        self.U = U
        self.T = T
        
        # Check that intermediates have been evaluated with ads_configs attribute, if not raise Warning
        if not all([inter.ads_configs for inter in self.intermediates.values()]):
            raise Warning(
                "Not all intermediates have been evaluated. Please evaluate all intermediates before evaluating reaction properties."
            )

    def adsorbate_domain(self):
        return ADSORBATE_ELEMS

    def surface_domain(self):
        return METALS

    def __repr__(self) -> str:
        return (
            f"GAME-Net-UQ ({int(self.num_params/1000)}K params, device={self.device})"
        )

    def __call__(self,
                 rxn: ElementaryReaction) -> None:
        self.eval(rxn)

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
                    for inter in self.intermediates.values()
                    if inter.formula == species_formula and inter.phase == "gas"
                ][0]
                energy_list = [
                    config["mu"] * x for config in gas_inter.ads_configs.values()
                ]
                s_list = [
                    config["s"] * x
                    for config in self.intermediates[
                        gas_inter.code
                    ].ads_configs.values()
                ]
            else:
                energy_list = [
                    config["mu"]
                    for config in self.intermediates[species.code].ads_configs.values()
                ]
                s_list = [
                    config["s"]
                    for config in self.intermediates[species.code].ads_configs.values()
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
    
    def ts_graph(self, step: ElementaryReaction) -> Data:
        """
        Generate transition state graph representing the surface bon-breaking
        elementary reaction A* + * -> B* + C*.

        Args:
            step (ElementaryReaction): Bond-breaking reaction

        Returns:
            Data: graph representing the TS graph
        """

        if not isinstance(step, BondBreaking):
            raise ValueError("Input reaction must be a bond-breaking reaction.")
        bond = tuple(step.r_type.split("-"))

        # 1) Select unfragmented adsorbate A*
        A_code = [
            inter.code for inter in list(step.reactants) if not inter.is_surface
        ][0]
        idx = min(
            self.intermediates[A_code].ads_configs,
            key=lambda x: self.intermediates[A_code].ads_configs[x]['s' if self.use_uq else 'mu'],
        )
        ts_graph = atoms_to_data(self.intermediates[A_code].ads_configs[idx]["ase"], 
                                 surface_order=-1)  # whole graph
        competitors = [
            inter
            for inter in list(step.products)
            if not inter.is_surface
        ]

        # 2) Build the NetworkX graph of the reaction component B* + C*
        if len(competitors) == 1:
            if abs(step.stoic[competitors[0].code]) == 2:  # A* -> 2B*
                nx_bc = [competitors[0].graph, competitors[0].graph]
                mapping = {n: n + nx_bc[0].number_of_nodes() for n in nx_bc[1].nodes()}
                nx_bc[1] = nx.relabel_nodes(nx_bc[1], mapping)
                nx_bc = nx.compose(nx_bc[0], nx_bc[1])
            elif abs(step.stoic[competitors[0].code]) == 1:  # A* -> B* (ring opening)
                nx_bc = competitors[0].graph
            else:
                raise ValueError("Reaction stoichiometry not supported.")
        else:  # A* -> B* + C* (B and C different)
            nx_bc = [competitors[0].graph, competitors[1].graph]
            mapping = {n: n + nx_bc[0].number_of_nodes() for n in nx_bc[1].nodes()}
            nx_bc[1] = nx.relabel_nodes(nx_bc[1], mapping)
            nx_bc = nx.compose(nx_bc[0], nx_bc[1])

        potential_edges = []
        for i in range(ts_graph.edge_index.shape[1]):
            edge_idxs = ts_graph.edge_index[:, i]
            atom1, atom2 = ts_graph.elem[edge_idxs[0]], ts_graph.elem[edge_idxs[1]]
            if (atom1, atom2) == bond or (atom2, atom1) == bond:
                potential_edges.append(i)

        # 3) Find TS edge via isomorphic comparison
        counter = 0
        while True:
            data = deepcopy(ts_graph)
            u, v = data.edge_index[:, potential_edges[counter]]
            mask = ~(
                (data.edge_index[0] == u) & (data.edge_index[1] == v)
                | (data.edge_index[0] == v) & (data.edge_index[1] == u)
            )
            data.edge_index = data.edge_index[:, mask]
            data.edge_attr = data.edge_attr[mask]
            adsorbate = extract_adsorbate(data, ["C", "H", "O", "N", "S"])
            nx_graph = pyg_to_nx(adsorbate)
            if nx.is_isomorphic(
                nx_bc, nx_graph, node_match=lambda x, y: x["elem"] == y["elem"]
            ):
                ts_graph.edge_attr[potential_edges[counter]] = 1
                idx = np.where(
                    (ts_graph.edge_index[0] == v) & (ts_graph.edge_index[1] == u)
                )[0].item()
                ts_graph.edge_attr[idx] = 1
                break
            else:
                counter += 1

        # 4) Assign each adsorbate node to one of the two fragments B* or C*
        adsorbate_node_indices = [
            i for i in range(ts_graph.x.shape[0]) if ts_graph.elem[i] in ADSORBATE_ELEMS
        ]
        node_indices_B, node_indices_C = [u.item()], [v.item()]
        for adsorbate_node_index in adsorbate_node_indices:
            if adsorbate_node_index in node_indices_B or adsorbate_node_index in node_indices_C:
                continue
            else:
                # check if the node is connected to node_indices_B or node_indices_C
                for i in range(ts_graph.edge_index.shape[1]):
                    edge_idxs = ts_graph.edge_index[:, i]
                    if (
                        (edge_idxs[0] == adsorbate_node_index and edge_idxs[1] in node_indices_B)
                        or (edge_idxs[1] == adsorbate_node_index and edge_idxs[0] in node_indices_B)
                    ):
                        node_indices_B.append(adsorbate_node_index)
                        break
                    elif (
                        (edge_idxs[0] == adsorbate_node_index and edge_idxs[1] in node_indices_C)
                        or (edge_idxs[1] == adsorbate_node_index and edge_idxs[0] in node_indices_C)
                    ):
                        node_indices_C.append(adsorbate_node_index)
                        break
        adsorbate_node_indices = list(set(adsorbate_node_indices))
        node_indices_B = list(set(node_indices_B))
        node_indices_C = list(set(node_indices_C))

        # 5) Find which of the two fragments is not connected to the surface
        connected_to_B, connected_to_C = False, False
        for i in range(ts_graph.edge_index.shape[1]):
            edge_idxs = ts_graph.edge_index[:, i]
            if (
                (edge_idxs[0] in node_indices_B and edge_idxs[1] not in adsorbate_node_indices)
                or (edge_idxs[1] in node_indices_B and edge_idxs[0] not in adsorbate_node_indices)
            ):
                connected_to_B = True
            if (
                (edge_idxs[0] in node_indices_C and edge_idxs[1] not in adsorbate_node_indices)
                or (edge_idxs[1] in node_indices_C and edge_idxs[0] not in adsorbate_node_indices)
            ):
                connected_to_C = True

        # 6) Find surface atom to connect to the unconnected fragment
        # Select the 2-hop surface atom with lowest coordination number
        min_gcn_idx, min_gcn = 1, 1.0
        for idx in ts_graph.surf_hops[2]: # Avoid surface atoms already interacting with the adsorbate
            if ts_graph.x[ts_graph.idx.index(idx), -1] < min_gcn:
                min_gcn = ts_graph.x[ts_graph.idx.index(idx), -1]
                min_gcn_idx = ts_graph.idx.index(idx)

        # 7) Add undirected edge between unconnected fragment and surface atom
        if not connected_to_B:
            ts_graph.edge_index = cat(
                (ts_graph.edge_index, tensor([[u, min_gcn_idx], [min_gcn_idx, u]])), dim=1
            )
            ts_graph.edge_attr = cat(
                (ts_graph.edge_attr, tensor([[0], [0]])), dim=0
            )
        if not connected_to_C:
            ts_graph.edge_index = cat(
                (ts_graph.edge_index, tensor([[v, min_gcn_idx], [min_gcn_idx, v]])), dim=1
            )
            ts_graph.edge_attr = cat(
                (ts_graph.edge_attr, tensor([[0], [0]])), dim=0
            )

        # 8) Remove from total graph the surface atoms which are not within the 2-hop neighborhood
        atoms_to_keep = []
        for i in range(ts_graph.x.shape[0]):
            if ts_graph.idx[i] in ts_graph.surf_hops[0] + ts_graph.surf_hops[1] + ts_graph.surf_hops[2]:
                atoms_to_keep.append(i)
        g = ts_graph.subgraph(tensor(atoms_to_keep))
        surf_hops_keys_to_delete = list(g.surf_hops.keys())
        for key in surf_hops_keys_to_delete:
            if key not in [0, 1, 2]:
                del g.surf_hops[key]  
        return g

    def eval(
        self,
        reaction: ElementaryReaction,
    ) -> None:
        """
        Estimate the reaction and the activation energies of a reaction step.

        Args:
            reaction (ElementaryReaction): The elementary reaction.
        """
        with no_grad():
            self.calc_reaction_energy(reaction)
            if isinstance(reaction, BondBreaking):  # GNN evaluates TS from bond-breaking direction
                ts_graph = self.ts_graph(reaction).to(self.device)  # unscaled output
                reaction.graph = ts_graph
                y = self.model(ts_graph)  # scaled output
                y_ts = y.mean.item() * self.model.y_scale_params["std"] + self.model.y_scale_params["mean"], y.scale.item() * self.model.y_scale_params["std"]
                if y_ts[0] > reaction.e_is[0] and y_ts[0] > reaction.e_fs[0]:  # correct predicted TS between IS and FS
                    reaction.e_ts = y_ts
                else: # wrong predicted TS between IS and FS, collapse to barrierless
                    reaction.e_ts = reaction.e_is if reaction.e_is[0] > reaction.e_fs[0] else reaction.e_fs
            else:  # barrierless, e_ts collapses to the highest among e_is and e_ts
                reaction.e_ts = reaction.e_is if reaction.e_is[0] > reaction.e_fs[0] else reaction.e_fs
            self.calc_reaction_barrier(reaction)
