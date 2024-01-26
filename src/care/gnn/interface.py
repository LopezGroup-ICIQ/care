import os
from typing import Optional

from ase.db import connect
from copy import deepcopy
import networkx as nx
import numpy as np
from torch import no_grad, where
from torch_geometric.data import Data


from care import Intermediate, ElementaryReaction, Surface, IntermediateEnergyEstimator, ReactionEnergyEstimator, DB_PATH
from care.adsorption.adsorbate_placement import ads_placement
from care.constants import METAL_STRUCT_DICT
from care.gnn import load_model
from care.gnn.graph import atoms_to_data
from care.gnn.graph_filters import extract_adsorbate
from care.gnn.graph_tools import pyg_to_nx


class GameNetUQ(IntermediateEnergyEstimator):
    def __init__(self,
                 model_path: str):
        self.path = model_path
        self.model = load_model(model_path)

    def estimate_energy(self,
                        intermediate: Intermediate,
                        surface: Optional[Surface] = None,
                        metal: Optional[str] = None,
                        facet: Optional[str] = None,
                        ) -> None:
        """
        Estimate the energy of a state.

        Parameters
        ----------
        intermediate : Intermediate
            The intermediate.
        surface : Optional[Surface], optional
            The surface. Defaults to None.

        Returns
        -------
        None
            Updates the Intermediate object with the estimated energy.
        """

        if surface is None and intermediate.phase != "gas":
            if metal is None or facet is None:
                raise ValueError(
                    "Either a surface or a metal and a facet must be provided.")

            # Loading surface from database
            metal_db = connect(os.path.abspath(DB_PATH))
            metal_structure = f"{METAL_STRUCT_DICT[metal]}({facet})"
            surface_ase = metal_db.get_atoms(
                calc_type="surface", metal=metal, facet=metal_structure)
            surface = Surface(surface_ase, str(facet))
        # Estimate the energy of the intermediate
        if intermediate.is_surface:
            intermediate.ads_configs = {
                "surf": {"conf": intermediate.molecule, "mu": 0.0, "s": 0.0}
            }

        elif intermediate.phase == "gas":
            # Obtaining the reference energy
            intermediate.ads_configs = {
                "gas": {"conf": intermediate.molecule, "mu": intermediate.ref_energy(), "s": 0.0}
            }

        else:
            # Adsorbate placement
            config_list = ads_placement(intermediate, surface)

            counter = 0
            ads_config_dict = {}
            for config in config_list:
                with no_grad():
                    ads_config_dict[f"{counter}"] = {}
                    ads_config_dict[f"{counter}"]["config"] = config
                    ads_config_dict[f"{counter}"]["pyg"] = (
                        atoms_to_data(config, self.model.graph_params)
                    )
                    y = self.model(ads_config_dict[f"{counter}"]["pyg"])
                    ads_config_dict[f"{counter}"]["mu"] = (
                        y.mean * self.model.y_scale_params["std"]
                        + self.model.y_scale_params["mean"]
                    ).item()  # eV
                    ads_config_dict[f"{counter}"]["s"] = (
                        y.scale * self.model.y_scale_params["std"]
                    ).item()  # eV
                    counter += 1
            intermediate.ads_configs = ads_config_dict


class ReactionEnergy(ReactionEnergyEstimator):
    """
    Base class for reaction energy estimators.
    """

    def __init__(self,
                 model_path: str,
                 intermediates: dict[str, Intermediate],
                 ):
        self.path = model_path
        self.model = load_model(model_path)
        self.intermediates = intermediates

    def calc_reaction_energy(
        self, reaction: ElementaryReaction, mean_field: bool = True
    ) -> None:
        """
        Get the reaction energy of the elementary reaction.

        Args:
            reaction (ElementaryReaction): Elementary reaction.
            mean_field (bool, optional): If True, the reaction energy will be
                calculated using the mean field approximation, with the
                smallest energy for each intermediate.
                Defaults to True.
        """
        if mean_field:
            mu_is, var_is, mu_fs, var_fs = 0, 0, 0, 0
            for reactant in reaction.reactants:
                if reactant.is_surface:
                    continue
                energy_list = [
                    config["mu"]
                    for config in self.intermediates[reactant.code].ads_configs.values()
                ]
                s_list = [
                    config["s"]
                    for config in self.intermediates[reactant.code].ads_configs.values()
                ]
                e_min_config = min(energy_list)
                s_min_config = s_list[energy_list.index(e_min_config)]
                mu_is += (
                    abs(reaction.stoic[self.intermediates[reactant.code].code])
                    * e_min_config
                )
                var_is += (
                    reaction.stoic[self.intermediates[reactant.code].code] ** 2
                    * s_min_config**2
                )
            for product in reaction.products:
                if product.is_surface:
                    continue
                energy_list = [
                    config["mu"]
                    for config in self.intermediates[product.code].ads_configs.values()
                ]
                s_list = [
                    config["s"]
                    for config in self.intermediates[product.code].ads_configs.values()
                ]
                e_min_config = min(energy_list)
                s_min_config = s_list[energy_list.index(e_min_config)]
                mu_fs += (
                    abs(reaction.stoic[self.intermediates[product.code].code])
                    * e_min_config
                )
                var_fs += (
                    reaction.stoic[self.intermediates[product.code].code] ** 2
                    * s_min_config**2
                )
            mu_rxn = mu_fs - mu_is
            std_rxn = (var_fs + var_is) ** 0.5
        else:
            pass
        reaction.e_is = mu_is, var_is**0.5
        reaction.e_fs = mu_fs, var_fs**0.5
        reaction.e_rxn = mu_rxn, std_rxn

    def calc_reaction_barrier(self, reaction: ElementaryReaction) -> None:
        """
        Get activation energy of the elementary reaction.

        Args:
            reaction (ElementaryReaction): Elementary reaction.
        """
        if "-" not in reaction.r_type:
            reaction.e_act = max(0, reaction.e_rxn[0]), reaction.e_rxn[1]
        else:  # bond-breaking reaction
            reaction.e_act = (
                reaction.e_ts[0] - reaction.e_is[0],
                (reaction.e_ts[1] ** 2 + reaction.e_is[1] ** 2) ** 0.5,
            )

            if reaction.e_act[0] < 0:  # Negative predicted barrier
                reaction.e_act = 0, 0
            if (
                reaction.e_act[0] < reaction.e_rxn[0]
            ):  # Barrier lower than reaction energy
                reaction.e_act = reaction.e_rxn[0], reaction.e_rxn[1]

    def ts_graph(self, step: ElementaryReaction) -> Data:
        """
        Given the bond-breaking reaction, detect the broken bond in the
        transition state and label the corresponding edge.
        Given A* + * -> B* + C* and the bond-breaking type X-Y, take the graph of A*,
        break all the potential X-Y bonds and perform isomorphism with B* + C*.
        When isomorphic, the broken edge is labelled.

        Args:
            graph (Data): adsorption graph of the intermediate which is fragmented in the reaction.
            reaction (ElementaryReaction): Bond-breaking reaction.

        Returns:
            Data: graph with the broken bond labeled.
        """

        if "-" not in step.r_type:
            raise ValueError("Input reaction must be a bond-breaking reaction.")
        bond = tuple(step.r_type.split("-"))

        # Select intermediate that is fragmented in the reaction (A*)
        inters = {
            inter.code: inter.graph.number_of_edges()
            for inter in list(step.reactants) + list(step.products)
            if not inter.is_surface
        }
        inter_code = max(inters, key=inters.get)
        idx = min(
            self.intermediates[inter_code].ads_configs,
            key=lambda x: self.intermediates[inter_code].ads_configs[x]["mu"],
        )
        ts_graph = deepcopy(self.intermediates[inter_code].ads_configs[idx]["pyg"])
        competitors = [
            inter
            for inter in list(step.reactants) + list(step.products)
            if not inter.is_surface and inter.code != inter_code
        ]

        # Build the nx graph of the competitors (B* + C*)
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
        else:  # asymmetric fragmentation
            nx_bc = [competitors[0].graph, competitors[1].graph]
            mapping = {n: n + nx_bc[0].number_of_nodes() for n in nx_bc[1].nodes()}
            nx_bc[1] = nx.relabel_nodes(nx_bc[1], mapping)
            nx_bc = nx.compose(nx_bc[0], nx_bc[1])

        # Lool for potential edges to break
        atom_symbol = lambda idx: ts_graph.node_feats[
            where(ts_graph.x[idx] == 1)[0].item()
        ]
        potential_edges = []
        for i in range(ts_graph.edge_index.shape[1]):
            edge_idxs = ts_graph.edge_index[:, i]
            atom1, atom2 = atom_symbol(edge_idxs[0]), atom_symbol(edge_idxs[1])
            if (atom1, atom2) == bond or (atom2, atom1) == bond:
                potential_edges.append(i)
        counter = 0

        # Find correct one via isomorphic comparison
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
            nx_graph = pyg_to_nx(adsorbate, data.node_feats)
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
        step.ts_graph = ts_graph

    def estimate_energy(self,
                        reaction: ElementaryReaction,
                        ) -> None:
        """
        Estimate the reaction and the activation energies of a reaction step.

        Args:
            reaction (ElementaryReaction): The ElementaryReaction.
            surf (Surface, optional): The surface. Defaults to None.
        """
        with no_grad():
            # Estimate the reaction energy
            self.calc_reaction_energy(reaction)
            if "-" in reaction.r_type:
                self.ts_graph(reaction)
                y = self.model(reaction.ts_graph)
                reaction.e_ts = (
                    y.mean.item() * self.model.y_scale_params["std"]
                    + self.model.y_scale_params["mean"],
                    y.scale.item() * self.model.y_scale_params["std"],
                )
            # Estimate the activation energy
            self.calc_reaction_barrier(reaction)
            print(reaction, reaction.r_type)
            print(
                "\nEact [eV]: N({:.2f}, {:.2f})    Erxn [eV]: N({:.2f}, {:.2f})".format(
                    reaction.e_act[0],
                    reaction.e_act[1],
                    reaction.e_rxn[0],
                    reaction.e_rxn[1],
                )
            )