import os
from typing import Optional

from ase import Atoms
from ase.db import connect
from copy import deepcopy
from itertools import chain
import networkx as nx
import numpy as np
from torch import no_grad, where, cuda
from torch_geometric.data import Data


from care import (
    Intermediate,
    ElementaryReaction,
    Surface,
)
from care.evaluators import IntermediateEnergyEstimator, ReactionEnergyEstimator
from care.evaluators.gamenet_uq import MODEL_PATH, ADSORBATE_ELEMS, METALS
from care.evaluators.gamenet_uq.adsorption.placement import place_adsorbate
from care.constants import INTER_ELEMS, K_B
from care.crn.utils.electro import Proton, Electron, Water
from care.evaluators.gamenet_uq import METAL_STRUCT_DICT
from care.evaluators.gamenet_uq.functions import load_model
from care.evaluators.gamenet_uq.graph import atoms_to_data
from care.evaluators.gamenet_uq.graph_filters import extract_adsorbate
from care.evaluators.gamenet_uq.graph_tools import pyg_to_nx
from care.crn.templates import BondBreaking


class GameNetUQInter(IntermediateEnergyEstimator):
    def __init__(
        self, surface: Surface, dft_db_path: Optional[str] = None, **kwargs
    ):
        """Interface for GAME-Net-UQ for intermediates.

        Args:
            surface (Surface, optional): Surface of interest.
            dft_db_path (Optional[str], optional): Path to ASE database for retrieving
                DFT data. Defaults to None.
        """

        self.model = load_model(MODEL_PATH)
        # self.device = "cuda" if cuda.is_available() else "cpu"  # TODO: combine CUDA and multiprocessing
        self.device = "cpu"
        self.num_params = sum(p.numel() for p in self.model.parameters())
        self.model.to(self.device)
        self.surface = surface

        if dft_db_path is not None and os.path.exists(dft_db_path):
            self.db = connect(dft_db_path)
        else:
            self.db = None

    def adsorbate_domain(self):
        return ADSORBATE_ELEMS

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
                    "pyg": atoms_to_data(stable_conf[-1][0], self.model.graph_params),
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
                    pyg = atoms_to_data(config, self.model.graph_params)
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
                config_list = place_adsorbate(intermediate, self.surface)
                counter, ads_config_dict = 0, {}
                for config in config_list:
                    with no_grad():
                        ads_config_dict[f"{counter}"] = {}
                        ads_config_dict[f"{counter}"]["ase"] = config
                        ads_config_dict[f"{counter}"]["pyg"] = atoms_to_data(
                            config, self.model.graph_params
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

                # Getting the top 10 most stable configurations
                ads_config_dict = dict(
                    sorted(ads_config_dict.items(), key=lambda item: item[1]["mu"])[:10]
                )
                intermediate.ads_configs = ads_config_dict
        else:
            raise ValueError("Phase not supported by the current estimator.")


class GameNetUQRxn(ReactionEnergyEstimator):
    """
    Interface for evaluating reaction properties using GAME-Net-UQ.

    Properties evaluated are transition state energy, reaction energy, and activation energy in eV.
    """

    def __init__(
        self,
        intermediates: dict[str, Intermediate],
        T: float = None,
        pH: float = None,
        U: float = None,
        **kwargs
    ):
        self.model = load_model(MODEL_PATH)
        self.device = "cuda" if cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.num_params = sum(p.numel() for p in self.model.parameters())
        self.intermediates = intermediates
        self.pH = pH
        self.U = U
        self.T = T

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
        if reaction.r_type == "PCET":
            for reactant in reaction.reactants:
                if reactant.is_surface or isinstance(reactant, Electron):
                    continue
                elif isinstance(reactant, Water):
                    H2O_gas = [
                        inter
                        for inter in self.intermediates.values()
                        if inter.formula == "H2O" and inter.phase == "gas"
                    ][0]
                    energy_list = [
                        config["mu"] for config in H2O_gas.ads_configs.values()
                    ]
                    s_list = [
                        config["s"]
                        for config in self.intermediates[
                            H2O_gas.code
                        ].ads_configs.values()
                    ]
                    e_min_config = min(energy_list)
                    s_min_config = s_list[energy_list.index(e_min_config)]

                    mu_is += abs(reaction.stoic[reactant.code]) * e_min_config
                    var_is += abs(reaction.stoic[reactant.code]) * s_min_config**2
                elif isinstance(reactant, Proton):
                    H2_gas = [
                        inter
                        for inter in self.intermediates.values()
                        if inter.formula == "H2" and inter.phase == "gas"
                    ][0]
                    energy_list = [
                        config["mu"] * 0.5 for config in H2_gas.ads_configs.values()
                    ]
                    s_list = [
                        config["s"]
                        for config in self.intermediates[
                            H2_gas.code
                        ].ads_configs.values()
                    ]
                    e_min_config = min(energy_list)
                    s_min_config = s_list[energy_list.index(e_min_config)]

                    mu_is += abs(reaction.stoic[reactant.code]) * e_min_config
                    var_is += abs(reaction.stoic[reactant.code]) * s_min_config**2
                else:
                    energy_list = [
                        config["mu"]
                        for config in self.intermediates[
                            reactant.code
                        ].ads_configs.values()
                    ]
                    s_list = [
                        config["s"]
                        for config in self.intermediates[
                            reactant.code
                        ].ads_configs.values()
                    ]
                    e_min_config = min(energy_list)
                    s_min_config = s_list[energy_list.index(e_min_config)]

                    mu_is += abs(reaction.stoic[reactant.code]) * e_min_config
                    var_is += abs(reaction.stoic[reactant.code]) * s_min_config**2
            for product in reaction.products:
                if product.is_surface or isinstance(product, Electron):
                    continue
                elif isinstance(product, Water):
                    H2O_gas = [
                        inter
                        for inter in self.intermediates.values()
                        if inter.formula == "H2O" and inter.phase == "gas"
                    ][0]
                    energy_list = [
                        config["mu"] for config in H2O_gas.ads_configs.values()
                    ]
                    s_list = [
                        config["s"]
                        for config in self.intermediates[
                            H2O_gas.code
                        ].ads_configs.values()
                    ]
                    e_min_config = min(energy_list)
                    s_min_config = s_list[energy_list.index(e_min_config)]

                    mu_fs += abs(reaction.stoic[product.code]) * e_min_config
                    var_fs += abs(reaction.stoic[product.code]) * s_min_config**2
                elif isinstance(product, Proton):
                    H2_gas = [
                        inter
                        for inter in self.intermediates.values()
                        if inter.formula == "H2" and inter.phase == "gas"
                    ][0]
                    energy_list = [
                        config["mu"] * 0.5 for config in H2_gas.ads_configs.values()
                    ]
                    s_list = [
                        config["s"]
                        for config in self.intermediates[
                            H2_gas.code
                        ].ads_configs.values()
                    ]
                    e_min_config = min(energy_list)
                    s_min_config = s_list[energy_list.index(e_min_config)]

                    mu_fs += abs(reaction.stoic[product.code]) * e_min_config
                    var_fs += abs(reaction.stoic[product.code]) * s_min_config**2
                else:
                    energy_list = [
                        config["mu"]
                        for config in self.intermediates[
                            product.code
                        ].ads_configs.values()
                    ]
                    s_list = [
                        config["s"]
                        for config in self.intermediates[
                            product.code
                        ].ads_configs.values()
                    ]
                    e_min_config = min(energy_list)
                    s_min_config = s_list[energy_list.index(e_min_config)]

                    mu_fs += abs(reaction.stoic[product.code]) * e_min_config
                    var_fs += abs(reaction.stoic[product.code]) * s_min_config**2
            reaction.e_is = mu_is, var_is**0.5
            reaction.e_fs = mu_fs, var_fs**0.5
            components = list(chain.from_iterable(reaction.components))
            for component in components:
                if isinstance(component, Electron):
                    stoic_electro = reaction.stoic[component.code]
            reaction.e_rxn = (
                mu_fs
                - mu_is
                - stoic_electro * (self.U + 2.303 * K_B * self.T * self.pH),
                (var_fs + var_is) ** 0.5,
            )

        else:
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

                mu_is += abs(reaction.stoic[reactant.code]) * e_min_config
                var_is += abs(reaction.stoic[reactant.code]) * s_min_config**2
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

                mu_fs += abs(reaction.stoic[product.code]) * e_min_config
                var_fs += abs(reaction.stoic[product.code]) * s_min_config**2
            reaction.e_is = mu_is, var_is**0.5
            reaction.e_fs = mu_fs, var_fs**0.5

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

        # Look for potential edges to break
        def atom_symbol(idx):
            return ts_graph.node_feats[where(ts_graph.x[idx] == 1)[0].item()]

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
        return ts_graph

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
                try:
                    y = self.model(self.ts_graph(reaction).to(self.device))  # scaled output
                    y_ts = y.mean.item() * self.model.y_scale_params["std"] + self.model.y_scale_params["mean"], y.scale.item() * self.model.y_scale_params["std"]
                    if y_ts[0] > reaction.e_is[0] and y_ts[0] > reaction.e_fs[0]:  # correct predicted TS between IS and FS
                        reaction.e_ts = y_ts
                    else: # wrong predicted TS between IS and FS, collapse to barrierless
                        reaction.e_ts = reaction.e_is if reaction.e_is[0] > reaction.e_fs[0] else reaction.e_fs
                except:
                    print("Error in transition state for {}.".format(reaction.code))
            else:  # barrierless, e_ts collapses to the highest among e_is and e_ts
                reaction.e_ts = reaction.e_is if reaction.e_is[0] > reaction.e_fs[0] else reaction.e_fs
            self.calc_reaction_barrier(reaction)
