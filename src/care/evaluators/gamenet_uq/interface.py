"""
Interface to GAME-Net-UQ graph neural network.
"""

from typing import Union, Any

from ase import Atoms

try:
    from gamenet_uq.constants import ADSORBATE_ELEMS, METALS
    from gamenet_uq.functions import load_model_from_url
    from gamenet_uq.graph import atoms_to_data
    from gamenet_uq.graph_filters import extract_adsorbate
    from gamenet_uq.graph_tools import convert_pyg_to_nx
    from torch_geometric.loader import DataLoader
    import torch
    torch.set_float32_matmul_precision('high')
    from torch import no_grad, tensor, cat, compile
    GAMENETUQ_AVAILABLE = True
except ImportError:
    GAMENETUQ_AVAILABLE = False

from care import Intermediate, ElementaryReaction
from care.evaluators import UniversalEvaluator
from care.adsorption import place_adsorbate
from care.crn.utils.graph import connectivity_signature
from care.crn.intermediate import SurfaceSite, AdsorbedSpecies, GasSpecies
from care.crn.templates.dissociation import BondBreaking, BondFormation, _build_fragment_nx


class GameNetUQevaluator(UniversalEvaluator):
    def __init__(
        self,
        device: str = "cpu",
        num_configs: int = 3,
        use_uq: bool = False,
        torch_compile: bool = False,
        **kwargs
    ):
        """Interface for GAME-Net-UQ evaluating intermediates and reaction barriers."""
        if not GAMENETUQ_AVAILABLE: 
            raise ImportError(
                "GameNet-UQ not installed. "
                "Install it using pip install gamenet-uq."
            )

        super().__init__(**kwargs)
        self.model = load_model_from_url()
        if torch_compile:
            self.model = compile(self.model, backend="inductor", fullgraph=False, mode="default")
            
        self.y_mean = self.model.y_scale_params["mean"]  
        self.y_std = self.model.y_scale_params["std"]
        self.device = device
        self.num_params = sum(p.numel() for p in self.model.parameters())
        self.model.to(self.device)
        self.num_configs = num_configs
        self.use_uq = use_uq
        self.is_mlp = False

    @property
    def adsorbate_domain(self):
        return ADSORBATE_ELEMS

    @property
    def surface_domain(self):
        return METALS

    def __repr__(self) -> str:
        return f"GAME-Net-UQ ({int(self.num_params/1000)}K params, device={self.device})"
        
    def eval(self, x: Union[Intermediate, ElementaryReaction, Atoms, list], **kwargs) -> None:
        """Override base eval to cleanly catch and process lists for fast GNN batching."""
        if isinstance(x, list):
            self._eval_reaction_batch(x, **kwargs)
        else:
            super().eval(x, **kwargs)

    def _eval_species(self, intermediate: Intermediate, **kwargs) -> None:
        """Evaluates species energy via GNN prediction (no structural relaxation)."""
        if not all([elem in self.adsorbate_domain for elem in intermediate.molecule.get_chemical_symbols()]):
            raise ValueError(
                f'GAME-Net-UQ can only evaluate adsorbates/molecules with {", ".join(self.adsorbate_domain)} elements.'
            )
            
        if isinstance(intermediate, SurfaceSite):
            intermediate.E = 0.0
            
        elif isinstance(intermediate, GasSpecies):
            with no_grad():
                pyg = atoms_to_data(intermediate.molecule, filter=False).to(self.device)
                y = self.model(pyg)  
                s = (y[1] * self.y_std).item()  #TODO: find way to store in Intermediate
                intermediate.E = (y[0] * self.y_std + self.y_mean).item()
                    
        elif isinstance(intermediate, AdsorbedSpecies):
            surf = getattr(intermediate, "catalyst", None)
            if surf is None:
                raise ValueError(f"Surface must be provided to evaluate {intermediate.code}.")
                
            adsorptions = place_adsorbate(intermediate, surf, self.num_configs)
            graphs = [atoms_to_data(adsorption, filter=False) for adsorption in adsorptions]
            loader = DataLoader(graphs, batch_size=len(graphs), shuffle=False)
            
            with no_grad():
                for batch in loader:
                    y = self.model(batch.to(self.device))
                    
            ads_config_dict = {}
            for i, adsorption in enumerate(adsorptions):
                ads_config_dict[f"{i}"] = {
                    "ase": adsorption,
                    "mu": (y[0][i] * self.y_std + self.y_mean).item(),
                    "s": (y[1][i] * self.y_std).item()
                }
                
            criterion = 's' if self.use_uq else 'mu'
            intermediate.ads_configs = dict(
                sorted(ads_config_dict.items(), key=lambda item: item[1][criterion])
            )
        else:
            raise ValueError("Phase not supported by GAME-Net-UQ.")

    def _eval_atoms(self, atoms: Atoms, **kwargs) -> None:
        if not all([elem in self.adsorbate_domain + self.surface_domain for elem in atoms.get_chemical_symbols()]):
            raise ValueError(
                f'GAME-Net-UQ can only evaluate adsorbates/molecules with {", ".join(self.adsorbate_domain + self.surface_domain)} elements.'
            )
        return self.model(atoms_to_data(atoms, filter=False).to(self.device))[0].item() * self.y_std + self.y_mean

    def _find_potential_edges(self, graph: Any, bond: tuple[str, str]) -> list[int]:
        potential_edges = []
        for i in range(graph.num_edges):
            edge_idxs = graph.edge_index[:, i]
            atom1, atom2 = graph.elem[edge_idxs[0]], graph.elem[edge_idxs[1]]
            if (atom1, atom2) == bond or (atom2, atom1) == bond:
                potential_edges.append(i)
        return potential_edges

    def ts_graph(self, step: ElementaryReaction) -> Any:
        """Generate TS graph representing the surface bond-breaking reaction."""
        bond = tuple(step.r_type.split("-"))
        A_state = list(step.reactants if isinstance(step, BondBreaking) else step.products)
        BC_state = list(step.products if isinstance(step, BondBreaking) else step.reactants)
        A = [x for x in A_state if isinstance(x, AdsorbedSpecies)][0]

        idx = min(A.ads_configs, key=lambda x: A.ads_configs[x]['s' if self.use_uq else 'mu'])
        ts_graph = atoms_to_data(A.ads_configs[idx]["ase"], surface_order=-1, filter=False, add_surf_hops_info=True) 
            
        n_nodes = ts_graph.num_nodes
        n_edges = ts_graph.num_edges
        potential_edges = self._find_potential_edges(ts_graph, bond)
        if len(potential_edges) == 0:
            raise RuntimeError(f"No edges found matching bond {bond} in ts_graph.")

        nx_bc = _build_fragment_nx(step.stoic, BC_state)
        nx_bc_signature = connectivity_signature(nx_bc)
        match_found = False
        for _, e_idx in enumerate(potential_edges):
            u = ts_graph.edge_index[0, e_idx].item()
            v = ts_graph.edge_index[1, e_idx].item()
            mask = ~(
                ((ts_graph.edge_index[0] == u) & (ts_graph.edge_index[1] == v)) |
                ((ts_graph.edge_index[0] == v) & (ts_graph.edge_index[1] == u))
            )
            data = ts_graph.clone()
            data.edge_index = ts_graph.edge_index[:, mask]
            data.edge_attr = ts_graph.edge_attr[mask]
            
            adsorbate = extract_adsorbate(data, ADSORBATE_ELEMS)
            nx_adsorbate = convert_pyg_to_nx(adsorbate)
            for _, node_data in nx_adsorbate.nodes(data=True):
                if 'elem' not in node_data:
                    node_data['elem'] = node_data.get('atom', node_data.get('symbol', node_data.get('element', 'X')))

            if connectivity_signature(nx_adsorbate) == nx_bc_signature:
                ts_graph.edge_attr[e_idx] = 1
                idx_rev = ((ts_graph.edge_index[0] == v) & (ts_graph.edge_index[1] == u)).nonzero(as_tuple=True)[0].item()
                ts_graph.edge_attr[idx_rev] = 1
                match_found = True
                break

        if not match_found:
            print(f"\n--- SIGNATURE MISMATCH FOR {step.repr_hr} ---")
            print(f"Target signature (Products): {nx_bc_signature}")
            print(f"Target nodes: {nx_bc.nodes(data=True)}")
            print(f"Candidate signature (Fragmented TS): {connectivity_signature(nx_adsorbate)}")
            print(f"Candidate nodes: {nx_adsorbate.nodes(data=True)}\n")
            raise RuntimeError(f"Could not identify the correct bond to break for reaction {step.repr_hr}. Signatures do not match.")

        adsorbate_node_indices = [i for i in range(n_nodes) if ts_graph.elem[i] in self.adsorbate_domain]
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

        connected_to_B, connected_to_C = False, False
        for node in list(node_indices_B):
            if any(nbr not in adsorbate_node_indices for nbr in neighbors[node]):
                connected_to_B = True
                break 

        for node in list(node_indices_C):
            if any(nbr not in adsorbate_node_indices for nbr in neighbors[node]):
                connected_to_C = True
                break

        if not connected_to_B or not connected_to_C:
            min_gcn_idx = min(ts_graph.surf_hops[2], key=lambda idx: ts_graph.x[idx, -1].item())

        for frag, connected in [(u, connected_to_B), (v, connected_to_C)]:
            if not connected:
                ts_graph.edge_index = cat((ts_graph.edge_index, tensor([[frag, min_gcn_idx], [min_gcn_idx, frag]])), dim=1)
                ts_graph.edge_attr = cat((ts_graph.edge_attr, tensor([[0], [0]])), dim=0)

        atoms_to_keep = set(ts_graph.surf_hops[0]) | set(ts_graph.surf_hops[1]) | set(ts_graph.surf_hops[2])
        g = ts_graph.subgraph(tensor(list(atoms_to_keep)))
        del g.surf_hops
        return g

    def _eval_reaction(self, reaction: ElementaryReaction, **kwargs) -> None:
        """Evaluates a single reaction using direct GNN inference."""
        for species in list(reaction.reactants) + list(reaction.products):
            if not isinstance(species, SurfaceSite) and getattr(species, 'E', None) is None:
                self._eval_species(species, **kwargs)

        if isinstance(reaction, BondBreaking) or isinstance(reaction, BondFormation):
            with no_grad():
                ts_graph = self.ts_graph(reaction).to(self.device)
                y = self.model(ts_graph)
                y_ts_mean = y[0].item() * self.y_std + self.y_mean
                y_ts_std = y[1].item() * self.y_std
                reaction.e_ts = y_ts_mean

    def _eval_reaction_batch(self, reactions: list[ElementaryReaction], **kwargs) -> None:
        """Highly optimized batch processing for list of reactions."""
        for rxn in reactions:
            for species in list(rxn.reactants) + list(rxn.products):
                if not isinstance(species, SurfaceSite) and getattr(species, 'E', None) is None:
                    self._eval_species(species, **kwargs)

        ts_graphs = []
        valid_rxns = []
        
        for rxn in reactions:
            if isinstance(rxn, BondBreaking) or isinstance(rxn, BondFormation):
                ts_graphs.append(self.ts_graph(rxn).to(self.device))
                valid_rxns.append(rxn)

        if not ts_graphs:
            return

        loader = DataLoader(ts_graphs, batch_size=len(ts_graphs), shuffle=False)
        with no_grad():
            for batch in loader:
                y = self.model(batch.to(self.device))
                for i, rxn in enumerate(valid_rxns):
                    y_ts = (
                        y[0][i].item() * self.y_std + self.y_mean, 
                        y[1][i].item() * self.y_std
                    )
                    rxn.e_ts = y_ts[0]