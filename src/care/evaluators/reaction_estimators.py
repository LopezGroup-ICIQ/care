from copy import deepcopy
from typing import Union

from ase.data import chemical_symbols
from ase.mep import NEB
from ase.optimize import BFGS, LBFGS
import networkx as nx
import numpy as np
from torch.cuda import empty_cache
from torch_geometric.data import Data

from care.crn.templates.dissociation import BondBreaking, BondFormation
from care import ElementaryReaction
from care.evaluators import ReactionEnergyEstimator, IntermediateEnergyEstimator
from care.evaluators.utils import atoms_to_data, pyg_to_nx, extract_adsorbate, is_adsorbate_fragmented, connectivity_signature
from care.constants import CORDERO


class BarrierlessReactionEnergyEstimator(ReactionEnergyEstimator):
    """
    Barrierless reaction energy estimator.
    No transition state evaluation is performed here.
    """
    def __init__(
        self, 
        T: float = 298.0,
        ref_electrode: str = "SHE",
        pH: float = 7.0,
        U: float = 0.0,
        **kwargs
    ):
        super().__init__(
            T=T,
            ref_electrode=ref_electrode,
            pH=pH,
            U=U,
            **kwargs
        )    

    @property
    def adsorbate_domain(self):
        """Returns the list of adsorbate elements that your model can handle."""
        return chemical_symbols

    @property
    def surface_domain(self):
        """Returns the list of surface elements that your model can handle."""
        return chemical_symbols
    
    def __repr__(self) -> str:
        return "Barrierless reaction energy estimator"


class NEBReactionEnergyEstimator(ReactionEnergyEstimator):
    """
    Class for estimating the transition state energy of surface reactions using the NEB method with 
    ML potentials within ASE calculators.
    """

    def __init__(
        self,
        mlp: IntermediateEnergyEstimator = None,
        num_images: int = 3,
        climb: bool = True,
        interpolation_method: str = "linear",
        neb_method: str = "aseneb",
        remove_rotation_and_translation: bool = True,
        allow_shared_calculator: bool = False,
        k: Union[float, list[float]] = 0.1,
        parallel: bool = True,
        dx: float = 1.5, 
        tol: float = 0.0, 
        max_steps: int = 100,
        optimizer: str = "BFGS",
        T: float = 298.0,
        ref_electrode: str = "SHE",
        pH: float = 7.0,
        U: float = 0.0,
        **kwargs
    ):
        super().__init__(
            T=T,
            ref_electrode=ref_electrode,
            pH=pH,
            U=U,
            **kwargs
        )
        if not mlp.is_mlp:
            raise ValueError("MLP must be a machine learning potential.")
        self.mlp = mlp
        self.device = mlp.device
        self.num_images = num_images
        self.dx = dx
        self.tol = tol
        self.max_steps = max_steps
        self.climb = climb
        self.interpolation_method = interpolation_method
        self.neb_method = neb_method
        self.remove_rotation_and_translation = remove_rotation_and_translation
        self.allow_shared_calculator = allow_shared_calculator
        self.k = k
        self.parallel = parallel
        self.supports_batching = False
        self.optimizer = optimizer

    @property
    def adsorbate_domain(self):
        """Returns the list of adsorbate elements that your model can handle."""
        return self.mlp.adsorbate_domain

    @property
    def surface_domain(self):
        """Returns the list of surface elements that your model can handle."""
        return self.mlp.surface_domain

    def __repr__(self) -> str:
        return f"NEB-based reaction energy estimator (MLP: {self.mlp})"
    
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

    def get_fs(self, reaction: ElementaryReaction) -> None:
        """
        Get the final state (FS) geometry for the given reaction.
        """
        reaction.bb()  # ensure starting always from reaction in bond-breaking direction (A* -> B* + C*)
        bond = tuple(reaction.r_type.split("-"))

        # 1) Get most stable configuration of initial state (A*)
        IS_intermediate = [
            inter for inter in list(reaction.reactants) if not inter.is_surface
        ][0]
        try:            
            idx = min(
                IS_intermediate.ads_configs,
                key=lambda x: IS_intermediate.ads_configs[x]['mu'],
            )
            IS = IS_intermediate.ads_configs[idx]["ase"]
            is_graph = atoms_to_data(IS, IS.get_array("atom_tags"), surface_order=-1, filter=True)
            reaction.is_graph = is_graph
            reaction.is_atoms = IS.copy()
            n_nodes = is_graph.num_nodes
            n_edges = is_graph.num_edges
        except:
            return

        potential_edges = self._find_potential_edges(is_graph, bond)
        nx_bc = self._build_product_nx(reaction)        

        # 2) Find broken bond in the graph of the IS via isomorphic comparison
        if len(potential_edges) == 0 and len(nx_bc) == 2: # edge case: H2, O2 not showing bond in the graph
            uvs = [is_graph.idx[i] for i in range(is_graph.num_nodes) if IS.get_array("atom_tags")[i] == 1]
            u, v = uvs[0], uvs[1]
        elif len(potential_edges) == 0 and len(nx_bc) != 2:
            return
        else:
            nx_bc_signature = connectivity_signature(nx_bc)
            for _, e_idx in enumerate(potential_edges):
                u = is_graph.edge_index[0, e_idx].item()
                v = is_graph.edge_index[1, e_idx].item()
                mask = ~(
                    ((is_graph.edge_index[0] == u) & (is_graph.edge_index[1] == v)) |
                    ((is_graph.edge_index[0] == v) & (is_graph.edge_index[1] == u))
                )
                edge_index_new = is_graph.edge_index[:, mask]
                data = is_graph.clone()
                data.edge_index = edge_index_new
                adsorbate = extract_adsorbate(data, IS.get_array("atom_tags"))
                nx_adsorbate = pyg_to_nx(adsorbate)
                if connectivity_signature(nx_adsorbate) == nx_bc_signature:
                    break
                if _ == len(potential_edges) - 1:
                    return

        # 3) Assign each adsorbate atom to one of the two fragments (B* or C*)
        adsorbate_node_indices = [
            i for i in range(n_nodes) if IS.get_array("atom_tags")[i] == 1
        ]
        node_indices_B, node_indices_C = {u}, {v}
        neighbors = {i: set() for i in range(n_nodes)}
        for i in range(n_edges):
            a, b = is_graph.edge_index[:, i].tolist()
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

        # 4) Get center of mass of fragments B and C and their distance from surface to choose which fragment to move
        slab_atoms = [idx for idx in is_graph.idx if idx not in node_indices_B and idx not in node_indices_C]
        z_max = max(IS.positions[i][2] for i in slab_atoms)
        cm_B = IS.get_center_of_mass(indices=node_indices_B)
        cm_C = IS.get_center_of_mass(indices=node_indices_C)
        dist_Bz = abs(z_max - cm_B[2])
        dist_Cz = abs(z_max - cm_C[2])
        atoms_to_move = node_indices_C if dist_Bz <= dist_Cz else node_indices_B

        # 5) construct displacement vector (direction: from fragment not moved to fragment moved)
        FS = deepcopy(IS)
        min_z = min(IS.positions[atoms_to_move, 2])
        z_vector = [0, 0, 2.0 + z_max - min_z]
        FS.positions[atoms_to_move] += z_vector
        vector = IS.positions[u] - IS.positions[v] if atoms_to_move == node_indices_B else IS.positions[v] - IS.positions[u]
        vector[2] = 0.0
        # prevent atomic clash when fragments are vertically aligned
        if abs(IS.positions[v][0] - IS.positions[u][0]) < 1.0 and abs(IS.positions[v][1] - IS.positions[u][1]) < 1.0:
            delta = CORDERO[IS[u].symbol] + CORDERO[IS[v].symbol] + self.tol
            vector[0] += delta
            vector[1] += delta
        direction = vector / np.linalg.norm(vector)
        increment = 0
        while True:  # if B* + C* relaxation results in connected adsorbate graph, increase increment
            if increment >= 3.0:
                print(f"{reaction.repr_hr}: Maximum increment reached ({increment}); displacement vector: {direction}")
                return
            FS.positions[atoms_to_move] += (self.dx + increment) * direction
            FS.wrap()
            # 7) Relax final state structure (B* + C*)
            FS.calc = deepcopy(self.mlp.calc)
            opt = BFGS(FS, 
                    logfile=None)
            opt.run(fmax=0.05, steps=self.mlp.max_steps)
            empty_cache()
            fs_graph = atoms_to_data(FS, FS.get_array("atom_tags"), surface_order=-1, filter=False)
            reaction.fs_graph = fs_graph
            reaction.fs_atoms = FS.copy()
            if is_adsorbate_fragmented(fs_graph, FS.get_array("atom_tags")):
                break
            else:
                increment += 0.5
        FS.calc = None

    def run_neb(self, reaction: ElementaryReaction):
        if reaction.fs_atoms is None or reaction.is_atoms is None:
            reaction.e_ts = reaction.e_is if reaction.e_is[0] > reaction.e_fs[0] else reaction.e_fs
            return
        images = [reaction.is_atoms] + [reaction.is_atoms.copy() for _ in range(self.num_images)] + [reaction.fs_atoms]
        neb = NEB(images, 
                k= self.k, 
                climb=self.climb, 
                parallel=self.parallel, 
                remove_rotation_and_translation=self.remove_rotation_and_translation, 
                method=self.neb_method, 
                allow_shared_calculator=self.allow_shared_calculator)
        neb.interpolate(method=self.interpolation_method, 
                        mic=True, 
                        apply_constraint=True)
        for image in images[1:self.num_images + 1]:
            image.calc = deepcopy(self.mlp.calc)
        if self.optimizer == "LBFGS":
            optimizer = LBFGS(neb, logfile=None)
        elif self.optimizer == "BFGS":
            optimizer = BFGS(neb, logfile=None)
        optimizer.run(fmax=0.05, steps=self.max_steps)

        final_NEB_frames = []
        final_NEB_energies = []
        for _, image in enumerate(neb.images):
            image.calc = deepcopy(self.mlp.calc)
            energy_image = image.get_potential_energy()
            final_NEB_energies.append(energy_image)
            image.calc = None
            final_NEB_frames.append(image)
        energy_TS = max(final_NEB_energies)
        reaction.neb_images = final_NEB_frames
        reaction.neb_energies = final_NEB_energies
        if type(self.mlp).__name__ != "OCPIntermediateEvaluator":
            referenced_ts_energy = energy_TS - self.mlp.surface.energy
        else:
            referenced_ts_energy = energy_TS + sum(
                reaction.is_atoms.get_chemical_symbols().count(el) * self.mlp.eref[el]
                for el in ['C', 'H', 'O', 'N']
            )
        if referenced_ts_energy > reaction.e_is[0] and referenced_ts_energy > reaction.e_fs[0]:
            reaction.e_ts = referenced_ts_energy, 0.0
        else:
            reaction.e_ts = reaction.e_is if reaction.e_is[0] > reaction.e_fs[0] else reaction.e_fs
        empty_cache()

    def eval(self, reaction: ElementaryReaction) -> None:
        """
        Estimate reaction properties. 
        This base implementation evaluates the elementary reactions as barrierless.

        Args:
            reaction (ElementaryReaction): The reaction.
        """
        for species in list(reaction.reactants) + list(reaction.products):
            if species.phase in ("gas", "ads") and species.ads_configs == {}:
                print(f"Species {species.formula} in {reaction.repr_hr} ElementaryReaction is not evaluated.")
                return
        self.calc_reaction_energy(reaction)
        if isinstance(reaction, (BondBreaking, BondFormation)):
            self.get_fs(reaction)
            self.run_neb(reaction)
        else:
            reaction.e_ts = reaction.e_is if reaction.e_is[0] > reaction.e_fs[0] else reaction.e_fs
        reaction.e_act = reaction.e_ts[0] - reaction.e_is[0], 0.0
