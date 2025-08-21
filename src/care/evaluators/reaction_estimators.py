from copy import deepcopy
from typing import Union

from ase.data import chemical_symbols
from ase.mep import NEB
from ase.optimize import BFGS
from ase.visualize.plot import plot_atoms
from ase.visualize import view
import networkx as nx
import numpy as np

from care.crn.templates.dissociation import BondBreaking, BondFormation
from care import Intermediate, ElementaryReaction
from care.evaluators import ReactionEnergyEstimator, IntermediateEnergyEstimator
from care.crn.visualize import plot_reaction_profile
from care.evaluators.utils import atoms_to_data, pyg_to_nx, extract_adsorbate, fragment_filter
from care.constants import CORDERO


class BarrierlessReactionEnergyEstimator(ReactionEnergyEstimator):
    """
    Barrierless reaction energy estimator.
    No transition state evaluation is performed here.
    """
    def __init__(
        self,
        intermediates: dict[str, Intermediate], 
        T: float = 298.0,
        ref_electrode: str = "SHE",
        pH: float = 7.0,
        U: float = 0.0,
        **kwargs
    ):
        super().__init__(
            intermediates=intermediates,
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
    Class for estimating the transition state energy of reactions using the NEB method with 
    ML potentials within ASE calculators.
    """

    def __init__(
        self,
        intermediates: dict[str, Intermediate],
        mlp: IntermediateEnergyEstimator,
        num_images: int = 5,
        climb: bool = True,
        interpolation_method: str = "linear",
        neb_method: str = "aseneb",
        remove_rotation_and_translation: bool = False,
        allow_shared_calculator: bool = True,
        k: Union[float, list[float]] = 0.1,
        parallel: bool = False,
        dx: float = 1.5, 
        tol: float = 0.0, 
        max_steps: int = 100,
        T: float = 298.0,
        ref_electrode: str = "SHE",
        pH: float = 7.0,
        U: float = 0.0,
        **kwargs
    ):
        super().__init__(
            intermediates=intermediates,
            T=T,
            ref_electrode=ref_electrode,
            pH=pH,
            U=U,
            **kwargs
        )
        if not mlp.is_mlp:
            raise ValueError("MLP must be a machine learning potential.")
        self.mlp = mlp
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

    def run_neb(self, reaction: ElementaryReaction) -> None:
        reaction.bb()  # ensure starting always from reaction in bond-breaking direction (A* -> B* + C*)
        bond = tuple(reaction.r_type.split("-"))

        # 1) Get most stable configuration of initial state (A*)
        IS_code = [
            inter.code for inter in list(reaction.reactants) if not inter.is_surface
        ][0]
        idx = min(
            self.intermediates[IS_code].ads_configs,
            key=lambda x: self.intermediates[IS_code].ads_configs[x]['mu'],
        )
        try:            
            IS = self.intermediates[IS_code].ads_configs[idx]["ase"]
            # energy_IS = self.intermediates[IS_code].ads_configs[idx]["mu"] + self.mlp.surface.energy
            is_graph = atoms_to_data(IS, IS.get_array("atom_tags"), surface_order=-1, filter=True)
            reaction.is_graph = is_graph
        except: # weird relaxation
            print(f"Weird initial state for {reaction.repr_hr}. Check intermediate {IS_code}, adsorption configuration {idx}. Skip TS evaluation (assumed barrierless)")
            reaction.e_ts = reaction.e_is if reaction.e_is[0] > reaction.e_fs[0] else reaction.e_fs
            return
        competitors = [
            inter
            for inter in list(reaction.products)
            if not inter.is_surface
        ]

        # 2) Build the NetworkX graph of the final state (B* + C*)
        if len(competitors) == 1:
            if abs(reaction.stoic[competitors[0].code]) == 2:  # A* -> 2B*
                nx_bc = [competitors[0].graph, competitors[0].graph]
                mapping = {n: n + nx_bc[0].number_of_nodes() for n in nx_bc[1].nodes()}
                nx_bc[1] = nx.relabel_nodes(nx_bc[1], mapping)
                nx_bc = nx.compose(nx_bc[0], nx_bc[1])
            elif abs(reaction.stoic[competitors[0].code]) == 1:  # A* -> B* (ring opening)
                nx_bc = competitors[0].graph
            else:
                raise ValueError("Reaction stoichiometry not supported.")
        else:  # A* -> B* + C* (B and C different)
            nx_bc = [competitors[0].graph, competitors[1].graph]
            mapping = {n: n + nx_bc[0].number_of_nodes() for n in nx_bc[1].nodes()}
            nx_bc[1] = nx.relabel_nodes(nx_bc[1], mapping)
            nx_bc = nx.compose(nx_bc[0], nx_bc[1])

        potential_edges = []
        for i in range(is_graph.edge_index.shape[1]):
            edge_idxs = is_graph.edge_index[:, i]
            atom1, atom2 = is_graph.elem[edge_idxs[0]], is_graph.elem[edge_idxs[1]]
            if (atom1, atom2) == bond or (atom2, atom1) == bond:
                potential_edges.append(i)

        print(f"Potential edges: {len(potential_edges)}")
        # 3) Find broken bond in the graph of the IS via isomorphic comparison
        counter = 0
        while True:
            g = deepcopy(is_graph)
            u, v = g.edge_index[:, potential_edges[counter]]
            mask = ~(
                (g.edge_index[0] == u) & (g.edge_index[1] == v)
                | (g.edge_index[0] == v) & (g.edge_index[1] == u)
            )
            g.edge_index = g.edge_index[:, mask]
            adsorbate = extract_adsorbate(g, IS.get_array("atom_tags"))
            nx_graph = pyg_to_nx(adsorbate)
            if nx.is_isomorphic(
                nx_bc, nx_graph, node_match=lambda x, y: x["elem"] == y["elem"]
            ):
                u, v = u.item(), v.item()
                break
            else:
                counter += 1

        # 4) Assign each adsorbate atom to one of the two fragments (B* or C*)
        adsorbate_node_indices = [
            i for i in range(is_graph.num_nodes) if IS.get_array("atom_tags")[i] == 1
        ]
        node_indices_B, node_indices_C = [u], [v]
        for adsorbate_node_index in adsorbate_node_indices:
            if adsorbate_node_index in node_indices_B or adsorbate_node_index in node_indices_C:
                continue
            else:
                # check if the node is connected to node_indices_B or node_indices_C
                for i in range(is_graph.edge_index.shape[1]):
                    edge_idxs = is_graph.edge_index[:, i]
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
        # adsorbate_node_indices = list(set(adsorbate_node_indices))
        node_indices_B, node_indices_C = list(set(node_indices_B)), list(set(node_indices_C))

        # 5) Get center of mass of fragments B and C and their distance from surface to choose which fragment to move
        slab_atoms = [idx for idx in is_graph.idx if idx not in node_indices_B and idx not in node_indices_C]
        z_max = max(IS.positions[i][2] for i in slab_atoms)
        cm_B = IS.get_center_of_mass(indices=node_indices_B)
        cm_C = IS.get_center_of_mass(indices=node_indices_C)
        dist_Bz = abs(z_max - cm_B[2])
        dist_Cz = abs(z_max - cm_C[2])
        FS = deepcopy(IS)
        atoms_to_move = node_indices_C if dist_Bz <= dist_Cz else node_indices_B

        # 6) construct displacement vector (direction: from fragment not moved to fragment moved)
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
                raise ValueError("Maximum increment reached")
            FS.positions[atoms_to_move] += (self.dx + increment) * direction
            FS.wrap()
            # 7) Relax final state structure (B* + C*)
            FS.calc = deepcopy(self.mlp.calc)
            opt = BFGS(FS, 
                    logfile=None)
            opt.run(fmax=0.05, steps=self.mlp.max_steps)
            fs_graph = atoms_to_data(FS, FS.get_array("atom_tags"), surface_order=-1, filter=False)
            reaction.fs_graph = fs_graph
            if not fragment_filter(fs_graph, FS.get_array("atom_tags")):
                break
            else:
                increment += 0.5

        # energy_FS = FS.get_potential_energy()

        # 8) Set up NEB simulation
        images = [IS] + [IS.copy() for _ in range(self.num_images)] + [FS]
        neb = NEB(images, 
                k= self.k, 
                climb=self.climb, 
                parallel=self.parallel, 
                remove_rotation_and_translation=self.remove_rotation_and_translation, 
                method=self.neb_method, 
                allow_shared_calculator=self.allow_shared_calculator)
        neb.interpolate(method=self.interpolation_method, 
                        mic=True, 
                        apply_constraint=None)
        for image in images[1:self.num_images + 1]:
            image.calc = deepcopy(self.mlp.calc)
        optimizer = BFGS(neb, logfile=None)
        optimizer.run(fmax=0.05, steps=self.max_steps)

        # 9) Collect final NEB frames and energies
        final_NEB_frames = []
        final_NEB_energies = []
        for idx, image in enumerate(neb.images):
            final_NEB_frames.append(image)
            image.calc = deepcopy(self.mlp.calc)
            energy_image = image.get_potential_energy()
            final_NEB_energies.append(energy_image)
        energy_TS = max(final_NEB_energies)
        reaction.neb_images = final_NEB_frames
        reaction.neb_energies = final_NEB_energies
        title = reaction.repr_hr + f" on {self.mlp.surface} ({reaction.r_type})"
        reaction.neb_figure = plot_reaction_profile(final_NEB_energies, title=title)
        referenced_ts_energy = energy_TS - self.mlp.surface.energy if type(self.mlp).__name__ != "OCPIntermediateEvaluator" else energy_TS + sum(IS[el] * self.mlp.eref[el] for el in ['C', 'H', 'O', 'N'])
        if referenced_ts_energy > reaction.e_is[0] and referenced_ts_energy > reaction.e_fs[0]:
            reaction.e_ts = referenced_ts_energy, 0.0
        else:
            reaction.e_ts = reaction.e_is if reaction.e_is[0] > reaction.e_fs[0] else reaction.e_fs

    def eval(self, reaction: ElementaryReaction) -> None:
        """
        Estimate reaction properties. 
        This base implementation evaluates the elementary reactions as barrierless.

        Args:
            reaction (ElementaryReaction): The reaction.
        """
        self.calc_reaction_energy(reaction)
        if isinstance(reaction, (BondBreaking, BondFormation)):
            self.run_neb(reaction)
        else:
            reaction.e_ts = reaction.e_is if reaction.e_is[0] > reaction.e_fs[0] else reaction.e_fs
        reaction.e_act = reaction.e_ts[0] - reaction.e_is[0], 0.0
