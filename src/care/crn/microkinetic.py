"""Helper functions for running microkinetic simulations."""

import networkx as nx
import numpy as np
from rdkit import Chem
from numba import njit, cuda
import math
from networkx.algorithms import shortest_path

from care import ElementaryReaction, Intermediate


def iupac_to_inchikey(iupac_name: str) -> str:
    mol = Chem.MolFromIUPACName(iupac_name)
    if mol is not None:
        return Chem.inchi.MolToInchiKey(mol)
    else:
        return "Invalid IUPAC name"
    

def gen_graph(inters: dict[str, Intermediate], 
              rxns: list[ElementaryReaction]) -> nx.DiGraph:
        """
        Generate a network graph representation.

        Returns:
            graph (obj:`nx.DiGraph`): Network graph. Nodes represent species and elementary reactions.
            Species nodes can only be linked to reaction nodes and vice versa.

        Notes:
            The returned graph is correctly directed at the elementary reaction level, 
            i.e. if A + B -> C + D is a reaction, the species on the same side (A and B)
            are linked to the reaction node in the same way (both entering or exiting).
            However, at the global level, the graph is ill-defined and further processing 
            (kinetic modeling) is needed to obtain a correct representation of the network.
        """
        graph = nx.DiGraph()

        for inter in inters.values():
            graph.add_node(
                inter.code,
                category="intermediate",
                phase=inter.phase,
                formula=inter.formula,
                nC=inter["C"],
                nH=inter["H"],
                nO=inter["O"],
                closed_shell=inter.closed_shell,
            )

        graph.add_node(
            "*",
            category="intermediate",
            phase="surf",
            formula="*",
            nC=0,
            nH=0,
            nO=0,
            closed_shell=False,
        )

        for idx, reaction in enumerate(rxns):
            r_type = (
                reaction.r_type
                if reaction.r_type in ("adsorption", "desorption", "eley_rideal", "PCET")
                else "surface_reaction"
            )
            graph.add_node(
                reaction.code,
                category="reaction",
                r_type=r_type,
                r=reaction.r_type,
                idx=idx
            )

            for inter in reaction.components[0]:
                if inter.phase in ("solv", "electro"):
                    if not graph.has_node(inter.code):
                        graph.add_node(inter.code, category="electro", phase=inter.phase, formula=inter.formula, nC=inter["C"], nH=inter["H"], nO=inter["O"])
                    graph.add_edge(inter.code, reaction.code, dir='in')
                else:
                    graph.add_edge(inter.code, reaction.code, dir='in')
            for inter in reaction.components[1]:
                if inter.phase in ("solv", "electro"):
                    if not graph.has_node(inter.code):
                        graph.add_node(inter.code, category="electro", phase=inter.phase, formula=inter.formula, nC=inter["C"], nH=inter["H"], nO=inter["O"])
                    graph.add_edge(reaction.code, inter.code, dir='out')
                else:
                    graph.add_edge(reaction.code, inter.code, dir='out')

        return graph       


def max_flux(graph: nx.DiGraph, source: str) -> list:
    """Collect edges defining the path with highest flux from source.

    Args:
        graph (nx.DiGraph): Reaction network graph. Each edge must have a 'rate' attribute
            defining the consumption rate of the connected intermediate.
        source (str): Source node formula given by ASE (e.g. CO2, CO) or InchiKey (e.g. QSHDAFOYLUJSOK-UHFFFAOYSA-N)

    Returns:
        list: List of edges which define the maximum flux path.
    """

    if len(source) == 27:
        # If InchiKey is given, convert to formula
        source = graph.nodes[source + "g"]["formula"]

    # Find the source node based on attributes
    for node in graph.nodes:
        if (
            graph.nodes[node]["category"] == "intermediate"
            and graph.nodes[node]["phase"] == "gas"
            and graph.nodes[node]["formula"] == source
        ):
            source_node = node
            break
    else:
        # If no source node is found
        return []

    path_edges = []
    current_node = source_node
    visited_nodes = {source_node}  # Set of visited nodes to avoid backtracking

    while True:
        # INTERMEDIATE -> REACTION
        # int_out_edges = [
        #     (u, v, data)
        #     for u, v, data in graph.edges(current_node, data=True)
        #     if u == current_node and v not in visited_nodes
        # ]
        int_out_edges = [edge for edge in graph.out_edges(current_node, data=True) if edge[1] not in visited_nodes]
        if not int_out_edges:
            path_edges = []
            print("No sink found", int_out_edges)
            break

        max_edge = max(int_out_edges, key=lambda edge: edge[2]["rate"])
        path_edges.append(max_edge)
        rxn_node = max_edge[1]
        visited_nodes.add(rxn_node)

        # REACTION -> INTERMEDIATE
        # rxn_out_edges = [
        #     (u, v, data)
        #     for u, v, data in graph.edges(rxn_node, data=True)
        #     if u == rxn_node and graph.nodes[v]["nC"] != 0 and v not in visited_nodes
        # ]

        rxn_out_edges = [edge for edge in graph.out_edges(rxn_node, data=True) if graph.nodes[edge[1]]["nC"] != 0 and edge[1] not in visited_nodes]

        try:
            max_edge = max(rxn_out_edges, key=lambda edge: edge[2]["rate"])
        except:
            print(rxn_out_edges)
            raise ValueError(f"No sink found for {rxn_node}")
        path_edges.append(max_edge)
        int_node = max_edge[1]
        visited_nodes.add(int_node)

        if (
            graph.nodes[int_node]["category"] == "intermediate"
            and graph.nodes[int_node]["phase"] == "gas"
        ):
            print("Found fastest path ({} steps) leading from {}(g) to {}(g)".format(int(len(path_edges)/2), graph.nodes[source_node]["formula"], graph.nodes[int_node]["formula"]))
            break

        current_node = int_node

    return path_edges

@njit
def net_rate(y, kd, kr, sf, sb):
    rates = np.empty_like(kd)
    for i in range(kd.shape[0]):  # Assuming kd and kr have the same shape
        forward_product = 1.0
        backward_product = 1.0
        for j in range(sf.shape[1]):  # Assuming sf and sb have the same shape [reactions, species]
            forward_product *= y[j] ** sf[i, j]
            backward_product *= y[j] ** sb[i, j]
        rates[i] = kd[i] * forward_product - kr[i] * backward_product
    return rates

def get_all_paths(g: nx.Graph, source:str):
    if len(source) == 27:
        source = g.nodes[source + "g"]["formula"]

    # Find the source node based on attributes
    for node in g.nodes:
        if (
            g.nodes[node]["category"] == "intermediate"
            and g.nodes[node]["phase"] == "gas"
            and g.nodes[node]["formula"] == source
        ):
            source_node = node
            break
    else:        
        return []  # If no source node is found
    
    gas_products = [n for n in g.nodes if g.nodes[n]['category'] == 'intermediate' and g.nodes[n]['phase'] == 'gas' and g.nodes[n]['molar_fraction'] == 0]
    products_dict = {}
    for product in gas_products:
        formula = g.nodes[product]['formula']
        print(f"Searching for paths leading from {source} to {formula}")
        try:
            path = shortest_path(g, source_node, product, weight='weight')
        except nx.NetworkXNoPath:
            print(f"No path found leading from {source} to {formula}")
            path = None
        if path:
            products_dict[formula] = path
            print(f"Found path leading from {source} to {formula}")
    return products_dict
