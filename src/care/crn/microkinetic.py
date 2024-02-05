"""Helper functions for running microkinetic simulations."""

import networkx as nx
import numpy as np
from rdkit import Chem
from numba import njit, cuda
import math


def iupac_to_inchikey(iupac_name: str) -> str:
    mol = Chem.MolFromIUPACName(iupac_name)
    if mol is not None:
        return Chem.inchi.MolToInchiKey(mol)
    else:
        return "Invalid IUPAC name"


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
        int_outgoing_edges = [
            (u, v, data)
            for u, v, data in graph.edges(current_node, data=True)
            if u == current_node and v not in visited_nodes
        ]
        if not int_outgoing_edges:
            print("No sink found", int_outgoing_edges)
            break

        max_edge = max(int_outgoing_edges, key=lambda edge: edge[2]["rate"])
        path_edges.append(max_edge)
        rxn_node = max_edge[1]
        visited_nodes.add(rxn_node)

        # REACTION -> INTERMEDIATE
        rxn_outgoing_edges = [
            (u, v, data)
            for u, v, data in graph.edges(rxn_node, data=True)
            if u == rxn_node and graph.nodes[v]["nC"] != 0 and v not in visited_nodes
        ]

        max_edge = max(rxn_outgoing_edges, key=lambda edge: edge[2]["rate"])
        path_edges.append(max_edge)
        int_node = max_edge[1]
        visited_nodes.add(int_node)

        if (
            graph.nodes[int_node]["category"] == "intermediate"
            and graph.nodes[int_node]["phase"] == "gas"
        ):
            print("Found sink", int_node)
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
