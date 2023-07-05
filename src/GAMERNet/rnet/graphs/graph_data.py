import networkx as nx

def molecular_formula(graph: nx.Graph) -> str:
    """Generates the molecular formula of a molecule from its graph representation.

    Parameters
    ----------
    graph : nx.Graph
        NetworkX Graph representing the molecule.

    Returns
    -------
    str
        Molecular formula.
    """

    num_C, num_H, num_O = 0, 0, 0
    for node in graph.nodes(data=True):
        elem_node = node[1]['elem']
        if elem_node == 'C':
            num_C += 1
        elif elem_node == 'H':
            num_H += 1
        elif elem_node == 'O':
            num_O += 1
    # Generating the formula
    formula = ""
    if num_C > 0:
        formula += "C" + (str(num_C) if num_C > 1 else "")
    if num_H > 0:
        formula += "H" + (str(num_H) if num_H > 1 else "")
    if num_O > 0:
        formula += "O" + (str(num_O) if num_O > 1 else "")
    return str(formula)