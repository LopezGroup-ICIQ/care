import pickle
from GAMERNet.rnet.networks.reaction_network import ReactionNetwork
import networkx as nx
import pubchempy as pcp
import pprint as pp

with open('../scripts/C1_Au100/rxn_net_bp.pkl', 'rb') as pickle_file:
    content = pickle.load(pickle_file)  # dict of elementary reactions

rxn_net = ReactionNetwork().from_dict(content)

def pcp_2_nx(compound: pcp.Compound) -> nx.Graph():
    """Converts a PubChemPy compound to a NetworkX graph.

    Parameters
    ----------
    compound : pubchempy.Compound
        PubChemPy compound object.

    Returns
    -------
    networkx.Graph
        NetworkX graph of the compound.
    """

    G=nx.DiGraph()
    for atom in compound.to_dict(properties=['atoms'])['atoms']:
        G.add_node(atom['aid'], elem=atom['element'])

    # Add bonds as edges
    for bond in compound.to_dict(properties=['bonds'])['bonds']:
        G.add_edge(bond['aid1'], bond['aid2'], order=bond['order'])

    return G

def get_codes(format: str, sources: list[str], targets: list[str], inters: list[str]) -> tuple[list[str], list[str], list[str]]:
    """Adapts the source, target and intermediate components label to the code format
    of the ReactionNetwork. Input format is a list of strings and can be either the 
    IUPAC name of the molecule, the SMILES, SDF, the InChI, the InChIKey or directly the code of
    the ReactionNetwork.
    Format input: 
        - IUPAC name: 'name'
        - SMILES: 'smiles'
        - SDF: 'sdf'
        - InChI: 'inchi'
        - InChIKey: 'inchikey'
        - Code: 'code'

    Parameters
    ----------
    format : str
        Format of the input sources and targets.
    sources : list[str]
        List of sources, starting point gas-phase molecules.
    targets : list[str]
        List of targets, final gas-phase molecules.
    inters : list[str]
        List of intermediates, intermediate molecules found within the chemical reaction.

    Returns
    -------
    tuple[list[str], list[str], list[str]]
        Tuple containing the adapted source, target and intermediate components to the code format.  
    """

    if format != 'code':
        source_graphs = [pcp_2_nx(pcp.get_compounds(source, format)[0]) for source in sources]
        target_graphs = [pcp_2_nx(pcp.get_compounds(target, format)[0]) for target in targets]
        # TODO: Double check this section
        inter_graphs = [pcp_2_nx(compound) for inter in inters for compound in pcp.get_compounds(inter, format, listkey_count=20)]
  
        source_codes = [rxn_net.search_graph_closed_shell(source_graph).code for source_graph in source_graphs]               
        target_codes = [rxn_net.search_graph_closed_shell(target_graph).code for target_graph in target_graphs]
        inter_codes = [rxn_net.search_graph(inter_graph).code for inter_graph in  inter_graphs]

    else:
        source_codes = [source for source in sources]
        target_codes = [target for target in targets]
        inter_codes = [inter for inter in inters]

    return source_codes, target_codes, inter_codes

def retrieve_data_from_net(rxn_net: ReactionNetwork, path: list[str]) -> tuple[dict, list]:
    """Retrieves the data from the ReactionNetwork for the given path to be used in the
    ReactionNetwork generation.

    Parameters
    ----------
    rxn_net : ReactionNetwork
        ReactionNetwork object.
    path : list[str]
        List of codes for the intermediates participating in the path.

    Returns
    -------
    tuple[dict, list]
        Tuple containing the Intermediates and ElementaryReactions participating in the path.
    """

    rxn_graph = rxn_net.gen_graph(show_steps=True)
    for edge in list(rxn_graph.edges):
        rxn_graph.add_edge(edge[1], edge[0])

    full_path = []
    for inter_1, inter_2 in zip(path[:-1], path[1:]):
        # Matching the nodes in the path to the nodes in the graph
        node_1 = rxn_graph.nodes[inter_1]
        node_2 = rxn_graph.nodes[inter_2]
        # Finding Elementary reactions that connect the nodes 
        elem_react = nx.shortest_path(rxn_graph, inter_1, inter_2)
        # Appending the first node only for the first iteration
        if len(full_path) == 0:
            full_path.append(node_1)
        full_path.append(elem_react)
        full_path.append(node_2)
    
    # Generating a list with only the elementary reactions for the given path
    elem_react_list = []
    for node in full_path:
        if isinstance(node, dict):
            continue
        else:
            elem_react_list.append(node)

    elem_react_for_rxn_net = []
    for elem_react in elem_react_list:
        elem_react_rxn_net = rxn_net.search_reaction(code=elem_react[1])[0]
        elem_react_for_rxn_net.append(elem_react_rxn_net)

    # Obtaining the Intermediates participating in the path
    all_inter_in_react = []
    for elem_react in elem_react_for_rxn_net:
        all_inter_in_react.extend([x for x in elem_react.reactants])
        all_inter_in_react.extend([x for x in elem_react.products])

    # Adapting the format for the ReactionNetwork generation
    inter_for_rxn_net = {}
    for inter in all_inter_in_react:
        inter_for_rxn_net[inter.code] = inter
    
    return inter_for_rxn_net, elem_react_for_rxn_net

    
source_codes, target_codes, inter_codes = get_codes('name', ['carbon monoxide'], ['methanol'], ['methoxide', 'formaldehyde', 'formate'])

all_paths = rxn_net.find_all_paths_from_sources_to_targets(source_codes, target_codes, cutoff=10)
all_paths = rxn_net.find_all_paths_from_sources_to_targets(source_codes, target_codes, inter_codes, cutoff=10)

all_rxn_nets = []
counter = 0
for path in all_paths.values():
    for path in path:
        inter_for_rxn_net, elem_react_for_rxn_net = retrieve_data_from_net(rxn_net, path)
        # Generating a new Reaction Network from full_path
        path_rxn_net = ReactionNetwork(inter_for_rxn_net, elem_react_for_rxn_net,rxn_net.surface)
        all_rxn_nets.append(path_rxn_net)
        # Saving the image of the Reaction Network
        path_rxn_net.write_dotgraph('.', f'rxn_net_path_{counter}.png', del_surf=False, show_steps=False)
        counter += 1