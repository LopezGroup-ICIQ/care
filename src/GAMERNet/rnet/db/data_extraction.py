import pickle
import networkx as nx

def read_db(pack: object, db_pickle: str, term: str, properties: list) -> dict:
    """
    Read the database and returns the desired properties of the intermediates and/or transition states.
    """

    with open(db_pickle, 'rb') as f:
        df = pickle.load(f)
    
    db_dict = {
        system: {
            saturation_family: {
                molpack.code: {} for molpack in pack[system][saturation_family]
            } for saturation_family in pack[system]
        } for system in pack
    }

    for system, data in pack.items():
        for saturation_family, molpacks in data.items():
            for molpack in molpacks:
                code = molpack.code
                molpack_graph = molpack.graph.to_undirected()
                for _, row in df[df['term'] == term].iterrows():
                    try:
                        df_graph = pickle.loads(row['conn_graph'])
                    except TypeError:
                        continue
                    if nx.is_isomorphic(molpack_graph, df_graph, node_match=lambda x, y: x["elem"] == y["elem"]):
                        for i in properties:
                            db_dict[system][saturation_family][code][i] = row[i]
                        break
    return db_dict