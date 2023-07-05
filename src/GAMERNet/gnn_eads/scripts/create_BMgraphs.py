import os
import sys
sys.path.insert(0, "../src")
from subprocess import Popen, PIPE
import argparse

from torch_geometric.loader import DataLoader
from torch import save

from gnn_eads.functions import get_graph_sample, structure_to_graph, get_graph_formula, get_id
from gnn_eads.constants import ENCODER

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description='Convert DFT samples of the FG dataset to graph formats')
    PARSER.add_argument('--data', type=str, dest="data", help='Path to data folder')
    PARSER.add_argument('--tol', type=float, default=0.5, dest="tol", help='Voronoi tolerance')
    PARSER.add_argument('--metal-hops', type=int, default=1, dest="metal_hops", help='Number of metal hops to consider in the graph')
    PARSER.add_argument('--sf', type=float, default=1.5, dest="sf", help='Scaling factor applied to metal atomic radii')
    ARGS = PARSER.parse_args()
    if ARGS.metal_hops == 1:
        sec_nn = False
    elif ARGS.metal_hops == 2:
        sec_nn = True
    else:
        print("Please choose a valid number of metal hops (1 or 2).")
    families = os.listdir(ARGS.data)
    BMgraphs = []
    for family in families:
        if "." in family:  # Hidden files
            continue
        print("Processing family: {}".format(family))
        for (dirpath, dirnames, filenames) in os.walk(ARGS.data+"/"+family):
            for dirname in dirnames:
                if "0000" in dirname:  # Slab
                    continue
                if "-" not in dirname: # Gas molecules
                    print("    {}".format(dirname))
                    graph = structure_to_graph(dirpath+"/"+dirname+"/CONTCAR", 
                                             voronoi_tolerance=ARGS.tol,
                                             scaling_factor=ARGS.sf, 
                                             second_order=sec_nn)
                    p1 = Popen(["grep", "energy  w", "{}/OUTCAR".format(dirpath+"/"+dirname)], stdout=PIPE)
                    p2 = Popen(["tail", "-1"], stdin=p1.stdout, stdout=PIPE)
                    p3 = Popen(["awk", "{print $NF}"], stdin=p2.stdout, stdout=PIPE)
                    graph.y = float(p3.communicate()[0].decode("utf-8"))
                    graph.formula = get_graph_formula(graph, ENCODER.categories_[0])
                    graph.family = family
                    BMgraphs.append(graph)
                else:
                    print("    {}".format(dirname))
                    metal = dirname.split("-")[0]
                    if family == "Plastics" or (family == "Polyurethanes" and metal == "au"):
                        BMgraphs.append(get_graph_sample(system=dirpath+"/"+dirname,
                                                     surface=dirpath + "/" + metal + "-0000",
                                                     voronoi_tolerance=ARGS.tol,
                                                     scaling_factor=ARGS.sf,
                                                     second_order=sec_nn,
                                                     family=family, 
                                                     surf_multiplier=4))
                    else:
                        BMgraphs.append(get_graph_sample(system=dirpath+"/"+dirname,
                                                     surface=dirpath + "/" + metal + "-0000",
                                                     voronoi_tolerance=ARGS.tol,
                                                     scaling_factor=ARGS.sf,
                                                     second_order=sec_nn,
                                                     family=family))
    BM_dataloader = DataLoader(BMgraphs, batch_size=len(BMgraphs), shuffle=False)
    save(BM_dataloader, 
         ARGS.data+"/BMdataloader_{}.pt".format(get_id({"voronoi_tol": ARGS.tol,
                                                 "scaling_factor": ARGS.sf,
                                                 "second_order_nn": sec_nn})))
    
    