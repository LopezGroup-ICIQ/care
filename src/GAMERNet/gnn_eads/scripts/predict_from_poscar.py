"""Script for the direct comparison of the GNN performance compared to DFT.
It works wirth systems containing the following elements: 
Adsorbate: C, H, O, N, S
Catalyst slab: Ag, Au, Cd, Co, Cu, Fe, Ir, Ni, Os, Pd, Pt, Rh, Ru, Zn
"""

import argparse
import sys
sys.path.insert(0, "../src")

import matplotlib.pyplot as plt

from gnn_eads.functions import structure_to_graph
from gnn_eads.graph_tools import visualize_graph, extract_adsorbate
from gnn_eads.nets import PreTrainedModel

if __name__ == "__main__":
    PARSER = argparse.ArgumentParser(description="Convert DFT system to graph and compare the GNN prediction to the DFT outcome.")
    PARSER.add_argument("-i", "--input", type=str, dest="input", 
                        help="Path to the POSCAR file of the specific adsorption system/gas-molecule.")
    ARGS = PARSER.parse_args()
    
    # 1) Load pre-trained GNN model on CPU
    MODEL_PATH = "../models/GAME-Net"
    model = PreTrainedModel(MODEL_PATH)
    
    # 2) Convert input DFT sample to graph object
    ads_graph = structure_to_graph(ARGS.input, model.g_tol, model.g_sf, model.g_metal_2nn)
    molecule_graph = extract_adsorbate(ads_graph)

    print(ads_graph)

    if ads_graph.num_nodes == molecule_graph.num_nodes:
        adsorption_system = False
    else:
        adsorption_system = True
    #print(graph)
    
    # 3) Get GNN prediction
    E_GNN = model.evaluate(ads_graph)
    if adsorption_system:
        E_mol = model.evaluate(molecule_graph)
        E_ads = E_GNN - E_mol
        print("Adsorption energy: {:.2f} eV".format(E_ads))
    else:
        print("Gas phase energy: {:.2f} eV".format(E_GNN))
    # visualize_graph(ads_graph, font_color="black")
    # plt.show()