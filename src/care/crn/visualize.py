"""Visualization modules for ReactionNetwork objects."""

import networkx as nx
import re
import numpy as np
import math
from pydot import Dot, Node, Edge, Subgraph

def write_dotgraph(graph, 
                   filename: str):
        pos = nx.kamada_kawai_layout(graph)
        nx.set_node_attributes(graph, pos, "pos")
        plot = nx.drawing.nx_pydot.to_pydot(graph)
        plot.set_orientation("landscape")
        subgraph_source = Subgraph("source", rank="source")
        subgraph_ads = Subgraph("ads", rank="same")
        subgraph_sink = Subgraph("sink", rank="sink")
        subgraph_des = Subgraph("des", rank="same")
        subgraph_same = Subgraph("same", rank="same")
        plot.rankdir = "LR"
        for node in plot.get_nodes():
            node.set_orientation("portrait")
            attrs = node.get_attributes()
            if attrs["category"] == 'intermediate':
                formula = node.get_attributes()["formula"]
                # add '(g)' in subscript if phase is gas else '*'
                formula += "" if attrs['phase'] == "gas" else "*"
                for num in re.findall(r"\d+", formula):
                    SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
                    formula = formula.replace(num, num.translate(SUB))
                node.set_fontname("Arial")
                node.set_label(formula)
                node.set_style("filled")
                if attrs['phase'] != "gas":
                    node.set_fillcolor("wheat")
                else:
                    node.set_fillcolor("lightpink")
                node.set_shape("ellipse")
                node.set_width("4/2.54")
                node.set_height("4/2.54")
                # node.set_fixedsize("true")
                node.set_fontsize("120")
                if attrs['phase'] == 'gas' and float(attrs['molar_fraction']) > 0.0:
                    # set node_shape to cylinder
                    node.set_width("5/2.54")
                    node.set_height("5/2.54")
                    node.set_shape("cylinder")
                    node.set_fillcolor("lightcoral")
                    node.set_fontsize("150")
                    # give rank
                    subgraph_source.add_node(node)
                elif attrs['phase'] == 'gas' and float(attrs['molar_fraction']) == 0.0:
                    # give lowest rank
                    subgraph_sink.add_node(node)
                else:
                    subgraph_same.add_node(node)
            else:  # REACTION
                node.set_shape("square")
                node.set_style("filled")
                node.set_label("")
                node.set_width("2/2.54")
                node.set_height("2/2.54")
                if attrs["r_type"] in ("adsorption", "desorption"):
                    if attrs['r_type'] == 'adsorption':
                        subgraph_ads.add_node(node)
                        node.set_fillcolor("palegreen2")
                    else:
                        subgraph_des.add_node(node)
                        node.set_fillcolor("palegreen3")
                elif attrs["r_type"] == "eley_rideal":
                    node.set_fillcolor("mediumpurple1")
                else:
                    node.set_fillcolor("steelblue3")

        # set edge width as function of consumption rate
        width_list = []
        max_scale, min_scale = 10, 1
        max_weight, min_weight = -np.log10(graph.min_rate), -np.log10(graph.max_rate)
        for edge in plot.get_edges():
            if edge.get_source() == '*' or edge.get_destination() == '*':
                plot.del_edge(edge.get_source(), edge.get_destination())
                continue
            # print(edge)
            rate = float(edge.get_attributes()["rate"])
            # Scale logarithmically the width of the edge considering graph.max_rate and graph.min_rate
            edge_width = -np.log10(rate)
            width = (max_scale - min_scale) / (max_weight - min_weight) * (
                edge_width - max_weight
            ) + max_scale
            # if width > 5:
                # set color red
                # edge.set_color("firebrick")
            edge.set_penwidth(width)
            width_list.append(width)
        
        plot.add_subgraph(subgraph_source)
        plot.add_subgraph(subgraph_sink)
        plot.add_subgraph(subgraph_ads)
        plot.add_subgraph(subgraph_des)
        # plot.add_subgraph(subgraph_same)
        # set min distance between nodes
        plot.set_overlap("false")
        plot.set_splines("true")
        plot.set_nodesep(0.3)    
        a4_dims = (120, 120)  # In inches
        
        # plot.set_size(f"{a4_dims[0],a4_dims[1]}!")
        plot.set_bgcolor("white")
        
        plot.set_dpi(20)
        plot.write_svg("./" + filename)
        return width_list
