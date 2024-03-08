"""Visualization modules for ReactionNetwork objects."""

import re
from os import makedirs
from os.path import abspath

import networkx as nx
import numpy as np
from pydot import Subgraph
from energydiagram import ED
from matplotlib import cm
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import BoxStyle, Rectangle
import matplotlib.pyplot as plt
from ase.io import write

from care import ElementaryReaction
from care.crn.microkinetic import max_flux


def write_dotgraph(graph: nx.DiGraph, filename: str, source: str = None):
    if source is not None:
        edge_list = max_flux(graph, source)
        for edge in graph.edges(data=True):
            if edge in edge_list:
                edge[2]["max"] = "max"
            else:
                edge[2]["max"] = "no"
    pos = nx.kamada_kawai_layout(graph)
    nx.set_node_attributes(graph, pos, "pos")
    plot = nx.drawing.nx_pydot.to_pydot(graph)
    subgraph_source = Subgraph("source", rank="source")
    subgraph_ads = Subgraph("ads", rank="same")
    subgraph_sink = Subgraph("sink", rank="sink")
    subgraph_des = Subgraph("des", rank="same")
    subgraph_same = Subgraph("same", rank="same")
    subgraph_electro = Subgraph("electro", rank="same")
    plot.rankdir = "TB"
    plot.set_dpi(1)
    for node in plot.get_nodes():
        node.set_orientation("portrait")
        attrs = node.get_attributes()
        if attrs["category"] == "intermediate":
            formula = node.get_attributes()["formula"]
            formula += "" if attrs["phase"] == "gas" else "*"
            for num in re.findall(r"\d+", formula):
                SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
                formula = formula.replace(num, num.translate(SUB))
            node.set_fontname("Arial")
            node.set_label(formula)
            node.set_style("filled")
            if attrs["phase"] != "gas":
                node.set_fillcolor("wheat")
            else:
                node.set_fillcolor("lightpink")
            node.set_shape("ellipse")
            node.set_width("4/2.54")
            node.set_height("4/2.54")
            # node.set_fixedsize("true")
            node.set_fontsize("120")
            if attrs["phase"] == "gas" and float(attrs["molar_fraction"]) > 0.0:
                # set node_shape to cylinder
                node.set_width("5/2.54")
                node.set_height("5/2.54")
                node.set_shape("cylinder")
                node.set_fillcolor("lightcoral")
                node.set_fontsize("150")
                subgraph_source.add_node(node)
            elif attrs["phase"] == "gas" and float(attrs["molar_fraction"]) == 0.0:
                # give lowest rank
                subgraph_sink.add_node(node)
            else:
                # subgraph_same.add_node(node)
                pass
        elif attrs["category"] == "electro":
            formula = node.get_attributes()["formula"]
            node.set_shape("diamond")
            node.set_style("filled")
            node.set_label(formula)
            node.set_width("2/2.54")
            node.set_height("2/2.54")
            node.set_fillcolor("yellow")
            node.set_fontsize("150")
            node.set_fontname("Arial")
            subgraph_electro.add_node(node)
        else:  # REACTION
            node.set_shape("square")
            node.set_style("filled")
            node.set_label("")
            node.set_width("2/2.54")
            node.set_height("2/2.54")
            if attrs["r_type"] in ("adsorption", "desorption"):
                if attrs["r_type"] == "adsorption":
                    subgraph_ads.add_node(node)
                    node.set_fillcolor("tomato1")
                else:
                    subgraph_des.add_node(node)
                    node.set_fillcolor("palegreen2")
            elif attrs["r_type"] == "eley_rideal":
                node.set_fillcolor("mediumpurple1")
            else:
                node.set_fillcolor("steelblue3")
                subgraph_same.add_node(node)

    # set edge width as function of consumption rate
    # width_list = []
    # max_scale, min_scale = 10, 1
    # max_weight, min_weight = -np.log10(graph.min_rate), -np.log10(graph.max_rate)
    for edge in plot.get_edges():
        if edge.get_source() == "*" or edge.get_destination() == "*":
            plot.del_edge(edge.get_source(), edge.get_destination())
            continue
        # rate = float(edge.get_attributes()["rate"])
        # # Scale logarithmically the width of the edge considering graph.max_rate and graph.min_rate
        # edge_width = -np.log10(rate)
        # width = (max_scale - min_scale) / (max_weight - min_weight) * (
        #     edge_width - max_weight
        # ) + max_scale
        # if edge.get_attributes()["max"] == "max":
        #     edge.set_color("firebrick")
        
        # edge.set_penwidth(30)
            # edge.set_penwidth(width)
            # width_list.append(width)

    plot.add_subgraph(subgraph_source)
    plot.add_subgraph(subgraph_sink)
    plot.add_subgraph(subgraph_ads)
    plot.add_subgraph(subgraph_des)
    plot.add_subgraph(subgraph_electro)
    # plot.add_subgraph(subgraph_same)
    # set min distance between nodes
    plot.set_overlap("false")
    plot.set_splines("true")
    plot.set_nodesep(0.5)
    plot.set_ranksep(1)
    plot.set_bgcolor("white")

    plot.write_svg("./" + filename)
    # return width_list


def visualize_reaction(step: ElementaryReaction, 
                        show_uncertainty: bool = True):
    # components = rxn.split("<->")
    # reactants, products = components[0].split("+"), components[1].split("+")
    # for i, inter in enumerate(reactants):
    #     if "0000000000*" in inter:
    #         where_surface = "reactants"
    #         surf_index = i
    #         break
    # for i, inter in enumerate(products):
    #     if "0000000000*" in inter:
    #         where_surface = "products"
    #         surf_index = i
    #         break
    # v_reactants = [
    #     re.findall(r"\[([a-zA-Z0-9])\]", reactant) for reactant in reactants
    # ]
    # v_products = [re.findall(r"\[([a-zA-Z0-9])\]", product) for product in products]
    # v_reactants = [item for sublist in v_reactants for item in sublist]
    # v_products = [item for sublist in v_products for item in sublist]
    # reactants = [re.findall(r"\((.*?)\)", reactant) for reactant in reactants]
    # products = [re.findall(r"\((.*?)\)", product) for product in products]
    # reactants = [item for sublist in reactants for item in sublist]
    # products = [item for sublist in products for item in sublist]
    # for i, reactant in enumerate(reactants):
    #     if v_reactants[i] != "1":
    #         reactants[i] = v_reactants[i] + reactant
    #     if "(g" in reactant:
    #         reactants[i] += ")"
    #     if where_surface == "reactants" and i == surf_index:
    #         reactants[i] = "*"
    # for i, product in enumerate(products):
    #     if v_products[i] != "1":
    #         products[i] = v_products[i] + product
    #     if "(g" in product:
    #         products[i] += ")"
    #     if where_surface == "products" and i == surf_index:
    #         products[i] = "*"
    # rxn_string = " + ".join(reactants) + " -> " + " + ".join(products)
    rxn_string = step.repr_hr
    where_surface = (
        "reactants"
        if any(inter.is_surface for inter in step.reactants)
        else "products"
    )
    diagram = ED()
    diagram.add_level(0, rxn_string.split(" <-> ")[0])
    diagram.add_level(round(step.e_act[0], 2), "TS", color="r")
    diagram.add_level(
        round(step.e_rxn[0], 2),
        rxn_string.split(" <-> ")[1],
    )
    diagram.add_link(0, 1)
    diagram.add_link(1, 2)
    y = diagram.plot(ylabel="Energy / eV")
    plt.title(rxn_string, fontname="Arial", fontweight="bold", y=1.05)
    artists = diagram.fig.get_default_bbox_extra_artists()
    size = artists[2].get_position()[0] - artists[3].get_position()[0]
    ap_reactants = (
        artists[3].get_position()[0],
        artists[3].get_position()[1] + 0.15,
    )
    ap_products = (
        artists[11].get_position()[0],
        artists[11].get_position()[1] + 0.15,
    )
    from matplotlib.patches import Rectangle
    makedirs("tmp", exist_ok=True)
    counter = 0
    for i, inter in enumerate(step.reactants):
        if inter.is_surface:
            pass
        else:
            fig_path = abspath("tmp/reactant_{}.png".format(i))
            write(fig_path, inter.molecule, show_unit_cell=0)
            arr_img = plt.imread(fig_path)
            im = OffsetImage(arr_img)
            if where_surface == "reactants":
                ab = AnnotationBbox(
                    im,
                    (
                        ap_reactants[0] + size / 2,
                        ap_reactants[1] + size * (0.5 + counter),
                    ),
                    frameon=False,
                )
                diagram.ax.add_artist(ab)
                counter += 1
            else:
                ab = AnnotationBbox(
                    im,
                    (
                        ap_reactants[0] + size / 2,
                        ap_reactants[1] + size * (0.5 + i),
                    ),
                    frameon=False,
                )
                diagram.ax.add_artist(ab)
    counter = 0
    for i, inter in enumerate(step.products):
        if inter.is_surface:
            pass
        else:
            fig_path = abspath("tmp/product_{}.png".format(i))
            write(fig_path, inter.molecule, show_unit_cell=0)
            arr_img = plt.imread(fig_path)
            im = OffsetImage(arr_img)
            if where_surface == "products":
                ab = AnnotationBbox(
                    im,
                    (
                        ap_products[0] + size / 2,
                        ap_products[1] + size * (0.5 + counter),
                    ),
                    frameon=False,
                )
                diagram.ax.add_artist(ab)
                counter += 1
            else:
                ab = AnnotationBbox(
                    im,
                    (ap_products[0] + size / 2, ap_products[1] + size * (0.5 + i)),
                    frameon=False,
                )
                diagram.ax.add_artist(ab)
    if show_uncertainty:
        from matplotlib.patches import Rectangle
        width = artists[2].get_position()[0] - artists[3].get_position()[0]
        height_ts = 1.96 * 2 * step.e_act[1]
        anchor_point_ts = (
            min(artists[6].get_position()[0], artists[7].get_position()[0]),
            round(step.e_act[0], 2) - 0.5 * height_ts,
        )
        ts_box = Rectangle(
            anchor_point_ts,
            width,
            height_ts,
            fill=True,
            color="#FFD1DC",
            linewidth=1.5,
            zorder=-1,
        )
        diagram.ax.add_patch(ts_box)
    return diagram


# def draw_graph(self):
    #     """Create a networkx graph representing the network.

    #     Returns:
    #         obj:`nx.DiGraph` with all the information of the network.
    #     """
    #     # norm_vals = self.get_min_max()
    #     colormap = cm.inferno_r
    #     # norm = mpl.colors.Normalize(*norm_vals)
    #     node_inf = {
    #         "inter": {"node_lst": [], "color": [], "size": []},
    #         "ts": {"node_lst": [], "color": [], "size": []},
    #     }
    #     edge_cl = []
    #     for node in self.graph.nodes():
    #         sel_node = self.graph.nodes[node]
    #         try:
    #             # color = colormap(norm(sel_node['energy']))
    #             if sel_node["category"] in ("gas", "ads", "surf"):
    #                 node_inf["inter"]["node_lst"].append(node)
    #                 node_inf["inter"]["color"].append("blue")
    #                 # node_inf['inter']['color'].append(mpl.colors.to_hex(color))
    #                 node_inf["inter"]["size"].append(20)
    #             # elif sel_node['category']  'ts':
    #             else:
    #                 if "electro" in sel_node:
    #                     if sel_node["electro"]:
    #                         node_inf["ts"]["node_lst"].append(node)
    #                         node_inf["ts"]["color"].append("red")
    #                         node_inf["ts"]["size"].append(5)
    #                 else:
    #                     node_inf["ts"]["node_lst"].append(node)
    #                     node_inf["ts"]["color"].append("green")
    #                     # node_inf['ts']['color'].append(mpl.colors.to_hex(color))
    #                     node_inf["ts"]["size"].append(5)
    #             # elif sel_node['electro']:
    #             #     node_inf['ts']['node_lst'].append(node)
    #             #     node_inf['ts']['color'].append('green')
    #             #     # node_inf['ts']['color'].append(mpl.colors.to_hex(color))
    #             #     node_inf['ts']['size'].append(10)
    #         except KeyError:
    #             node_inf["ts"]["node_lst"].append(node)
    #             node_inf["ts"]["color"].append("green")
    #             node_inf["ts"]["size"].append(10)

    #     # for edge in self.graph.edges():
    #     #     sel_edge = self.graph.edges[edge]
    #     # color = colormap(norm(sel_edge['energy']))
    #     # color = mpl.colors.to_rgba(color, 0.2)
    #     # edge_cl.append(color)

    #     fig = plt.Figure()
    #     axes = fig.gca()
    #     axes.get_xaxis().set_visible(False)
    #     axes.get_yaxis().set_visible(False)
    #     fig.patch.set_visible(False)
    #     axes.axis("off")

    #     pos = nx.drawing.layout.kamada_kawai_layout(self.graph)

    #     nx.drawing.draw_networkx_nodes(
    #         self.graph,
    #         pos=pos,
    #         ax=axes,
    #         nodelist=node_inf["ts"]["node_lst"],
    #         node_color=node_inf["ts"]["color"],
    #         node_size=node_inf["ts"]["size"],
    #     )

    #     nx.drawing.draw_networkx_nodes(
    #         self.graph,
    #         pos=pos,
    #         ax=axes,
    #         nodelist=node_inf["inter"]["node_lst"],
    #         node_color=node_inf["inter"]["color"],
    #         node_size=node_inf["inter"]["size"],
    #     )
    #     #    node_shape='v')
    #     nx.drawing.draw_networkx_edges(
    #         self.graph,
    #         pos=pos,
    #         ax=axes,
    #         #    edge_color=edge_cl,
    #         width=0.3,
    #         arrowsize=0.1,
    #     )
    #     # add white background to the plot
    #     axes.set_facecolor("white")
    #     fig.tight_layout()
    #     return fig

def build_energy_profile(graph: nx.DiGraph, path: list[str]):
    """
    Generate energy profile with the energydiagram package.
    """
    ed = ED()
    ed.round_energies_at_digit=2
    ed.add_level(0)
    counter = 0
    ref = 0
    for item in path:
        if len(item[0]) == 28:  # Intermediate -> step (Add TS)
            inter, step = item[0], item[1]
            delta = graph.edges[(inter, step)]['delta']
            ed.add_level(ref + delta, 'TS', color='r')
            ref += delta
            counter += 1 
            ed.add_link(counter-1, counter)
        else: # Step -> intermediate (Add intermediate always)
            step, inter = item[0], item[1]
            delta = graph.edges[(step, inter)]['delta']
            ed.add_level(ref + delta, 'int')
            ref += delta
            counter += 1
            ed.add_link(counter-1, counter)
    return ed