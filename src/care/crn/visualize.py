"""Visualization modules for ReactionNetwork objects."""

from os import makedirs
from os.path import abspath
import re

from ase.io import write
from ase import Atoms
from ase.visualize import view
from energydiagram import ED
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from pydot import Subgraph
from scipy.interpolate import CubicSpline

from care import ElementaryReaction, format_reaction, Intermediate, ReactionNetwork


def write_dotgraph(graph: ReactionNetwork, 
                   filename: str, 
                   figsize:tuple=(18, 15), 
                   rankdir: str="TB", 
                   rank_sep: float=0.3, 
                   node_sep: float=0.15, 
                   fontsize: int=50, 
                   layout_engine: str="dot", 
                   dpi:int=150):
    """
    Write a dot graph representing the reaction network.

    Args:
        graph (ReactionNetwork): The reaction network graph.
        filename (str): The output filename for the dot graph.
        figsize (tuple): Figure size in cm.
        rankdir (str): Rank direction for the graph layout. Options are "TB" (top-bottom), "LR" (left-right), etc.
        rank_sep (float): Separation between ranks in the graph.
        node_sep (float): Separation between nodes in the graph.
        fontsize (int): Font size for the graph labels.
        layout_engine (str): Layout engine to use (e.g., "dot", "neato", "fdp", "circo", "twopi").
    """
    g = nx.DiGraph()
    for node, _ in graph.nodes(data=True):
        if isinstance(node, Intermediate):
            g.add_node(node, category="intermediate", formula=node.formula, phase=node.phase)
        elif isinstance(node, ElementaryReaction):
            g.add_node(node, category="reaction", r_type=node.r_type, repr_hr=node.repr_hr)
            for predecessor in graph.predecessors(node):
                g.add_edge(predecessor, node)
            for successor in graph.successors(node):
                g.add_edge(node, successor)
        else:
            pass
    g.remove_node("*")
    plot = nx.drawing.nx_pydot.to_pydot(g)
    subgraph_source = Subgraph("source", rank="source")
    subgraph_ads = Subgraph("ads", rank="same")
    subgraph_sink = Subgraph("sink", rank="sink")
    subgraph_des = Subgraph("des", rank="same")
    subgraph_same = Subgraph("same", rank="same")
    color_code_species = {"gas": "lightpink", "ads": "wheat"}
    for node in plot.get_nodes():
        try:
            node.set_orientation("portrait")
            attrs = node.get_attributes()
            node.set_penwidth("2")
            if attrs["category"] == "intermediate":
                formula = attrs["formula"]
                formula += "" if attrs["phase"] == "gas" else "*"
                for num in re.findall(r"\d+", formula):
                    SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
                    formula = formula.replace(num, num.translate(SUB))
                node.set_shape("ellipse")
                node.set_style("filled")
                node.set_label(formula)
                node.set_fillcolor(color_code_species[attrs["phase"]])
            elif attrs["category"] == "reaction":
                node.set_shape("square")
                node.set_style("filled")
                node.set_label("")
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
        except KeyError:
            pass
    for edge in plot.get_edges():
        edge.set_penwidth("2")      
        edge.set_arrowsize("1.5")   
        edge.set_arrowhead("vee")

    plot.add_subgraph(subgraph_source)
    plot.add_subgraph(subgraph_sink)
    plot.add_subgraph(subgraph_ads)
    plot.add_subgraph(subgraph_des)
    plot.set_overlap("false")
    plot.set_splines("true")
    plot.set_bgcolor("white")
    x, y = (figsize[0]/2.54, figsize[1]/2.54)
    plot.set_size(f"{x},{y}!")
    plot.set_ratio("fill")
    plot.set_nodesep(node_sep)
    plot.set_ranksep(rank_sep)
    plot.set_rankdir(rankdir)
    plot.set_fontname("Arial")
    plot.set_fontsize(str(fontsize))
    plot.set_dpi(str(dpi))
    print(f"Writing graph to {filename} using layout engine: {layout_engine}")
    try:
        if filename.endswith(".svg"):
            plot.write_svg("./" + filename, prog=layout_engine)
        elif filename.endswith(".png"):
            plot.write_png("./" + filename, prog=layout_engine)
        elif filename.endswith(".dot"):
            plot.write_dot("./" + filename, prog=layout_engine)
        else:
            print(f"Warning: Unknown file extension. Writing to {filename}.svg")
            plot.write_svg("./" + filename + ".svg", prog=layout_engine)
    except FileNotFoundError:
        print(f"Error: Layout engine '{layout_engine}' not found.")
        print("Please ensure Graphviz is installed and in your system's PATH.")
        print("You can download it from: https://graphviz.org/download/")


def visualize_reaction(step: ElementaryReaction, 
                       show_uncertainty: bool = True) -> ED:
    """Visualize a reaction step with an energy diagram.
    Based on PyEnergyDiagrams package.

    Args:
        step (ElementaryReaction): The reaction step to visualize.
        show_uncertainty (bool): Whether to show uncertainty in the energy values.
    Returns:        
        ED: An energy diagram object representing the reaction step."""
    rxn_string = step.repr_hr
    where_surface = (
        "reactants" if any(inter.is_surface for inter in step.reactants) else "products"
    )
    diagram = ED()  # Energy Diagram object
    diagram.add_level(0, format_reaction(rxn_string.split(" \u27F9 ")[0]))
    diagram.add_level(round(step.e_act[0], 2), "TS", color="r")
    diagram.add_level(
        round(step.e_rxn[0], 2),
        format_reaction(rxn_string.split(" \u27F9 ")[1]),
    )
    diagram.add_link(0, 1)
    diagram.add_link(1, 2)
    y = diagram.plot(ylabel="Energy / eV")
    plt.title(format_reaction(step.repr_hr), fontname="DejaVu Sans", fontweight="bold", y=1.05)
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


def build_energy_profile(graph: nx.DiGraph, path: list[str]):
    """
    Generate energy profile with the energydiagram package.
    """
    ed = ED()
    ed.round_energies_at_digit = 2
    ed.add_level(0)
    counter = 0
    ref = 0
    for item in path:
        if len(item[0]) == 28:  # Intermediate -> step (Add TS)
            inter, step = item[0], item[1]
            delta = graph.edges[(inter, step)]["delta"]
            ed.add_level(ref + delta, "TS", color="r")
            ref += delta
            counter += 1
            ed.add_link(counter - 1, counter)
        else:  # Step -> intermediate (Add intermediate always)
            step, inter = item[0], item[1]
            delta = graph.edges[(step, inter)]["delta"]
            ed.add_level(ref + delta, "int")
            ref += delta
            counter += 1
            ed.add_link(counter - 1, counter)
    return ed


def plot_reaction_profile(energies, title="", num_points=100):
    """
    Plots a smooth reaction energy profile using cubic spline interpolation,
    ensuring the curve starts and ends with a zero slope (stationary points).

    Args:
        energies (list or np.array): A list of energy values for key points.
        num_points (int): The number of points for the smooth curve.
    """
    energies = [x - energies[0] for x in energies]  # reference wrt IS
    x_data = np.arange(len(energies))
    
    # Create the cubic spline interpolation function with boundary conditions
    cs = CubicSpline(x_data, energies, bc_type=((1, 0.0), (1, 0.0)))
    
    x_smooth = np.linspace(0, len(energies) - 1, num_points)
    
    y_smooth = cs(x_smooth)
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot the smooth curve on the axes
    ax.plot(x_smooth, y_smooth, label='Smooth Curve', color='blue')
    
    # Plot the original data points on the axes
    ax.plot(x_data, energies, 'o', label='Images', color='red')
    
    # Set labels and other properties using the axes object
    ax.set_xlabel('Reaction Coordinate')
    ax.set_ylabel('Energy / eV')
    ax.set_title(title)
    ax.grid(True)
    ax.legend()
    plt.close(fig)
    return fig


def visualize_intermediate(x: Intermediate):
    """Visualize the molecule of an intermediate.

    Args:
        inter_code (str): Code of the intermediate.
    """
    configs = [
        config["ase"]
        for config in x.ads_configs.values()
    ]
    if len(configs) == 0 and type(configs[0]) == Atoms:
        view(x.molecule)
    else:
        view(configs)
