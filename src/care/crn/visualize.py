"""Visualization modules for ReactionNetwork objects."""

import re
from io import BytesIO 
from PIL import Image
import os
import shutil
import tempfile

from ase.io import write
from ase import Atoms
from ase.visualize import view
from energydiagram import ED
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pydot
from pydot import Subgraph
from scipy.interpolate import CubicSpline

from care import ElementaryReaction, format_reaction, Intermediate, ReactionNetwork
from care.crn.intermediate import SurfaceSite, AdsorbedSpecies, GasSpecies


def plot_crn(graph, 
            filename: str = None, 
            figsize: tuple = (18, 15), 
            rankdir: str = "TB", 
            rank_sep: float = 0.3, 
            node_sep: float = 0.15, 
            fontsize: int = 50, 
            legend_fontsize: int = 20,
            layout_engine: str = "dot", 
            show_species_labels: bool = True,
            arrowhead: str = "normal",
            fontname: str = "Arial",
            dpi: int = 100) -> None:
    """
    Write a dot graph representing the reaction network.
    """
    if shutil.which(layout_engine) is None:
        raise RuntimeError("Graphviz binaries not found. You must install Graphviz "
        "(https://graphviz.org/download/) and ensure it is in your system PATH.")

    color_code_species = {
        "gas": {"color": "lightpink", "label": "Gas Species", "shape": "ellipse"},
        "ads": {"color": "wheat", "label": "Adsorbed Species", "shape": "ellipse"},
        "solv": {"color": "lightblue", "label": "Solvated Species", "shape": "ellipse"},
        "electro": {"color": "gold", "label": "Electron", "shape": "ellipse"}
    }
    
    color_code_reactions = {
        "Adsorption": {"color": "tomato1", "label": "Adsorption", "shape": "square"},
        "Desorption": {"color": "palegreen2", "label": "Desorption", "shape": "square"},
        "EleyRideal": {"color": "mediumpurple1", "label": "Eley-Rideal", "shape": "square"},
        "PCET": {"color": "lightsalmon", "label": "PCET", "shape": "square"},
        "BondBreaking": {"color": "steelblue3", "label": "Surface Step", "shape": "square"},
        "BondFormation": {"color": "steelblue3", "label": "Surface Step", "shape": "square"}
    }
    
    generic_step = {"color": "steelblue3", "label": "Surface Step", "shape": "square"}

    active_phases = set()
    active_rxn_classes = set()

    g = nx.DiGraph()
    for node, _ in graph.nodes(data=True):
        cls_name = node.__class__.__name__
        if type(node).__name__ in ("GasSpecies", "AdsorbedSpecies", "SurfaceSite", "Proton", "Water", "Electron") or hasattr(node, "phase"):
            g.add_node(node, category="intermediate", formula=node.formula, phase=node.phase, cls_name=cls_name)
            active_phases.add(node.phase)
        elif type(node).__name__ in ("ElementaryReaction", "Adsorption", "Desorption", "PCET", "BondBreaking", "BondFormation", "Rearrangement") or hasattr(node, "r_type"):
            g.add_node(node, category="reaction", r_type=node.r_type, repr_hr=node.repr_hr, cls_name=cls_name)
            active_rxn_classes.add(cls_name)
            for predecessor in graph.predecessors(node):
                g.add_edge(predecessor, node)
            for successor in graph.successors(node):
                g.add_edge(node, successor)
                
    if "*" in g:
        g.remove_node("*")
        
    plot = nx.drawing.nx_pydot.to_pydot(g)
    plot.set_fontname(fontname)
    plot.set_node_defaults(fontname=fontname)
    plot.set_edge_defaults(fontname=fontname)
    
    subgraph_source = Subgraph("source", rank="source")
    subgraph_ads = Subgraph("ads", rank="same")
    subgraph_sink = Subgraph("sink", rank="sink")
    subgraph_des = Subgraph("des", rank="same")
    subgraph_same = Subgraph("same", rank="same")
                            
    for node in plot.get_nodes():
        node.set_fontname(fontname)
        try:
            node.set_orientation("portrait")
            attrs = node.get_attributes()
            node.set_penwidth("2")
            category = attrs.get("category", "").strip('"')
            
            if category == "intermediate":  
                node.set_shape("ellipse")
                node.set_style("filled")
                phase_key = attrs.get("phase", "").strip('"')
                
                if show_species_labels:
                    formula = attrs.get("formula", "").strip('"')
                    formula += "" if phase_key == "gas" else "*"
                    for num in re.findall(r"\d+", formula):
                        SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
                        formula = formula.replace(num, num.translate(SUB))
                    node.set_label(formula)
                else:
                    node.set_label("")
                    
                node.set_fillcolor(color_code_species.get(phase_key, {}).get("color", "white"))
                
            elif category == "reaction":
                node.set_shape("square")
                node.set_style("filled")
                node.set_label("")
                r_type = attrs.get("cls_name", "").strip('"')
                
                reaction_data = color_code_reactions.get(r_type, generic_step)
                node.set_fillcolor(reaction_data["color"])
                
                if r_type == "Adsorption":
                    subgraph_ads.add_node(node)
                elif r_type == "Desorption":
                    subgraph_des.add_node(node)
                else:
                    subgraph_same.add_node(node)
                    
        except Exception:
            pass
            
    for edge in plot.get_edges():
        edge.set_fontname(fontname)
        edge.set_penwidth("2")      
        edge.set_arrowsize("1.5")   
        edge.set_arrowhead(arrowhead)

    # --- Dynamic HTML Legend Generation ---
    legend_items = []
    
    for phase in ["gas", "ads", "solv", "electro"]:
        if phase in active_phases and phase in color_code_species:
            legend_items.append(color_code_species[phase])
            
    added_rxn_labels = set()
    for r_type in active_rxn_classes:
        data = color_code_reactions.get(r_type, generic_step)
        if data["label"] not in added_rxn_labels:
            legend_items.append(data)
            added_rxn_labels.add(data["label"])

    rows_html = ""
    for i in range(0, len(legend_items), 2):
        rows_html += "<TR>"
        
        # Insert Unicode shapes to visually indicate Ovals vs Squares in the HTML table
        item1 = legend_items[i]
        rows_html += f'<TD BGCOLOR="{item1["color"]}" BORDER="1"><FONT POINT-SIZE="{legend_fontsize}">{item1["label"]}</FONT></TD>'
        
        if i + 1 < len(legend_items):
            item2 = legend_items[i+1]
            rows_html += f'<TD BGCOLOR="{item2["color"]}" BORDER="1"><FONT POINT-SIZE="{legend_fontsize}">{item2["label"]}</FONT></TD>'
        else:
            rows_html += '<TD BORDER="0"></TD>'
        rows_html += "</TR>\n"
        
    legend_html = f"""<
    <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="6" CELLPADDING="6">
      <TR><TD COLSPAN="2"><FONT POINT-SIZE="{legend_fontsize + 4}"><B>Legend</B></FONT></TD></TR>
      {rows_html}
    </TABLE>>"""
    
    legend_node = pydot.Node("Legend", shape="none", margin="0", label=legend_html)
    subgraph_sink.add_node(legend_node)
    # -------------------------------------

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
    plot.set_fontname(fontname)
    plot.set_fontsize(str(fontsize))
    plot.set_dpi(str(dpi))
    
    if filename is None:
        png_str = plot.create_png(prog=layout_engine)
        sio = BytesIO(png_str)
        img = Image.open(sio)
        plt.figure(figsize=(figsize[0]/2.54, figsize[1]/2.54), dpi=dpi)
        plt.imshow(img)
        plt.axis('off')
        plt.show() 
    else:
        print(f"Writing graph to {filename} using layout engine: {layout_engine}")
        try:
            if filename.endswith(".svg"):
                plot.write_svg(filename, prog=layout_engine)
            elif filename.endswith(".png"):
                plot.write_png(filename, prog=layout_engine)
            elif filename.endswith(".dot"):
                plot.write_dot(filename, prog=layout_engine)
            else:
                plot.write_svg("./" + filename + ".svg", prog=layout_engine)
        except FileNotFoundError:
            print(f"Error: Layout engine '{layout_engine}' not found.")

def visualize_reaction(step: ElementaryReaction) -> ED:
    """Visualize a reaction step with an energy diagram.
    Based on PyEnergyDiagrams package.

    Args:
        step (ElementaryReaction): The reaction step to visualize.
        
    Returns:        
        ED: An energy diagram object representing the reaction step.
    """
    rxn_string = step.repr_hr
    reactants_str, products_str = rxn_string.split(" \u27F9 ")
    
    where_surface = "reactants" if any(isinstance(inter, SurfaceSite) for inter in step.reactants) else "products"
    
    diagram = ED()
    diagram.add_level(0, format_reaction(reactants_str))
    diagram.add_level(round(step.e_act, 2), "TS", color="r")
    diagram.add_level(round(step.e_rxn, 2), format_reaction(products_str))
    diagram.add_link(0, 1)
    diagram.add_link(1, 2)
    
    diagram.plot(ylabel="Energy / eV")
    plt.title(format_reaction(step.repr_hr), fontname="DejaVu Sans", fontweight="bold", y=1.05)
    
    artists = diagram.fig.get_default_bbox_extra_artists()
    
    # Warning: Hardcoded indices (2, 3, 11) depend strictly on PyEnergyDiagrams' internal plotting order.
    size = artists[2].get_position()[0] - artists[3].get_position()[0]
    ap_reactants = (artists[3].get_position()[0], artists[3].get_position()[1] + 0.15)
    ap_products = (artists[11].get_position()[0], artists[11].get_position()[1] + 0.15)

    def _place_images(species_list, base_pos, is_surface_side, tmp_dir, prefix):
        counter = 0
        for i, inter in enumerate(species_list):
            if isinstance(inter, SurfaceSite):
                continue
                
            fig_path = os.path.join(tmp_dir, f"{prefix}_{i}.png")
            write(fig_path, inter.molecule, show_unit_cell=0)
            
            arr_img = plt.imread(fig_path)
            im = OffsetImage(arr_img)
            
            y_offset = counter if is_surface_side else i
            ab = AnnotationBbox(
                im,
                (base_pos[0] + size / 2, base_pos[1] + size * (0.5 + y_offset)),
                frameon=False,
            )
            diagram.ax.add_artist(ab)
            
            if is_surface_side:
                counter += 1

    with tempfile.TemporaryDirectory() as tmp_dir:
        _place_images(step.reactants, ap_reactants, where_surface == "reactants", tmp_dir, "reactant")
        _place_images(step.products, ap_products, where_surface == "products", tmp_dir, "product")

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
        x(Intermediate): Intermediate species.
    """
    if isinstance(x, AdsorbedSpecies):
        configs = [
            config["ase"]
            for config in x.ads_configs.values()
        ]
        if len(configs) == 0 and type(configs[0]) == Atoms:
            view(x.molecule)
        else:
            view(configs)
    elif isinstance(x, GasSpecies):
        view(x.molecule)
