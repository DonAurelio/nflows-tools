#!/usr/bin/env python3

"""
@authors: DeepSeek
@edited_by: Aurelio Vivas
@promt:
"""

import argparse
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from tabulate import tabulate

def find_data_distribution_nodes(digraph, min_fanout=2):
    """
    Find nodes that distribute data to multiple destinations (one-to-many).
    Returns a list of tuples: (focal_node, [origin_nodes], [destination_nodes])
    """
    distributors = []

    for node in digraph.nodes():
        out_degree = digraph.out_degree(node)

        if out_degree >= min_fanout:
            origins = []
            destinations = list(digraph.successors(node))
            distributors.append((node, origins, destinations))

    return distributors

def find_data_redistribution_nodes(digraph, min_degree=2):
    """
    Find data redistribution points with their origins and destinations.
    Returns a list of tuples: (focal_node, [origin_nodes], [destination_nodes])
    """
    redistributions = []

    for node in digraph.nodes():
        in_degree = digraph.in_degree(node)
        out_degree = digraph.out_degree(node)

        if in_degree >= min_degree and out_degree >= min_degree:
            origins = list(digraph.predecessors(node))
            destinations = list(digraph.successors(node))
            redistributions.append((node, origins, destinations))

    return redistributions

def find_data_aggregation_nodes(digraph, min_fanin=2):
    """
    Find nodes that aggregate data from multiple sources (many-to-one).
    Returns a list of tuples: (focal_node, [origin_nodes], [destination_nodes])
    """
    aggregators = []

    for node in digraph.nodes():
        in_degree = digraph.in_degree(node)
        
        if in_degree >= min_fanin:
            origins = list(digraph.predecessors(node))
            destinations = []
            aggregators.append((node, origins, destinations))

    return aggregators

def find_data_structures(digraph, patterns=None, **kwargs):
    """
    Find specified data processing microstructures in a graph.

    Parameters:
    -----------
    digraph : DiGraph
        The input directed graph
    patterns : list of str, optional
        List of patterns to find. If None, finds all patterns.
        Possible values: "distribution", "redistribution", "aggregation"
    **kwargs : 
        Additional parameters to pass to individual pattern finders:
        - min_fanout: For distribution patterns (default: 2)
        - min_degree: For redistribution patterns (default: 2)
        - min_fanin: For aggregation patterns (default: 2)
        - min_length: For pipeline patterns (default: 3)
        - min_size: For process patterns (default: 3)

    Returns:
    --------
    dict
        Dictionary containing found patterns (only requested patterns are included)
    """
    # Define all available pattern finders
    pattern_finders = {
        "distribution": lambda: find_data_distribution_nodes(
            digraph, 
            min_fanout=kwargs.get('min_fanout', 2)
        ),
        "redistribution": lambda: find_data_redistribution_nodes(
            digraph, 
            min_degree=kwargs.get('min_degree', 2)
        ),
        "aggregation": lambda: find_data_aggregation_nodes(
            digraph, 
            min_fanin=kwargs.get('min_fanin', 2)
        ),
    }

    # If no patterns specified, use all available patterns
    patterns = pattern_finders.keys() if patterns is None else patterns

    # Find requested patterns
    results = {pattern: pattern_finders[pattern]() for pattern in patterns}

    return results

def get_clean_subgraph(digraph, node, origins, destinations):
    # Ensure the source is a MultiDiGraph
    assert isinstance(digraph, nx.MultiDiGraph), "Source graph must be a MultiDiGraph"

    # Create the destination graph as a plain DiGraph
    cleaned_graph = nx.DiGraph()

    # Add focal node with its attributes
    cleaned_graph.add_node(node, **digraph.nodes[node])

    # Add edges from origins to focal node
    for origin in origins:
        if digraph.has_edge(origin, node):
            cleaned_graph.add_node(origin, **digraph.nodes[origin])
            # Pick the first edge's attributes (could also merge if needed)
            key, attrs = next(iter(digraph[origin][node].items()))
            cleaned_graph.add_edge(origin, node, **attrs)

    # Add edges from focal node to destinations
    for dest in destinations:
        if digraph.has_edge(node, dest):
            cleaned_graph.add_node(dest, **digraph.nodes[dest])
            key, attrs = next(iter(digraph[node][dest].items()))
            cleaned_graph.add_edge(node, dest, **attrs)

    return cleaned_graph

# Helper function to get edge size safely
def get_edge_size(u, v):
    try:
        if graph.is_multigraph():
            # For multigraphs, get sum of all parallel edges' sizes
            return sum(int(data.get('size')) for data in graph[u][v].values())
        else:
            return int(graph.edges[u, v].get('size'))
    except KeyError:
        raise ValueError(f"Edge size not found for edge ({u}, {v})")

def get_dataframe(graph, pattern_results, sort_by=None, sort_by_ascending=False, slice=":"):
    """
    Create a DataFrame from pattern results with node, origin_count, and destination_count,
    along with computation and communication metrics.
    
    Parameters:
    -----------
    pattern_results : list of tuples: (focal_node, [origin_nodes], [destination_nodes])
        The results of the pattern search.
    graph : networkx.DiGraph
        The directed graph containing node and edge attributes
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with columns: node, in, out, flops, bytes, in_bytes, out_bytes
    """
    rows = []

    for focal_node, origins, destinations in pattern_results:
        flops = int(graph.nodes[focal_node].get('size'))
        flops += sum(int(graph.nodes[o].get('size')) for o in origins)
        flops += sum(int(graph.nodes[d].get('size')) for d in destinations)

        in_edge_bytes = sum(get_edge_size(o, focal_node) for o in origins)
        out_edge_bytes = sum(get_edge_size(focal_node, d) for d in destinations)
        bytes = in_edge_bytes + out_edge_bytes

        rows.append({
            'node': focal_node,
            'in': len(origins),
            'out': len(destinations),
            'flops': flops,
            'bytes': bytes,
            'in_bytes': in_edge_bytes,
            'out_bytes': out_edge_bytes
        })

    start, end = map(lambda x: int(x.strip()) if x else None, slice.split(':'))
    df = pd.DataFrame(rows).sort_values(by=sort_by, ascending=sort_by_ascending).reset_index(drop=True)[start:end]

    return df

def print_dataframe(data, title):
    print(f"\n{title.replace('_', ' ').title()}: {data.shape[0]} found")
    table_str = tabulate(data, headers=data.columns, tablefmt="grid", showindex=True)
    print("\n" + "\n".join(f"  {line}" for line in table_str.split("\n")))

def analyze_structures(graph, patterns, **kwargs):
    """Analyze the graph for specified patterns."""
    return find_data_structures(graph, patterns=patterns, **kwargs)

def display_structures(graph, results, **kwargs):
    """Display analysis results in the console."""
    for pattern_type, pattern_results in results.items():
        if pattern_results:
            df = get_dataframe(graph, pattern_results, **kwargs)
            print_dataframe(df, title=pattern_type)

def plot_dag(digraph, focal_node, output_file):
    """Improved DAG plotting with optimal node separation and clear labels."""
    plt.figure(figsize=(14, 10))
    
    # Assign layers using topological generations
    layers = list(nx.topological_generations(digraph))
    for layer, nodes in enumerate(layers):
        for node in nodes:
            digraph.nodes[node]["layer"] = layer
    
    # Compute initial multipartite layout
    pos = nx.multipartite_layout(
        digraph,
        subset_key="layer",
        align='horizontal',
        scale=2.0,
        center=None
    )
    
    # Enhanced node separation algorithm
    for layer_idx, layer_nodes in enumerate(layers):
        # Sort nodes by x-position to space them evenly
        sorted_nodes = sorted(layer_nodes, key=lambda x: pos[x][0])
        
        # Calculate horizontal spacing based on node count
        layer_width = len(layer_nodes) * 1.2  # Base spacing multiplier
        x_start = -layer_width/2
        
        # Apply new x-positions with better spacing
        for i, node in enumerate(sorted_nodes):
            pos[node][0] = x_start + i * (layer_width/max(1, len(layer_nodes)-1))
            
        # Increase vertical spacing between layers
        for node in layer_nodes:
            pos[node][1] = -layer_idx * 2.5  # Increased vertical spacing
    
    # Dynamic node sizing and styling
    max_label_len = max(len(str(node)) for node in digraph.nodes)
    node_sizes = [800 + 200 * (len(str(node))/max_label_len) for node in digraph.nodes]

    # Add subtle grid lines for layer alignment
    # for layer_idx in range(len(layers)):
    #     plt.axhline(y=-layer_idx*2.5, color='lightgray', linestyle='--', alpha=0.3, zorder=-1)

    # Draw all nodes without labels
    nx.draw_networkx_nodes(
        digraph,
        pos=pos,
        node_size=node_sizes,
        node_color='#1f78b4',
        edgecolors='black',
        linewidths=1,
        alpha=0.95
    )
    
    # Draw all edges
    nx.draw_networkx_edges(
        # digraph,
        # pos=pos,
        # width=1.5,
        # edge_color='#666666',
        # arrowsize=20
        digraph,
        pos,
        arrows=True,
        arrowstyle='-|>',  # Clean arrow style
        arrowsize=15,
        edge_color='#666666',
        width=1.5,
        node_size=node_sizes,  # Must match node_size above!
        connectionstyle='arc3,rad=0.1',  # Curved edges
    )
    
    # Draw ONLY the focus node's label
    nx.draw_networkx_labels(
        digraph,
        pos=pos,
        labels={focal_node: focal_node},  # Only show this label
        font_size=10,
        font_weight='bold',
    )
    
    plt.axis('off')
    plt.tight_layout()

    # # Add this before plt.savefig() for very large graphs
    # if len(digraph.nodes) > 50:
    #     plt.figure(figsize=(20, 14))
    #     nx.draw_networkx(..., font_size=8, node_size=600)
    
    # Save with professional quality
    plt.savefig(
        output_file,
        format=output_file.split('.')[-1],
        dpi=300,
        bbox_inches='tight',
        transparent=False,
    )
    plt.close()
    print(f"Figure created: '{output_file}'.")

def save_structure(graph, structures, focal_node, output_file, add_root_end=False):
    """Save the specified pattern and node to an output file."""
    selected_structure = None
    file_format = output_file.split('.')[-1]

    # selected_structure = next((structure for structure in structures if focal_node == structure[0]), None)

    for structure in structures:
        if focal_node in structure[0]:
            selected_structure = structure
            break
    else:
        raise ValueError(f"Node '{focal_node}' not found.")

    nodes =  [selected_structure[0], *selected_structure[1], *selected_structure[2]]

    # Create subgraph for the selected structure
    digraph = get_clean_subgraph(graph.subgraph(nodes).copy(), *selected_structure)

    # Add root and end nodes if requested
    if add_root_end:
        # Add root node connected to tasks with no parents
        root_node = "root"
        digraph.add_node(root_node, size=2)
        for node in digraph.nodes():
            if node != root_node and not any(True for _ in digraph.predecessors(node)):
                digraph.add_edge(root_node, node, size=2)

        # Add end node connected to tasks with no children
        end_node = "end"
        digraph.add_node(end_node, size=2)
        for node in digraph.nodes():
            if node != end_node and not any(True for _ in digraph.successors(node)):
                digraph.add_edge(node, end_node, size=2)

    # Save in appropriate format
    if file_format == 'dot':
        nx.drawing.nx_pydot.write_dot(digraph, output_file)
        print(f"Structure saved to '{output_file}'")
        return

    plot_dag(digraph, focal_node, output_file)

def main():
    parser = argparse.ArgumentParser(description="Analyze graph microstructures.")
    parser.add_argument("input_dot_file", help="Path to input DOT file")
    parser.add_argument("--sort_by", nargs='+', default=['node'], choices=['node', 'in', 'out', 'flops', 'bytes', 'in_byes', 'out_bytes'], help="Sort criteria: specify one or more of 'node', 'in', 'out'.")
    parser.add_argument("--sort_by_ascending", action="store_true", help="Sort values in ascending order (default: False).")
    parser.add_argument("--slice", type=str, default=":", help="Slice rows using 'start:end' format (e.g., '0:10').")
    parser.add_argument("--output_file", help="Path to output file (DOT, PNG, PDF, etc.)")
    parser.add_argument("--add_root_end", action="store_true", help="Add 'root' and 'end' nodes to the graph. (default: False).")
    parser.add_argument("--patterns", nargs='+', choices=['distribution', 'redistribution', 'aggregation'], help="Patterns to analyze (default: all)")
    parser.add_argument("--pattern", choices=['distribution', 'redistribution', 'aggregation'], help="Single pattern for output (required with --output_file)")
    parser.add_argument("--node", type=str, help="Specific node to visualize (required with --output_file)")

    args = parser.parse_args()

    if args.output_file and (not args.pattern or not args.node):
        parser.error("--output_file requires both --pattern and --node arguments")
    
    # Load and analyze the graph
    graph = nx.nx_pydot.read_dot(args.input_dot_file)

    for u, v in graph.edges():
        if len(graph[u][v].values()) != 1:
            raise ValueError(f"Graph is not a simple graph: multiple edges between {u} and {v}")

    results = analyze_structures(graph, [args.pattern] if args.pattern else args.patterns)

     # Handle output based on arguments
    if not args.output_file:
        display_structures(graph, results, sort_by=args.sort_by, sort_by_ascending=args.sort_by_ascending, slice=args.slice)
    else:
        save_structure(graph, results[args.pattern], args.node, args.output_file, add_root_end=args.add_root_end)

if __name__ == "__main__":
    main()
