#!/usr/bin/env python3

"""
@authors: ChatGPT, Aurelio Vivas <aa.vivas@uniandes.edu.co>
"""

import yaml
import pandas as pd
import numpy as np
from math import log2
import networkx as nx
import argparse

from sklearn.cluster import KMeans
from itertools import combinations
from networkx.algorithms.isomorphism import GraphMatcher
from networkx.drawing.nx_pydot import write_dot
from collections import defaultdict, deque
from tabulate import tabulate

def scale_time(value, unit):
    scale_factors = {'us': 1, 'ms': 1e3, 's': 1e6, 'min': 6e7}
    return float(value) / float(scale_factors[unit])

def scale_payload(value, unit):
    scale_factors = {'B': 1, 'KB': 1e3, 'MB': 1e6, 'GB': 1e9}
    return float(value) / float(scale_factors[unit])

def build_digraph(data, edge_strategy='combined', time_unit='us', payload_unit='B', **kwargs):
    G = nx.DiGraph()

    trace = data['trace']
    tasks = trace['exec_name_compute_offsets']
    name_to_thread_locality = trace['name_to_thread_locality']
    numa_mappings_write = trace.get('numa_mappings_write', {})
    numa_mappings_read = trace.get('numa_mappings_read', {})

    # ---- Add vertices (tasks) with NUMA and core locality ----
    for task_name, props in tasks.items():
        start, end = float(props['start']), float(props['end'])
        payload = float(props['payload'])

        locality = name_to_thread_locality.get(task_name, {})
        numa_id = locality['numa_id']
        core_id = locality['core_id']

        G.add_node(
            task_name,
            payload=scale_payload(payload, payload_unit),
            start=scale_time(start, time_unit),
            end=scale_time(end, time_unit),
            dur=scale_time(end - start, time_unit),
            numa_id=numa_id,
            core_id=core_id
        )

    write_offsets = trace['comm_name_write_offsets']
    read_offsets = trace['comm_name_read_offsets']

    # ---- Helper: determine locality for write or read edges ----
    def determine_locality(edge_key, edge_type, u, v):
        if edge_type == 'write':
            mapping = numa_mappings_write.get(edge_key)
            numa = G.nodes[u].get('numa_id', None)
        else:
            mapping = numa_mappings_read.get(edge_key)
            numa = G.nodes[v].get('numa_id', None)

        if not mapping:
            return None  # no locality data

        numa_ids = mapping.get('numa_ids', [])
        if not numa_ids:
            return None
            
        if len(set(numa_ids)) > 1:
            return 'mixed'
        elif numa is None:
            return None
        elif numa_ids[0] == numa:
            return 'local'
        else:
            return 'remote'

    # ---- Add edges according to strategy ----
    if edge_strategy == 'write':
        for edge_key, props in write_offsets.items():
            u, v = edge_key.split("->")
            start, end = float(props['start']), float(props['end'])
            payload = float(props['payload'])

            write_locality = determine_locality(edge_key, 'write', u, v)

            G.add_edge(
                u, v,
                write_payload=scale_payload(payload, payload_unit),
                write_start=scale_time(start, time_unit),
                write_end=scale_time(end, time_unit),
                write_dur=scale_time(end - start, time_unit),
                write_locality=write_locality,
                dur=scale_time(end - start, time_unit)
            )

    elif edge_strategy == 'read':
        for edge_key, props in read_offsets.items():
            u, v = edge_key.split("->")
            start, end = float(props['start']), float(props['end'])
            payload = float(props['payload'])

            read_locality = determine_locality(edge_key, 'read', u, v)

            G.add_edge(
                u, v,
                read_payload=scale_payload(payload, payload_unit),
                read_start=scale_time(start, time_unit),
                read_end=scale_time(end, time_unit),
                read_dur=scale_time(end - start, time_unit),
                read_locality=read_locality,
                dur=scale_time(end - start, time_unit)
            )

    elif edge_strategy == 'combined':
        common_keys = set(write_offsets) & set(read_offsets)
        for edge_key in common_keys:
            write = write_offsets[edge_key]
            read = read_offsets[edge_key]

            start = float(write['start'])
            end = float(read['end'])
            dur_with_wait = end - start

            # The variable wait represents the idle or waiting time that 
            # occurs between the end of the write phase and the beginning 
            # of the read phase for a communication edge.
            dur_write = float(write['end']) - float(write['start'])
            dur_read = float(read['end']) - float(read['start'])
            dur_without_wait = dur_write + dur_read
            wait = dur_with_wait - dur_without_wait

            u, v = edge_key.split("->")
            write_payload = float(write['payload'])
            read_payload = float(read['payload'])
            payload = float(write['payload']) + float(read['payload'])

            write_locality = determine_locality(edge_key, 'write', u, v)
            read_locality = determine_locality(edge_key, 'read', u, v)

            G.add_edge(
                u, v,

                write_payload=scale_payload(write_payload, payload_unit),
                write_start=scale_time(float(write['start']), time_unit),
                write_end=scale_time(float(write['end']), time_unit),
                write_dur=scale_time(dur_write, time_unit),
                write_locality=write_locality,

                read_payload=scale_payload(read_payload, payload_unit),
                read_start=scale_time(float(read['start']), time_unit),
                read_end=scale_time(float(read['end']), time_unit),
                read_dur=scale_time(dur_read, time_unit),
                read_locality=read_locality,

                start=scale_time(start, time_unit),
                end=scale_time(end, time_unit),
                dur=scale_time(dur_without_wait, time_unit),
                wait=scale_time(wait, time_unit),
            )

            # The desition to consider dur_without_wait as main duration is based on the idea that
            # First the critical path is the longest path through the network that establioshes the "minimum" time
            # overall proyect duration. That minimum time do not consider the waiting times.
            # Any waiting time will increase this minimum, becoming the makespan, maximum task compeltion time.
            # Reference: Chapter 1 - Project Scheduling (Theodore J.  et al) - Redefining the Critical Path
            # DOI: https://doi.org/10.1016/B978-1-85617-677-4.00001-5

    return G

# ================================================================
# --------------------- Basic Structural Metrics -----------------
# ================================================================

def num_vertices(G):
    """Return number of vertices (|V|).
    Meaning: Reflects workflow size or granularity."""
    return G.number_of_nodes()

def num_edges(G):
    """Return number of edges (|E|).
    Meaning: Indicates number of dependencies."""
    return G.number_of_edges()

def edge_density(G):
    """Compute edge density = |E| / [|V|*(|V|-1)].
    High value → more dependencies, denser DAG."""
    n = G.number_of_nodes()
    return 0 if n <= 1 else G.number_of_edges() / (n * (n - 1))

def average_degree(G):
    """Return average in-degree and out-degree.
    High value → more interdependent tasks.
    High fan-in → synchronization points; High fan-out → branching.
    """
    indeg = np.mean([d for _, d in G.in_degree()])
    outdeg = np.mean([d for _, d in G.out_degree()])
    return indeg, outdeg

def nodes_by_level(G):
    """Create a tree-like hierarchical layout for a DAG."""

    # Compute levels using topological sort
    levels = defaultdict(int)
    in_degree = {n: 0 for n in G.nodes}
    for u, v in G.edges:
        in_degree[v] += 1

    # BFS to assign levels
    queue = deque([n for n in G.nodes if in_degree[n] == 0])
    while queue:
        node = queue.popleft()
        for neighbor in G.successors(node):
            levels[neighbor] = max(levels[neighbor], levels[node] + 1)
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # Group nodes by level
    level_nodes = defaultdict(list)
    for node, level in levels.items():
        level_nodes[level].append(node)

    results = {}
    for level in sorted(level_nodes.keys()):
        nodes_at_level = level_nodes[level]
        results[level] = len(nodes_at_level)

    return results

# ================================================================
# ------------------- Hierarchical / Topological -----------------
# ================================================================

def topological_depth(G):
    """
    Compute the longest path length (height) in a DAG.

    Meaning:
    - High value → long sequential chain (deep workflow)
    - Low value → shallow DAG (more parallelism)
    """
    if G.number_of_nodes() == 0:
        return 0

    # Initialize distance for each node
    dist = {v: 0 for v in G.nodes()}

    # Traverse in topological order
    for v in nx.topological_sort(G):
        for succ in G.successors(v):
            dist[succ] = max(dist[succ], dist[v] + 1)

    return max(dist.values()) if dist else 0

def topological_width(G):
    """Compute maximum width (max number of nodes in a topological level).
    High value → high potential parallelism."""
    levels = list(nx.topological_generations(G))
    return max(len(lvl) for lvl in levels)

def shape_factor(G):
    """Return (width, height) and shape factor = width / height.
    High value → flat DAG; Low value → deep DAG."""
    height = topological_depth(G)
    width = topological_width(G)
    return width, height, width / height if height > 0 else np.nan

# ================================================================
# ----------------------- Regularity -----------------------------
# ================================================================

def degree_entropy_regularity(G):
    """Entropy of degree distribution (in+out).

    | Regularity score (R) | Structural meaning | Description                                           |
    | -------------------- | ------------------ | ----------------------------------------------------- |
    | **≈ 1.0**            | Perfectly regular  | All nodes have identical degrees → minimum entropy.   |
    | **≈ 0.5**            | Moderately regular | Degree distribution shows some variation.             |
    | **≈ 0.0**            | Highly irregular   | Degree probabilities are widely spread (max entropy). |
    
    Shannon entropy formula:
    H = - Σ (p_i * log2(p_i))
    where p_i is the probability of degree d_i.
    TODO: I checked this and sees like it is correct. The ecuation is mentioned in literature.
    I just need to double check the implementation and find the defining paper.
    """
    degs = [G.in_degree(n) + G.out_degree(n) for n in G.nodes()]
    total = sum(degs)
    if total == 0:
        return 0
    probs = [d / total for d in degs if d > 0]
    H = -sum(p * log2(p) for p in probs)
    return 1 - (H / log2(len(probs)))

def degree_distribution_regularity(G):
    """Measures how uniform in/out-degree distribution is (1 = perfectly regular).
    A graph is regular if every node has the same in-degree and out-degree.
    This metric quantifies how close the graph is to that ideal.
    TODO: Seems coherent with the concept of regularity.
    """
    indeg = np.array([d for _, d in G.in_degree()])
    outdeg = np.array([d for _, d in G.out_degree()])
    total_deg = indeg + outdeg
    if len(G) == 0 or total_deg.sum() == 0:
        return 1.0
    # Coefficient of variation: std / mean (lower = more regular)
    cv = np.std(total_deg) / np.mean(total_deg)
    # Transform into [0,1] where 1 = perfectly regular
    return 1 / (1 + cv)

def level_distribution_regularity(G):
    """Regularity across topological levels of DAG.
    Quantify the balance of nodes across topological levels.
    - If each level has roughly the same number of nodes → high regularity (value close to 1).
    - If some levels are dense and others sparse → low regularity (value closer to 0).
    TODO: Seems coherent with the concept of regularity.
    """
    levels = {}
    for node in nx.topological_sort(G):
        preds = list(G.predecessors(node))
        levels[node] = 0 if not preds else max(levels[p] for p in preds) + 1
    counts = np.array(list(np.unique(list(levels.values()), return_counts=True)[1]))
    return 1 / (1 + np.std(counts) / np.mean(counts))

# ================================================================
# ------------------------- Symmetry -----------------------------
# ================================================================

def discriete_structural_symmetry(G):
    """Fraction of nodes sharing identical structural signatures.

    | Graph type                        | Expected value | Interpretation                               |
    | --------------------------------- | -------------- | -------------------------------------------- |
    | Linear chain (A→B→C→D)            | 0              | Every node has a distinct signature          |
    | Star (center→leaves)              | ≈ 0.8          | All leaves share identical signatures        |
    | Perfectly regular k×k lattice DAG | 1              | All nodes have identical structural patterns |

    - Range: [0, 1]
    - Not continuous: small structural changes can drop the score abruptly
    - Normalized meaning: yes, higher = more symmetric structure
    - Sensitive to graph size and discrete degree changes

    The current approximate_symmetry() is binary: two nodes are either identical or not, based on discrete equality 
    of their signatures. That’s fine for coarse symmetry detection, but it misses subtle similarities (e.g., two nodes 
    with nearly equal degrees or similar ancestor/descendant counts).
    """
    sigs = {}
    for n in G.nodes():
        sig = (
            G.in_degree(n),
            G.out_degree(n),
            len(nx.ancestors(G, n)),
            len(nx.descendants(G, n))
        )
        sigs.setdefault(sig, []).append(n)
    symmetric_pairs = sum(len(v) for v in sigs.values() if len(v) > 1)
    return symmetric_pairs / len(G)

def continuous_structural_symmetry(G):
    """Estimates structural symmetry in a continuous way.
    Returns a normalized score in [0,1].
    
    - 1 → perfectly symmetric structure
    - 0 → completely asymmetric

    | Value        | Meaning                                    |
    | ------------ | ------------------------------------------ |
    | **1.0**      | All nodes are structurally identical       |
    | **≈0.8–0.9** | Highly regular (like grids, stars)         |
    | **≈0.5**     | Some repeating patterns but not uniform    |
    | **≈0.0–0.2** | Irregular DAG with distinct roles per node |

    | Aspect         | Old `approximate_symmetry`  | New `continuous_symmetry`         |
    | -------------- | --------------------------- | --------------------------------- |
    | Comparison     | Binary (equal / not equal)  | Continuous (degree of similarity) |
    | Range          | [0, 1]                      | [0, 1]                            |
    | Sensitivity    | Discrete jumps              | Smooth variation                  |
    | Interpretation | Fraction of identical nodes | Average pairwise similarity       |
    """

    if len(G) <= 1:
        return 1.0

    # Feature vectors per node
    feats = []
    for n in G.nodes():
        feats.append([
            G.in_degree(n),
            G.out_degree(n),
            len(nx.ancestors(G, n)),
            len(nx.descendants(G, n))
        ])
    feats = np.array(feats, dtype=float)

    # Normalize features to avoid scale dominance
    if np.any(feats):
        feats = (feats - feats.min(axis=0)) / (np.ptp(feats, axis=0) + 1e-9)

    # Compute pairwise cosine similarity between all node signatures
    sims = []
    for i, j in combinations(range(len(G)), 2):
        a, b = feats[i], feats[j]
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        sims.append(np.dot(a, b) / denom if denom else 1.0)

    # Average similarity as overall symmetry measure
    return np.mean(sims)

def entropy_structural_symmetry(G, k=8, random_state=0):
    """Approximates structural symmetry via clustering of node signatures.

    | Symmetry score | Meaning                                            |
    | -------------- | -------------------------------------------------- |
    | **≈1.0**       | Most nodes share similar connectivity patterns     |
    | **≈0.7–0.8**   | Graph has repeated motifs or balanced structure    |
    | **≈0.3–0.5**   | Moderate diversity in node structures              |
    | **≈0.0–0.2**   | Highly irregular; few or no structural repetitions |
    
    Returns a normalized score in [0,1]:
    - 1 → highly symmetric (many nodes share similar structure)
    - 0 → highly asymmetric (each node structurally unique)
    
    Parameters:
    - k: number of clusters (higher captures more nuance)
    - random_state: ensures reproducibility
    """
    n = len(G)
    if n <= 1:
        return 1.0

    # --- Step 1: Build feature vectors ---
    feats = np.array([
        [G.in_degree(n), G.out_degree(n),
         len(nx.ancestors(G, n)), len(nx.descendants(G, n))]
        for n in G.nodes()
    ], dtype=float)

    # --- Step 2: Normalize features ---
    if np.any(np.ptp(feats, axis=0)):
        feats = (feats - feats.min(axis=0)) / (np.ptp(feats, axis=0) + 1e-9)

    # --- Step 3: Cluster node signatures ---
    k = min(k, n)  # can't have more clusters than nodes
    km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    labels = km.fit_predict(feats)

    # --- Step 4: Measure balance of cluster sizes ---
    counts = np.bincount(labels)
    probs = counts / n

    # Shannon entropy normalized by maximum possible entropy (log2(k))
    H = -np.sum(probs * np.log2(probs + 1e-12))
    H_norm = H / np.log2(k)

    # High entropy (uniform distribution) → many unique patterns (low symmetry)
    # Low entropy (few dense clusters) → high symmetry
    symmetry_score = 1 - H_norm
    return float(symmetry_score)

# ================================================================
# --------------------- Execution Metrics ------------------------
# ================================================================

def longest_path_with_nodes(G, edge_weight="dur", node_weight="dur"):
    """Compute the longest path in a DAG considering both edge and node weights."""
    if not nx.is_directed_acyclic_graph(G):
        raise nx.NetworkXError("Input graph must be a DAG.")

    # Initialize distance and predecessor maps
    dist = {}
    pred = {}

    # Process nodes in topological order
    for node in nx.topological_sort(G):
        node_cost = G.nodes[node][node_weight]

        # Compute best predecessor distance
        max_dist = 0
        best_pred = None
        for u in G.predecessors(node):
            edge_cost = G[u][node][edge_weight]
            cand = dist[u] + edge_cost
            if cand > max_dist:
                max_dist = cand
                best_pred = u

        dist[node] = max_dist + node_cost
        pred[node] = best_pred

    # Find the node with the maximum distance
    end_node = max(dist, key=dist.get)

    # Reconstruct the path
    path = []
    while end_node is not None:
        path.append(end_node)
        end_node = pred[end_node]
    path.reverse()

    # Calculate total duration
    node_attrs = []
    edge_attrs = []

    for i, node in enumerate(path):
        val = G.nodes[node][node_weight]
        node_attrs.append((node, val))

        if i < len(path) - 1:
            u, v = path[i], path[i+1]
            val = G[u][v][edge_weight]
            write_locality = G[u][v].get('write_locality', '-')
            write_dur = G[u][v].get('write_dur', 0)
            read_locality = G[u][v].get('read_locality', '-')
            read_dur = G[u][v].get('read_dur', 0)
            edge_attrs.append((f'{u}->{v}', val, write_locality, write_dur, read_locality, read_dur))

    return path, dist[path[-1]], node_attrs, edge_attrs

def longest_path_edges_locality_summary(edges_df):
    """
    Compute plain and weighted percentages of write/read locality types.

    Weighted percentages are based on their respective durations:
      - write_locality → weighted by 'write_dur'
      - read_locality  → weighted by 'read_dur'

    Returns:
        summary_df : pd.DataFrame with columns:
            ['write_% (plain)', 'write_% (weighted)',
             'read_% (plain)', 'read_% (weighted)']
    """

    def plain_percentage(series):
        counts = series.value_counts(dropna=False)
        total = counts.sum()
        return (counts / total).round(4)

    def weighted_percentage(df, col, dur_col):
        total = df[dur_col].sum()
        if total == 0:
            return pd.Series(dtype=float)
        return (
            df.groupby(col)[dur_col].sum() / total
        ).round(4)

    # ---- Plain percentages ----
    write_plain = plain_percentage(edges_df["write_locality"])
    read_plain = plain_percentage(edges_df["read_locality"])

    # ---- Weighted percentages ----
    write_weighted = weighted_percentage(edges_df, "write_locality", "write_dur")
    read_weighted = weighted_percentage(edges_df, "read_locality", "read_dur")

    # ---- Combine all results ----
    summary = pd.DataFrame({
        "write_% (plain)": write_plain,
        "write_% (weighted)": write_weighted,
        "read_% (plain)": read_plain,
        "read_% (weighted)": read_weighted,
    }).fillna(0)

    # Ensure consistent order
    summary = summary.reindex(["local", "remote", "mixed"]).fillna(0)

    return summary

# ================================================================
# ----------------------- Visualization --------------------------
# ================================================================

# def visualize_graph(G, critical_path_nodes=None, critical_path_edges=None):
#     pos = tree_layout(G)
#     plt.figure(figsize=(12, 8))

#     # Default all nodes and edges
#     all_nodes = set(G.nodes)
#     all_edges = set(G.edges)
    
#     # Critical path elements
#     cp_nodes = set(critical_path_nodes or [])
#     cp_edges = set(critical_path_edges or [])

#     # Draw non-critical nodes
#     nx.draw_networkx_nodes(G, pos, nodelist=list(all_nodes - cp_nodes), node_size=700,
#                            node_color='lightgray', edgecolors='black')
#     # Draw critical path nodes
#     nx.draw_networkx_nodes(G, pos, nodelist=list(cp_nodes), node_size=700,
#                            node_color='tomato', edgecolors='black')

#     # Labels for all nodes
#     nx.draw_networkx_labels(G, pos, font_size=9)

#     # Draw non-critical edges
#     nx.draw_networkx_edges(G, pos, edgelist=list(all_edges - cp_edges),
#                            edge_color='lightgray', arrows=True, arrowstyle='->', arrowsize=15)
#     # Draw critical path edges
#     nx.draw_networkx_edges(G, pos, edgelist=list(cp_edges),
#                            edge_color='red', width=2.5, arrows=True, arrowstyle='->', arrowsize=20)

#     # Optionally draw durations
#     edge_labels = {
#         (u, v): f"{d['dur']:.1f}" for u, v, d in G.edges(data=True)
#     }
#     nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)

#     plt.title("Task Graph")
#     plt.axis('off')
#     plt.tight_layout()
#     plt.show()

# ================================================================
# -------------------------- Reporting ---------------------------
# ================================================================

def print_dict(data, title):
    print(f"\n{title.replace('_', ' ').title()}")
    for key, value in data.items():
        if isinstance(value, str):
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value:.4f}")

def print_df(data, title):
    print(f"\n{title.replace('_', ' ').title()}: {eval(data[2])}")
    table_str = tabulate(data[0], headers=data[1], tablefmt="grid", showindex=True)
    print("\n" + "\n".join(f"  {line}" for line in table_str.split("\n")))

def print_profile(output_scalars, output_matrices):
    for key, value in output_scalars.items():
        print_dict(value, key)

    for key, value in output_matrices.items():
        print_df(value, key)
    print("")

def get_graph_profile(G):
    # ---- Nodes ----
    node_headers = ["node", "payload", "start", "end", "dur", "numa_id", "core_id"]
    node_table = []

    for n, d in G.nodes(data=True):
        node_table.append([
            n,
            d.get("payload", "-"),
            d.get("start", "-"),
            d.get("end", "-"),
            d.get("dur", "-"),
            d.get("numa_id", "-"),
            d.get("core_id", "-"),
        ])

    nodes_df = pd.DataFrame(node_table, columns=node_headers)

    # ---- Edges ----
    edge_headers = [
        "edge",
        "write_payload", "write_start", "write_end", "write_dur", "write_locality",
        "read_payload", "read_start", "read_end", "read_dur", "read_locality",
        "dur", "wait"
    ]
    edge_table = []

    for u, v, d in G.edges(data=True):
        edge_table.append([
            f"{u}->{v}",
            d.get("write_payload", "-"),
            d.get("write_start", "-"),
            d.get("write_end", "-"),
            d.get("write_dur", "-"),
            d.get("write_locality", "-"),
            d.get("read_payload", "-"),
            d.get("read_start", "-"),
            d.get("read_end", "-"),
            d.get("read_dur", "-"),
            d.get("read_locality", "-"),
            d.get("dur", "-"),
            d.get("wait", "-"),
        ])

    edges_df = pd.DataFrame(edge_table, columns=edge_headers)

    return nodes_df, edges_df

def build_profile(data, edge_strategy='combined', time_unit='us', payload_unit='B'):
    G = build_digraph(data, edge_strategy=edge_strategy, time_unit=time_unit, payload_unit=payload_unit)
    nodes_df, edges_df = get_graph_profile(G)

    path, total_dur, node_attrs, edge_attrs = longest_path_with_nodes(G)

    longest_path_nodes_df = pd.DataFrame(node_attrs, columns=["node", 'weight'])
    longest_path_edges_df = pd.DataFrame(edge_attrs, columns=["edge", 'weight', "write_locality", "write_dur", "read_locality", "read_dur"])

    longest_path_edges_locality_df = longest_path_edges_locality_summary(longest_path_edges_df)

    # # Extract entities
    # critical_nodes = [n for n, _ in node_data]
    # critical_edges = [(u, v) for u, v, _ in edge_data]

    # # Visualize
    # visualize_graph_with_critical_path(G, critical_path_nodes=critical_nodes, critical_path_edges=critical_edges)

    output_scalars = {
        "user": {
            "time_unit": time_unit,
            "payload_unit": payload_unit,
            "edge_strategy": edge_strategy,
        },
        "basic": {
            'num_vertices': num_vertices(G),
            'num_edges': num_edges(G),
            'edge_density': edge_density(G),
            'avg_in_degree': average_degree(G)[0],
            'avg_out_degree': average_degree(G)[1],
        },
        "hierarchical": {  
            'width': shape_factor(G)[0],
            'height': shape_factor(G)[1],
            'shape_factor': shape_factor(G)[2],
        },
        "regularity": {
            'degree_entropy': degree_entropy_regularity(G),
            'degree_distribution': degree_distribution_regularity(G),
            'level_distribution': level_distribution_regularity(G),
        },
        "symmetry": {
            "discrete_structural": discriete_structural_symmetry(G),
            "continuous_structural": continuous_structural_symmetry(G),
            "entropy_structural": entropy_structural_symmetry(G),
        },
        "levels": nodes_by_level(G),
        "critical_path": {
            "length": len(path),
            "duration": total_dur,
            "read_accesses_local_perc":  longest_path_edges_locality_df.loc["local",  "read_% (plain)"],
            "read_accesses_remote_perc": longest_path_edges_locality_df.loc["remote", "read_% (plain)"],
            "read_accesses_mixed_perc":  longest_path_edges_locality_df.loc["mixed",  "read_% (plain)"],

            "write_accesses_local_perc": longest_path_edges_locality_df.loc["local",  "write_% (plain)"],
            "write_accesses_remote_perc":longest_path_edges_locality_df.loc["remote", "write_% (plain)"],
            "write_accesses_mixed_perc": longest_path_edges_locality_df.loc["mixed",  "write_% (plain)"],

            "read_time_local_perc":      longest_path_edges_locality_df.loc["local",  "read_% (weighted)"],
            "read_time_remote_perc":     longest_path_edges_locality_df.loc["remote", "read_% (weighted)"],
            "read_time_mixed_perc":      longest_path_edges_locality_df.loc["mixed",  "read_% (weighted)"],

            "write_time_local_perc":     longest_path_edges_locality_df.loc["local",  "write_% (weighted)"],
            "write_time_remote_perc":    longest_path_edges_locality_df.loc["remote", "write_% (weighted)"],
            "write_time_mixed_perc":     longest_path_edges_locality_df.loc["mixed",  "write_% (weighted)"],
        },
    }

    output_matrices = {
        "longest_path_nodes": (longest_path_nodes_df, longest_path_nodes_df.columns, 'data[0].shape[0]'),
        "longest_path_edges": (longest_path_edges_df, longest_path_edges_df.columns, 'data[0].shape[0]'),
        "longest_path_edges_locality": (longest_path_edges_locality_df, longest_path_edges_locality_df.columns, 'data[0].shape[0]'),
        "nodes": (nodes_df, nodes_df.columns, 'data[0].shape[0]'),
        "edges": (edges_df, edges_df.columns, 'data[0].shape[0]'),
    }

    return output_scalars, output_matrices, G

# def print_graph(G):
#     # ---- Print nodes ----
#     print("\nNodes (Tasks):")
#     node_table = []
#     for n, d in G.nodes(data=True):
#         node_table.append([
#             n,
#             d.get('start'),
#             d.get('end'),
#             d.get('payload', '-'),
#             d.get('dur'),
#             d.get('numa_id', '-'),
#             d.get('core_id', '-')
#         ])

#     print(tabulate(
#         node_table,
#         headers=["Task", "Start", "End", "Payload", "Duration", "NUMA ID", "Core ID"]
#     ))

#     # ---- Print edges ----
#     print("\nEdges (Communications):")
#     edge_table = []
#     for u, v, d in G.edges(data=True):
#         row = [
#             f"{u}->{v}",
#             d.get('start'),
#             d.get('end'),
#             d.get('payload', '-'),
#             d.get('dur'),
#         ]
#         # Optional columns
#         if 'wait' in d:
#             row.append(d['wait'])
#         if 'write_locality' in d:
#             row.append(d['write_locality'])
#         if 'read_locality' in d:
#             row.append(d['read_locality'])
#         edge_table.append(row)

#     # Determine headers dynamically
#     headers = ["Edge", "Start", "End", "Payload", "Duration"]
#     if any('wait' in d for _, _, d in G.edges(data=True)):
#         headers.append("Wait")
#     if any('write_locality' in d for _, _, d in G.edges(data=True)):
#         headers.append("Write Locality")
#     if any('read_locality' in d for _, _, d in G.edges(data=True)):
#         headers.append("Read Locality")

#     print(tabulate(edge_table, headers=headers))

def main():
    parser = argparse.ArgumentParser(description="Process NUMA access data.")
    parser.add_argument("input_yaml", help="Path to input YAML profile file")
    parser.add_argument("--time_unit", type=str, choices=['us', 'ms', 's', 'min'], default='us', help="Time unit for scaling.")
    parser.add_argument("--payload_unit", type=str, choices=['B','KB', 'MB', 'GB'], default='KB', help="Payload unit for scaling.")
    parser.add_argument("--edge_strategy", type=str, choices=['write', 'read', 'combined'], default='combined', help="Edge strategy for building the graph.")
    args = parser.parse_args()

    with open(args.input_yaml, "r") as file:
        data = yaml.load(file, Loader=yaml.FullLoader)

    output_scalars, output_matrices, G = build_profile(data, args.edge_strategy, args.time_unit, args.payload_unit)
    print_profile(output_scalars, output_matrices)

if __name__ == "__main__":
    main()
