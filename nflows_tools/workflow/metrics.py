#!/usr/bin/env python3
"""
DAG Structure Analyzer
======================
Reads a Directed Acyclic Graph (DAG) from a DOT file and computes
various structural and regularity metrics that describe its topology.

By default, artificial "root" and "end" nodes and their adjacent edges
are excluded from the calculations.

Usage (CLI):
------------
python dag_metrics.py --dot_file example.dot --metrics all
python dag_metrics.py --dot_file example.dot --metrics regularity --include-root-end
"""

import argparse
import networkx as nx
import pandas as pd
import numpy as np
from math import log2
import math
from statistics import variance, mean, pstdev
from networkx.algorithms.isomorphism import GraphMatcher


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

def structural_entropy(G):
    """Entropy of degree distribution (in+out).
    High entropy → irregular structure; Low → regular.
    
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
    return -sum(p * log2(p) for p in probs)


def level_regularization(G):
    """Regularity across topological levels of DAG.
    TODO: Needs review.
    """
    levels = {}
    for node in nx.topological_sort(G):
        preds = list(G.predecessors(node))
        levels[node] = 0 if not preds else max(levels[p] for p in preds) + 1
    counts = np.array(list(np.unique(list(levels.values()), return_counts=True)[1]))
    return 1 / (1 + np.std(counts) / np.mean(counts))


def degree_regularization(G):
    """Measures how uniform in/out-degree distribution is (1 = perfectly regular).
    TODO: Needs review.
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


# ================================================================
# ------------------------- Symmetry -----------------------------
# ================================================================

def symmetry_score(G):
    """Estimates structural symmetry based on automorphism orbits.
    TODO: Needs review.
    """
    UG = G.to_undirected()
    matcher = GraphMatcher(UG, UG)
    # Get all automorphisms (expensive for large graphs)
    auts = list(matcher.isomorphisms_iter())
    n = len(auts)
    return np.log10(n + 1) / np.log10(len(G) + 1)

def approximate_symmetry(G):
    """Fraction of nodes sharing identical structural signatures.
    TODO: Needs review.
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


# ================================================================
# ---------------------- Entry Point Function --------------------
# ================================================================

def preprocess_graph(G, include_root_end=False):
    """Remove artificial root and end nodes if requested."""
    if include_root_end:
        return G.copy()

    # Explicitly remove nodes named "root" or "end" (case-insensitive)
    named_nodes = [n for n in G.nodes() if str(n).lower() in {"root", "end"}]

    # Create a copy and remove these nodes
    H = G.copy()
    H.remove_nodes_from(named_nodes)
    return H


def compute_metrics(G, selected_groups='all', include_root_end=False):
    """Compute requested metric groups and return a pandas Series."""
    G = preprocess_graph(G, include_root_end=include_root_end)
    results = {}

    if selected_groups in ('all', 'basic'):
        results.update({
            'num_vertices': num_vertices(G),
            'num_edges': num_edges(G),
            'edge_density': edge_density(G),
            'avg_in_degree': average_degree(G)[0],
            'avg_out_degree': average_degree(G)[1],
        })

    if selected_groups in ('all', 'hierarchical'):
        width, height, sf = shape_factor(G)
        results.update({
            'width': width,
            'height': height,
            'shape_factor': sf,
        })

    if selected_groups in ('all', 'regularity'):
        results.update({
            'structural_entropy': structural_entropy(G),
            'level_regularization': level_regularization(G),
            'degree_regularization': degree_regularization(G),
        })

    if selected_groups in ('all', 'symmetry'):
        results.update({
            'symmetry_score': symmetry_score(G),
            'approx_symmetry': approximate_symmetry(G),
        })

    return pd.Series(results)


# ================================================================
# ------------------------- CLI Interface ------------------------
# ================================================================

def main():
    parser = argparse.ArgumentParser(description="Compute structural metrics for a DAG (.dot).")
    parser.add_argument('--dot_file', required=True, help='Path to the input DOT file.')
    parser.add_argument('--metrics', default='all', help='Metric groups: all,basic,hierarchical,fanin_fanout,entropy,regularity')
    parser.add_argument('--include-root-end', action='store_true', help='Include root and end nodes in calculations (default: False).')
    args = parser.parse_args()

    # Load graph
    G = nx.drawing.nx_pydot.read_dot(args.dot_file)
    G = nx.DiGraph(G)  # Ensure directed
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("Input graph must be a DAG.")

    # Compute metrics
    result = compute_metrics(G, selected_groups=args.metrics, include_root_end=args.include_root_end)
    print(result.to_string())


if __name__ == "__main__":
    main()
