#!/usr/bin/env python3

"""
@authors: ChatGPT, Aurelio Vivas <aa.vivas@uniandes.edu.co>
"""

import os
import yaml
import pandas as pd
from tqdm import tqdm
import argparse
from nflows_tools import profile_compact
from nflows_tools import profile_graph

from common import flatten_dict

def collect_profiles(root_dir, mode, rel_lat_matrix, time_unit, payload_unit, edge_strategy=None):
    """Collects profiling results from YAML files and aggregates them into a DataFrame."""
    yaml_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".yaml") or file.endswith(".yml"):
                yaml_files.append(os.path.join(root, file))

    print(f"Found {len(yaml_files)} YAML files to process.\n")

    records = []

    for input_file in tqdm(yaml_files, desc="Processing YAML files", unit="file"):
        parts = input_file.split(os.sep)

        try:
            workflow_name = parts[-4]
            algorithm_and_cores = parts[-3]
            config = parts[-2]
            file_name = parts[-1]
        except IndexError:
            print(f"Malformed path: '{input_file}'")
            continue

        with open(input_file, "r") as file:
            data = yaml.load(file, Loader=yaml.FullLoader)

        try:
            # --- Unified selection of profiling mode ---
            if mode == "graph":
                output = profile_graph.build_profile(data, edge_strategy, time_unit, payload_unit)
                output_scalars = output[0]
                output_scalars.pop("levels")

                output_data = flatten_dict(output_scalars)
            elif mode == "compact":
                output = profile_compact.build_profile(data, rel_lat_matrix, time_unit, payload_unit)
                output_scalars = output[0]

                output_data = flatten_dict(output_scalars)
            else:
                raise ValueError(f"Unknown mode '{mode}'. Expected 'graph' or 'compact'.")

            # --- Common metadata ---
            output_data["workflow_name"] = workflow_name
            output_data["algorithm_and_cores"] = algorithm_and_cores
            output_data["config"] = config
            output_data["file_name"] = file_name

            records.append(output_data)

        except Exception as e:
            print(f"Error processing {input_file}: {e}")

    df = pd.DataFrame(records)
    return df

def main():
    parser = argparse.ArgumentParser(description="Collect and aggregate profiling results from YAML files.")
    parser.add_argument("root_dir", help="Root directory containing YAML profiling results.")
    parser.add_argument("--mode", choices=["graph", "compact"], required=True, help="Choose profiling mode.")
    parser.add_argument("--edge_strategy", default="combined", help="Edge strategy for graph mode.")
    parser.add_argument("--input_file_rel_lat", default=None, help="Path to relative latency matrix (for compact mode).")
    parser.add_argument("--time_unit", type=str, choices=['us', 'ms', 's', 'min'], default='us', help="Time unit for scaling.")
    parser.add_argument("--payload_unit", type=str, choices=['B','KB','MB','GB'], default='B', help="Payload unit for scaling.")
    parser.add_argument("--output", default="summary.csv", help="Output CSV file.")

    args = parser.parse_args()

    if args.input_file_rel_lat:
        with open(args.input_file_rel_lat, 'r') as file:
            dimension = int(file.readline().strip())
            rel_lat_matrix = pd.DataFrame([list(map(float, file.readline().split())) for _ in range(dimension)])
    else:
        rel_lat_matrix = pd.DataFrame()

    # --- Run selected mode ---
    df = collect_profiles(
        args.root_dir,
        mode=args.mode,
        rel_lat_matrix=rel_lat_matrix,
        time_unit=args.time_unit,
        payload_unit=args.payload_unit,
        edge_strategy=args.edge_strategy,
    )

    # --- Save and report ---
    df.to_csv(args.output, index=False)
    print(f"Results saved to '{args.output}' ({len(df)} records).")

if __name__ == "__main__":
    main()
