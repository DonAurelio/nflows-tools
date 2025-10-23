#!/usr/bin/env python3

"""
@authors: ChatGPT, Aurelio Vivas <aa.vivas@uniandes.edu.co>
"""

import os
import yaml
import pandas as pd
from tqdm import tqdm
import argparse
from nflows_tools.profile import profile_compute


def profiles_collect(root_dir, rel_lat_matrix, time_unit, payload_unit, labels_to_extract=None):
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
            workflow_name = parts[1]
            algorithm_and_cores = parts[2]
            config = parts[3]
            file_name = parts[-1]
        except IndexError:
            print(f"Malformed path: '{input_file}'")
            continue

        with open(input_file, "r") as file:
            data = yaml.load(file, Loader=yaml.FullLoader)

        try:
            output_scalars, output_matrices, output_data = profile_compute(data, rel_lat_matrix, time_unit, payload_unit)

            output_scalars["workflow_name"] = workflow_name
            output_scalars["algorithm_and_cores"] = algorithm_and_cores
            output_scalars["config"] = config
            output_scalars["file_name"] = file_name

            records.append(output_scalars)

        except Exception as e:
            print(f"Error processing {input_file}: {e}")

    df = pd.DataFrame(records)
    return df


def main():
    parser = argparse.ArgumentParser(description="Collect and aggregate profiling results from YAML files.")
    parser.add_argument("root_dir", help="Root directory containing YAML profiling results.")
    parser.add_argument("--output", default="summary.csv", help="Path to output CSV file.")
    parser.add_argument("--input_file_rel_lat", default=None, help="Path to input TXT file with hwloc/numactl relative latencies.")
    parser.add_argument("--time_unit", type=str, choices=['us', 'ms', 's', 'min'], default='us', help="Time unit for scaling.")
    parser.add_argument("--payload_unit", type=str, choices=['K', 'M', 'G'], default='', help="Payload unit for scaling.")

    args = parser.parse_args()

    if args.input_file_rel_lat:
        with open(args.input_file_rel_lat, 'r') as file:
            dimension = int(file.readline().strip())
            rel_lat_matrix = pd.DataFrame([list(map(float, file.readline().split())) for _ in range(dimension)])
    else:
        rel_lat_matrix = pd.DataFrame()

    # Run profiling collection
    df = profiles_collect(args.root_dir, rel_lat_matrix, args.time_unit, args.payload_unit)

    # Save and report
    df.to_csv(args.output, index=False)
    print(f"Results saved to '{args.output}' ({len(df)} records).")


if __name__ == "__main__":
    main()
