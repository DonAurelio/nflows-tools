#!/usr/bin/env python3

"""
@authors: ChatGPT, DeepSeek
@edited_by: Aurelio Vivas
@promt:
"""

import yaml
import pandas as pd
import numpy as np
import argparse

from tabulate import tabulate
from colorama import Fore, Style

def scale_time(value, unit):
    scale_factors = {'us': 1, 'ms': 1e3, 's': 1e6, 'min': 6e7}
    return float(value) / float(scale_factors[unit])

def scale_payload(value, unit):
    scale_factors = {'K': 1e3, 'M': 1e6, 'G': 1e9}
    return float(value) / float(scale_factors.get(unit, 1))

def get_machine_average_numa_factor(data):
    """
    Computes the average NUMA factor (ratio of remote to local memory access latency)
    for matrices of size n x n.
    """

    matrix = data["distance_lat_ns"]

    # Convert the matrix into a Pandas DataFrame
    df = pd.DataFrame(matrix)

    # Extract the diagonal (local latencies)
    local_latencies = pd.Series(df.values.diagonal())

    # Compute the remote-to-local latency ratios
    ratios = []
    for i in range(df.shape[0]):  # Iterate through rows
        for j in range(df.shape[1]):  # Iterate through columns
            if i != j:  # Only consider off-diagonal elements (remote latencies)
                remote_latency = df.iloc[i, j]
                local_latency = local_latencies[i]
                ratios.append(remote_latency / local_latency)

    # Compute the average NUMA factor
    average_numa_factor = sum(ratios) / len(ratios)

    return average_numa_factor

def get_data_access_profile(data):
    name_to_thread_locality = data["name_to_thread_locality"]
    numa_mappings_write = data["numa_mappings_write"]
    numa_mappings_read = data["numa_mappings_read"]

    rows = []

    # Helper function to process data spred across multiple NUMA nodes.
    def process_access(data_item, task_name, cpu_node, mem_nodes, core_id, access_type):
        for mem_node in mem_nodes:
            rows.append([data_item, task_name, cpu_node, mem_node, core_id, access_type])

    # Process write access operations from numa_mappings_write
    for comm_name, mapping in numa_mappings_write.items():
        mem_nodes = mapping["numa_ids"]
        for task_name, locality in name_to_thread_locality.items():
            cpu_node = locality["numa_id"]
            core_id = locality["core_id"]
            if task_name in comm_name:
                if comm_name.split("->")[0] == task_name:  # Write access (task_name on the left)
                    process_access(comm_name, task_name, cpu_node, mem_nodes, core_id, "write")

    # Process read access operations from numa_mappings_read
    for comm_name, mapping in numa_mappings_read.items():
        mem_nodes = mapping["numa_ids"]
        for task_name, locality in name_to_thread_locality.items():
            cpu_node = locality["numa_id"]
            core_id = locality["core_id"]
            if task_name in comm_name:
                if comm_name.split("->")[1] == task_name:  # Read access (task_name on the right)
                    process_access(comm_name, task_name, cpu_node, mem_nodes, core_id, "read")

    # Create a DataFrame
    return pd.DataFrame(rows, columns=["data_item", "task_name", "cpu_node", "mem_node", "core_id", "access_type"])

def get_aggregation_matrix(df_profile, equal=None):
    """
    Creates a matrix aggregating counts based on cpu_node vs mem_node.
    
    Parameters:
        df_profile (pd.DataFrame): The input DataFrame containing 'cpu_node' and 'mem_node' columns.
        equal (bool or None): 
            - If True, aggregate where cpu_node == mem_node.
            - If False, aggregate where cpu_node != mem_node.
            - If None, aggregate all rows regardless of equality.
    
    Returns:
        pd.DataFrame: A matrix with aggregated counts (cpu_node vs mem_node).
    """
    # Ensure cpu_node and mem_node are treated as integers for matrix aggregation
    df_profile['cpu_node'] = df_profile['cpu_node'].astype(int)
    df_profile['mem_node'] = df_profile['mem_node'].astype(int)
    
    # Apply filtering based on the equality parameter
    if equal is True:
        filtered_df = df_profile[df_profile['cpu_node'] == df_profile['mem_node']]
    elif equal is False:
        filtered_df = df_profile[df_profile['cpu_node'] != df_profile['mem_node']]
    else:  # equal is None
        filtered_df = df_profile
    
    # Create a pivot table for the matrix (cpu_node vs mem_node)
    matrix = filtered_df.pivot_table(
        index='cpu_node',
        columns='mem_node',
        values='task_name',
        aggfunc='count',
        fill_value=0
    )
    
    return matrix

def get_memory_migrations_profile(df_profile, migrations=None):
    # Step 1: Create the 'writings' dataframe
    writings = df_profile[df_profile['access_type'] == 'write'][['data_item', 'mem_node']]
    writings = writings.groupby('data_item')['mem_node'].apply(lambda x: ','.join(map(str, x))).reset_index()

    # Step 2: Create the 'readings' dataframe
    readings = df_profile[df_profile['access_type'] == 'read'][['data_item', 'mem_node']]
    readings = readings.groupby('data_item')['mem_node'].apply(lambda x: ','.join(map(str, x))).reset_index()

    # Step 3: Compare 'mem_node' in 'writings' and 'readings'
    merged_df = pd.merge(writings, readings, on='data_item', how='outer', suffixes=('_write', '_read'))
    merged_df['migration'] = merged_df.apply(
        lambda row: 'no' if row['mem_node_write'] == row['mem_node_read'] else 'yes', axis=1
    )

    # Step 4: Apply filtering based on the parameter
    if migrations is True:
        return merged_df[merged_df['migration'] == 'yes']
    elif migrations is False:
        return merged_df[merged_df['migration'] == 'no']
    else:  # migrations is None
        pass

    return merged_df

def get_memory_spreading_profile(df_profile, filter_spread=False):
    # Step 1: Create the 'writings' dataframe
    writings = df_profile[df_profile['access_type'] == 'write'][['data_item', 'mem_node']]
    writings = writings.groupby('data_item')['mem_node'].apply(lambda x: ','.join(map(str, x))).reset_index()
    
    if filter_spread:
        # Step 2: Filter out items where "mem_node" contains a comma
        writings = writings[~writings['mem_node'].str.contains(',', na=False)]
    else:
        # Step 2: Filter out items that do not include a comma in 'mem_node'
        writings = writings[writings['mem_node'].str.contains(',')]

    return writings

def compute_data_access_pattern_performance(accesses_matrix, relatve_latencies):
    """
    Computes the NUMA metric based on the provided matrices.
    
    Parameters:
        accesses_matrix (pd.DataFrame): Matrix of task accesses.
        relatve_latencies (pd.DataFrame): Matrix of distances (hwloc/numactl relative latencies).
        
    Returns:
        float: The calculated NUMA metric.
    """
    # Step 2: Compute q_distance_matrix (d_distance_matrix with diagonal set to 0)
    q_distance_matrix = relatve_latencies.copy()
    np.fill_diagonal(q_distance_matrix.values, 0)

    t_accesses_matrix = accesses_matrix.copy().reindex(index=q_distance_matrix.index, columns=q_distance_matrix.columns, fill_value=0)

    # Step 3: Compute T and Q
    # Ensure the matrices have the same shape
    T = t_accesses_matrix.values.sum()
    Q = q_distance_matrix.values.sum()

    # Step 4: Compute weighted_sum
    weighted_sum = (t_accesses_matrix * q_distance_matrix).values.sum()

    # Step 5: Compute the metric
    numa_metric = (1 / (T * Q)) * weighted_sum if T > 0 and Q > 0 else 0

    return numa_metric

def compute_durations(data, df_profile, time_unit):
    """
    Computes local and remote read/write times based on the access patterns.
    
    Parameters:
        data (dict): The input data containing the access offsets.
        df_profile (pd.DataFrame): The DataFrame containing the access patterns.
        
    Returns:
        dict: .
    """
    # Initialize local and remote times
    read_time_local = 0.0
    read_time_remote = 0.0
    read_time_total = 0.0
    write_time_local = 0.0
    write_time_remote = 0.0
    write_time_total = 0.0
    compute_time_total = 0.0

    # Filter local and remote accesses
    local_accesses = df_profile[df_profile['cpu_node'] == df_profile['mem_node']]
    remote_accesses = df_profile[df_profile['cpu_node'] != df_profile['mem_node']]

    # Compute local and remote read times
    read_time_local = sum(
        float(data['comm_name_read_offsets'][f"{row['data_item']}"]['end']) -
        float(data['comm_name_read_offsets'][f"{row['data_item']}"]['start'])
        for _, row in local_accesses[local_accesses['access_type'] == 'read'].iterrows()
    )

    read_time_remote = sum(
        float(data['comm_name_read_offsets'][f"{row['data_item']}"]['end']) -
        float(data['comm_name_read_offsets'][f"{row['data_item']}"]['start'])
        for _, row in remote_accesses[remote_accesses['access_type'] == 'read'].iterrows()
    )

    read_time_total = sum(float(entry['end']) - float(entry['start']) for entry in data['comm_name_read_offsets'].values())

    # Compute local and remote write times
    write_time_local = sum(
        float(data['comm_name_write_offsets'][f"{row['data_item']}"]['end']) -
        float(data['comm_name_write_offsets'][f"{row['data_item']}"]['start'])
        for _, row in local_accesses[local_accesses['access_type'] == 'write'].iterrows()
    )

    write_time_remote = sum(
        float(data['comm_name_write_offsets'][f"{row['data_item']}"]['end']) -
        float(data['comm_name_write_offsets'][f"{row['data_item']}"]['start'])
        for _, row in remote_accesses[remote_accesses['access_type'] == 'write'].iterrows()
    )

    write_time_total = sum(float(entry['end']) - float(entry['start']) for entry in data['comm_name_write_offsets'].values())

    # Compute total compute time
    compute_time_total = sum(float(entry['end']) - float(entry['start']) for entry in data['exec_name_compute_offsets'].values())

    makespan = max(float(entry['end']) for entry in data['exec_name_total_offsets'].values())

    return {
        "makespan": scale_time(makespan, time_unit),
        "read_time_local": scale_time(read_time_local, time_unit),
        "read_time_remote": scale_time(read_time_remote, time_unit),
        "read_time_total": scale_time(read_time_total, time_unit),
        "write_time_local": scale_time(write_time_local, time_unit),
        "write_time_remote": scale_time(write_time_remote, time_unit),
        "write_time_total": scale_time(write_time_total, time_unit),
        "compute_time_total": scale_time(compute_time_total, time_unit),
    }

def compute_payloads(data, df_profile, payload_unit):
    """
    Computes local and remote read/write times based on the access patterns.
    
    Parameters:
        data (dict): The input data containing the access offsets.
        df_profile (pd.DataFrame): The DataFrame containing the access patterns.
        
    Returns:
        dict: .
    """
    # Initialize local and remote times
    read_payload_local = 0.0
    read_payload_remote = 0.0
    read_payload_total = 0.0
    write_payload_local = 0.0
    write_payload_remote = 0.0
    write_payload_total = 0.0
    accesses_payload_total = 0.0
    compute_payload_total = 0.0

    # Filter local and remote accesses
    local_accesses = df_profile[df_profile['cpu_node'] == df_profile['mem_node']]
    remote_accesses = df_profile[df_profile['cpu_node'] != df_profile['mem_node']]

    # Compute local and remote read times
    read_payload_local = sum(
        float(data['comm_name_read_offsets'][f"{row['data_item']}"]['payload'])
        for _, row in local_accesses[local_accesses['access_type'] == 'read'].iterrows()
    )

    read_payload_remote = sum(
        float(data['comm_name_read_offsets'][f"{row['data_item']}"]['payload'])
        for _, row in remote_accesses[remote_accesses['access_type'] == 'read'].iterrows()
    )

    read_payload_total = sum(float(entry['payload']) for entry in data['comm_name_read_offsets'].values())

    # Compute local and remote write times
    write_payload_local = sum(
        float(data['comm_name_write_offsets'][f"{row['data_item']}"]['payload'])
        for _, row in local_accesses[local_accesses['access_type'] == 'write'].iterrows()
    )

    write_payload_remote = sum(
        float(data['comm_name_write_offsets'][f"{row['data_item']}"]['payload'])
        for _, row in remote_accesses[remote_accesses['access_type'] == 'write'].iterrows()
    )

    write_payload_total = sum(float(entry['payload']) for entry in data['comm_name_write_offsets'].values())

    # Access payload total
    accesses_payload_total = read_payload_total + write_payload_total

    # Compute total compute time
    compute_payload_total = sum(float(entry['payload']) for entry in data['exec_name_compute_offsets'].values())

    return {
        "read_payload_local": scale_payload(read_payload_local, payload_unit),
        "read_payload_remote": scale_payload(read_payload_remote, payload_unit),
        "read_payload_total": scale_payload(read_payload_total, payload_unit),
        "write_payload_local": scale_payload(write_payload_local, payload_unit),
        "write_payload_remote": scale_payload(write_payload_remote, payload_unit),
        "write_payload_total": scale_payload(write_payload_total, payload_unit),
        "accesses_payload_local": scale_payload(read_payload_local + write_payload_local, payload_unit),
        "accesses_payload_remote": scale_payload(read_payload_remote + write_payload_remote, payload_unit),
        "accesses_payload_total": scale_payload(accesses_payload_total, payload_unit),
        "compute_payload_total": scale_payload(compute_payload_total, payload_unit),
    }

def compute_accesses(df_profile):
    # Initialize local and remote times
    read_accesses_local = 0.0
    read_accesses_remote = 0.0
    read_accesses_total = 0.0
    write_accesses_local = 0.0
    write_accesses_remote = 0.0
    write_accesses_total = 0.0
    accesses_total = 0.0

    # Filter local and remote accesses
    local_accesses = df_profile[df_profile['cpu_node'] == df_profile['mem_node']]
    remote_accesses = df_profile[df_profile['cpu_node'] != df_profile['mem_node']]

    # Compute local and remote read times
    read_accesses_local = len(local_accesses[local_accesses['access_type'] == 'read'])
    read_accesses_remote = len(remote_accesses[remote_accesses['access_type'] == 'read'])
    read_accesses_total = len(df_profile[df_profile['access_type'] == 'read'])

    write_accesses_local= len(local_accesses[local_accesses['access_type'] == 'write'])
    write_accesses_remote = len(remote_accesses[remote_accesses['access_type'] == 'write'])
    write_accesses_total = len(df_profile[df_profile['access_type'] == 'write'])

    accesses_total = len(df_profile)

    return {
        "read_accesses_local": read_accesses_local,
        "read_accesses_remote": read_accesses_remote,
        "read_accesses_total": read_accesses_total,
        "write_accesses_local": write_accesses_local,
        "write_accesses_remote": write_accesses_remote,
        "write_accesses_total": write_accesses_total,
        "accesses_local": len(local_accesses),
        "accesses_remote": len(remote_accesses),
        "accesses_total": accesses_total,
    }

# def print_dict(data, title):
#     print(f"\n{Fore.CYAN}{title.replace('_', ' ').title()}{Style.RESET_ALL}")
#     for key, value in data.items():
#         if isinstance(value, str):
#             print(f"  {Fore.YELLOW}{key}:{Style.RESET_ALL} {value}")
#         else:
#             print(f"  {Fore.YELLOW}{key}:{Style.RESET_ALL} {value:.4f}")

# def print_df(data, title):
#     print(f"\n{Fore.CYAN}{title.replace('_', ' ').title()}:{Fore.YELLOW} {eval(data[2])}{Style.RESET_ALL}")
#     table_str = tabulate(data[0], headers=data[1], tablefmt="grid", showindex=True)
#     print("\n" + "\n".join(f"  {line}" for line in table_str.split("\n")))

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


def print_profile(data, matrix_relative_latencies, time_unit, payload_unit, export_csv):
    user_data = data['user']
    trace_data = data['trace']
    runtime_data = data['runtime']
    workflow_data = data['workflow']

    df_profile = get_data_access_profile(trace_data)
    durations = compute_durations(trace_data, df_profile, time_unit)
    payloads = compute_payloads(trace_data, df_profile, payload_unit)
    accesses = compute_accesses(df_profile)

    matrix_accesses_total = get_aggregation_matrix(df_profile, equal=None)

    matrix_memory_pages_migrations = get_memory_migrations_profile(df_profile, migrations=True)
    
    data_access_pattern_performance = compute_data_access_pattern_performance(
        matrix_accesses_total, matrix_relative_latencies) if not matrix_relative_latencies.empty else -1

    df_output_profile = df_profile[["task_name", "core_id", "cpu_node", "mem_node", "data_item", "access_type"]]

    df_accesses_spread = get_memory_spreading_profile(df_profile)

    system = {
        "data_pages_migrations": len(matrix_memory_pages_migrations),
        "data_pages_spreadings": len(df_accesses_spread),
    }

    workflow = {
        "execs_count": workflow_data.get("execs_count", None),
        "reads_count": workflow_data.get("reads_count", None),
        "writes_count": workflow_data.get("writes_count", None),
    }

    runtime = {
        "threads_checksum": runtime_data.get("threads_checksum", None),
        "threads_active": runtime_data.get("threads_active", None),
        "execs_active_count": runtime_data.get("tasks_active_count", None),
        "reads_active_count": runtime_data.get("reads_active_count", None),
        "writes_active_count": runtime_data.get("writes_active_count", None),
    }

    profile = {
        "time_unit": time_unit,
        "data_payload_unit": f"{payload_unit}bytes",
        "compute_payload_unit": f"{payload_unit}flops",
    }

    machine = {
        "numa_factor": get_machine_average_numa_factor(user_data),
    }

    ratios = {
        "comp_to_comm": durations["compute_time_total"] / (durations["read_time_total"] + durations["write_time_total"]),
        "comm_to_comp": (durations["read_time_total"] + durations["write_time_total"]) / durations["compute_time_total"],
    }

    metrics = {
        "numa_awareness": data_access_pattern_performance
    }

    output_scalars = {
        "system": system,
        "workflow": workflow,
        "runtime": runtime,
        "machine": machine,
        "profile": profile,
        "metrics": metrics,
        "durations": durations,
        "payloads": payloads,
        "accesses": accesses,
        "ratios": ratios,
    }

    # Ensure the total matrix is square by reindexing
    max_dim = max(len(matrix_accesses_total.index), len(matrix_accesses_total.columns))
    full_index = range(max_dim)
    matrix_accesses_total = matrix_accesses_total.reindex(index=full_index, columns=full_index, fill_value=0)
    matrix_accesses_total_percent = matrix_accesses_total / matrix_accesses_total.values.sum()

    # Prepare output matrices for printing
    output_matrices = {
        "accesses_compact_profile": (df_output_profile, df_output_profile.columns, 'data[0].shape[0]'),
        "data_pages_spreadings_write": (df_accesses_spread, df_accesses_spread.columns, 'data[0].shape[0]'),
        "data_pages_migrations_read": (matrix_memory_pages_migrations, matrix_memory_pages_migrations.columns, 'data[0].shape[0]'),
        "matrix_accesses_total": (matrix_accesses_total, matrix_accesses_total.columns, 'data[0].sum().sum()'),
        "matrix_accesses_total_percent": (matrix_accesses_total_percent, matrix_accesses_total_percent.columns, 'data[0].sum().sum()'),
    }

    if export_csv:
        output_data = {
            **system, **runtime, **machine, **ratios, 
            **metrics, **durations, **payloads, **accesses
        }

        output_series = pd.Series(output_data)
        output_series.to_csv(export_csv, header=False)
        print(f"Profile exported: {export_csv}")

        return

    for key, value in output_scalars.items():
        print_dict(value, key)

    for key, value in output_matrices.items():
        print_df(value, key)
    print("")

def main():
    parser = argparse.ArgumentParser(description="Process NUMA access data.")
    parser.add_argument("input_file", help="Path to input YAML profile file")
    parser.add_argument("--input_file_rel_lat", default=None, help="Path to input TXT file with hwloc/numactl relative latencies")
    parser.add_argument("--time_unit", type=str, choices=['us', 'ms', 's', 'min'], default='us', help="Time unit for scaling.")
    parser.add_argument("--payload_unit", type=str, choices=['K', 'M', 'G'], default='', help="Payload unit for scaling.")
    parser.add_argument("--export_csv", type=str, help="Path to export CSV file.", default=None)
    args = parser.parse_args()

    with open(args.input_file, "r") as file:
        data = yaml.load(file, Loader=yaml.FullLoader)

    if args.input_file_rel_lat:
        with open(args.input_file_rel_lat, 'r') as file:
            dimension = int(file.readline().strip())
            rel_lat_matrix = pd.DataFrame([list(map(float, file.readline().split())) for _ in range(dimension)])
    else:
        rel_lat_matrix = pd.DataFrame()

    print_profile(data, rel_lat_matrix, args.time_unit, args.payload_unit, args.export_csv)

if __name__ == "__main__":
    main()
