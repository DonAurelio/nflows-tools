#!/usr/bin/env python3

"""
@authors: ChatGPT
@edited_by: Aurelio Vivas
@promt:
"""

import yaml
import random
import numpy as np
import matplotlib.pyplot as plt
import argparse

def load_yaml(filename):
    with open(filename, 'r') as file:
        return yaml.safe_load(file)

def scale_time(value, unit):
    scale_factors = {'us': 1, 'ms': 1e3, 's': 1e6, 'min': 6e7}
    return float(value) / float(scale_factors[unit])

def scale_payload(value, unit):
    scale_factors = {'B': 1, 'KB': 1e3, 'MB': 1e6, 'GB': 1e9}
    return float(value) / float(scale_factors[unit])

def get_unique_color(existing_colors):
    while True:
        color = (random.random(), random.random(), random.random())
        if color not in existing_colors:
            existing_colors.add(color)
            return color

def compute_figure_size(resource_count):
    width = 12
    height = max(6, resource_count * 0.5)
    return (width, height)

def plot_gantt(yaml_data, output_file, time_unit='us', payload_unit='B', use_numa=True, title=None, xlabel=None, ylabel=None, resource_label=None):
    trace_data = yaml_data['trace']
    resources = sorted(set(
        locality['numa_id'] if use_numa else locality['core_id']
        for locality in trace_data['name_to_thread_locality'].values()
    ))
    resource_map = {r: i for i, r in enumerate(resources)}
    fig, ax = plt.subplots(figsize=compute_figure_size(len(resources)))
    existing_colors = set()
    offset = 1  # Adjust offset to ensure one tick per resource
    
    for resource in resources:
        ax.axhline(y=resource_map[resource] * offset, color='gray', linestyle='--', linewidth=0.5)
    
    for task, times in trace_data['exec_name_compute_offsets'].items():
        locality = trace_data['name_to_thread_locality'][task]
        resource_id = locality['numa_id'] if use_numa else locality['core_id']
        y_position = resource_map[resource_id] * offset
        start = scale_time(times['start'], time_unit)
        end = scale_time(times['end'], time_unit)
        color = get_unique_color(existing_colors)
        ax.barh(y_position, end - start, left=start, color=color, edgecolor='black', alpha=0.7)
        ax.text((start + end) / 2, y_position, f"{task}", ha='center', va='center', fontsize=8, color='black', weight='bold')
    
    for task in trace_data['exec_name_compute_offsets'].keys():
        read_offsets = trace_data['comm_name_read_offsets']
        write_offsets = trace_data['comm_name_write_offsets']
        
        largest_read = max((read_offsets[comm] for comm in read_offsets if comm.endswith(f"->{task}")), key=lambda x: scale_time(x['end'] - x['start'], time_unit), default=None)
        largest_write = max((write_offsets[comm] for comm in write_offsets if comm.startswith(f"{task}->")), key=lambda x: scale_time(x['end'] - x['start'], time_unit), default=None)
        
        if largest_read:
            resource_id = trace_data['name_to_thread_locality'][task]['numa_id'] if use_numa else trace_data['name_to_thread_locality'][task]['core_id']
            y_position = resource_map[resource_id] * offset
            start = scale_time(largest_read['start'], time_unit)
            end = scale_time(largest_read['end'], time_unit)
            ax.barh(y_position, end - start, left=start, color='blue', edgecolor='black', alpha=0.7, hatch='//')
        
        if largest_write:
            resource_id = trace_data['name_to_thread_locality'][task]['numa_id'] if use_numa else trace_data['name_to_thread_locality'][task]['core_id']
            y_position = resource_map[resource_id] * offset
            start = scale_time(largest_write['start'], time_unit)
            end = scale_time(largest_write['end'], time_unit)
            ax.barh(y_position, end - start, left=start, color='red', edgecolor='black', alpha=0.7, hatch='\\')
    
    ax.set_yticks(np.arange(len(resources)) * offset)
    ax.set_yticklabels([f"{resource_label} {r}" for r in resources] if resource_label else [f"Resource {r}" for r in resources])
    ax.set_xlabel(xlabel if xlabel else f"Time ({time_unit})")
    ax.set_ylabel(ylabel if ylabel else "Resources")
    ax.set_title(title if title else "Gantt Chart of Task Execution and Communication")
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.savefig(output_file, format=output_file.split('.')[-1])
    plt.close()

    print(f"Gantt created: '{output_file}'.")

def main():
    parser = argparse.ArgumentParser(description="Generate a Gantt chart from a YAML file.")
    parser.add_argument("yaml_file", type=str, help="Path to the YAML file containing scheduling data.")
    parser.add_argument("output_file", type=str, help="Output file (PNG or PDF).")
    parser.add_argument("--time_unit", type=str, choices=['us', 'ms', 's', 'min'], default='us', help="Time unit for scaling.")
    parser.add_argument("--payload_unit", type=str, choices=['B', 'KB', 'MB', 'GB'], default='B', help="Payload unit for scaling.")
    parser.add_argument("--use_numa", action="store_true", help="Use NUMA instead of core ID for resource mapping.")
    parser.add_argument("--title", type=str, help="Title of the plot.")
    parser.add_argument("--xlabel", type=str, help="Label for the x-axis.")
    parser.add_argument("--ylabel", type=str, help="Label for the y-axis.")
    parser.add_argument("--resource_label", type=str, help="Label for resource names.")

    args = parser.parse_args()

    data = load_yaml(args.yaml_file)
    plot_gantt(data, args.output_file, args.time_unit, args.payload_unit, args.use_numa, args.title, args.xlabel, args.ylabel, args.resource_label)

if __name__ == "__main__":
    main()
