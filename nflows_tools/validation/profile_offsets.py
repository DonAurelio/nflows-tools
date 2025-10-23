#!/usr/bin/env python3

"""
@authors: ChatGPT
@edited_by: Aurelio Vivas
@promt: 
"""

import yaml
import argparse

def load_yaml_data(file_path):
    """Loads YAML data from the given file."""
    with open(file_path, "r") as file:
        return yaml.safe_load(file)

def determine_key_type(key, right_hand_in_read, left_hand_in_write):
    """Determines if a key is a root, intermediate, or end node."""
    if key not in right_hand_in_read:
        return "root"
    elif key not in left_hand_in_write:
        return "end"
    return "intermediate"

def compute_max_offset_time(dependencies, target_key, position):
    """
    Computes the maximum offset time for dependencies.
    - `dependencies`: Dictionary containing offset start and end times.
    - `target_key`: The key whose dependencies are analyzed.
    - `position`: "left" for write dependencies, "right" for read dependencies.
    """
    max_time = 0
    for dep_key, dep in dependencies.items():
        left, right = dep_key.split("->")
        if (position == "right" and right == target_key) or (position == "left" and left == target_key):
            max_time = max(max_time, float(dep["end"]) - float(dep["start"]))
    return max_time

def validate_offsets(data):
    """Validates execution offsets based on computed vs expected total times."""
    trace = data.get("trace", {})
    comm_read_offsets = trace.get("comm_name_read_offsets", {})
    comm_write_offsets = trace.get("comm_name_write_offsets", {})
    exec_compute_offsets = trace["exec_name_compute_offsets"]
    exec_total_offsets = trace["exec_name_total_offsets"]

    right_hand_in_read = {key.split("->")[1] for key in comm_read_offsets.keys()}
    left_hand_in_write = {key.split("->")[0] for key in comm_write_offsets.keys()}

    for key, total in exec_total_offsets.items():
        start, end = float(total["start"]), float(total["end"])
        key_type = determine_key_type(key, right_hand_in_read, left_hand_in_write)

        # Compute offsets
        max_read_time = compute_max_offset_time(comm_read_offsets, key, "right") if key_type in ["intermediate", "end"] else 0
        max_write_time = compute_max_offset_time(comm_write_offsets, key, "left") if key_type in ["root", "intermediate"] else 0
        compute_time = float(exec_compute_offsets[key]["end"]) - float(exec_compute_offsets[key]["start"])

        computed_sum = max_read_time + compute_time + max_write_time

        if start + computed_sum != end:
            print(f"Offsets validation failed for {key}: expected {end}, got {start + computed_sum}")

def main():
    parser = argparse.ArgumentParser(description="Validate execution offsets from YAML file.")
    parser.add_argument("yaml_path", type=str, help="Path to the YAML file")
    args = parser.parse_args()

    yaml_data = load_yaml_data(args.yaml_path)
    validate_offsets(yaml_data)

if __name__ == "__main__":
    main()