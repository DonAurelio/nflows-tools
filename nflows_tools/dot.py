#!/usr/bin/env python3

"""
@authors: ChatGPT, DeepSeek
@edited_by: Aurelio Vivas
@promt:

I need a Python script that takes a workflow represented in JSON and parses it into a dot_file. These are the steps to be performed in the algorithm.

The tasks are defined in workflow.specification.tasks.
The files are defined in workflow.specification.files.

1. Parse the tasks field and create all the nodes in the graph using a networkx digraph. 
2. Use the property id of every task as a node identifier.
3. Look at workflow.execution.tasks and use the runtimeInSeconds property to calculate the size of the node. The size of the node is calculated by converting the runtimeInSeconds property into flops when considering a hypothetical core processing speed. Use this value as the size property of the node.
4. Use the children's property in every task to create the dependencies. The size of the dependencies is calculated according to the file property in the parent and child task. If the parent and child share the same file name, the size will be the sum of the sizeInBytes property of the files they share.
5. Add a root node in the graph.
6. Add dependencies between the root node and the tasks whose parents' property is an empty list. The size of these dependencies is calculated from the file property of these tasks. Sum the sizeInBytes of the input files and set this value as the dependency size.
7. Add an end node in the graph.
8. Add dependencies between the leave nodes and the end node. Leave nodes are those whose children's properties are an empty list. The size of these dependencies is calculated from the file property of these tasks. Sum the sizeInBytes of the output files and set this value as the dependency size.

This is an example of a json file

I need to update the following script by:

1. Providing an option to change the scale of the calculated flops between two values, [X, Y].
2. Providing an option to change the scale of the calculated dependency size between two values [W, Z]
3. Provide an option to place a constant value for all flops.
4. Provide an option to place a constant value for all deps.
"""

import json
import networkx as nx
import argparse

def scale_value(value, old_min, old_max, new_min, new_max):
    """Scale a value from [old_min, old_max] to [new_min, new_max]"""
    if old_max == old_min:  # avoid division by zero
        return int(new_min)
    return int( ((value - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min )

def main():
    parser = argparse.ArgumentParser(description="Generate a DOT file from a JSON in wfformat_v1.5.")
    parser.add_argument("input_file", help="Path to the JSON file in wfformat_v1.5.")
    parser.add_argument("output_file", help="Path to the DOT output file.")
    parser.add_argument("--flops_per_cycle", default=32, type=float, help="CPU flops per clock cycle (default: 32).")
    parser.add_argument("--cycles_per_second", default=1e9, type=float, help="CPU clock frequency in Hz (1/s) (default: 1e9).")

    # Options for FLOPS (node sizes)
    flops_group = parser.add_mutually_exclusive_group()
    flops_group.add_argument("--flops_scale_range", nargs=2, metavar=('MIN', 'MAX'), type=float, help="Scale FLOPS values to range [MIN, MAX].")
    flops_group.add_argument("--flops_scale_percent", type=float, help="Percentage to scale FLOPS values (e.g., 100 for no change, 50 to reduce by half).")
    flops_group.add_argument("--flops_constant", type=int, help="Set a constant value for all FLOPS (node sizes).")

    # Options for dependency sizes (edge sizes)
    dep_group = parser.add_mutually_exclusive_group()
    dep_group.add_argument("--dep_scale_range", nargs=2, metavar=('MIN', 'MAX'), type=float, help="Scale dependency sizes to range [MIN, MAX].")
    dep_group.add_argument("--dep_constant", type=int, help="Set a constant value for all dependency sizes (edge sizes).")

    args = parser.parse_args()

    # Load the JSON data
    with open(args.input_file) as f:
        data = json.load(f)

    # Extract tasks and files
    tasks = data['workflow']['specification']['tasks']
    files = data['workflow']['specification']['files']
    execution_tasks = data['workflow']['execution']['tasks']

    # Create a directed graph
    G = nx.DiGraph()

    # Create a dictionary to map file IDs to their sizes
    file_sizes = {file['id']: file['sizeInBytes'] for file in files}

    # Create a dictionary to map task IDs to their runtime in seconds
    task_runtimes = {task['id']: task['runtimeInSeconds'] for task in execution_tasks}

    # Calculate raw FLOPS values for all tasks
    raw_flops = {task['id']: int(task_runtimes[task['id']] * args.flops_per_cycle * args.cycles_per_second) for task in tasks}

    # Add nodes to the graph
    for task in tasks:
        task_id = task['id']
        flops = raw_flops[task_id]

        # Scale FLOPS if requested
        if args.flops_scale_range:
            min_flops, max_flops = min(raw_flops.values()), max(raw_flops.values())
            new_min_flops, new_max_flops = args.flops_scale_range
            flops = scale_value(flops, min_flops, max_flops, new_min_flops, new_max_flops)
            assert new_min_flops <= flops <= new_max_flops, f"flops {flops} out of range [{new_min_flops}, {new_max_flops}]"

        if args.flops_scale_percent:
            flops *= args.flops_scale_percent

        if args.flops_constant:
            flops = args.flops_constant
        # If a constant value is provided, use it
        # If no scaling is requested, use the raw FLOPS value

        G.add_node(task_id, size=int(flops))

    # Collect all dependency sizes to calculate scaling parameters if needed
    dependency_sizes = []

    # First pass to collect all dependency sizes if scaling is requested
    if args.dep_scale_range:
        for task in tasks:
            task_id = task['id']

            # Check parent dependencies
            for parent in task['parents']:
                shared_files = set(task['inputFiles']).intersection(set(next(t for t in tasks if t['id'] == parent)['outputFiles']))
                dependency_size = sum(file_sizes[file] for file in shared_files)
                dependency_sizes.append(dependency_size)

            # Check child dependencies
            for child in task['children']:
                shared_files = set(task['outputFiles']).intersection(set(next(t for t in tasks if t['id'] == child)['inputFiles']))
                dependency_size = sum(file_sizes[file] for file in shared_files)
                dependency_sizes.append(dependency_size)

        min_dep, max_dep = min(dependency_sizes), max(dependency_sizes)
        new_min_dep, new_max_dep = args.dep_scale_range

    # Add edges to the graph based on children and parents
    for task in tasks:
        task_id = task['id']

        # Add edges from parents to the current task
        for parent in task['parents']:
            # Calculate the size of the dependency based on shared files
            shared_files = set(task['inputFiles']).intersection(set(next(t for t in tasks if t['id'] == parent)['outputFiles']))
            dependency_size = sum(file_sizes[file] for file in shared_files)
            
            # Scale dependency size if requested
            if args.dep_scale_range:
                dependency_size = scale_value(dependency_size, min_dep, max_dep, new_min_dep, new_max_dep)
                assert new_min_dep <= dependency_size <= new_max_dep, f"dependency_size {dependency_size} out of range [{new_min_dep}, {new_max_dep}]"

            G.add_edge(parent, task_id, size = args.dep_constant or dependency_size)

        # Add edges from the current task to its children
        for child in task['children']:
            # Calculate the size of the dependency based on shared files
            shared_files = set(task['outputFiles']).intersection(set(next(t for t in tasks if t['id'] == child)['inputFiles']))
            dependency_size = sum(file_sizes[file] for file in shared_files)

            # Scale dependency size if requested
            if args.dep_scale_range:
                dependency_size = scale_value(dependency_size, min_dep, max_dep, new_min_dep, new_max_dep)
                assert new_min_dep <= dependency_size <= new_max_dep, f"dependency_size {dependency_size} out of range [{new_min_dep}, {new_max_dep}]"

            G.add_edge(task_id, child, size = args.dep_constant or dependency_size)

    # Add a root node and connect it to tasks with no parents
    root_node = 'root'
    G.add_node(root_node, size=2)
    for task in tasks:
        if not task['parents']:
            # Calculate the size of the dependency based on input files
            # dependency_size = sum(file_sizes[file] for file in task['inputFiles'])

            # Scale dependency size if requested
            # if args.dep_scale_range:
                # dependency_size = scale_value(dependency_size, min_dep, max_dep, new_min_dep, new_max_dep)
                # assert new_min_dep <= dependency_size <= new_max_dep, f"dependency_size {dependency_size} out of range [{new_min_dep}, {new_max_dep}]"

            # G.add_edge(root_node, task['id'], size = args.dep_constant or dependency_size)
            G.add_edge(root_node, task['id'], size = 2) # Placeholder size for root dependencies

    # Add an end node and connect it to tasks with no children
    end_node = 'end'
    G.add_node(end_node, size=2)
    for task in tasks:
        if not task['children']:
            # Calculate the size of the dependency based on output files
            # dependency_size = sum(file_sizes[file] for file in task['outputFiles'])
            
            # Scale dependency size if requested
            # if args.dep_scale_range:
            #     dependency_size = scale_value(dependency_size, min_dep, max_dep, new_min_dep, new_max_dep)
            #     assert new_min_dep <= dependency_size <= new_max_dep, f"dependency_size {dependency_size} out of range [{new_min_dep}, {new_max_dep}]"
            
            # G.add_edge(task['id'], end_node, size = args.dep_constant or dependency_size)
            G.add_edge(task['id'], end_node, size = 2) # Placeholder size for end dependencies

    # Write the graph to a DOT file
    nx.drawing.nx_pydot.write_dot(G, args.output_file)

    print(f"DOT file created: '{args.output_file}'.")

if __name__ == "__main__":
    main()
