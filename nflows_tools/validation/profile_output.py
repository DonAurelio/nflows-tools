#!/usr/bin/env python3

"""
@authors: ChatGPT
@edited_by: Aurelio Vivas
@promt: 
"""

import yaml
import sys
import argparse

def validate_yaml(output_path, expected_path, check_order_keys):
    with open(output_path, 'r') as output_file, open(expected_path, 'r') as expected_file:
        output_data = yaml.safe_load(output_file)
        expected_data = yaml.safe_load(expected_file)
    
    def compare(expected, actual, path="", depth=0):
        if isinstance(expected, dict):
            expected_keys = list(expected.keys())
            actual_keys = list(actual.keys()) if isinstance(actual, dict) else []
            
            if path in check_order_keys and expected_keys != actual_keys[:len(expected_keys)]:
                print(f"Key order mismatch at {path}: expected {expected_keys}, got {actual_keys[:len(expected_keys)]}")
                return False

            for key, value in expected.items():
                new_path = f"{path}.{key}" if path else key
                if key not in actual:
                    print(f"Missing key: {new_path}")
                    return False
                if not compare(value, actual[key], new_path, depth + 1):
                    return False
        elif isinstance(expected, list):
            if not isinstance(actual, list) or len(expected) != len(actual):
                print(f"Mismatch in list length at {path}")
                return False
            for i, (exp_item, act_item) in enumerate(zip(expected, actual)):
                if not compare(exp_item, act_item, f"{path}[{i}]", depth + 1):
                    return False
        else:
            if expected is not None and expected != actual:
                print(f"Mismatch at {path}: expected {expected}, got {actual}")
                return False
        return True
    
    return compare(expected_data, output_data)

def main():
    parser = argparse.ArgumentParser(description="Validate YAML files for structural and content correctness.")
    parser.add_argument("--check-order", metavar="KEYS", type=str, help="Comma-separated list of keys to check order")
    parser.add_argument("output_yaml", help="Path to the output YAML file")
    parser.add_argument("expected_yaml", help="Path to the expected YAML file")

    args = parser.parse_args()

    check_order_keys = set(args.check_order.split(",")) if args.check_order else set()
    output_file = args.output_yaml
    expected_file = args.expected_yaml

    if validate_yaml(output_file, expected_file, check_order_keys):
        print(f"Output validation successful: '{output_file}' matches '{expected_file}'.")
    else:
        print(f"Output validation failed: '{output_file}' does not match '{expected_file}'.")
        sys.exit(1)

if __name__ == "__main__":
    main()
