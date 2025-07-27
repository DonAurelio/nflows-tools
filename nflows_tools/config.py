#!/usr/bin/env python3

"""
@authors: ChatGPT
@edited_by: Aurelio Vivas
@promt:
"""

import json
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Generate a JSON configuration from a template and parameter variations.")
    parser.add_argument("--template", required=True, help="Path to the JSON template file.")
    parser.add_argument("--output_file", required=True, help="Full path for the output JSON file, without extension.")
    parser.add_argument("--params", nargs='+', required=True, help="Named parameters in key=value format.")
 
    args = parser.parse_args()

    with open(args.template, 'r') as f:
        template = json.load(f)

    params_dict = {}
    for param in args.params:
        key, value = param.split("=")
        params_dict[key] = value
    
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    template.update(params_dict)
    
    with open(args.output_file, 'w') as f:
        json.dump(template, f, indent=4)

    print(f"Config file created: {args.output_file}")

if __name__ == "__main__":
    main()
