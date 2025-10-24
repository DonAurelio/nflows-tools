"""
@authors: ChatGPT, Aurelio Vivas <aa.vivas@uniandes.edu.co>
"""

import pandas as pd
from tabulate import tabulate
from colorama import Fore, Style

def scale_time(value, unit):
    scale_factors = {'us': 1, 'ms': 1e3, 's': 1e6, 'min': 6e7}
    return float(value) / float(scale_factors[unit])

def scale_payload(value, unit):
    scale_factors = {'K': 1e3, 'M': 1e6, 'G': 1e9}
    return float(value) / float(scale_factors.get(unit, 1))

def flatten_dict(d, parent_key='', sep='_'):
    """
    Recursively flattens a nested dictionary using the given separator.

    Example:
        flatten_dict({"a": {"b": 1, "c": 2}}, parent_key="x")
        -> {"x_a_b": 1, "x_a_c": 2}
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

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

def print_profile(output_scalars, output_matrices):
    for key, value in output_scalars.items():
        print_dict(value, key)

    for key, value in output_matrices.items():
        print_df(value, key)
    print("")

def export_profile(output_data, export_csv):
    output_series = pd.Series(output_data)
    output_series.to_csv(export_csv, header=False)
    print(f"Profile exported: {export_csv}")