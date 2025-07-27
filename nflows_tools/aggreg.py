#!/usr/bin/env python3

"""
@authors: ChatGPT
@edited_by: Aurelio Vivas
@promt: 
"""

import os
import argparse
import pandas as pd

def load_csv_series(file_path):
    """Load a CSV file as a series where the first column is the index and the second column is the value."""
    df = pd.read_csv(file_path, header=None, index_col=0)
    return df.squeeze("columns")

def compile_series_to_dataframe(input_folder):
    """Compile all CSV series in a folder into a single DataFrame."""
    series_list = []
    filenames = []

    for file in os.listdir(input_folder):
        if file.endswith(".csv"):
            file_path = os.path.join(input_folder, file)
            series = load_csv_series(file_path)
            series_list.append(series)
            filenames.append(file)

    return pd.DataFrame(series_list, index=filenames)

def aggregate_dataframe(df):
    """Compute the mean, standard deviation, and variance for each column and return as a Series."""
    mean_series = df.mean()
    std_series = df.std()
    var_series = df.var()

    aggregated_series = pd.Series()
    for col in df.columns:
        aggregated_series[f"{col}_mean"] = mean_series[col]
        aggregated_series[f"{col}_std"] = std_series[col]
        aggregated_series[f"{col}_variance"] = var_series[col]

    return aggregated_series

def main():
    parser = argparse.ArgumentParser(description="Process CSV files into a summary and aggregated data.")
    parser.add_argument("input_folder", type=str, help="Path to the folder containing CSV files")
    parser.add_argument("output_summary_name", type=str, help="Path to the folder where results will be saved")
    parser.add_argument("output_aggreg_name", type=str, help="Path to the folder where results will be saved")
    args = parser.parse_args()

    summary_csv = args.output_summary_name
    aggregated_csv = args.output_aggreg_name

    df = compile_series_to_dataframe(args.input_folder)
    df.to_csv(summary_csv, index=False)

    aggregated_df = aggregate_dataframe(df)
    aggregated_df.to_csv(aggregated_csv, header=False)
    
    print(f"Summary saved to: {summary_csv}")
    print(f"Aggregated data saved to: {aggregated_csv}")

if __name__ == "__main__":
    main()
