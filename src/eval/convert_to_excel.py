import argparse
import json
import os
import sys

import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Convert evaluation JSON files to an Excel report."
    )
    parser.add_argument("--output_file", help="Path to the output .xlsx file")
    parser.add_argument(
        "--input_files", nargs="+", help="List of JSON files to process"
    )
    args = parser.parse_args()

    key_mapping = {
        "average": "Average",
        "math": "Math",
        "amc23": "AMC23",
        "aime24": "AIME24",
        "college_math": "college_math",
        "olympiadbench": "olympiadbench",
        "minerva_math": "minerva_math",
    }

    ordered_keys = [
        "average",
        "math",
        "amc23",
        "aime24",
        "college_math",
        "olympiadbench",
        "minerva_math",
    ]

    metrics_main = ["pass@3", "acc@3"]
    metrics_token = ["avg_tokens"]

    rows_main = []
    rows_token = []

    for file_path in args.input_files:
        try:
            with open(file_path, "r") as f:
                data = json.load(f)

            clean_path = os.path.normpath(file_path)
            dirname, filename = os.path.split(clean_path)
            parent_dir = os.path.basename(dirname)
            base_name = filename[:-5] if filename.endswith(".json") else filename
            if parent_dir not in [".", ""]:
                row_label = f"{parent_dir}/{base_name}"
            else:
                row_label = base_name

            row_data_main = {"Model": row_label}
            row_data_token = {"Model": row_label}

            for key in ordered_keys:
                if key in data:

                    for metric in metrics_main:
                        val = data[key].get(metric, None)
                        if val is not None:
                            row_data_main[(key_mapping[key], metric)] = val * 100
                        else:
                            row_data_main[(key_mapping[key], metric)] = None

                    # Process Token Metrics
                    val_tok = data[key].get("avg_tokens", None)
                    row_data_token[(key_mapping[key], "avg_tokens")] = val_tok

                    val_think = data[key].get("avg_thinking_ratio", None)
                    if val_think is not None:
                        row_data_token[(key_mapping[key], "avg_thinking_ratio")] = (
                            val_think * 100
                        )
                    else:
                        row_data_token[(key_mapping[key], "avg_thinking_ratio")] = None

                else:
                    for metric in metrics_main:
                        row_data_main[(key_mapping[key], metric)] = None
                    for metric in metrics_token:
                        row_data_token[(key_mapping[key], metric)] = None

            rows_main.append(row_data_main)
            rows_token.append(row_data_token)

        except Exception as e:
            print(f"Error processing {file_path}: {e}", file=sys.stderr)

    if not rows_main:
        print("No valid data found to export.")
        return

    # --- Helper to save DataFrame ---
    def save_df(rows, metrics, filename):
        df_data = pd.DataFrame(rows)
        if "Model" in df_data.columns:
            df_data.set_index("Model", inplace=True)

        tuple_cols = [c for c in df_data.columns if isinstance(c, tuple)]

        desired_columns = []
        for key in ordered_keys:
            header = key_mapping[key]
            for metric in metrics:
                desired_columns.append((header, metric))

        df_data = df_data.reindex(columns=desired_columns)
        df_data.columns = pd.MultiIndex.from_tuples(df_data.columns)

        try:
            df_data.to_excel(filename)
            print(f"Successfully saved to {filename}")
        except ImportError:
            print(
                "Error: `openpyxl` library is missing. Please install it with `pip install openpyxl`."
            )
        except Exception as e:
            print(f"Error saving to Excel {filename}: {e}")

    # Save Main Report
    save_df(rows_main, metrics_main, args.output_file)

    # Save Token Report
    base, ext = os.path.splitext(args.output_file)
    output_file_token = f"{base}_token{ext}"
    save_df(rows_token, metrics_token, output_file_token)


if __name__ == "__main__":
    main()
