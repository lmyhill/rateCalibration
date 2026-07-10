#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt


def json_default(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def format_summary_value(value):
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, default=json_default)
    return str(value)


def load_config(config_path):
    with open(config_path, "r") as config_file:
        return json.load(config_file)


def resolve_output_root(output_root):
    return os.path.abspath(output_root)


def find_raw_result_files(output_root):
    return sorted(Path(output_root).rglob("raw_results.json"))


def write_summary_file(raw_results, summary_dir):
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / f"results_seed_{raw_results['seed']}_row_{raw_results['row']}.txt"

    with open(summary_path, "w") as summary_file:
        for key, value in raw_results.items():
            summary_file.write(f"{key}: {format_summary_value(value)}\n")

    return summary_path


def save_line_plot(x_values, y_values, title, xlabel, ylabel, figure_path):
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(figure_path)
    plt.close()


def process_raw_result(raw_file, output_settings):
    with open(raw_file, "r") as f:
        raw_results = json.load(f)

    raw_results["row"] = int(raw_results["row"])
    raw_results["seed"] = int(raw_results["seed"])

    simulation_results_dir = raw_file.parent
    summary_dir = simulation_results_dir
    summary_path = write_summary_file(raw_results, summary_dir)

    seed_dir = raw_file.parents[1]

    if output_settings.get("output_betaP_figure", False):
        figure_path = seed_dir / "betaP_figures" / f"betaP_figure_row_{raw_results['row']}_seed_{raw_results['seed']}.png"
        save_line_plot(
            raw_results.get("time [s]", []),
            raw_results.get("betaP_1", []),
            f"betaP row {raw_results['row']} seed {raw_results['seed']}",
            "time [s]",
            "betaP_1",
            figure_path,
        )

    if output_settings.get("output_dotBetaP_figure", False):
        figure_path = seed_dir / "dotBetaP_figures" / f"dotBetaP_figure_row_{raw_results['row']}_seed_{raw_results['seed']}.png"
        save_line_plot(
            raw_results.get("time [s]", []),
            raw_results.get("dotBetaP", []),
            f"dotBetaP row {raw_results['row']} seed {raw_results['seed']}",
            "time [s]",
            "dotBetaP",
            figure_path,
        )

    return summary_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config_sweep.json",
        help="Path to sweep configuration file",
    )
    parser.add_argument(
        "--output-root",
        default=None,
        help="Root directory containing simulation raw results",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    output_settings = config.get("outputSettings", {})

    output_root = args.output_root or os.path.join(
        os.environ.get("SLURM_SUBMIT_DIR", os.getcwd()),
        "outputs",
    )
    output_root = resolve_output_root(output_root)

    raw_result_files = find_raw_result_files(output_root)
    if not raw_result_files:
        print(f"No raw_results.json files found under {output_root}")
        return

    for raw_file in raw_result_files:
        summary_path = process_raw_result(raw_file, output_settings)
        print(f"Wrote post-processed results to {summary_path}")


if __name__ == "__main__":
    main()
