#!/usr/bin/env python3

import argparse
import json
import os
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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


def copy_tutorial_artifacts(simulation_root, simulation_results_dir):
    for folder_name in ("evl", "F", "inputFiles"):
        source_dir = simulation_root / folder_name
        if not source_dir.exists():
            continue

        destination_dir = simulation_results_dir / folder_name
        if destination_dir.exists():
            shutil.rmtree(destination_dir)

        shutil.copytree(source_dir, destination_dir)


def find_simulation_result_file(raw_data_dir, row, seed):
    row_dir_candidates = sorted(raw_data_dir.glob(f"row_{row}_job_*"))
    if not row_dir_candidates:
        return None

    for row_dir in row_dir_candidates:
        candidate = row_dir / f"row_{row}" / f"seed_{seed}" / "simulation_results" / "raw_results.json"
        if candidate.exists():
            return candidate

    return None


def collect_grid_bounds(raw_data_dir):
    raw_result_files = find_raw_result_files(raw_data_dir)
    if not raw_result_files:
        return 0, 0

    rows = []
    seeds = []
    for raw_file in raw_result_files:
        try:
            rows.append(int(raw_file.parents[2].name.split("_", 1)[1]))
            seeds.append(int(raw_file.parents[1].name.split("_", 1)[1]))
        except (IndexError, ValueError):
            continue

    if not rows or not seeds:
        return 0, 0

    return max(rows) + 1, max(seeds)


def plot_grid(raw_data_dir, numRows=None, numSeeds=None, save_path=None):
    if numRows is None or numSeeds is None:
        inferred_rows, inferred_seeds = collect_grid_bounds(raw_data_dir)
        numRows = inferred_rows if numRows is None else numRows
        numSeeds = inferred_seeds if numSeeds is None else numSeeds

    if numRows <= 0 or numSeeds <= 0:
        return None

    fig, axes = plt.subplots(
        numRows,
        numSeeds,
        figsize=(3.8 * numSeeds, 2.2 * numRows),
        sharex=True,
        sharey=True,
    )

    if numRows == 1 and numSeeds == 1:
        axes = np.array([[axes]])
    elif numRows == 1 or numSeeds == 1:
        axes = np.asarray(axes).reshape(numRows, numSeeds)

    for row in range(numRows):
        for seed in range(1, numSeeds + 1):
            col = seed - 1
            ax = axes[row, col]

            raw_results_file = find_simulation_result_file(raw_data_dir, row, seed)
            if raw_results_file is None:
                ax.set_visible(False)
                continue

            try:
                with open(raw_results_file, "r") as f:
                    raw_results = json.load(f)

                time_values = raw_results.get("time [s]", [])
                beta_p_values = raw_results.get("betaP_1", [])

                ax.plot(time_values, beta_p_values, color="k", lw=1)
                ax.set_title(f"Row {row}, Seed {seed}", fontsize=9)

                text = (
                    f"alphaLT={raw_results.get('alphaLineTension', 'n/a')}\n"
                    f"B0e={raw_results.get('B0e_SI', 'n/a')}\n"
                    f"B0s={raw_results.get('B0s_SI', 'n/a')}\n"
                    f"stress={raw_results.get('stress (MPa)', 'n/a')} MPa\n"
                    f"T={raw_results.get('temperature [K]', 'n/a')} K"
                )
                ax.text(
                    0.02,
                    0.98,
                    text,
                    transform=ax.transAxes,
                    fontsize=7,
                    verticalalignment="top",
                    bbox=dict(facecolor="white", alpha=0.6, lw=0),
                )
            except Exception as exc:
                ax.text(0.5, 0.5, f"Error:\n{exc}", ha="center", va="center", fontsize=7)
                ax.set_visible(True)
                continue

            if row == numRows - 1:
                ax.set_xlabel("Time [s]", fontsize=8)
            if col == 0:
                ax.set_ylabel(r"$\beta^p_{12}$", fontsize=8)

            ax.tick_params(axis="both", labelsize=7)
            ax.grid(True, alpha=0.2)

    plt.subplots_adjust(hspace=0.25, wspace=0.25)
    plt.suptitle(r"$\beta^p_{12}$ vs Time - Drag Line Tension Calibration", fontsize=14, y=0.995)

    if save_path:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path / "globalEventFigure.png", dpi=300, bbox_inches="tight")

    plt.close(fig)
    return fig


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

    simulation_root = raw_file.parents[3]
    copy_tutorial_artifacts(simulation_root, simulation_results_dir)

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

    plot_grid(output_root, save_path=output_root)
    print(f"Wrote grid figure to {Path(output_root) / 'globalEventFigure.png'}")


if __name__ == "__main__":
    main()
