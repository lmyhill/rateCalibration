#!/usr/bin/env python3

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np

from funct import run_arrhenius_simulation


def json_default(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return str(value)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="config_sweep.json",
        help="Path to sweep configuration file"
    )
    parser.add_argument(
        "--row",
        type=int,
        required=True,
        help="Row index in theta_samples_generated.csv"
    )

    args = parser.parse_args()

    # ----------------------------------------------------------
    # Read configuration
    # ----------------------------------------------------------

    with open(args.config, "r") as f:
        config = json.load(f)

    row = args.row

    ufl_base_directory = config["ufl"]

    output_settings = config["outputSettings"]

    DD_settings = config["DD_settings"]
    noise_settings = config["noise_settings"]
    material_settings = config["material_settings"]
    elasticDeformation_settings = config["elasticDeformation_settings"]
    polycrystal_settings = config["polycrystal_settings"]
    microstructure_settings = config["microstructure_settings"]

    library_driven = config.get("library_driven", True)

    detection_method = config.get(
        "detectionMethod",
        "custom"
    )

    step_detection_settings = config.get(
        "step_detection_settings",
        {}
    )

    crss_settings = config.get(
        "crss_settings",
        {
            "compute_crss": False
        }
    )

    build_dir = config.get(
        "build_dir",
        False
    )

    stress_component = config.get(
        "stress_component",
        3
    )

    # ----------------------------------------------------------
    # Read parameter samples
    # ----------------------------------------------------------

    sample_file = config["samplingSettings"]["inputFile"]

    if not os.path.exists(sample_file):
        raise FileNotFoundError(
            f"Could not find sample file: {sample_file}"
        )

    df = pd.read_csv(sample_file)

    if row >= len(df):
        raise IndexError(
            f"Requested row {row}, but sample file only "
            f"contains {len(df)} rows."
        )

    latin_hypercube_sample = (
        df.iloc[row]
        .to_dict()
    )

    # ----------------------------------------------------------
    # Determine seed
    # ----------------------------------------------------------

    #
    # If you later decide to use
    # multiple seeds per parameter set,
    # add another command line argument.
    #
    seed = 1

    # ----------------------------------------------------------
    # Create isolated run directory under the submission outputs
    # ----------------------------------------------------------

    slurm_job_id = os.environ.get(
        "SLURM_JOB_ID",
        "local"
    )

    slurm_submit_dir = os.environ.get(
        "SLURM_SUBMIT_DIR",
        os.getcwd()
    )

    output_root = os.path.join(
        slurm_submit_dir,
        "outputs"
    )

    run_directory = os.path.join(
        output_root,
        f"row_{row}_job_{slurm_job_id}"
    )

    os.makedirs(run_directory, exist_ok=True)

    #
    output_settings["outputPath"] = run_directory

    print(
        f"Running row {row} "
        f"in {ufl_base_directory}"
    )

    # ----------------------------------------------------------
    # Run simulation
    # ----------------------------------------------------------

    results = run_arrhenius_simulation(
        latin_hypercube_sample=latin_hypercube_sample,
        stress_component=stress_component,
        ufl=ufl_base_directory,
        DD_settings=DD_settings,
        noise_settings=noise_settings,
        material_settings=material_settings,
        elasticDeformation_settings=elasticDeformation_settings,
        polycrystal_settings=polycrystal_settings,
        microstructure_settings=microstructure_settings,
        output_settings=output_settings,
        row=row,
        seed=seed,
        detectionMethod=detection_method,
        step_detction_settings=step_detection_settings,
        library_driven=library_driven,
        build_dir=build_dir,
        crss_settings=crss_settings
    )

    # ----------------------------------------------------------
    # Save raw simulation results for post-processing
    # ----------------------------------------------------------

    results_dir = os.path.join(
        run_directory,
        f"row_{row}",
        f"seed_{seed}",
        "simulation_results"
    )

    os.makedirs(results_dir, exist_ok=True)

    results_file = os.path.join(
        results_dir,
        "raw_results.json"
    )

    with open(results_file, "w") as f:
        json.dump(
            results,
            f,
            indent=4,
            default=json_default
        )

    print(
        f"Completed row {row}; raw results saved to {results_file}"
    )

    return


if __name__ == "__main__":
    main()