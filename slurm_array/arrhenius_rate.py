#!/usr/bin/env python3

import os
import sys
import json
import argparse
import shutil
import pandas as pd
import numpy as np

from funct import run_arrhenius_simulation


def json_default(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return str(value)


def resolve_tutorial_root(ufl_path):
    normalized_path = os.path.normpath(ufl_path)
    if os.path.basename(normalized_path) == "tutorials":
        return normalized_path

    candidate = os.path.join(normalized_path, "tutorials")
    if os.path.isdir(candidate):
        return candidate

    return normalized_path

def extract_sample_parameters(config):
    """
    Extract parameters marked for LHS sampling.

    Returns:
        dict:
            {
                parameter_name: {
                    min_value: ...,
                    max_value: ...
                }
            }
    """

    sampled_parameters = {}

    def recurse(settings):
        for key, value in settings.items():

            if isinstance(value, dict):

                if value.get("sample", False):
                    sampled_parameters[key] = {
                        "min_value": value["min_value"],
                        "max_value": value["max_value"]
                    }

                else:
                    recurse(value)

    recurse(config)

    return sampled_parameters

def resolve_settings(settings, sample):
    """
    Replace sampled parameters with LHS values.
    """

    resolved = {}

    for key, value in settings.items():

        if isinstance(value, dict):

            if value.get("sample", False):

                resolved[key] = sample[key]

            else:
                resolved[key] = resolve_settings(
                    value,
                    sample
                )

        else:
            resolved[key] = value

    return resolved

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

    ufl_base_directory = resolve_tutorial_root(config["ufl"])

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

    stress_component = config["application_domain"]["appliedStress"].get(
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

    raw_sample = (
        df.iloc[row]
        .to_dict()
    )


    sampling_parameters = extract_sample_parameters(config)


    latin_hypercube_sample = {}

    for parameter in sampling_parameters:

        if parameter not in raw_sample:
            raise KeyError(
                f"Sample parameter '{parameter}' "
                "missing from CSV"
            )

        latin_hypercube_sample[parameter] = (
            raw_sample[parameter]
        )

    resolved_DD_settings = resolve_settings(
        DD_settings,
        latin_hypercube_sample
    )

    resolved_material_settings = resolve_settings(
        material_settings,
        latin_hypercube_sample
    )

    resolved_application_domain = resolve_settings(
        config["application_domain"],
        latin_hypercube_sample
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

    simulation_run_directory = os.path.join(
        ufl_base_directory,
        f"row_{row}_job_{slurm_job_id}"
    )

    if os.path.exists(simulation_run_directory):
        shutil.rmtree(simulation_run_directory)

    os.makedirs(simulation_run_directory, exist_ok=True)

    run_directory = os.path.join(
        output_root,
        f"row_{row}_job_{slurm_job_id}"
    )

    os.makedirs(run_directory, exist_ok=True)

    output_settings["outputPath"] = run_directory

    print(
        f"Running row {row} "
        f"in {simulation_run_directory}"
    )

    # ----------------------------------------------------------
    # Run simulation
    # ----------------------------------------------------------

    results = run_arrhenius_simulation(
        application_domain=resolved_application_domain,
        sampled_parameters=latin_hypercube_sample,
        ufl=simulation_run_directory,
        DD_settings=resolved_DD_settings,
        noise_settings=noise_settings,
        material_settings=resolved_material_settings,
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