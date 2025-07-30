import os
import sys
import json
import numpy as np
from scipy.stats import qmc
from concurrent.futures import ProcessPoolExecutor, as_completed
import re
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import shutil
from funct import run_arrhenius_simulation

def simulation_wrapper(args):
    """Wrapper to pass multiple arguments to the simulation function."""
    (latin_hypercube_sample, stress_component, ufl_base, DD_settings, noise_settings,
     material_settings, elasticDeformation_settings, polycrystal_settings, microstructure_settings,
     output_settings, row, seed, detectionMethod, step_detection_settings, library_driven, build_dir) = args
    
    job_id = None
    if build_dir:
        match = re.search(r'build_(\d+)$', build_dir)
        if match:
            job_id = match.group(1)
        else:
            job_id = "unknown"
    else:
        job_id = "default_job_id"

    # Create a unique simulation directory for this row and seed
    unique_ufl = os.path.join(ufl_base, f'row_{row}_seed_{seed}_job_{job_id}')
    os.makedirs(unique_ufl, exist_ok=True)

    # Call your simulation function
    return run_arrhenius_simulation(
        latin_hypercube_sample, stress_component, unique_ufl, DD_settings, noise_settings,
        material_settings, elasticDeformation_settings, polycrystal_settings, microstructure_settings,
        output_settings, row, seed, detectionMethod, step_detection_settings, library_driven, build_dir
    )



def main():
    
    # Load configuration from config.json
    config_path = "config.json"
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found.")
        sys.exit(1)

    with open(config_path, "r") as config_file:
        config = json.load(config_file)

    # Extract settings from the configuration
    ufl_base_directory = config.get("ufl", "")
    run_simulation = config.get("run_arrhenius_simulation", False)
    application_domain= config.get("application_domain", "")
    input_metrics = config.get("inputMetrics", {})
    number_of_simulations = config.get("numberOfSimulations", 100)
    number_of_seeds_per_simulation= config.get("numberofSeedsPerSimulation", 1)
    output_settings = config.get("outputSettings", {})
    
    DD_settings = config.get("DD_settings", {})
    noise_settings = config.get("noise_settings", {})
    material_settings = config.get("material_settings", {})
    elasticDeformation_settings = config.get("elasticDeformation_settings", {})
    polycrystal_settings = config.get("polycrystal_settings", {})
    microstructure_settings = config.get("microstructure_settings", {})
    
    library_driven = config.get("library_driven", True)
    
    step_detection_settings = config.get("step_detection_settings", {})

    speedup_settings = config.get("speedup_settings", {})
    
    build_dir = config.get("build_dir", False)

    copy_config = config.get("copy_config", False)

    detectionMethod = config.get("detectionMethod", "custom")  
    stress_component = config.get("stress_component", 3)     
    
    # Ensure the output directory exists
    os.makedirs(output_settings["outputPath"], exist_ok=True)
    
    # Combine the application domain and input metrics into a unified dictionary
    combined_metrics = {}

    # Add application domain metrics
    if isinstance(application_domain, dict):
        for key, value in application_domain.items():
            combined_metrics[key] = value
    else:
        print("Warning: application_domain is not a dictionary. Skipping.")

    # Add input metrics
    combined_metrics.update(input_metrics)

    # Generate the latin hypercube using scipy.stats.qmc.LatinHypercube
    sampler = qmc.LatinHypercube(d=len(combined_metrics))
    sample = sampler.random(n=number_of_simulations)
    
    # Scale the sample to the specified bounds
    bounds = np.array([[metric_bounds.get("min_value", 0), metric_bounds.get("max_value", 1)] 
                        for metric_bounds in combined_metrics.values()])
    
    # Print the bounds for debugging
    print("Bounds:")
    for i, (metric, metric_bounds) in enumerate(input_metrics.items()):
        print(f"{metric}: min={metric_bounds.get('min_value', 0)}, max={metric_bounds.get('max_value', 1)}")
    
    scaled_sample = qmc.scale(sample, bounds[:, 0], bounds[:, 1])
    
    # Map the scaled sample to the corresponding metrics
    latin_hypercube = {metric: scaled_sample[:, i] for i, metric in enumerate(combined_metrics.keys())}
    
    # Prepare list of per-row samples
    latin_hypercube_sample_list = []
    num_rows = len(next(iter(latin_hypercube.values())))

    for i in range(num_rows):
        sample = {key: latin_hypercube[key][i] for key in latin_hypercube}
        latin_hypercube_sample_list.append(sample)

    seeds = list(range(number_of_seeds_per_simulation))

    jobs = []
    for row in range(num_rows):
        for seed in seeds:
            jobs.append((
                latin_hypercube_sample_list[row],
                stress_component,
                ufl_base_directory,
                DD_settings,
                noise_settings,
                material_settings,
                elasticDeformation_settings,
                polycrystal_settings,
                microstructure_settings,
                output_settings,
                row,
                seed,
                detectionMethod,
                step_detection_settings,
                library_driven,
                build_dir
            ))

    results = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = [executor.submit(simulation_wrapper, job) for job in jobs]
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Simulation failed with error: {e}")

    print(f"Completed {len(results)} simulations.")



    return

if __name__ == "__main__":
    main()