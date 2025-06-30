import os
import sys
import json
import numpy as np
from scipy.stats import qmc
from concurrent.futures import ProcessPoolExecutor, as_completed
import re
import pickle
# from rateCalibration import run_arrhenius_simulation

def simulation_wrapper(args):
    """Wrapper to pass multiple arguments to the simulation function."""
    (latin_hypercube_sample, stress_component, ufl_base, DD_settings, noise_settings,
     material_settings, elasticDeformation_settings, polycrystal_settings, microstructure_settings,
     output_settings, row, seed, detectionMethod, library_driven, build_dir) = args
    
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
        output_settings, row, seed, detectionMethod, library_driven, build_dir
    )

#define a function to run the langevin thermostat simulation and extract the rate
def run_arrhenius_simulation(latin_hypercube_sample,stress_component,ufl,DD_settings,noise_settings,material_settings,elasticDeformation_settings,polycrystal_settings,microstructure_settings,output_settings,row,seed,detectionMethod,library_driven=True,build_dir=False):
    
    returnDict = {}
    
    stress=latin_hypercube_sample["appliedStress"]
    
    formattedStress=formatStress(stress_component,stress/material_settings["mu_0"])
    
    os.chdir(ufl)
    
    #clear old simulation results
    if os.path.exists('inputFiles'):
        shutil.rmtree('inputFiles')
    if os.path.exists('F'):
        shutil.rmtree('F')
    if os.path.exists('evl'):
        shutil.rmtree('evl')
        
    os.makedirs('evl', exist_ok=True)
    os.makedirs('F', exist_ok=True)
    os.makedirs('inputFiles', exist_ok=True)
        
    writeElasticDeformationFile(elasticDeformation_settings["elasticDeformationFile"],formattedStress)
    writeDipoleMicrostructureFile(microstructure_settings["microstructureFile1"],microstructure_settings["slipSystemIDs"],microstructure_settings["exitFaceIDs"],microstructure_settings["dipoleCenters"],microstructure_settings["nodesPerLine"],microstructure_settings["dipoleHeights"],microstructure_settings["glideSteps"])
    writeDDfile(DD_settings["useFEM"],DD_settings["useDislocations"],DD_settings["useInclusions"],DD_settings["useElasticDeformation"],DD_settings["useClusterDynamics"],DD_settings["quadPerLength"],DD_settings["periodic_image_size"],DD_settings["EwaldLengthFactor"],latin_hypercube_sample["coreSize"],latin_hypercube_sample["alphaLineTension"],DD_settings["remeshFrequency"],DD_settings["timeSteppingMethod"],DD_settings["dtMax"],DD_settings["dxMax"],DD_settings["maxJunctionIterations"],DD_settings["use_velocityFilter"],DD_settings["use_stochasticForce"],str(seed),DD_settings["Lmin"],DD_settings["Lmax"],DD_settings["outputFrequency"],DD_settings["outputQuadraturePoints"],DD_settings["glideSolverType"],DD_settings["climbSolverType"],int(DD_settings['Nsteps']))
    writeMaterialFile(material_settings["materialFile"],material_settings["enabledSlipSystems"],material_settings["glidePlaneNoise"],material_settings["atomsPerUnitCell"],material_settings["dislocationMobilityType"],latin_hypercube_sample["B0e_SI"],latin_hypercube_sample["B1e_SI"],latin_hypercube_sample["B0s_SI"],latin_hypercube_sample["B1s_SI"],material_settings["rho"],material_settings["mu_0"],material_settings["mu_1"],material_settings["nu"])
    writePolyCrystalFile(polycrystal_settings["meshFile"],material_settings["materialFile"],latin_hypercube_sample["appliedTemperature"],np.array(polycrystal_settings["grain1globalX1"]),np.array(polycrystal_settings["grain1globalX3"]),np.array(polycrystal_settings["boxEdges"]),np.array(polycrystal_settings["boxScaling"]),np.array(polycrystal_settings["X0"]),np.array(polycrystal_settings["periodicFaceIDs"]),np.array(polycrystal_settings["gridSize_poly"]),np.array(polycrystal_settings["gridSpacing_SI_poly"]))
    writeNoiseFile(noise_settings["noiseFile"],noise_settings["type"],row,noise_settings["seed"],noise_settings["correlationFile_L"],noise_settings["correlationFile_T"],noise_settings["gridSize"],noise_settings["gridSpacing_SI"],noise_settings["a_cai_SI"])

    b=material_settings["b_SI"]
    cs=computeShearWaveSpeed(material_settings["mu_0"],material_settings["rho"])
    
    slipsystemIDs=microstructure_settings["slipSystemIDs"]
    exitFaceIDs=microstructure_settings["exitFaceIDs"]
    dipoleCenters=microstructure_settings["dipoleCenters"]
    dipoleHeights=microstructure_settings["dipoleHeights"]
    nodesPerLine=microstructure_settings["nodesPerLine"]
    glideSteps=microstructure_settings["glideSteps"]
    
    if library_driven:
        # Run the simulation
        ddBase=pyMoDELib.DislocationDynamicsBase(ufl)
        defectiveCrystal=pyMoDELib.DefectiveCrystal(ddBase)
        microstructureGenerator=pyMoDELib.MicrostructureGenerator(ddBase)
        
        spec=pyMoDELib.PeriodicDipoleIndividualSpecification()  #defines the specification of the dipole
        spec.slipSystemIDs=slipsystemIDs
        spec.exitFaceIDs=exitFaceIDs
        # spec.dipoleCenters=np.array([xCenter,xCenter])
        spec.dipoleCenters=dipoleCenters
        spec.dipoleHeights=dipoleHeights
        spec.nodesPerLine=nodesPerLine
        spec.glideSteps=glideSteps
        
        microstructureGenerator.addPeriodicDipoleIndividual(spec)  #adds the periodic dipole to the microstructure
        
        defectiveCrystal.initializeConfiguration(microstructureGenerator.configIO)
        
        defectiveCrystal.runSteps() #may be causing segfault, try ddomp
    else:
        if build_dir:
            # Write the Microstructure file evl with mg
            MG(os.path.abspath(os.path.join(build_dir, "tools")),ufl)
            # Run the simulation using DDomp
            DDomp(os.path.abspath(os.path.join(build_dir, "tools")),ufl)
        else:
            # Write the Microstructure file evl with mg
            MG(os.path.abspath(os.path.join(ufl, "../../build/tools")),ufl)
            # Run the simulation using DDomp
            DDomp(os.path.abspath(os.path.join(ufl, "../../build/tools")),ufl)
    
    print("Simulation completed.")
    
    
    # Extract the time, plastic strain, and plastic strain rate, from the simulation
    F,Flabels=readFfile('./F')
    
    if stress_component==3:
        time_bCs=getFarray(F,Flabels,'time [b/cs]')
        time_s = np.array(time_bCs) / float(cs) * float(b)
        dotBetaP=getFarray(F,Flabels,'dotBetaP_12 [cs/b]')
        betaP_1=getFarray(F,Flabels,'betaP_12')
        
    # Store time_s and betaP_1 as columns in a pandas DataFrame
    simulation_data = pd.DataFrame({
        "time_s": time_s,
        "betaP_1": betaP_1
    })
    
    if detectionMethod=="autoStepFinder":
        
        # Create a directory for the AutoStepFinder data
        auto_stepfinder_dir = os.path.join(outputPath, "autostepfinderData")
        os.makedirs(auto_stepfinder_dir, exist_ok=True)

        # Define the path for the AutoStepFinder output file
        # auto_stepfinder_file_path = os.path.join(auto_stepfinder_dir, f"autoSteppyFinder_{tag}.txt")
        
        formatAutoSteppyFinderTxt(simulationData, auto_stepfinder_dir, f'autoSteppyFinder_{row}.txt')
        
        os.chdir(auto_stepfinder_dir)
        # calculate the rate from the plastic strain
        multiPassCustom(demo=1.1, tresH=0.15, N_iter=N_iter, path=os.path.join(auto_stepfinder_dir,f'autoSteppyFinder_{row}.txt')) #runs autoSteppyFinder

        temperature=latin_hypercube_sample["appliedTemperature"]
        invTemp=1/(temperature)
        
        waiting_times,numEvents=compute_average_dwell_threshold_correlatedJumpsIncluded(name,betaPthreshold)
    elif detectionMethod=="custom":
        print("Using custom step detection for rate calculation")
        figure_dir = os.path.join(output_settings["outputPath"],f"row_{row}",f"seed_{seed}","step_detection_figures")
        os.makedirs(figure_dir, exist_ok=True)
        waiting_times, step_indices, fig = rudimentary_multistep_waiting_times(
        betaP_1, time_s, step_height=0.0085, tolerance=0.002, start_value=0.001,output_stepfit=output_settings["outputStepFitPlots"],plotName=os.path.join(figure_dir, f'step_detection_seed_{seed}_row_{row}.png'))
        if waiting_times:
            print(f"Waiting times detected: {waiting_times}")
            rate= compute_rate(waiting_times)
        else:
            print("No waiting times detected, setting rate to NaN")
            rate = np.nan
        temperature=latin_hypercube_sample["appliedTemperature"]
        invTemp=1/(temperature)
    else:
        print("Using ruptures for step detection")
        # Set the figure path to outputPath/step_detection_figures/seed_{seed}_row_{tag}
        figure_dir = os.path.join(output_settings["outputPath"],f"seed_{seed}",f"row_{row}","step_detection_figures")
        os.makedirs(figure_dir, exist_ok=True)
        step_indices, waiting_times = detect_large_steps(
            betaP_1, time_s,
            min_step_height=0.0001,
            min_distance=15000,
            save_plot=True,
            label=f'stepFit__seed_{seed}_row_{row}',
            figurePath=figure_dir
        )
        temperature=latin_hypercube_sample["appliedTemperature"]
        invTemp=1/(temperature)
        
    print(f"Number of events: {len(step_indices)}")
    print(f"Waiting times: {waiting_times}")
        
     
    #Extract the quantities related to the rate from the simulation
    returnDict['rate']=rate
    returnDict['waitingTimes'] = waiting_times
    returnDict['numEvents'] = len(step_indices)
    returnDict['inverseTemperature'] = invTemp
    returnDict['time [s]'] = time_s
    
    # Extract other simulation results
    returnDict['stress'] = stress
    returnDict['betaP_1'] = betaP_1
    returnDict['dotBetaP'] = dotBetaP
    returnDict['stress_component'] = stress_component
    returnDict['slipSystemIDs'] = slipsystemIDs
    returnDict['exitFaceIDs'] = exitFaceIDs
    returnDict['dipoleCenters'] = dipoleCenters
    returnDict['dipoleHeights'] = dipoleHeights
    returnDict['nodesPerLine'] = nodesPerLine
    returnDict['glideSteps'] = glideSteps
    returnDict['slipsystemIDs'] = slipsystemIDs
    returnDict['exitFaceIDs'] = exitFaceIDs
    returnDict['dipoleCenters'] = dipoleCenters
    returnDict['dipoleHeights'] = dipoleHeights
    returnDict['nodesPerLine'] = nodesPerLine
    returnDict['glideSteps'] = glideSteps
    returnDict['coreSize'] = latin_hypercube_sample["coreSize"]
    returnDict['seed'] = seed

    # Save the returnDict to a text file in the figure_dir
    output_file = os.path.join(figure_dir, f"results_seed_{seed}_row_{row}.txt")
    with open(output_file, "w") as f:
        for key, value in returnDict.items():
            f.write(f"{key}: {value}\n")
        
    # print("returnDict Written")
    # # Print the returnDict for debugging
    # print("Return Dictionary:")
    # for key, value in returnDict.items():
    #     print(f"{key}: {value}")
    
    
    return(returnDict)



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