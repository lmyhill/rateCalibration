#The purpose of this file is to calibrate the arrhenius rate of DDD data to MD ground truth
#The script runs a series of modelib simulations at various input settings to generate a latin hypercube

#import necessary libraries
import os
import sys
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import qmc
import shutil

config_path = "config.json"
if not os.path.exists(config_path):
    print(f"Error: {config_path} not found.")
    sys.exit(1)

with open(config_path, "r") as config_file:
    config = json.load(config_file)
    if config["step_detection_settings"]["detectionMethod"] == "autoStepFinder":
        autoSteppyFinderLoc=config["arrheniusSettings"]["autoStepFinderPath"]
        sys.path.insert(1,autoSteppyFinderLoc)
        from AutoSteppyfinder import *
    if config["run_arrhenius_simulation"]:
        sys.path.append(config["modelLibraryPath"])
        sys.path.append(config["modelPythonPath"])
        import pyMoDELib
        from modlibUtils import *

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
    figure_dir = os.path.join(output_settings["outputPath"],f"row_{row}",f"seed_{seed}","simulation_results")
    os.makedirs(figure_dir, exist_ok=True)
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

def runCRSS_simulation(latin_hypercube_sample,stress_component,ufl,DD_settings,noise_settings,material_settings,elasticDeformation_settings,polycrystal_settings,microstructure_settings,output_settings,row,seed,detectionMethod,library_driven=True):
    
    return()

def rudimentary_multistep_waiting_times(signal, time, step_height, tolerance, start_value=None,output_stepfit=True,plotName='stepFitOutput'):
    """
    Detects waiting times where the signal stays within a band of width 2*tolerance,
    with bands spaced by step_height, starting from start_value (or min(signal) if None).
    The number of bands is determined by the signal's min and max values.

    Parameters:
        signal (np.ndarray): 1D array of signal values.
        time (np.ndarray): 1D array of time values (same length as signal).
        step_height (float): The target step height to define bands.
        tolerance (float): Allowed deviation from the band center.
        start_value (float or None): Center of the first band. If None, uses min(signal).

    Returns:
        waiting_times (list): List of waiting times (float) for each band visit.
        band_indices (list): List of (start_idx, end_idx, band_center) for each waiting time.
        fig (matplotlib.figure.Figure): The plot figure object.
    """
    import matplotlib.pyplot as plt

    signal = np.asarray(signal)
    time = np.asarray(time)
    waiting_times = []
    band_indices = []

    if start_value is None:
        base = np.min(signal)
    else:
        base = start_value

    max_val = np.max(signal)
    n_bands = int(np.ceil((max_val - base) / step_height)) + 1
    band_centers = [base + i * step_height for i in range(n_bands)]

    idx = 0
    n = len(signal)
    while idx < n:
        # Find which band the current value is in
        for band_center in band_centers:
            if abs(signal[idx] - band_center) <= tolerance:
                # Entered a band, now find how long it stays in this band
                start_idx = idx
                while idx < n and abs(signal[idx] - band_center) <= tolerance:
                    idx += 1
                end_idx = idx - 1
                waiting_time = time[end_idx] - time[start_idx]
                waiting_times.append(waiting_time)
                band_indices.append((start_idx, end_idx, band_center))
                break
        else:
            idx += 1

    # If only one band is detected, return NaN for waiting times and band indices
    if len(waiting_times) == 1:
        waiting_times = [np.nan]

    if len(waiting_times) > 1:
        #ignore the last waiting time because no event is detected at the tail of the signal
        waiting_times = waiting_times[:-1]


    if output_stepfit:
    # Plotting
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(time, signal, label='Signal')
        colors = plt.cm.viridis(np.linspace(0, 1, n_bands))
        for i, (start_idx, end_idx, band_center) in enumerate(band_indices):
            ax.axvspan(time[start_idx], time[end_idx], color=colors[i % n_bands], alpha=0.2)
            ax.axhspan(band_center - tolerance, band_center + tolerance, color=colors[i % n_bands], alpha=0.15)
            ax.axhline(band_center, color=colors[i % n_bands], linestyle='--', alpha=0.5)
        ax.set_xlabel('time [s]')
        ax.set_ylabel('Signal')
        ax.set_title('Waiting Times in Bands (step_height={}, tolerance={})'.format(step_height, tolerance))
        ax.legend()
        plt.tight_layout()
        # plt.show()
        plt.savefig(plotName)

    return waiting_times, band_indices, fig

def compute_rate(waiting_times):
    """
    Compute the rate as the inverse of the average waiting time.

    Parameters:
        waiting_times (array-like): Collection of waiting times (list, numpy array, etc.)

    Returns:
        rate (float): Computed rate (1 / average waiting time). Returns np.nan if input is empty.
    """
    waiting_times = np.asarray(waiting_times)
    if waiting_times.size == 0:
        return np.nan
    return 1.0 / np.mean(waiting_times)

#define a function to write the input settings of DD.txt
def writeDDfile(useFEM,useDislocations,useInclusions,useElasticDeformation,useClusterDynamics,quadPerLength,periodicImageSize,EwaldLengthFactor,coreSize,alphaLineTension,remeshFrequency,timeSteppingMethod,dtMax,dxMax,maxJunctionIterations,use_velocityFilter,use_stochasticForce,stochasticForce_seed,Lmin,Lmax,outputFrequency,outputQuadraturePoints,glideSolverType,climbSolverType,Nsteps):
    
    # Make a local copy of DD parameters file and modify that copy if necessary
    DDfile='DD.txt'
    DDfileTemplate='../../Library/DislocationDynamics/'+DDfile
    print("\033[1;32mCreating  DDfile\033[0m")
    shutil.copy2(DDfileTemplate,'inputFiles/'+DDfile)
    setInputVariable('inputFiles/'+DDfile,'useFEM',useFEM)
    setInputVariable('inputFiles/'+DDfile,'useDislocations',useDislocations)
    setInputVariable('inputFiles/'+DDfile,'useInclusions',useInclusions)
    setInputVariable('inputFiles/'+DDfile,'useElasticDeformation',useElasticDeformation)
    setInputVariable('inputFiles/'+DDfile,'useClusterDynamics',useClusterDynamics)
    setInputVariable('inputFiles/'+DDfile,'quadPerLength',quadPerLength)
    setInputVariable('inputFiles/'+DDfile,'periodicImageSize',periodicImageSize)
    setInputVariable('inputFiles/'+DDfile,'EwaldLengthFactor',EwaldLengthFactor)
    setInputVariable('inputFiles/'+DDfile,'coreSize',str(coreSize))
    setInputVariable('inputFiles/'+DDfile,'alphaLineTension',str(alphaLineTension)) # dimensionless scale factor in for line tension forces
    setInputVariable('inputFiles/'+DDfile,'remeshFrequency',remeshFrequency)
    setInputVariable('inputFiles/'+DDfile,'timeSteppingMethod',timeSteppingMethod) # adaptive or fixed
    setInputVariable('inputFiles/'+DDfile,'dtMax',dtMax)
    setInputVariable('inputFiles/'+DDfile,'dxMax',dxMax) # max nodal displacement for when timeSteppingMethod=adaptive
    setInputVariable('inputFiles/'+DDfile,'maxJunctionIterations',maxJunctionIterations) # dimensionless scale factor in for line tension forces
    setInputVariable('inputFiles/'+DDfile,'use_velocityFilter',use_velocityFilter) # don't filter velocity if noise is enabled
    setInputVariable('inputFiles/'+DDfile,'use_stochasticForce',use_stochasticForce) # Langevin thermal noise enabled
    setInputVariable('inputFiles/'+DDfile,'stochasticForceSeed',stochasticForce_seed) # Langevin thermal noise enabled
    setInputVariable('inputFiles/'+DDfile,'Lmin',Lmin)  # min segment length (in Burgers vector units)
    setInputVariable('inputFiles/'+DDfile,'Lmax',Lmax)  # max segment length (in Burgers vector units)
    setInputVariable('inputFiles/'+DDfile,'outputFrequency',outputFrequency)  # output frequency
    setInputVariable('inputFiles/'+DDfile,'outputQuadraturePoints',outputQuadraturePoints)  # output quadrature data
    setInputVariable('inputFiles/'+DDfile,'glideSolverType',glideSolverType)  # type of glide solver, or none
    setInputVariable('inputFiles/'+DDfile,'climbSolverType',climbSolverType)  # type of clim solver, or none
    setInputVariable('inputFiles/'+DDfile,'Nsteps',str(Nsteps))  # number of simulation steps

    
    
    return()
    
#define a function to modify only the number of simulation steps
def addSimSteps(DDfile,Nsteps):
   
    setInputVariable('inputFiles/'+DDfile,'Nsteps',str(Nsteps))  # number of simulation steps
    
    return()

# define a function to write the input settings of material.txt
def writeMaterialFile(MaterialFile,enabledSlipSystems,glidePlaneNoise,atomsPerUnitCell,dislocationMobilityType,B0e_SI,B1e_SI,B0s_SI,B1s_SI,rho,mu_0,mu_1,nu):
    materialFile=MaterialFile;
    materialFileTemplate='../../Library/Materials/'+materialFile;
    print("\033[1;32mCreating  materialFile\033[0m")
    shutil.copy2(materialFileTemplate,'inputFiles/'+materialFile)
    setInputVariable('inputFiles/'+materialFile,'enabledSlipSystems',enabledSlipSystems)
    setInputVariable('inputFiles/'+materialFile,'glidePlaneNoise',glidePlaneNoise)
    setInputVariable('inputFiles/'+materialFile,'atomsPerUnitCell',atomsPerUnitCell)
    setInputVariable('inputFiles/'+materialFile,'dislocationMobilityType',dislocationMobilityType)
    setInputVariable('inputFiles/'+materialFile,'B0e_SI',str(B0e_SI))
    setInputVariable('inputFiles/'+materialFile,'B1e_SI',str(B1e_SI))
    setInputVariable('inputFiles/'+materialFile,'B0s_SI',str(B0s_SI))
    setInputVariable('inputFiles/'+materialFile,'B1s_SI',str(B1s_SI))
    setInputVariable('inputFiles/'+materialFile,'rho_SI',str(rho))
    setInputVariable('inputFiles/'+materialFile,'mu0_SI',str(mu_0))
    setInputVariable('inputFiles/'+materialFile,'mu1_SI',str(mu_1))
    setInputVariable('inputFiles/'+materialFile,'nu',str(nu))
    return()

# define a function to write the input settings of polycrystal.txt
def writePolyCrystalFile(meshFile,materialFile,absoluteTemperature,grain1globalX1,grain1globalX3,boxEdges,boxScaling,X0,periodicFaceIDs,gridSize_poly,gridSpacing_SI_poly):
    # Create polycrystal.txt using local material file
    meshFile=meshFile;
    meshFileTemplate='../../Library/Meshes/'+meshFile;
    print("\033[1;32mCreating  polycrystalFile\033[0m")
    shutil.copy2(meshFileTemplate,'inputFiles/'+meshFile)
    pf=PolyCrystalFile(materialFile);
    pf.absoluteTemperature=absoluteTemperature;
    pf.meshFile=meshFile
    pf.grain1globalX1=grain1globalX1     # global x1 axis. Overwritten if alignToSlipSystem0=true
    pf.grain1globalX3=grain1globalX3    # global x3 axis. Overwritten if alignToSlipSystem0=true
    pf.boxEdges=boxEdges # i-throw is the direction of i-th box edge
    pf.boxScaling=boxScaling # must be a vector of integers
    pf.X0=X0 # Centering unitCube mesh. Mesh nodes X are mapped to x=F*(X-X0)
    pf.periodicFaceIDs=periodicFaceIDs
    pf.gridSize=gridSize_poly
    pf.gridSpacing_SI=gridSpacing_SI_poly
    pf.write('inputFiles')

#define a function to write the input settings of noise.txt
def writeNoiseFile(noiseFile,type,tag,seed,correlationFile_L,correlationFile_T,gridSize,gridSpacing_SI,a_cai_SI):
    # Make a local copy of noise file, and modify that copy if necessary
    if noiseFile:
        noiseFile=noiseFile;
        noiseFileTemplate='../../Library/GlidePlaneNoise/'+noiseFile;
        print("\033[1;32mCreating  noiseFile\033[0m")
        shutil.copy2(noiseFileTemplate,'inputFiles/'+noiseFile) # target filename is /dst/dir/file.ext
        setInputVariable('inputFiles/'+noiseFile,'type',type)
        setInputVariable('inputFiles/'+noiseFile,'tag',tag)
        setInputVariable('inputFiles/'+noiseFile,'seed',seed)
        setInputVariable('inputFiles/'+noiseFile,'correlationFile_L',correlationFile_L)
        setInputVariable('inputFiles/'+noiseFile,'correlationFile_T',correlationFile_T)
        setInputVector('inputFiles/'+noiseFile,'gridSize',gridSize,'number of grid points in each direction')
        gridSpacing_SI=gridSpacing_SI
        # gridSpacing_burgers=gridSpacing_SI*2.796
        setInputVector('inputFiles/'+noiseFile,'gridSpacing_SI',gridSpacing_SI,'grid spacing in each direction')
        setInputVariable('inputFiles/'+noiseFile,'a_cai_SI',a_cai_SI)
    
        if noiseFile=='MDSolidSolution.txt' and not os.path.exists('inputFiles/MDSolidSolutionCorrelations_L_MoNbTi.vtk') and not os.path.exists('inputFiles/MDSolidSolutionCorrelations_T_MoNbTi.vtk'):
            shutil.copy2('../../Library/GlidePlaneNoise/MoDELCompatible_MoNbTi_xy.vtk','inputFiles/MDSolidSolutionCorrelations_L_MoNbTi.vtk')
            shutil.copy2('../../Library/GlidePlaneNoise/MoDELCompatible_MoNbTi_xz.vtk','inputFiles/MDSolidSolutionCorrelations_T_MoNbTi.vtk')
    return()
    
# define a function to write the input settings of microstructure.txt (assuming it is a dipole file)
def writeDipoleMicrostructureFile(microstructureFile,slipSystemIDs,exitFaceIDs,dipoleCenters,nodesPerLine,dipoleHeights,glideSteps):
    # make a local copy of microstructure file, and modify that copy if necessary
    microstructureFile1=microstructureFile;
    microstructureFileTemplate='../../Library/Microstructures/'+microstructureFile1;
    print("\033[1;32mCreating  microstructureFile\033[0m")
    shutil.copy2(microstructureFileTemplate,'inputFiles/'+microstructureFile1) # target filename is /dst/dir/file.ext
    setInputVector('inputFiles/'+microstructureFile1,'slipSystemIDs',slipSystemIDs,'slip system IDs for each dipole')
    setInputVector('inputFiles/'+microstructureFile1,'exitFaceIDs',exitFaceIDs,'0 is for edge, 4 for screw')
    if np.array(dipoleCenters).ndim == 2 and all(len(row) == len(dipoleCenters[0]) for row in dipoleCenters):
        setInput2DVector('inputFiles/'+microstructureFile1, 'dipoleCenters', np.array(dipoleCenters), 'center of each dipole')
    else:
        setInputVector('inputFiles/'+microstructureFile1, 'dipoleCenters', dipoleCenters, 'center of each dipole')
    setInputVector('inputFiles/'+microstructureFile1,'nodesPerLine',nodesPerLine,'number of extra nodes on each dipole')
    setInputVector('inputFiles/'+microstructureFile1,'dipoleHeights',dipoleHeights,'height of each dipole, in number of planes')
    setInputVector('inputFiles/'+microstructureFile1,'glideSteps',glideSteps,'step of each dipole in the glide plane')

    print("\033[1;32mCreating  initialMicrostructureFile\033[0m")
    with open('inputFiles/initialMicrostructure.txt', "w") as initialMicrostructureFile:
        initialMicrostructureFile.write('microstructureFile='+microstructureFile1+';\n')
    return()

# define a function to write the input settings of elasticDeformation.txt
def writeElasticDeformationFile(elasticDeformationFile,ExternalStress0):
    # Make a local copy of ElasticDeformation file, and modify that copy if necessary
    elasticDeformationFile=elasticDeformationFile;
    elasticDeformationFileTemplate='../../Library/ElasticDeformation/'+elasticDeformationFile;
    print("\033[1;32mCreating  elasticDeformationFile\033[0m")
    shutil.copy2(elasticDeformationFileTemplate,'inputFiles/'+elasticDeformationFile)
    setInputVector('inputFiles/'+elasticDeformationFile,'ExternalStress0',ExternalStress0,'applied stress')
    return()

#define a function to format stress (voigt notation)
#return(stress)
def formatStress(stress_component,stress_mu):
    stress=np.zeros(6)
    stress_component=int(stress_component)
    if stress_component==0:
        stress[0]=stress_mu
    elif stress_component==1:
        stress[1]=stress_mu
    elif stress_component==2:
        stress[2]=stress_mu
    elif stress_component==3:
        stress[3]=stress_mu
    elif stress_component==4:
        stress[4]=stress_mu
    elif stress_component==5:
        stress[5]=stress_mu
    return(stress)

#define a function to compute the shear wave speed
def computeShearWaveSpeed(mu,rho):
    # Compute the shear wave speed
    c_s = np.sqrt(float(mu) / float(rho))
    return c_s

def compute_average_dwell_threshold_correlatedJumpsIncluded(filename, threshold):
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(filename, sep='\t')
    
    # Initialize a list to store indices of rows to be eliminated
    rows_to_drop = []
    
    # Iterate through each row in the DataFrame
    for i in range(len(df)):
        # If the 'step' value is below the first critical value
        if abs(df.loc[i, 'step']) < threshold:
            # If 'dwell after' is below the second critical value
            if df.loc[i, 'dwell after'] < 100:
                # Initialize a variable to store the sum of 'steps'
                sum_steps = abs(df.loc[i, 'step'])
                # Check subsequent rows until the sum of 'steps' is above the first critical value
                j = i + 1
                while j < len(df) and sum_steps < threshold:
                    # If 'dwell after' is below the second critical value, add 'step' to sum
                    if df.loc[j, 'dwell after'] < 100:
                        sum_steps += abs(df.loc[j, 'step'])
                    else:
                        break  # Exit loop if 'dwell after' exceeds second critical value
                    j += 1
                
                # If the sum of 'steps' is above or equal to the first critical value
                if sum_steps >= threshold:
                    # Keep the 'dwell before' of the first identified row in the column average
                    break
                else:
                    rows_to_drop.append(i)  # Mark the current row for elimination
    
    # Drop rows marked for elimination
    df = df.drop(rows_to_drop)
    
    # Compute the average of the column 'dwell before'
    average_dwell = df['dwell before'].mean()

    print(f'average_dwell:{average_dwell}')
    
    numEvents = df.shape[0]  # Number of remaining events

    print(f'numEvents: {numEvents}')
    
    # Example usage:
    # filename = "your_csv_file.csv"
    # threshold = 1.4  # Example threshold value
    # average_dwell, numEvents = compute_average_dwell_threshold(filename, threshold)
    # print("Average dwell before:", average_dwell)
    # print("Number of events:", numEvents)
    
    return(average_dwell, numEvents)

#define a function to format the df of a dataframe such that it is compatible with autoSteppyFinder.py [@author: Jacob Kerssemakers]
#return: nothing
def formatAutoSteppyFinderTxtDDD(df,filePath,outputFileName):

    cwd=os.getcwd()
    
    os.chdir(filePath)
    
    with open(f'{outputFileName}','w') as f:
        for i in range(1,np.size(df['time [s]'])):
            f.write(str(str(df['betaP_12'][i])+'\n'))

    os.chdir(cwd)
    return()

def detect_large_steps(signal, time, min_step_height=0.0001, min_distance=15000,save_plot=True,label='betaP_12',figurePath=None):
    """
    Detect large step changes in a signal based on first differences.
    
    Parameters:
        signal (np.ndarray): The signal data.
        time (np.ndarray): Corresponding time values.
        min_step_height (float): Minimum height of step to detect.
        min_distance (int): Minimum number of points between steps.

    Returns:
        step_indices (np.ndarray): Indices where steps occur.
        waiting_times (np.ndarray): Waiting times before each step.
    """
    # First difference
    delta = np.diff(signal)
    abs_delta = np.abs(delta)
    
    # Candidate indices where jump exceeds threshold
    candidates = np.where(abs_delta > min_step_height)[0] + 1  # +1 to shift index after diff

    # Filter based on minimum distance
    if len(candidates) == 0:
        return np.array([]), np.array([])

    filtered = [candidates[0]]
    for idx in candidates[1:]:
        if idx - filtered[-1] >= min_distance:
            filtered.append(idx)

    step_indices = np.array(filtered)
    waiting_times = np.diff(np.insert(time[step_indices], 0, time[0]))
    
    # Ignore the first detected step (if any)
    if len(step_indices) > 1:
        step_indices = step_indices[1:]
        waiting_times = waiting_times[1:]

    print(f"Number of large steps detected (ignoring first): {len(step_indices)}")
    print(f"Waiting times between large steps (ignoring first): {waiting_times}")


    if save_plot:
        plt.figure(figsize=(10,6))
        plt.plot(time, signal, label='Signal', alpha=0.6)
        plt.scatter(time[step_indices.astype(int)], signal[step_indices.astype(int)], color='orange', label='Steps (First Ignored)')
        plt.xlabel("time [b/cs]")
        plt.ylabel("betaP_12")
        plt.title("Step Detection via First Differences (First Step Ignored)")
        plt.legend()
        if figurePath is not None:
            plt.savefig(os.path.join(figurePath, f'step_detection_{label}.png'))
    
    # if save_plot:
    #     plt.figure(figsize=(10,6))
    #     plt.plot(time, signal, label='Signal', alpha=0.6)
    #     plt.scatter(time[step_indices.astype(int)], signal[step_indices.astype(int)], color='red', label='Detected Steps')
    #     plt.xlabel("time [b/cs]")
    #     plt.ylabel("betaP_12")
    #     plt.title("Step Detection via First Differences")
    #     plt.legend()
    #     plt.savefig(f'step_detection_{label}.png')
    #     # plt.show()
        

    return step_indices, waiting_times

#function to run DDomp
#return()
def DDomp(toolsDir,ufl):

    print('Running DDomp')

    currentDir=os.getcwd()

    os.chdir(toolsDir+'/DDomp')
    os.system('./DDomp '+ufl)

    os.chdir(currentDir)

    return()

def MG(toolsDir,ufl):

    print('Running microstructureGenerator...')

    currentDir=os.getcwd()

    os.chdir(toolsDir+'/MicrostructureGenerator/')
    os.system('./microstructureGenerator '+ufl)

    os.chdir(currentDir)
    return()

def plot_data(x, y, title, filename):
    """
    Plots two vectors of data x and y and saves the figure to a file.

    Parameters:
    - x: Vector of x-axis data.
    - y: Vector of y-axis data.
    - filename: The filename to save the figure as (including extension, e.g., 'figure.png').
    - title: The title of the plot.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(x, y, marker='o', linestyle='-', color='b', label='Data')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def main():
    
    # Load configuration from config.json
    config_path = "config.json"
    if not os.path.exists(config_path):
        print(f"Error: {config_path} not found.")
        sys.exit(1)

    with open(config_path, "r") as config_file:
        config = json.load(config_file)

    # Extract settings from the configuration
    ufl = config.get("ufl", "")
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

    # Validate and print the configuration
    # print("Configuration loaded:")
    # print(f"ufl: {ufl}")
    # print(f"Input Metrics: {input_metrics}")
    # print(f"Number of Simulations: {number_of_simulations}")
    # print(f"Output Settings: {output_settings}")
    
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
    
    # Print the keys and items in combined_metrics
    print("Combined Metrics:")
    for key, value in combined_metrics.items():
        print(f"{key}: {value}")
    
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
    
    print("Generated Latin Hypercube:")
    for metric, values in latin_hypercube.items():
        print(f"{metric}: {values[:5]}...")  # Print first 5 values for brevity

    if copy_config:
        # Copy the config.json file to the output directory
        config_output_path = os.path.join(output_settings["outputPath"], "config.json")
        shutil.copy(config_path, config_output_path)
        print(f"Config file copied to {config_output_path}")

    if output_settings.get("outputHypercube", False):

        # Save the generated Latin Hypercube
        output_hypercube_file_name = output_settings.get("outputHypercubeFileName", "latin_hypercube")
        output_hypercube_format = output_settings.get("outputHypercubeFormat", "txt").lower()
        output_hypercube_path = output_settings.get("outputPath", ".")
        
        # Create a directory for the hypercube info in the outputPath
        hypercube_dir = os.path.join(output_hypercube_path, "hypercube_info")
        os.makedirs(hypercube_dir, exist_ok=True)

        output_hypercube_file_path = os.path.join(hypercube_dir, f"{output_hypercube_file_name}.{output_hypercube_format}")

        if output_hypercube_format == "txt":
            pd.DataFrame(latin_hypercube).to_csv(output_hypercube_file_path, sep='\t', index=False)
            # Rename the .csv file to .txt
            txt_file_path = output_hypercube_file_path.replace(".csv", ".txt")
            os.rename(output_hypercube_file_path, txt_file_path)
            output_file_path = txt_file_path
            
        elif output_hypercube_format == "json":
            with open(output_hypercube_file_path, "w") as output_file:
                json.dump(latin_hypercube, output_file, indent=4)
        else:
            print(f"Error: Unsupported output format '{output_hypercube_format}'.")
            sys.exit(1)

        print(f"Latin Hypercube saved to {output_hypercube_file_path}")
    
    if run_simulation and not speedup_settings.get("useSpeedup", False):
        
        #create an id
        id_tag = 0
        # creat a dictionary to append the simulation results
        simulation_results = {}
        
        # Run the simulation with the generated Latin Hypercube
        # This is a placeholder for the actual simulation code
        print("Running simulation...")
        
        # Transpose the latin_hypercube dictionary to group values by rows
        grouped_values = zip(*latin_hypercube.values())
        
        for row_index, row_values in enumerate(grouped_values):
            for seed in range(1, number_of_seeds_per_simulation + 1):
                print(f'Running simulation for row {row_index} with values: {row_values} for seed {seed}')
                
                # Set the seed for reproducibility
                np.random.seed(seed)
                
                #create directories for each seed of the simulation
                row_dir = os.path.join(output_settings["outputPath"], f"row_{row_index}")
                seed_dir = os.path.join(output_settings["outputPath"], f"seed_{seed}")
                os.makedirs(seed_dir, exist_ok=True)
                seed_row_dir = os.path.join(row_dir, f"seed_{seed}")
                
                # Map the row values back to their corresponding metrics
                latin_hypercube_sample = {metric: value for metric, value in zip(latin_hypercube.keys(), row_values)}
                print(f"Sample: {latin_hypercube_sample}")
                stress_component = config["application_domain"]["appliedStress"]["stress_component"]
                
                detectionMethod = config["step_detection_settings"]["detectionMethod"]
                
                single_sim_dict = {}
                single_sim_dict = run_arrhenius_simulation(latin_hypercube_sample,stress_component,ufl,DD_settings,noise_settings,material_settings,elasticDeformation_settings,polycrystal_settings,microstructure_settings,output_settings,row_index,seed,detectionMethod,library_driven,build_dir)
                if output_settings.get("outputInputFilesDirectory", False):
                    # Create a directory for the input files
                    input_files_dir = os.path.join(seed_row_dir, "input_files")
                    os.makedirs(input_files_dir, exist_ok=True)
                    
                    # Copy the input files to the input_files directory
                    os.chdir(ufl)
                    for file_name in os.listdir("inputFiles"):
                        source_file_path = os.path.join("inputFiles", file_name)
                        destination_file_path = os.path.join(input_files_dir, file_name)
                        shutil.copy2(source_file_path, destination_file_path)
                
                if output_settings.get("outputEVL", False):
                    # Create a directory for the EVL files
                    evl_dir = os.path.join(seed_row_dir, "evl_files")
                    os.makedirs(evl_dir, exist_ok=True)
                    
                    # Copy the entire EVL directory from ufl to the evl_files directory
                    src_evl_dir = os.path.join(ufl, "evl")
                    if os.path.exists(src_evl_dir):
                        dest_evl_dir = os.path.join(evl_dir, "evl")
                        if os.path.exists(dest_evl_dir):
                            shutil.rmtree(dest_evl_dir)
                        shutil.copytree(src_evl_dir, dest_evl_dir)
                        
                        
                #if no events are detected, skip the simulation and record the input parameters
                
                if output_settings.get("outputF", False):
                    # Create a directory for the F files
                    f_dir = os.path.join(seed_row_dir, "f_files")
                    os.makedirs(f_dir, exist_ok=True)
                    
                    # Copy the entire F directory from ufl to the f_files directory
                    src_f_dir = os.path.join(ufl, "F")
                    if os.path.exists(src_f_dir):
                        dest_f_dir = os.path.join(f_dir, "F")
                        if os.path.exists(dest_f_dir):
                            shutil.rmtree(dest_f_dir)
                        shutil.copytree(src_f_dir, dest_f_dir)
                
                if output_settings.get("output_betaP_figure", False):
                    # Create a directory for the betaP figure
                    betaP_dir = os.path.join(seed_row_dir, "betaP_figures")
                    os.makedirs(betaP_dir, exist_ok=True)
                    
                    # Define the filename for the betaP figure
                    betaP_figure_file_name = f"betaP_figure_row_{row_index}_seed_{seed}.png"
                    betaP_figure_path = os.path.join(betaP_dir, betaP_figure_file_name)
                    
                    # Plot the data and save the figure
                    stress = single_sim_dict.get("stress", "unknown") #Stress in MPa
                    invTemperature = single_sim_dict.get("inverseTemperature", "unknown") #inverse temperature
                    temperature= 1/invTemperature
                    title = f"{stress:.2f}MPa_{temperature}K_row{row_index}_seed{seed}_betaP"
                    plot_data(single_sim_dict["time [s]"], single_sim_dict["betaP_1"], title, betaP_figure_path)
                    print(f"BetaP figure saved to {betaP_figure_path}")
                
                if output_settings.get("output_dotBetaP_figure", False):
                    # Create a directory for the dotBetaP figure
                    stress = single_sim_dict.get("stress", "unknown") #Stress in MPa
                    stress= stress/1e6 #convert to MPa
                    
                    dotBetaP_dir = os.path.join(seed_row_dir, "dotBetaP_figures")
                    os.makedirs(dotBetaP_dir, exist_ok=True)
                    
                    # Define the filename for the dotBetaP figure
                    dotBetaP_figure_file_name = f"dotBetaP_figure_row_{row_index}.png"
                    dotBetaP_figure_path = os.path.join(dotBetaP_dir, dotBetaP_figure_file_name)
                    invTemperature = single_sim_dict.get("inverseTemperature", "unknown") #inverse temperature
                    temperature= 1/invTemperature
                    title = f"{stress:.2f}MPa_{temperature}K_{row_index}_{seed}_dotBetaP"
                    # Plot the data and save the figure
                    plot_data(single_sim_dict["time [s]"], single_sim_dict["dotBetaP"],title, dotBetaP_figure_path)
                    print(f"dotBetaP figure saved to {dotBetaP_figure_path}")
                
                
                # Output individual simulation results
                if output_settings.get("outputIndividualSimulationDictionaries", False):
                # Create a directory for individual simulation results
                    individual_sim_dir = os.path.join(seed_row_dir, "individual_simulation_results")
                    os.makedirs(individual_sim_dir, exist_ok=True)
                    # Save the individual simulation results
                    if output_settings.get("singleSimulationDictionaryFormat", "txt").lower() == "txt":
                        output_file_path = os.path.join(individual_sim_dir, f"row_{row_index}_simulation_results.txt")
                        with open(output_file_path, "w") as output_file:
                            for key, value in single_sim_dict.items():
                                output_file.write(f"{key}: {value}\n")
                    elif output_settings.get("singleSimulationDictionaryFormat", "json").lower() == "json":
                        output_file_path = os.path.join(individual_sim_dir, f"row_{row_index}_simulation_results.json")
                        with open(output_file_path, "w") as output_file:
                            json.dump(single_sim_dict, output_file, indent=4)
                    else:
                        print(f"Error: Unsupported output format '{output_settings.get('singleSimulationDictionaryFormat')}'.")
                        sys.exit(1)


                # Write simulation results to a populated hypercube file in the hypercube_info directory
                populated_hypercube_path = os.path.join(hypercube_dir, 'populated_hypercube.txt')
                write_header = not os.path.exists(populated_hypercube_path)
                with open(populated_hypercube_path, 'a') as f:
                    if write_header:
                        f.write("rowID\tseed\tappliedStress\ttemperature\tB_0e\tB_1e\tB_0s\tB1_s\trate\n")
                    rowID = row_index
                    appliedStress = latin_hypercube_sample.get("appliedStress", "")
                    temperature = latin_hypercube_sample.get("appliedTemperature", "")
                    B_0e = latin_hypercube_sample.get("B0e_SI", "")
                    B_1e = latin_hypercube_sample.get("B1e_SI", "")
                    B_0s = latin_hypercube_sample.get("B0s_SI", "")
                    B1_s = latin_hypercube_sample.get("B1s_SI", "")
                    rate = single_sim_dict.get("rate", "")
                    f.write(f"{rowID}\t{seed}\t{appliedStress}\t{temperature}\t{B_0e}\t{B_1e}\t{B_0s}\t{B1_s}\t{rate}\n")
                    
                
                # Append the results to the simulation_results dictionary
                simulation_results[row_index] = single_sim_dict
                id_tag += 1

                # If the rate for this simulation is zero or Nan, skip the rest of the seeds for this simulation
                if single_sim_dict.get("rate", 0) <= 0 or np.isnan(single_sim_dict.get("rate", 0)):
                    print(f"Skipping remaining seeds for row {row_index} due to zero or NaN rate.")
                    break



            
        # organize the simulation results in the desired format
        # Compute the average waiting time per simulation
        average_waiting_times = {}
        for sim_index, sim_result in simulation_results.items():
            waiting_times = sim_result.get("waitingTimes", [])
            if len(waiting_times) > 0:
                average_waiting_time = np.mean(waiting_times)
                average_waiting_times[sim_index] = average_waiting_time
                simRate = 1/average_waiting_time
                print(f"Simulation {sim_index}: Average Waiting Time = {average_waiting_time}, Simulation Rate = {simRate}")
            else:
                average_waiting_times[sim_index] = None  # Handle cases with no waiting times
                print(f"Simulation {sim_index}: No waiting times available.")

        # Output the average waiting times and the rate
        # Create a DataFrame to store the results
        results_df = pd.DataFrame(latin_hypercube)

        # Append the rate column to the DataFrame
        results_df["rate"] = [1 / average_waiting_times.get(sim_index, np.nan) for sim_index in range(len(average_waiting_times))]

        # Save the results to a file
        output_results_file_name = output_settings.get("outputResultsFileName", "simulation_results")
        output_results_format = output_settings.get("outputResultsFormat", "txt").lower()
        output_results_path = output_settings.get("outputPath", ".")

        results_file_path = os.path.join(output_results_path, f"{output_results_file_name}.{output_results_format}")

        if output_results_format == "txt":
            results_df.to_csv(results_file_path, sep='\t', index=False)
        elif output_results_format == "json":
            results_df.to_json(results_file_path, orient="records", indent=4)
        else:
            print(f"Error: Unsupported output format '{output_results_format}'.")
            sys.exit(1)

        print(f"Simulation results saved to {results_file_path}")
        
    elif run_simulation and speedup_settings.get("useSpeedup", False):

        if speedup_settings.get("option") == "multiEVL":

            print(f'The simulation will be run in individaul evl directories simultaneously.')
            # # Create a unique evl directory for each row
            # for row_index, row_values in enumerate(latin_hypercube.values()):
            #     for seed in range(1, number_of_seeds_per_simulation + 1):
            #         print(f'Running simulation for row {row_index} with values: {row_values} for seed {seed}')
                    
            #         # Set the seed for reproducibility
            #         # np.random.seed(seed)
                    
            #         # Create directories for each seed of the simulation
            #         row_dir = os.path.join(output_settings["outputPath"], f"row_{row_index}")
            #         seed_dir = os.path.join(row_dir, f"seed_{seed}")
            #         os.makedirs(seed_dir, exist_ok=True)
                    
            #         # Map the row values back to their corresponding metrics
            #         latin_hypercube_sample = {metric: value for metric, value in zip(latin_hypercube.keys(), row_values)}
            #         print(f"Sample: {latin_hypercube_sample}")
                    
            #         stress_component = config["application_domain"]["appliedStress"]["stress_component"]
                    
            #         detectionMethod = config["step_detection_settings"]["detectionMethod"]
                    
            #         run_arrhenius_simulation(latin_hypercube_sample,stress_component,ufl,DD_settings,noise_settings,material_settings,elasticDeformation_settings,polycrystal_settings,microstructure_settings,output_settings,row_index,seed,detectionMethod,library_driven,build_dir,speedup=True)



    return()

if __name__ == "__main__":
    main()