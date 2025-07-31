import os
import sys
import json
import numpy as np
from scipy.stats import qmc
import matplotlib.pyplot as plt
import shutil
import pandas as pd

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
        #import pyMoDELib
        from modlibUtils import *

#define a function to run the langevin thermostat simulation and extract the rate
def run_arrhenius_simulation(latin_hypercube_sample,stress_component,ufl,DD_settings,noise_settings,material_settings,elasticDeformation_settings,polycrystal_settings,microstructure_settings,output_settings,row,seed,detectionMethod,step_detction_settings,library_driven=True,build_dir=False):
    
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
    writeMaterialFile(material_settings["materialFile"],material_settings["enabledSlipSystems"],material_settings["glidePlaneNoise"],material_settings["atomsPerUnitCell"],material_settings["dislocationMobilityType"],material_settings["B0e_SI"],material_settings["B1e_SI"],material_settings["B0s_SI"],material_settings["B1s_SI"],material_settings["rho"],material_settings["mu_0"],material_settings["mu_1"],material_settings["nu"])
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
    
    
    if detectionMethod=="custom":
        print("Using custom step detection for rate calculation")
        figure_dir = os.path.join(output_settings["outputPath"],f"row_{row}",f"seed_{seed}","step_detection_figures")
        step_height = step_detction_settings["step_height"]
        step_tolerance = step_detction_settings["tolerance"]
        start_value = step_detction_settings["start_value"]
        os.makedirs(figure_dir, exist_ok=True)
        waiting_times, step_indices, fig = rudimentary_multistep_waiting_times(
        betaP_1, time_s, step_height=step_height, tolerance=step_tolerance, start_value=start_value,output_stepfit=output_settings["outputStepFitPlots"],plotName=os.path.join(figure_dir, f'step_detection_seed_{seed}_row_{row}.png'))
        if waiting_times:
            print(f"Waiting times detected: {waiting_times}")
            rate= compute_rate(waiting_times)
        else:
            print("No waiting times detected, setting rate to NaN")
            rate = np.nan
        temperature=latin_hypercube_sample["appliedTemperature"]
        invTemp=1/(temperature)
    
        
     
    #Extract the quantities related to the rate from the simulation
    returnDict['rate']=rate
    returnDict['waitingTimes'] = waiting_times
    returnDict['numEvents'] = len(step_indices)
    returnDict['inverseTemperature'] = invTemp
    returnDict['time [s]'] = time_s
    
    # Extract other simulation results
    returnDict['stress (Pa)'] = stress
    returnDict['stress (MPa)'] = stress / 1e6
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
    returnDict['alphaLineTension'] = latin_hypercube_sample["alphaLineTension"]
    returnDict['seed'] = seed
    returnDict['B0e_SI'] = latin_hypercube_sample["B0e_SI"]
    returnDict['B1e_SI'] = latin_hypercube_sample["B1e_SI"]
    returnDict['B0s_SI'] = latin_hypercube_sample["B0s_SI"]
    returnDict['B1s_SI'] = latin_hypercube_sample["B1s_SI"]

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