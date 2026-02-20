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
def run_arrhenius_simulation(latin_hypercube_sample,stress_component,ufl,DD_settings,noise_settings,material_settings,elasticDeformation_settings,polycrystal_settings,microstructure_settings,output_settings,row,seed,detectionMethod,step_detction_settings,library_driven=True,build_dir=False,crss_settings=False):
    
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
        
    writeElasticDeformationFile(elasticDeformation_settings["elasticDeformationFile"],
                                formattedStress)
    
    writeDipoleMicrostructureFile(microstructure_settings["microstructureFile1"],
                                  microstructure_settings["slipSystemIDs"],
                                  microstructure_settings["exitFaceIDs"],
                                  microstructure_settings["dipoleCenters"],
                                  microstructure_settings["nodesPerLine"],
                                  microstructure_settings["dipoleHeights"],
                                  microstructure_settings["glideSteps"])
    
    writeDDfile(DD_settings["useFEM"],
                DD_settings["useDislocations"],
                DD_settings["useInclusions"],
                DD_settings["useElasticDeformation"],
                DD_settings["useClusterDynamics"],
                DD_settings["quadPerLength"],
                DD_settings["periodic_image_size"],
                DD_settings["EwaldLengthFactor"],
                DD_settings["coreSize"],
                latin_hypercube_sample["alphaLineTension"],
                DD_settings["remeshFrequency"],
                DD_settings["timeSteppingMethod"],
                DD_settings["dtMax"],
                DD_settings["dxMax"],
                DD_settings["maxJunctionIterations"],
                DD_settings["use_velocityFilter"],
                DD_settings["use_stochasticForce"],
                str(seed),
                DD_settings["Lmin"],
                DD_settings["Lmax"],
                DD_settings["outputFrequency"],
                DD_settings["outputQuadraturePoints"],
                DD_settings["glideSolverType"],
                DD_settings["climbSolverType"],
                int(DD_settings['Nsteps']))
    
    writeMaterialFile(material_settings["materialFile"],
                      material_settings["enabledSlipSystems"],
                      material_settings["glidePlaneNoise"],
                      material_settings["atomsPerUnitCell"],
                      material_settings["dislocationMobilityType"],
                      latin_hypercube_sample["B0e_SI"],
                      material_settings["B1e_SI"],
                      latin_hypercube_sample["B0s_SI"],
                      material_settings["B1s_SI"],
                      material_settings["rho"],
                      material_settings["mu_0"],
                      material_settings["mu_1"],
                      material_settings["nu"])
    
    writePolyCrystalFile(polycrystal_settings["meshFile"],
                         material_settings["materialFile"],
                         latin_hypercube_sample["appliedTemperature"],
                         np.array(polycrystal_settings["grain1globalX1"]),
                         np.array(polycrystal_settings["grain1globalX3"]),
                         np.array(polycrystal_settings["boxEdges"]),
                         np.array(polycrystal_settings["boxScaling"]),
                         np.array(polycrystal_settings["X0"]),
                         np.array(polycrystal_settings["periodicFaceIDs"]),
                         np.array(polycrystal_settings["gridSize_poly"]),
                         np.array(polycrystal_settings["gridSpacing_SI_poly"]))
    
    writeNoiseFile(noise_settings["noiseFile"],
                   noise_settings["type"],
                   row,
                   noise_settings["seed"],
                   noise_settings["correlationFile_L"],
                   noise_settings["correlationFile_T"],
                   noise_settings["gridSize"],
                   noise_settings["gridSpacing_SI"],
                   noise_settings["a_cai_SI"])

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

    # --- Record the measured separation height ---
    filtered_nodes = {}

    # process the evl files to find the exact separation distance of the dislocations in burgers vector units
    evl_dict = readEVLtxtLoopNode(os.path.join(ufl, "evl/evl_0"))

    # Get glissile nodes
    glissile_nodes = find_glissile_nodes(ufl, 0)

    # Group node positions by loop id
    # Each node: [loop_id, node_id, x, y, z, ...]
    for node in glissile_nodes:
        loop_id = int(node[0])
        pos = [node[2], node[3], node[4]]  # columns 2,3,4
        filtered_nodes.setdefault(loop_id, []).append(pos)

    # print(f"filtered_nodes: {filtered_nodes}")

    # Debug: print all y positions for each loop
    for loop_id, nodes in filtered_nodes.items():
        y_positions = [pos[1] for pos in nodes]
        # print(f"Loop {loop_id} y positions: {y_positions}")

    # Improved separation calculation: group loops by sign of mean y
    loop_means = {loop_id: np.mean([pos[1] for pos in nodes]) for loop_id, nodes in filtered_nodes.items()}
    positive_means = [mean for mean in loop_means.values() if mean > 0]
    negative_means = [mean for mean in loop_means.values() if mean < 0]

    # print(f"Loop means: {loop_means}")
    # print(f"Positive loop means: {positive_means}")
    # print(f"Negative loop means: {negative_means}")

    if positive_means and negative_means:
        separation = np.mean(positive_means) - np.mean(negative_means)
        separation_list = [separation]
    else:
        separation_list = []
    # print(f"Raw separation list (before filtering): {separation_list}")

    separation_measured = np.unique([
        sep for sep in separation_list if sep > 1e-6 and not np.isnan(sep)
    ])

    
    if detectionMethod=="custom":
        print("Using custom step detection for rate calculation")
        step_detection_figure_dir = os.path.join(output_settings["outputPath"],f"row_{row}",f"seed_{seed}","step_detection_figures")
        psd_figure_dir = os.path.join(output_settings["outputPath"],f"row_{row}",f"seed_{seed}","psd")
        crss_figure_dir = os.path.join(output_settings["outputPath"],f"row_{row}",f"seed_{seed}","crss")
        step_height = step_detction_settings["step_height"]
        step_tolerance = step_detction_settings["tolerance"]
        start_value = step_detction_settings["start_value"]
        #make the necessary directories
        os.makedirs(step_detection_figure_dir, exist_ok=True)
        os.makedirs(psd_figure_dir, exist_ok=True)
        os.makedirs(crss_figure_dir, exist_ok=True)
        waiting_times, step_indices, fig = rudimentary_multistep_waiting_times(
        betaP_1, time_s, step_height=step_height, tolerance=step_tolerance, start_value=start_value,output_stepfit=output_settings["outputStepFitPlots"],plotName=os.path.join(step_detection_figure_dir, f'step_detection_seed_{seed}_row_{row}.png'))
        if waiting_times:
            print(f"Waiting times detected: {waiting_times}")
            rate= compute_rate(waiting_times)
        else:
            print("No waiting times detected, setting rate to NaN")
            rate = np.nan
        compute_psd(ufl,os.path.join(psd_figure_dir,f"row_{row}",f"seed_{seed}","psd"))
        temperature=latin_hypercube_sample["appliedTemperature"]
        invTemp=1/(temperature)

    if crss_settings["compute_crss"]:
        # Compute the CRSS based on the current simulation parameters
        crss = compute_crss(latin_hypercube_sample,stress_component,ufl,DD_settings,noise_settings,material_settings,elasticDeformation_settings,polycrystal_settings,microstructure_settings,crss_settings,output_settings,row,seed,build_dir)
        returnDict['CRSS'] = crss

    #Extract the quantities related to the rate from the simulation
    returnDict['rate']=rate
    returnDict['waitingTimes'] = waiting_times
    returnDict['numEvents'] = len(step_indices)
    returnDict['temperature [K]'] = latin_hypercube_sample["appliedTemperature"]
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
    returnDict['coreSize'] = DD_settings["coreSize"]
    returnDict['alphaLineTension'] = latin_hypercube_sample["alphaLineTension"]
    returnDict['seed'] = seed
    returnDict['B0e_SI'] = latin_hypercube_sample["B0e_SI"]
    # returnDict['B1e_SI'] = latin_hypercube_sample["B1e_SI"]
    returnDict['B0s_SI'] = latin_hypercube_sample["B0s_SI"]
    # returnDict['B1s_SI'] = latin_hypercube_sample["B1s_SI"]
    returnDict['measuredSeparation']= separation_measured

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
        # setInputNDvector('inputFiles/'+microstructureFile1, 'dipoleCenters', np.array(dipoleCenters), 'center of each dipole')
        setInputNDvector('inputFiles/'+microstructureFile1,'dipoleCenters',*np.array(dipoleCenters),newCom='center of each dipole')
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


def compute_crss(latin_hypercube_sample,stress_component,ufl,DD_settings,noise_settings,material_settings,elasticDeformation_settings,polycrystal_settings,microstructure_settings,crss_settings,crss_fig_dir,row,seed,build_dir):
    """
    Compute the Critical Resolved Shear Stress (CRSS) based on the current simulation parameters.

    Parameters:
        latin_hypercube_sample (dict): Sample parameters from the Latin hypercube.
        stress_component (int): The stress component to consider.
        ufl (str): Path to the UFL directory.
        DD_settings (dict): Settings for dislocation dynamics.
        noise_settings (dict): Settings for noise.
        material_settings (dict): Material properties.
        elasticDeformation_settings (dict): Elastic deformation settings.
        polycrystal_settings (dict): Polycrystal settings.
        microstructure_settings (dict): Microstructure settings.

    Returns:
        crss (float): Computed CRSS value.
    """
    start_stress = crss_settings["stress_min"] #MPa
    end_stress = crss_settings["stress_max"] #MPa
    stress_increment = crss_settings["stress_increment"] #MPa

    mu=material_settings["mu_0"]

    #Create a datastructure for a uniform distribution of the stress states
    stress_states = []
    for i in range(int(start_stress), int(end_stress), int(stress_increment)):
        stress_states.append(formatStress(stress_component,i/mu))
    stress_states = np.array(stress_states)

    # Initialize variables for the bisection algorithm
    low = 0
    high = len(stress_states) - 1
    crss_found = False
    count=0

    #create a duplicate of the ufl directory and add _crss to the end of the path. Use this new directory for crss simulations
    crss_ufl = f"{ufl}_crss"
    os.system(f"cp -r {ufl} {crss_ufl}")

    while low <= high:
        os.chdir(crss_ufl)
        # Clean old simulation data
        os.system('rm -rf F/*')
        os.system('rm -rf evl/*')

        #write the simulation input files

        writeElasticDeformationFile(elasticDeformation_settings["elasticDeformationFile"],stress_states[count])
        writeDipoleMicrostructureFile(microstructure_settings["microstructureFile1"],microstructure_settings["slipSystemIDs"],microstructure_settings["exitFaceIDs"],microstructure_settings["dipoleCenters"],microstructure_settings["nodesPerLine"],microstructure_settings["dipoleHeights"],microstructure_settings["glideSteps"])
        writeDDfile(DD_settings["useFEM"],DD_settings["useDislocations"],DD_settings["useInclusions"],DD_settings["useElasticDeformation"],DD_settings["useClusterDynamics"],DD_settings["quadPerLength"],DD_settings["periodic_image_size"],DD_settings["EwaldLengthFactor"],latin_hypercube_sample["coreSize"],latin_hypercube_sample["alphaLineTension"],DD_settings["remeshFrequency"],DD_settings["timeSteppingMethod"],DD_settings["dtMax"],DD_settings["dxMax"],DD_settings["maxJunctionIterations"],DD_settings["use_velocityFilter"],'0','0',DD_settings["Lmin"],DD_settings["Lmax"],DD_settings["outputFrequency"],DD_settings["outputQuadraturePoints"],DD_settings["glideSolverType"],DD_settings["climbSolverType"],int(DD_settings['Nsteps']))
        writeMaterialFile(material_settings["materialFile"],material_settings["enabledSlipSystems"],material_settings["glidePlaneNoise"],material_settings["atomsPerUnitCell"],material_settings["dislocationMobilityType"],material_settings["B0e_SI"],material_settings["B1e_SI"],material_settings["B0s_SI"],material_settings["B1s_SI"],material_settings["rho"],material_settings["mu_0"],material_settings["mu_1"],material_settings["nu"])
        writePolyCrystalFile(polycrystal_settings["meshFile"],material_settings["materialFile"],latin_hypercube_sample["appliedTemperature"],np.array(polycrystal_settings["grain1globalX1"]),np.array(polycrystal_settings["grain1globalX3"]),np.array(polycrystal_settings["boxEdges"]),np.array(polycrystal_settings["boxScaling"]),np.array(polycrystal_settings["X0"]),np.array(polycrystal_settings["periodicFaceIDs"]),np.array(polycrystal_settings["gridSize_poly"]),np.array(polycrystal_settings["gridSpacing_SI_poly"]))
        writeNoiseFile(noise_settings["noiseFile"],noise_settings["type"],'0',noise_settings["seed"],noise_settings["correlationFile_L"],noise_settings["correlationFile_T"],noise_settings["gridSize"],noise_settings["gridSpacing_SI"],noise_settings["a_cai_SI"])

        
        mid = (low + high) // 2
        current_stress_Mu = stress_states[mid]
        current_stress_MPa = current_stress*mu*1e-6

        # Run simulation with the current stress state
        print(f"Running simulation for stress state: {current_stress_Mu}")
        MG(os.path.abspath(os.path.join(build_dir, "tools")),crss_ufl)
        DDomp(os.path.abspath(os.path.join(build_dir, "tools")),crss_ufl)

        # Get the plastic strain rate over the last 1000 steps
        F,Flabels=readFfile('./F')
        
        if stress_component==3:
            dotBetaP=getFarray(F,Flabels,'dotBetaP_12 [cs/b]')
            betaP_1=getFarray(F,Flabels,'betaP_12')
            
        # Create subplots for betaP_1 and dotBetaP
        fig, axs = plt.subplots(2, 1, figsize=(10, 12))

        # Plot betaP_1
        axs[0].plot(betaP_1, label="betaP_1", color="blue")
        axs[0].set_xlabel("Steps")
        axs[0].set_ylabel("BetaP_1")
        axs[0].set_title("BetaP_1 Evolution")
        axs[0].legend()
        axs[0].grid()

        # Plot dotBetaP
        axs[1].plot(dotBetaP, label="dotBetaP", color="green")
        axs[1].set_xlabel("Steps")
        axs[1].set_ylabel("dotBetaP")
        axs[1].set_title("dotBetaP Evolution")
        axs[1].legend()
        axs[1].grid()

        plt.tight_layout()
        plt.savefig(f"{crss_fig_dir}/betaP_dotBetaP_evolution_row_{row}_seed_{seed}_{current_stress_MPa}MPa.png")

        dotBetaP_1=dotBetaP[-10:]
        print(f'mean(dotBetaP_1) = {np.mean(dotBetaP_1)}')
        x = np.arange(len(betaP_1) - 10, len(betaP_1))
        y = betaP_1[-10:]
        slope, _ = np.polyfit(x, y, 1)

        #if slope is above threshold, mark CRSS as found
        if slope > crss_settings["slope_threshold"]:
            min_crss = current_stress_MPa
            crss_found = True
            print(f"CRSS found at stress state: {current_stress_MPa} MPa")
            break

        count += 1

    if crss_found:
        print(f"Minimum CRSS found at stress state: {min_crss} MPa")
    else:
        print("CRSS not found within the given stress range.")


    return min_crss  # Return CRSS in MPa



def compute_psd(ufl,savepath):

    # ----------------------
    # EVL Formatting
    # ----------------------

    LOOP_ID_COL = 0
    LOOP_TYPE_COL = 11
    NETWORK_NODE_COL = 5
    SESSILE_TYPE = 1

    # ----------------------
    # Directories & EVL files
    # ----------------------
    evl_dir = os.path.join(ufl, 'evl')
    evlFiles = [f for f in os.listdir(evl_dir) if f.endswith('.txt') and f.startswith('evl_')]

    # ----------------------
    # Rotation matrix from polycrystal.txt
    # ----------------------
    C2G1 = None
    with open(os.path.join(ufl, 'inputFiles', 'polycrystal.txt'), 'r') as f:
        for line in f:
            if 'C2G1' in line:
                C2G1 = np.fromstring(line.split('=')[1], sep=' ')
                C2G1 = np.append(C2G1, np.fromstring(f.readline().strip(), sep=' '))
                C2G1 = np.append(C2G1, np.fromstring(f.readline().strip(), sep=' '))
                C2G1 = C2G1.reshape((3, 3))
                break

    # ----------------------
    # Storage
    # ----------------------
    dataDict = {}
    filtered_rotated_nodes = {}

    # For k-space averaging
    all_k_vals = []
    all_fft_vals = []
    all_rg_vals = []

    # ----------------------
    # Process each EVL file
    # ----------------------
    for evlFile in evlFiles:
        runID = evlFile.split('_')[1].strip('.txt')

        # Load EVL object
        dataDict[runID] = readEVLtxtLoopNode(os.path.join(evl_dir, evlFile.strip('.txt')))
        loop_nodes = dataDict[runID].loopNodes

        # Get glissile nodes
        glissile_nodes = find_glissile_nodes(ufl, runID)

        # Apply rotation matrix
        if C2G1 is not None:
            for node in glissile_nodes:
                rotated_node = np.dot(C2G1, node[:3])
                loop_id = int(node[LOOP_ID_COL])
                filtered_rotated_nodes.setdefault(loop_id, []).append(rotated_node)

    # ----------------------
    # Loop-by-loop analysis with physical k
    # ----------------------
    all_k_vals = []
    all_fft_vals = []
    all_rg_vals = []

    for loop_id, nodes in filtered_rotated_nodes.items():
        nodes = np.array(nodes)  # shape: (n_nodes, 3)

        # --- Total loop length (close loop for distance calculation) ---
        closed_nodes = np.vstack([nodes, nodes[0]])
        segment_lengths = np.linalg.norm(np.diff(closed_nodes, axis=0), axis=1)
        L = np.sum(segment_lengths)  # total loop length in simulation length units
        avg_spacing = L / nodes.shape[0]

        # --- Center of mass ---
        com = np.mean(nodes, axis=0)

        # --- Positions relative to COM ---
        rel_positions = nodes - com

        # --- FFT along node index dimension ---
        fft_vals = np.fft.fft(rel_positions, axis=0)     # shape: (n_nodes, 3)
        power_spectrum = np.sum(np.abs(fft_vals) ** 2, axis=1)  # sum over xyz

        # --- Physical wave vectors ---
        n_nodes = nodes.shape[0]
        m_vals = np.fft.fftfreq(n_nodes) * n_nodes  # integer mode numbers
        k_vals = (2.0 * np.pi * m_vals) / L         # physical wave vectors [1/length_unit]

        # --- Radius of gyration ---
        squared_distances = np.sum(rel_positions ** 2, axis=1)
        rg = np.sqrt(np.mean(squared_distances))

        # --- Store all data for binning ---
        all_k_vals.extend(k_vals)
        all_fft_vals.extend(power_spectrum)
        all_rg_vals.extend([rg] * len(k_vals))

    # ----------------------
    # Bin & average in physical k-space
    # ----------------------
    all_k_vals = np.array(all_k_vals)
    all_fft_vals = np.array(all_fft_vals)
    all_rg_vals = np.array(all_rg_vals)

    # Only positive k for plotting
    mask_pos_k = all_k_vals >= 0
    k_vals_abs = all_k_vals[mask_pos_k]
    fft_vals_arr = all_fft_vals[mask_pos_k]
    rg_vals_arr = all_rg_vals[mask_pos_k]

    # Bin edges in physical k (auto-spacing)
    k_min, k_max = np.min(k_vals_abs), np.max(k_vals_abs)
    num_bins = 30
    k_bins = np.linspace(k_min, k_max, num_bins + 1)
    k_bin_centers = 0.5 * (k_bins[:-1] + k_bins[1:])

    fft_avg = []
    rg_avg = []
    for i in range(len(k_bins) - 1):
        bin_mask = (k_vals_abs >= k_bins[i]) & (k_vals_abs < k_bins[i + 1])
        if np.any(bin_mask):
            fft_avg.append(np.mean(fft_vals_arr[bin_mask]))
            rg_avg.append(np.mean(rg_vals_arr[bin_mask]))
        else:
            fft_avg.append(np.nan)
            rg_avg.append(np.nan)

    fft_avg = np.array(fft_avg)
    rg_avg = np.array(rg_avg)

    # ----------------------
    # Plots in physical units using log-log
    # ----------------------
    # for psd plot, add a trendline and equation to the data
    plt.figure()
    plt.loglog(k_bin_centers, fft_avg, marker='o')
    plt.xlabel(r'Wave vector $k$ [1/unit length]')
    plt.ylabel('Average Power Spectrum')
    plt.title('Average Power Spectrum vs Physical k')
    plt.savefig(os.path.join(savepath, 'psd_plot.png'))

    # Fit a line to the log-log data
    log_k = np.log(k_bin_centers)
    log_fft = np.log(fft_avg)
    fit = np.polyfit(log_k[~np.isnan(log_fft)], log_fft[~np.isnan(log_fft)], 1)
    fit_line = np.exp(fit[1]) * k_bin_centers ** fit[0]
    plt.loglog(k_bin_centers, fit_line, linestyle='--', color='red')
    plt.legend(['Data', f'Fit: y = {fit[0]:.2f}x + {fit[1]:.2f}'])
    plt.grid(True)

    plt.figure()
    plt.plot(k_bin_centers, rg_avg, marker='o', color='orange')
    plt.xlabel(r'Wave vector $k$ [1/unit length]')
    plt.ylabel('Average Radius of Gyration')
    plt.title('Radius of Gyration vs Physical k')
    plt.grid(True)
    plt.savefig(os.path.join(savepath, 'rg_plot.png'))
    return()


### functions for parsing nodal info from evl files

class EVL:
    nodes=np.empty([0,0])

def readEVLtxtLoopNode(filename: str) -> EVL:

    evlFile = open(filename+'.txt', "r")

    numNetNodes=int(evlFile.readline().rstrip())

    numLoops=int(evlFile.readline().rstrip())

    numLinks=int(evlFile.readline().rstrip())

    numLoopNodes=int(evlFile.readline().rstrip())

    numSpInc=int(evlFile.readline().rstrip())

    numPolyInc=int(evlFile.readline().rstrip())

    numPolyIncNodes=int(evlFile.readline().rstrip())

    numPolyIncEdges=int(evlFile.readline().rstrip())

    numEDrow=int(evlFile.readline().rstrip())

    numCDrow=int(evlFile.readline().rstrip())

    evl=EVL();



    dislocLoopDataSpan = 17

    evl.dislocLoop=np.empty([numLoops, dislocLoopDataSpan], dtype=np.float64)

    # Read file line by line until we fill the array

    for n in range(numLoops):

        while True:  # Keep reading until we get a valid line

            line = evlFile.readline()

            # End of file check

            if not line:

                raise ValueError(f"Reached EOF before reading {numLoops} loops")

            data = np.fromstring(line.rstrip(), sep=' ')

            if len(data) == dislocLoopDataSpan:

                evl.dislocLoop[n] = data

                # Move to next node after successful read 

                break



    # init connectivity data

    loopNodeDataSpan = 11

    evl.loopNodes=np.empty([numLoopNodes, loopNodeDataSpan], dtype=np.float64)

    # Read file line by line until we fill the array

    for m in range(numLoopNodes):

        while True:  # Keep reading until we get a valid line

            line = evlFile.readline()

            # End of file check

            if not line:

                raise ValueError(f"Reached EOF before reading {numLoopNodes} nodes")

            data = np.fromstring(line.rstrip(), sep=' ')

            if len(data) == loopNodeDataSpan:

                evl.loopNodes[m] = data

                # Move to next node after successful read 

                break



    return evl

def find_glissile_nodes(dataDir, runID):

    LOOP_ID_COL = 0
    LOOP_TYPE_COL = 11
    NETWORK_NODE_COL = 5
    SESSILE_TYPE = 1


    evl_path = os.path.join(dataDir, 'evl', f'evl_{runID}')
    evl = readEVLtxtLoopNode(evl_path)
    loop_data = evl.dislocLoop
    loop_nodes = evl.loopNodes

    # Get sessile loop IDs (vectorized)
    sessile_mask = loop_data[:, LOOP_TYPE_COL] == SESSILE_TYPE
    sessile_loop_ids = loop_data[sessile_mask, LOOP_ID_COL].astype(int)

    # Handle case where sessile loops are not detected correctly
    if not len(sessile_loop_ids):
        unique_loop_ids, node_count_per_loop = np.unique(loop_nodes[:, LOOP_ID_COL], return_counts=True)
        all_nodes_per_loop = {int(k): v for k, v in zip(unique_loop_ids, node_count_per_loop)}
        sessile_loop_ids = [key for key, count in all_nodes_per_loop.items() if count < 12]

    # Get all unique loop IDs (vectorized)
    all_loop_ids = np.unique(loop_data[:, LOOP_ID_COL]).astype(int)

    # Filter non-sessile loop IDs (vectorized)
    non_sessile_loop_ids = all_loop_ids[~np.isin(all_loop_ids, sessile_loop_ids)]
    #print the non-sessile loop IDs
    # print(f"Non-sessile loop IDs for runID {runID}: {non_sessile_loop_ids}")

    # Get sessile network nodes (vectorized)
    sessile_network_mask = np.isin(loop_nodes[:, LOOP_ID_COL], sessile_loop_ids)
    sessile_network_nodes = np.unique(loop_nodes[sessile_network_mask, NETWORK_NODE_COL]).astype(int)

    # Precompute masks for fast filtering
    non_sessile_node_mask = ~np.isin(loop_nodes[:, NETWORK_NODE_COL], sessile_network_nodes)
    valid_loop_mask = np.isin(loop_nodes[:, LOOP_ID_COL], non_sessile_loop_ids)
    combined_mask = non_sessile_node_mask & valid_loop_mask

    # Group nodes by loop ID
    filtered_nodes = loop_nodes[combined_mask]
    return filtered_nodes




