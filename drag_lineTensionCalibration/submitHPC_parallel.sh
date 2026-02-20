#!/bin/bash

#SBATCH --job-name=rateCalibration_drag_lineTension
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=50gb
#SBATCH --time=72:00:00
#SBATCH --output=out_%j.out
#SBATCH --error=err_%j.err
#SBATCH --mail-user=lmyhill@clemson.edu
#SBATCH --mail-type=FAIL,END     # Only mail on failure (success handled below)

# ----------- Variables -----------
SANDBOXDIR="/home/${USER}/github/Palmetto-Apptainer-Builds/DDD/archDDD.sandbox"
SCRIPT_PY="parallel_run.py"
JOB_BUILDDIR="/scratch/lmyhill/DDD/MoDELib2-lmyhill/build_${SLURM_JOB_ID}"
VELOCITY_PROJECTION="X"   # Options: X, Y, Z
SRC_FILE="/root/MoDELib2-lmyhill/src/DislocationDynamics/DislocationNode.cpp"

# Optional: If script relies on current working directory
WORKDIR="$SLURM_SUBMIT_DIR"

# Optional: If you need to bind extra directories
BIND_PATHS="/scratch"

# BUILDDIR="/root/MoDELib2/build"

# ----------- Adjust the fixed nodal velocity before compile -----------

if [[ "$VELOCITY_PROJECTION" == "None" ]]; then
  # Comment out any uncommented UnitX/Y/Z line
  sed -i 's|^\(\s*\)temp.push_back(VectorDim::Unit[XYZ]());|\1// temp.push_back(VectorDim::UnitX());|' "$SRC_FILE"
else
  # First, uncomment if it is commented
  sed -i "s|^\(\s*\)//\s*temp.push_back(VectorDim::Unit[XYZ]());|\1temp.push_back(VectorDim::Unit${VELOCITY_PROJECTION}());|" "$SRC_FILE"

  # Then, replace any still-uncommented line with the chosen axis
  sed -i "s|^\(\s*\)temp.push_back(VectorDim::Unit[XYZ]());|\1temp.push_back(VectorDim::Unit${VELOCITY_PROJECTION}());|" "$SRC_FILE"
fi

# -----------Compile Tools -----

apptainer exec --writable --fakeroot --bind $BIND_PATHS "$SANDBOXDIR" bash -c "
  echo 'Cleaning and recompiling...'
  mkdir $JOB_BUILDDIR
  cd $JOB_BUILDDIR || exit 1
  rm -rf *
  export CXXFLAGS='-march=x86-64 -mtune=generic'
  export CFLAGS='-march=x86-64 -mtune=generic'
  cmake ..
  make -j8
"

#----------- Modify config.json with updated build directory and job ID -----------
sed -i "s|\"build_dir\": \".*\"|\"build_dir\": \"$JOB_BUILDDIR\"|" config.json


#----------Modify config.json with the current path as the output directory ----------------
sed -i "s|\"outputPath\":.*|\"outputPath\": \"$WORKDIR\",|" config.json


# ----------- Run Job -----------


cd "$WORKDIR" || { echo "Cannot cd to $WORKDIR"; exit 1; }

echo "[INFO] Running inside container: $SANDBOXDIR"
echo "[INFO] Current working directory: $(pwd)"
echo "[INFO] Script to run: $SCRIPT_PY"

apptainer exec \
    --writable \
    --fakeroot \
    --bind $BIND_PATHS \
    "$SANDBOXDIR" \
    python "$SCRIPT_PY"