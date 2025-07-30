#!/bin/bash

#SBATCH --job-name=rateCalibration
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=50gb
#SBATCH --time=72:00:00
#SBATCH --output=out_%j.out
#SBATCH --error=err_%j.err

# ----------- Variables -----------
SANDBOXDIR="/scratch/${USER}/DDD/Palmetto-Apptainer-Builds/DDD/archDDD.sandbox"
SCRIPT_PY="parallel_run.py"
JOB_BUILDDIR="/scratch/lmyhill/DDD/MoDELib2/build_${SLURM_JOB_ID}"


# Optional: If script relies on current working directory
WORKDIR="$SLURM_SUBMIT_DIR"

# Optional: If you need to bind extra directories
BIND_PATHS="/scratch"

# BUILDDIR="/root/MoDELib2/build"

# -----------Compile Tools -----

apptainer exec --writable --fakeroot --bind $BIND_PATHS "$SANDBOXDIR" bash -c "
  echo 'Cleaning and recompiling...'
  mkdir $JOB_BUILDDIR
  cd $JOB_BUILDDIR || exit 1
  rm -rf *
  export CXXFLAGS='-O2 -march=x86-64 -mtune=generic'
  export CFLAGS='-O2 -march=x86-64 -mtune=generic'
  cmake ..
  make -j$SLURM_CPUS_PER_TASK
"

#----------- Modify config.json with updated build directory and job ID -----------
sed -i "s|\"build_dir\": \".*\"|\"build_dir\": \"$JOB_BUILDDIR\"|" config.json


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