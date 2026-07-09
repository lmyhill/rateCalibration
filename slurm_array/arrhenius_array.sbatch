#!/bin/bash

#SBATCH --job-name=rate_array
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4gb
#SBATCH --time=72:00:00
#SBATCH --array=0-14
#SBATCH --output=logs/out_%A_%a.out
#SBATCH --error=logs/err_%A_%a.err

SANDBOXDIR="/home/${USER}/github/Palmetto-Apptainer-Builds/DDD/archDDD.sandbox"

WORKDIR="$SLURM_SUBMIT_DIR"

BIND_PATHS="/scratch"

cd "$WORKDIR" || exit 1

# ----------------------------------------
# Read shared build directory
# ----------------------------------------

BUILD_DIR=$(cat latest_build_dir.txt)

echo "Using build:"
echo "$BUILD_DIR"

# ----------------------------------------
# Pass runtime variables to Python
# ----------------------------------------

export BUILD_DIR
export OUTPUT_DIR="${WORKDIR}/outputs"

mkdir -p "$OUTPUT_DIR"

# ----------------------------------------
# Run simulation
# ----------------------------------------

apptainer exec \
    --writable \
    --fakeroot \
    --bind $BIND_PATHS \
    "$SANDBOXDIR" \
    python arrhenius_rate.py

