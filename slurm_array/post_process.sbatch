#!/bin/bash

#SBATCH --job-name=rate_post_process
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2gb
#SBATCH --time=12:00:00
#SBATCH --output=logs/post_process_%j.out
#SBATCH --error=logs/post_process_%j.err

SANDBOXDIR="/home/${USER}/github/Palmetto-Apptainer-Builds/DDD/archDDD.sandbox"
WORKDIR="$SLURM_SUBMIT_DIR"
BIND_PATHS="/scratch"
CONFIG_PATH="${WORKDIR}/config_sweep.json"
OUTPUT_ROOT="${WORKDIR}/outputs"

cd "$WORKDIR" || exit 1

apptainer exec \
    --writable \
    --fakeroot \
    --bind "$BIND_PATHS" \
    --env CONFIG_PATH="$CONFIG_PATH" \
    "$SANDBOXDIR" \
    bash -lc "cd '$WORKDIR' && python post_process.py --config '$CONFIG_PATH' --output-root '$OUTPUT_ROOT'"
