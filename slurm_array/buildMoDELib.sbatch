#!/bin/bash

#SBATCH --job-name=build_modelib
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16gb
#SBATCH --time=04:00:00
#SBATCH --output=logs/build_%j.out
#SBATCH --error=logs/build_%j.err

# ============================================================
# Strict shell behavior
# ============================================================

set -euo pipefail
set -x

# ============================================================
# Variables
# ============================================================

SANDBOXDIR="/home/${USER}/github/Palmetto-Apptainer-Builds/DDD/archDDD.sandbox"

WORKDIR="$SLURM_SUBMIT_DIR"

BIND_PATHS="/scratch"

SOURCE_DIR="/root/MoDELib2-lmyhill"

SRC_FILE="${SOURCE_DIR}/src/DislocationDynamics/DislocationNode.cpp"

VELOCITY_PROJECTION="X"

BUILD_ROOT="/scratch/lmyhill/DDD/MoDELib2"

BUILD_DIR="${BUILD_ROOT}/build_${SLURM_JOB_ID}"

# ============================================================
# Host-side diagnostics
# ============================================================

echo "=================================================="
echo "HOST ENVIRONMENT"
echo "=================================================="

echo "Hostname:"
hostname

echo "Date:"
date

echo "User:"
whoami

echo "SLURM job ID:"
echo "$SLURM_JOB_ID"

echo "Working directory:"
echo "$WORKDIR"

echo "Sandbox:"
echo "$SANDBOXDIR"

echo "Source directory:"
echo "$SOURCE_DIR"

echo "Build root:"
echo "$BUILD_ROOT"

echo "Build dir:"
echo "$BUILD_DIR"

echo "=================================================="

echo "Checking apptainer executable..."

which apptainer

apptainer --version

echo "=================================================="

echo "Checking sandbox existence..."

ls -ld "$SANDBOXDIR"

echo "=================================================="

mkdir -p "$BUILD_ROOT"

cd "$WORKDIR"

# Save build path for later array jobs
echo "$BUILD_DIR" > latest_build_dir.txt

# ============================================================
# Container diagnostics
# ============================================================

echo "=================================================="
echo "CONTAINER DIAGNOSTICS"
echo "=================================================="

apptainer exec \
    --writable \
    --fakeroot \
    --bind "$BIND_PATHS" \
    "$SANDBOXDIR" \
    bash -c "
        set -e
        set -x

        export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

        echo 'Inside container'

        echo 'PATH:'
        echo \$PATH

        echo 'Checking source directory...'
        ls -ld $SOURCE_DIR

        echo 'Checking source file...'
        ls -l $SRC_FILE

        echo 'Checking build root...'
        ls -ld $BUILD_ROOT || true

        echo 'Current velocity projection lines:'
        grep 'temp.push_back(VectorDim::Unit' $SRC_FILE || true
    "

# ============================================================
# Modify velocity projection
# ============================================================

echo "=================================================="
echo "MODIFYING VELOCITY PROJECTION"
echo "=================================================="

apptainer exec \
    --writable \
    --fakeroot \
    --bind "$BIND_PATHS" \
    "$SANDBOXDIR" \
    bash -c "
        set -e
        set -x

        export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

        if [[ \"$VELOCITY_PROJECTION\" == \"None\" ]]; then

            sed -i \
            's|^\(\s*\)temp.push_back(VectorDim::Unit[XYZ]());|\1// temp.push_back(VectorDim::UnitX());|' \
            $SRC_FILE

        else

            sed -i \
            \"s|^\(\s*\)//\s*temp.push_back(VectorDim::Unit[XYZ]());|\1temp.push_back(VectorDim::Unit${VELOCITY_PROJECTION}());|\" \
            $SRC_FILE

            sed -i \
            \"s|^\(\s*\)temp.push_back(VectorDim::Unit[XYZ]());|\1temp.push_back(VectorDim::Unit${VELOCITY_PROJECTION}());|\" \
            $SRC_FILE

        fi

        echo 'Updated velocity projection lines:'

        grep 'temp.push_back(VectorDim::Unit' $SRC_FILE || true
    "

# ============================================================
# Compile
# ============================================================

echo "=================================================="
echo "STARTING COMPILATION"
echo "=================================================="

apptainer exec \
    --writable \
    --fakeroot \
    --bind "$BIND_PATHS" \
    "$SANDBOXDIR" \
    bash -c "
        set -e
        set -x

        export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

        mkdir -p $BUILD_DIR

        cd $BUILD_DIR

        echo 'Current build directory:'
        pwd

        export CXXFLAGS='-O2 -march=x86-64 -mtune=generic'
        export CFLAGS='-O2 -march=x86-64 -mtune=generic'

        echo 'Checking cmake executable...'
        which cmake

        echo 'Checking make executable...'
        which make

        echo 'Running cmake...'

        cmake $SOURCE_DIR

        echo 'Running make...'

        make -j8

        echo 'Compilation finished.'

        echo 'Build directory contents:'

        ls -lah
    "

echo "=================================================="
echo "BUILD COMPLETE"
echo "=================================================="
