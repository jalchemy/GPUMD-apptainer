#!/bin/bash

# This script is a wrapper to run gpumd and nep commands inside the Apptainer container.

# Determine the command from the script name
COMMAND=$(basename "$0")

# Get the directory where the script is located
SCRIPT_DIR=$(dirname "$(realpath "$0")")
SIF_FILE="$SCRIPT_DIR/gpumd.sif"

# Check if the SIF file exists
if [ ! -f "$SIF_FILE" ]; then
    echo "Error: SIF file '$SIF_FILE' not found."
    exit 1
fi

# Check if docker is available
if command -v docker &> /dev/null; then
    # The docker command to run apptainer
    docker run --rm --privileged \
      -v $(pwd):/work \
      apptainer:1.4.4 exec --nv "/work/gpumd.sif" "$COMMAND" "$@"
else
    # The bare apptainer command
    apptainer exec --nv "$SIF_FILE" "$COMMAND" "$@"
fi
