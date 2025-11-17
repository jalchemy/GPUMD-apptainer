#!/bin/bash

# This script is a wrapper to run gpumd and nep commands inside the Apptainer container.

# Determine the command from the script name
COMMAND=$(basename "$0")

# Get the directory where the script is located
SCRIPT_DIR=$(dirname "$(realpath "$0")")
SIF_FILE="$SCRIPT_DIR/gpumd.sif"
GENERATED_NVLIBLIST_FILE="$SCRIPT_DIR/generated.nvliblist.conf"

# Check if the SIF file exists
if [ ! -f "$SIF_FILE" ]; then
    echo "Error: SIF file '$SIF_FILE' not found."
    exit 1
fi

# Check if docker is available
if command -v docker &> /dev/null; then
    # Prepare Docker arguments for nvliblist
    DOCKER_NV_ARGS=""
    if [ -f "$GENERATED_NVLIBLIST_FILE" ]; then
        DOCKER_NV_ARGS="-v $GENERATED_NVLIBLIST_FILE:/etc/apptainer/nvliblist.conf:ro"
    fi

    # The docker command to run apptainer
    docker run -it --privileged --runtime=nvidia  --gpus all -v $(pwd):/work \
        --mount type=bind,src=/var/log/,dst=/var/log/ \
        --mount type=bind,src=/etc/nvidia-container-runtime/,dst=/etc/nvidia-container-runtime/ \
        --mount type=bind,src=/usr/bin/nvidia-container-cli,dst=/usr/bin/nvidia-container-cli \
        --mount type=bind,src=/usr/bin/../lib/x86_64-linux-gnu/libnvidia-container.so.1,dst=/usr/bin/../lib/x86_64-linux-gnu/libnvidia-container.so.1 \
        --mount type=bind,src=/usr/lib/x86_64-linux-gnu/libnvidia-container-go.so.1,dst=/usr/lib/x86_64-linux-gnu/libnvidia-container-go.so.1 \
        apptainer:1.4.4.deb13slim apptainer run --nvccli --bind /var/log/ --bind /etc/nvidia-container-runtime/ --bind /var/log/ --bind /usr/bin/ /work/gpumd.sif "$COMMAND" "$@"
    # docker run --rm --privileged --runtime=nvidia --gpus all \
    #   $DOCKER_NV_ARGS \
    #   -v $(pwd):/work \
    #   -v /etc/nvidia-container-runtime/ \
    #   apptainer:1.4.4.deb13slim apptainer exec --nv --bind /etc/nvidia-container-runtime/ "/work/gpumd.sif" nvidia-smi
    #   apptainer:1.4.4.deb13slim apptainer exec --nv --bind /etc/nvidia-container-runtime "/work/gpumd.sif" "$COMMAND"
    #   "$@"
else
    # The bare apptainer command
    apptainer exec --nv "$SIF_FILE" "$COMMAND" "$@"
fi
