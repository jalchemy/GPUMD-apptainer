#!/bin/bash

# Default CUDA SM architecture
CUDA_SM_ARCH=89

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --cuda_arch) CUDA_SM_ARCH="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

echo "======================================="
echo " GPUMD Apptainer Installer"
echo " Using CUDA SM Architecture: $CUDA_SM_ARCH"
echo "======================================="
echo ""

# --- 1. Create Local Dirs ---
APPTAINER_CACHE_DIR="$HOME/.apptainer_cache"
ARTIFACTS_DIR="$(pwd)/build_artifacts"
mkdir -p "$APPTAINER_CACHE_DIR"
mkdir -p "$ARTIFACTS_DIR"

# --- 2. Generate NVIDIA Library List for Docker ---
echo "--- Generating NVIDIA Library List for Docker ---"
GENERATED_NVLIBLIST_FILE="$(pwd)/generated.nvliblist.conf"
DOCKER_NV_ARGS=""

if command -v docker &> /dev/null; then
    echo "-> Discovering GPU libraries inside Docker NVIDIA runtime..."
    # Find all libraries with 'nvidia' or 'cuda' in their path from ldconfig
    docker run --rm --runtime=nvidia --gpus=all apptainer:1.4.4.deb13slim ldconfig -p | \
        grep -E 'libnvidia|libcuda' | \
        awk 'NF>1 {print $NF}' > "$GENERATED_NVLIBLIST_FILE"

    if [ -s "$GENERATED_NVLIBLIST_FILE" ]; then
        echo "-> Successfully generated library list."
        DOCKER_NV_ARGS="-v $GENERATED_NVLIBLIST_FILE:/etc/apptainer/nvliblist.conf:ro"
    else
        echo "-> WARNING: Failed to generate NVIDIA library list. GPU support may not work."
    fi
else
    echo "-> Docker not found, skipping library list generation."
fi
echo ""


# --- 3. Build the Apptainer Image ---
echo "--- Building Container ---"
SIF_FILE="gpumd.sif"

if command -v docker &> /dev/null; then
    echo "-> Docker found, building with Docker..."
    docker run --rm --runtime=nvidia --gpus=all --privileged \
      $DOCKER_NV_ARGS \
      -v $(pwd):/work \
      -v "$APPTAINER_CACHE_DIR":/cache \
      -v "$ARTIFACTS_DIR":/artifacts \
      -e APPTAINER_CACHEDIR=/cache \
      apptainer:1.4.4.deb13slim apptainer build --ignore-subuid --bind "/artifacts:/tmp/artifacts" --build-arg "CUDA_SM_ARCH=$CUDA_SM_ARCH" "$SIF_FILE" gpumd.def
else
    echo "-> Docker not found, building with Apptainer directly..."
    apptainer build --ignore-subuid --bind "/artifacts:/tmp/artifacts" --build-arg "CUDA_SM_ARCH=$CUDA_SM_ARCH" "$SIF_FILE" gpumd.def
fi

if [ $? -ne 0 ]; then
    echo "-> ERROR: Failed to build the Apptainer image."
    exit 1
else
    echo "-> Build successful: $SIF_FILE created."
fi
echo ""

# --- 3. Create Symbolic Links ---
echo "--- Creating Symbolic Links ---"
INSTALL_DIR="$HOME/.local/bin"
echo "-> Creating symbolic links in $INSTALL_DIR..."
mkdir -p "$INSTALL_DIR"

SCRIPT_PATH=$(realpath run_gpumd.sh)
ln -sf "$SCRIPT_PATH" "$INSTALL_DIR/gpumd"
ln -sf "$SCRIPT_PATH" "$INSTALL_DIR/nep"
echo "-> Symbolic links created."
echo ""

# --- 4. Update PATH ---
echo "--- Updating PATH ---"
BIN_DIR="$HOME/.local/bin"
BASHRC_FILE="$HOME/.bashrc"
PATH_STRING="export PATH=\"$BIN_DIR:\$PATH\""

if grep -q "${BIN_DIR}" "$BASHRC_FILE" 2>/dev/null; then
    echo "-> Your PATH is already configured in $BASHRC_FILE."
else
    echo "-> Your PATH needs to be updated."
    read -p "   Add the required line to your ${BASHRC_FILE}? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "" >> "$BASHRC_FILE"
        echo "# Add GPUMD Suite to PATH" >> "$BASHRC_FILE"
        echo "$PATH_STRING" >> "$BASHRC_FILE"
        echo "-> Successfully updated ${BASHRC_FILE}."
        echo "   Please restart your shell or run 'source ~/.bashrc'."
    else
        echo "-> OK. Please add the following line to your ~/.bashrc file:"
        echo "   $PATH_STRING"
    fi
fi
echo ""
echo "======================================="
echo " Installation Complete"
echo "======================================="
