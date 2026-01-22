#!/bin/bash

# --- 3. Create Symbolic Links ---
echo "--- Creating Symbolic Links ---"
INSTALL_DIR="/mnt/parscratch/users/mtp23jls/public/bin"
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
        echo "# Add LAMMPS binary to PATH" >> "$BASHRC_FILE"
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
