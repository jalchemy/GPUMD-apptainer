# GPUMD with Apptainer
 
This repository provides a simple way to build and run [GPUMD](https://github.com/brucefan1983/GPUMD) using Apptainer. This approach ensures a reliable, portable, and reproducible environment for your simulations across different systems.
 
## Prerequisites
 
Before you begin, ensure you have the following:

*   **Apptainer:** For running the containerized application.
    > [!NOTE]
    > The `install.sh` script can either use Apptainer directly or within a Docker container like [this one](https://github.com/jalchemy/apptainer-in-docker). If Docker is detected, it
    > will be used for the build process (since it is usually not present on HPC systems and present on end-user systems).
*   **NVIDIA GPU and Drivers:** Required for GPUMD (as one might expect from the name). These only need to be installed on the **host** system, not within the docker or apptainer images.
*   **Docker (Optional):** If you choose to run the Apptainer container within a Docker environment.
*   **nvidia-container-toolkit (Optional):** Required if using Apptainer in Docker in order to pass through GPU drivers correctly. Installation instructions [here](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).


## Installation

The `install.sh` script automates the setup process, including building the Apptainer image and creating symbolic links
for the `gpumd` and `nep` commands so they can be run without modification from the command line and in scripts.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/GPUMD-apptainer.git
    cd GPUMD-apptainer
    ```

2.  **Run the installation script:**
    ```bash
    ./install.sh
    ```

> [!TIP]
> The CUDA architecture is set in `install.sh` as `CUDA_SM_ARCH=89` by default. To override this, you can either edit
> `install.sh` or use the `--cuda_arch` flag:
> ```bash
> ./install.sh --cuda_arch 80 # e.g. for A100 GPUs
> ```

## Quick Start

After successful installation, we can run a simulation. The below test uses the provided `run.in`, `nep.txt` and `model.xyz` files in the `example` directory.

### Navigate to the example directory and run the simulation:

```bash
cd example
gpumd # or gpumd <run_filename.in> if not named run.in
```

### Check the output

You will see output files generated in the `example` directory, such as `dump.xyz` and `thermo.out`.

## Usage

### `gpumd | nep`

The commands `gpumd` or `nep` are symlinks to the `run_gpumd.sh` script which parses the command and any arguments and runs them inside the `gpumd.sif` container. When you run `gpumd` or `nep`, you are actually calling this script.

#### Points to note

*   The script automatically detects whether to use Docker to run the Apptainer container.
*   The current working directory is mounted into the container so that GPUMD can read input files and write results. The commands should work in any directory, however, not just ones within this repository, so there is no need to move the container image.

### `run.in`

This is the main input file for a GPUMD simulation. The provided `run.in` in the `example` directory is a simple example:

```
potential nep.txt  # Specifies the potential file
time_step 0.5      # Simulation time step in fs

velocity 300       # Initialize velocities at 300K

ensemble nve       # Run in the NVE ensemble

dump_exyz 10 1 1   # Write to output files every
dump_position 10   # 10 steps
dump_thermo 10

run 100            # Run for 10 steps
```

> [!IMPORTANT]
> Make sure your potential file (e.g. `nep.txt`) is in the same directory as your `run.in` file, or provide a valid path to it


For more information on the usage of GPUMD, see the [official docs](https://gpumd.org/index.html).