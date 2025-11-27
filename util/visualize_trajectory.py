import jax.numpy as jnp
from jax_md import space, partition

import sys

sys.path.append(".")
from util.sim_utils import read_xyz_traj
from util.plotting import animate_trajectory


def main():
    if len(sys.argv) > 1:
        xyz_file = sys.argv[1]
    else:
        # Default to the example file if no argument is provided
        xyz_file = "example_mc/dump.xyz"
        print("Usage: python visualize_trajectory.py <path_to_xyz_file>")
        print(f"Using default file: {xyz_file}")

    try:
        positions, species, energies, boxes = read_xyz_traj(xyz_file)
    except FileNotFoundError:
        print(f"Error: File not found at {xyz_file}")
        sys.exit(1)

    # For a single frame, we can duplicate it to create a short trajectory for visualization
    if len(positions.shape) == 2:
        positions = jnp.stack([positions, positions])
        species = jnp.stack([species, species])
        energies = jnp.stack([energies, energies])
        boxes = jnp.stack([boxes, boxes])

    box = boxes[0]
    displacement_fn, shift_fn = space.periodic_general(box)

    # Since atoms don't move, we can compute neighbor list once for the first frame.
    neighbor_fn = partition.neighbor_list(
        displacement_fn, box, r_cutoff=3.0, dr_threshold=0.5
    )
    nbrs = neighbor_fn.allocate(positions[0])

    fig = animate_trajectory(
        traj=positions,
        es=energies,
        d_fn=displacement_fn,
        box_size=box,
        traj_species=species,
        nbrs_g=nbrs,
        Ts=jnp.linspace(1100, 800, len(energies)),
        annotation_data=None,
    )

    fig.write_html("trajectory_visualization.html")


if __name__ == "__main__":
    main()
