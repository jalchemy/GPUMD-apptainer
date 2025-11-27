# jax imports
import jax
import jax.numpy as jnp
from jax import random

# jax_md imports
from jax_md import quantity, space, partition, smap, util

# other imports
import numpy as onp

# jax.config.update("jax_enable_x64", True)
# jax.config.update("jax_debug_nans", False)

# Type defs #
Array = util.Array


def binary_site_potential(dr: Array, epsilon=None, species=None):
    return -epsilon


def bsp_neighbor_list(
    displacement_or_metric,
    box_size,
    species=None,
    epsilon=1.0,
    r_cutoff=1.2,
    dr_threshold=0.0,
    format=partition.Dense,
    **kwargs,
):
    # epsilon = jnp.array(epsilon, dtype=jnp.float32)
    # r_cutoff = jnp.array(r_cutoff, dtype=jnp.float32)

    neighbor_fn = partition.neighbor_list(
        displacement_or_metric,
        box_size,
        r_cutoff,
        dr_threshold,
        format=format,
        capacity_multiplier=1.0,
    )

    energy_fn = smap.pair_neighbor_list(
        binary_site_potential,
        space.canonicalize_displacement_or_metric(displacement_or_metric),
        species=species,
        epsilon=epsilon,
    )

    return neighbor_fn, energy_fn


def square_lattice(N, box_size):
    Nx = int(jnp.sqrt(N))
    Ny, ragged = divmod(N, Nx)
    if Ny != Nx or ragged:
        assert ValueError("Particle count should be a square. Found {}.".format(N))
    length_scale = box_size / Nx
    R = []
    for i in range(Nx):
        for j in range(Ny):
            R.append([i * length_scale, j * length_scale])
    return jnp.array(R)


def generate_random_species(key, N, ps, atomic_numbers):
    species = random.permutation(
        key,
        onp.hstack(
            [
                onp.ones((onp.array(onp.round(p * N), dtype=int),)) * atomic_numbers[i]
                for i, p in enumerate(ps)
            ]
        ),
        independent=True,
    )
    return species


def ternary_comps_uniform(comp_step):
    # Return all unique ternary compositions with each element's at% in range(comp_step, 1, comp_step)
    stop = int(onp.round(1 / comp_step, 2))

    comps = onp.zeros((int((stop - 2) * (stop - 1) / 2), 3))
    for i in onp.arange(1, stop):
        for j in onp.arange(1, stop - i):
            idx = int(
                ((stop - 2) * (stop - 1) - (stop - 1 - i) * (stop - i)) / 2 + (j - 1)
            )
            comps[idx] = onp.array([i, j, stop - (i + j)]) * comp_step

    return comps


def calc_rdf_pair(
    displacement_fn,
    r,
    species=None,
    r_min=0.1,
    r_max=2.0,
    n_bins=1000,
    dr=0.01,
    total_rdf=True,
):
    # if dr:
    #     n_bins = int((r_max - r_min) / dr)
    # else:
    #     dr = (r_max - r_min) / n_bins

    eval_radii = jnp.linspace(r_min, r_max, n_bins)
    # Assume that the deviation of atoms due to thermal oscillation etc. is much (5x) larger than the resolution from
    # the bin spacing. In practice this just gives nice smooth peaks which look nice and smooth and I like them :)
    # neighbor_fn is discarded because in pure MC the nbrs don't change, only species occupation. Will need to change in
    # the future
    g = quantity.pair_correlation(
        space.metric(displacement_fn),
        eval_radii,
        dr,
        species=species,
        compute_average=total_rdf,
    )

    # return eval_radii, g(r)
    return g(r)
    # return eval_radii, jnp.array([jnp.mean(gs, axis=0) for gs in g(r, neighbor=nbrs)])


def calc_rdf(
    displacement_fn,
    box_size,
    r,
    nbrs,
    species=None,
    r_min=0.1,
    r_max=2.0,
    n_bins=1000,
    dr=0.01,
    total_rdf=True,
):
    # if dr:
    #     n_bins = int((r_max - r_min) / dr)
    # else:
    #     dr = (r_max - r_min) / n_bins

    eval_radii = jnp.linspace(r_min, r_max, n_bins)
    # Assume that the deviation of atoms due to thermal oscillation etc. is much (5x) larger than the resolution from
    # the bin spacing. In practice this just gives nice smooth peaks which look nice and smooth and I like them :)
    # neighbor_fn is discarded because in pure MC the nbrs don't change, only species occupation. Will need to change in
    # the future
    _, g = quantity.pair_correlation_neighbor_list(
        space.metric(displacement_fn),
        box_size,
        eval_radii,
        dr,
        species=species,
        compute_average=total_rdf,
    )

    return eval_radii, g(r, nbrs, species)
    # return eval_radii, jnp.array([jnp.mean(gs, axis=0) for gs in g(r, neighbor=nbrs)])



from ase.io import read as ase_read

def read_xyz_traj(file_path):
    """
    Reads a trajectory from an xyz file and returns the positions, species, energies, and boxes.

    Args:
        file_path (str): Path to the xyz file.

    Returns:
        tuple: A tuple containing:
            - traj (jnp.ndarray): Trajectory of particle positions.
            - species (jnp.ndarray): Atomic numbers of particles.
            - energies (jnp.ndarray): Energy of each frame.
            - boxes (jnp.ndarray): Box matrix for each frame.
    """
    frames = ase_read(file_path, index=":")
    if not frames:
        raise FileNotFoundError(f"Could not read frames from {file_path}. It might be empty or in a wrong format.")

    positions = jnp.array([frame.get_positions() for frame in frames])
    species = jnp.array([frame.get_atomic_numbers() for frame in frames])
    energies = jnp.array([frame.get_potential_energy() for frame in frames])
    boxes = jnp.array([frame.cell[:] for frame in frames])
    
    return positions, species, energies, boxes

def s_q_from_g_r(G_r, eval_radii):
    # Scale the raw g_r to the form required for the Fourier transform as per literature
    # G_r = 4 * np.pi * eval_radii * rho_0 * (g_r - 1)
    G_r_padded = jnp.pad(G_r, (0, 1000), mode="constant")

    dr = eval_radii[1] - eval_radii[0]

    # The input to the FFT must be an odd function G_odd(eval_radii) = -G_odd(-eval_radii)
    # We construct it as [G(0), G(r_1), ..., G(r_N), G(r_{N+1})=0, -G(r_N), ..., -G(r_1)]
    # Note: G(0) is always 0 by definition.
    # The `[::-1]` syntax reverses the array.
    g_odd = jnp.concatenate(
        (jnp.array([0.0]), G_r_padded, jnp.array([0.0]), -G_r_padded[::-1])
    )

    # The total number of points in the odd g array
    M = len(g_odd)
    # M = len(G_r)

    # Compute the Fourier transform
    F_odd = jnp.fft.fft(jnp.fft.fftshift(g_odd))

    # The output of the FFT needs a corresponding frequency (Q) axis.
    # np.fft.fftfreq is the perfect tool for this. It generates the frequency bins
    # in the correct order for the FFT output.
    # The frequency f is 1/(M*dr). The scattering vector Q is 2*pi*f.
    # Q_raw = 2 * jnp.pi * jnp.fft.fftfreq(M, d=dr)
    Q_raw = 2 * jnp.pi * jnp.fft.fftfreq(M, d=dr)

    # From theory, the sine transform is related to the imaginary part of the FFT
    # of the odd function.
    # Integral[G(eval_radii)sin(Qr)dr] ≈ -Im[FFT(G_odd)] * dr
    # S(Q) = 1 + (1/Q) * Integral[...]
    # So, S(Q) ≈ 1 - (1/Q) * Im[F_g_odd] * dr

    # We need to handle Q=0 separately to avoid division by zero.
    # S(0) has a thermodynamic meaning but is problematic here, so we handle the warning.
    # At Q=0, S(Q) is undefined by this formula. We can set it to a special
    # value like NaN or 0. For plotting, it's often ignored.
    S_Q_raw = jnp.where(Q_raw == 0, 0, 1 - (F_odd.imag * dr) / Q_raw)

    # The output of np.fft functions puts the zero-frequency component first,
    # followed by positive frequencies, then negative frequencies.
    # `np.fft.fftshift` rearranges this into a more intuitive order:
    # negative frequencies, zero, then positive frequencies.

    Q = jnp.fft.fftshift(Q_raw)
    S_Q = jnp.fft.fftshift(S_Q_raw)

    print(Q)

    return Q, S_Q


def calc_s_q(
    displacement_fn,
    box_size,
    r,
    nbrs,
    species=None,
    n_rdf_bins=1000,
    dr_rdf=0.01,
    total_rdf=True,
    return_g_r=False,
):
    eval_radii = jnp.linspace(0.1, box_size / 2, n_rdf_bins)

    _, g = quantity.pair_correlation_neighbor_list(
        space.metric(displacement_fn),
        box_size,
        eval_radii,
        dr_rdf,
        species=species,
        compute_average=total_rdf,
    )

    G_r = g(r, nbrs, species)

    Q, S_Q = s_q_from_g_r(G_r, eval_radii)

    if return_g_r:
        return Q, S_Q, eval_radii, G_r
    else:
        return Q, S_Q
