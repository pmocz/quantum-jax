import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
import matplotlib.pyplot as plt
import time
import os
import argparse
import json
import h5py

"""
A simple Schrodinger-Poisson solver written in JAX
to simulate fuzzy dark matter in a cosmological (comoving) volume

Philip Mocz (2025), @pmocz
Flatiron Institute

Simulate the Schrodinger-Poisson system with the spectral method described in:

Mocz, P., et al. (2017)
Galaxy Formation with BECDM: I. Turbulence and relaxation of idealised haloes
Monthly Notices of the Royal Astronomical Society, 471(4), 4559-4570
https://doi.org/10.1093/mnras/stx1887
https://arxiv.org/abs/1705.05845


Example Usage:

python quantum-cosmo.py --res_factor 1

"""


########################
# Unit System (comoving)
# [L] = kpc
# [V] = km/s
# [M] = Msun
# ==> [T] = kpc / (km/s) = 0.9778 Gyr


######################################
# Global Simulation Parameters (input)

# command line input:
parser = argparse.ArgumentParser(description="Simulate the Schrodinger-Poisson system.")
parser.add_argument("--res_factor", type=int, default=1, help="Resolution factor")
args = parser.parse_args()

# Enable for double precision
# jax.config.update("jax_enable_x64", True)

# resolution
nx = 256 * args.res_factor

# box dimensions (in units of h^-1 kpc)
Lx = 1000.0

# start/stop time (in units of scale factor)
z_start = 127
z_end = 0
a_start = 1.0 / (1.0 + z_start)
a_end = 1.0 / (1.0 + z_end)

# axion mass (in units of 10^-22 eV)
m_22 = 2.5


##################
# Global Constants

G = 4.30241002e-6  # gravitational constant in kpc (km/s)^2 / Msun  |  [V^2][L]/[M]  |  (G / (km/s)^2 * (mass of sun) / kpc)
hbar = 1.71818134e-87  # in [V][L][M] | (hbar / ((km/s) * kpc * mass of sun))
ev_to_msun = 8.96215334e-67  # mass of electron volt in [M] | (eV/c^2/mass of sun)
ev_to_internal = 8.05478173e-56  # eV to internal energy unites
c = 299792.458  # speed of light in km/s
m = m_22 * 1.0e-22 * ev_to_msun  # axion mass in [M]
m_per_hbar = m / hbar  # (~0.052 1/([V][M]))

h = 1.0  # little-h (dimensionless) -- set to 1 for now, units are in h^-1
H0 = 0.1 * h  # Hubble constant in (km/s)/kpc
rho_crit = 3.0 * H0**2 / (8.0 * jnp.pi * G)  # critical density in Msun/kpc^3 (~136)
omega_matter = 0.27
omega_lambda = 0.73
omega_baryon = 0.046

# average density of all matter (dm) in the simulation (in units of Msun / kpc^3)
rho_bar = omega_matter * rho_crit


######
# Mesh

# Domain [0,Lx] x [0,Ly] x [0,Lz]
dx = Lx / nx
vol = dx * dx * dx  # volume of each cell
x_lin = jnp.linspace(0.5 * dx, Lx - 0.5 * dx, nx)
X, Y, Z = jnp.meshgrid(x_lin, x_lin, x_lin, indexing="ij")

# checks
v_resolved = (hbar / m) * jnp.pi / dx

# Fourier Space Variables
kx_lin = 2.0 * jnp.pi / Lx * jnp.arange(-nx / 2, nx / 2)
kx, ky, kz = jnp.meshgrid(kx_lin, kx_lin, kx_lin, indexing="ij")
kx = jnp.fft.ifftshift(kx)
ky = jnp.fft.ifftshift(ky)
kz = jnp.fft.ifftshift(kz)
k_sq = kx**2 + ky**2 + kz**2

# Time step (fixed)
da_ref = 1e-4
# round up to the nearest multiple of 100
nt = int(jnp.ceil(jnp.ceil((a_end - a_start) / da_ref) / 100.0) * 100)
nt_sub = int(jnp.round(nt / 100.0))
da = (a_end - a_start) / nt


##############
# Checkpointer
options = ocp.CheckpointManagerOptions()
checkpoint_dir = f"checkpoints_cosmo{args.res_factor}"
path = ocp.test_utils.erase_and_create_empty(os.getcwd() + "/" + checkpoint_dir)
async_checkpoint_manager = ocp.CheckpointManager(path, options=options)


############
# Parameters
params = {}
params["Lx"] = Lx
params["nx"] = nx
params["m_22"] = m_22
params["rho_bar"] = rho_bar
params["a_start"] = a_start
params["a_end"] = a_end


#########
# Gravity


def get_potential(rho):
    """Solve the Poisson equation."""
    V_hat = -jnp.fft.fftn(4.0 * jnp.pi * G * (rho - rho_bar)) / (k_sq + (k_sq == 0))
    V = jnp.real(jnp.fft.ifftn(V_hat))
    return V


def compute_dt(a, da):
    n_quad = 1000
    dx = 1.0 / n_quad
    lin = jnp.linspace(0.5 * dx, 1.0 - 0.5 * dx, n_quad)

    a_tmp = a * lin
    adot = H0 * jnp.sqrt((omega_matter / a_tmp) + (omega_lambda * a_tmp**2))
    t = jnp.mean(1.0 / adot) * (a - 0.0)

    a_tmp = (a + da) * lin
    adot = H0 * jnp.sqrt((omega_matter / a_tmp) + (omega_lambda * a_tmp**2))
    dt = jnp.mean(1.0 / adot) * ((a + da) - 0.0) - t

    return dt


#######################
# Main part of the code


def compute_step(psi, a):
    """Compute the next step in the simulation."""

    # cosmological factors
    kin_fac = 0.5 * (a**-2 + (a + da) ** -2)
    pot_fac = 0.5 * (a**-1 + (a + da) ** -1)

    dt = compute_dt(a, da)

    # (1/2) kick
    rho_tot = jnp.abs(psi) ** 2
    V = get_potential(rho_tot)
    psi = jnp.exp(-1.0j * m_per_hbar * dt * pot_fac / 2.0 * V) * psi

    # drift
    psi_hat = jnp.fft.fftn(psi)
    psi_hat = jnp.exp(dt * kin_fac * (-1.0j * k_sq / m_per_hbar / 2.0)) * psi_hat
    psi = jnp.fft.ifftn(psi_hat)

    # (1/2) kick
    rho_tot = jnp.abs(psi) ** 2
    V = get_potential(rho_tot)
    psi = jnp.exp(-1.0j * m_per_hbar * dt * pot_fac / 2.0 * V) * psi

    # update time
    a += da

    return psi, a


@jax.jit
def update(_, state):
    """Update the state of the system by one time step."""
    (
        state["psi"],
        state["a"],
    ) = compute_step(
        state["psi"],
        state["a"],
    )
    return state


def plot_sim(state):
    """Plot the simulation state."""
    # DM projection
    rho_proj_dm = jnp.log10(jnp.mean(jnp.abs(state["psi"]) ** 2, axis=2))
    vmin = 1.7
    vmax = 3.7
    plt.imshow(
        rho_proj_dm,
        cmap="inferno",
        origin="lower",
        vmin=vmin,
        vmax=vmax,
        extent=(0, nx, 0, nx),
    )
    plt.colorbar(label="log10(rho_dm)")
    ax = plt.gca()
    ax.set_aspect("equal")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.tight_layout()


def main():
    """Main physics simulation."""

    # Initial Conditions
    a = a_start

    # dark matter
    # read in hdf5 file (which is in units of 1e10 Msun/kpc^2)
    with h5py.File("data/fdm_1mpc_256_m1e-21_z127_ic.hdf5", "r") as f:
        psi = jnp.array(f["psiRe"]) + 1.0j * jnp.array(f["psiIm"])
        psi *= 1e5

    # diagnostics
    print(f"size of psi: {psi.shape}")
    rho_init = jnp.mean(jnp.abs(psi) ** 2, axis=(0, 1, 2))
    print(f"initial average density: {rho_init:.5f} Msun/kpc^3")
    print(f"rho_bar: {rho_bar:.5f} Msun/kpc^3")
    assert nx == psi.shape[0]

    # Construct initial simulation state
    state = {}
    state["a"] = a
    state["psi"] = psi

    # Simulation Main Loop
    plt.figure(figsize=(6, 4), dpi=80)
    print("Starting simulation ...")
    with open(checkpoint_dir + "/params.json", "w") as f:
        json.dump(params, f, indent=2)
    t_start_timer = time.time()
    for i in range(100):
        z = 1.0 / state["a"] - 1.0
        print(f"step {i}, z={z:.2f}")
        state = jax.lax.fori_loop(0, nt_sub, update, init_val=state)
        async_checkpoint_manager.save(i, args=ocp.args.StandardSave(state))
        plot_sim(state)
        plt.savefig(checkpoint_dir + "/snap{:03d}.png".format(i))
        plt.clf()
        async_checkpoint_manager.wait_until_finished()
    jax.block_until_ready(state)
    print("Simulation Run Time (s): ", time.time() - t_start_timer)

    # Plot final state
    plot_sim(state)
    plt.savefig(checkpoint_dir + "/final.png", dpi=240)


if __name__ == "__main__":
    main()
