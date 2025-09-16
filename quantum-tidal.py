import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import argparse
import json

"""
A simple Schrodinger-Poisson + Static Potential solver written in JAX
to simulate fuzzy dark matter + in an external potential.

Philip Mocz (2025), @pmocz
Flatiron Institute

Simulate the Schrodinger-Poisson system with the spectral method described in:

Mocz, P., et al. (2017)
Galaxy Formation with BECDM: I. Turbulence and relaxation of idealised haloes
Monthly Notices of the Royal Astronomical Society, 471(4), 4559-4570
https://doi.org/10.1093/mnras/stx1887
https://arxiv.org/abs/1705.05845


Example Usage:

python quantum-tidal.py --res 1

"""


#############
# Unit System
# [L] = kpc
# [V] = km/s
# [M] = Msun
# ==> [T] = kpc / (km/s) = 0.9778 Gyr


######################################
# Global Simulation Parameters (input)

# command line input:
parser = argparse.ArgumentParser(description="Simulate the Schrodinger-Poisson system.")
parser.add_argument("--res", type=int, default=1, help="Resolution factor")
args = parser.parse_args()

# Enable for double precision
# jax.config.update("jax_enable_x64", True)

# resolution
nx = 64 * args.res

# box dimensions (in units of kpc)
Lx = 10.0

# stop time (in units of kpc / (km/s) = 0.9778 Gyr)
t_end = 1.0  # 10.0

# axion mass (in units of 10^-22 eV)
m_22 = 1.0

# dark matter
M_soliton = 1.0e9  # mass of soliton in Msun
k_soliton = 4.0  # wave-number for orbital motion of soliton
r_separation = 0.2 * Lx  # initial separation of soliton from center (in kpc)


##################
# Global Constants

G = 4.30241002e-6  # gravitational constant in kpc (km/s)^2 / Msun  |  [V^2][L]/[M]  |  (G / (km/s)^2 * (mass of sun) / kpc)
hbar = 1.71818134e-87  # in [V][L][M] | (hbar / ((km/s) * kpc * mass of sun))
ev_to_msun = 8.96215334e-67  # mass of electron volt in [M] | (eV/c^2/mass of sun)
ev_to_internal = 8.05478173e-56  # eV to internal units (eV / (mass of sun * (km/s)^2))
c = 299792.458  # speed of light in km/s (c / (km/s))
m = m_22 * 1.0e-22 * ev_to_msun  # axion mass in [M]
m_per_hbar = m / hbar  # (~0.052 1/([V][M]))

h = 0.7  # little-h (dimensionless)
H0 = 0.1 * h  # Hubble constant in (km/s)/kpc
rho_crit = 3.0 * H0**2 / (8.0 * np.pi * G)  # critical density in Msun/kpc^3 (~136)

# soliton properties
r_soliton = 2.2e8 * m_22**-2 / M_soliton  # in kpc
assert r_soliton < 0.5 * Lx
v_vir = G * M_soliton * m_per_hbar * np.sqrt(0.10851)

# check that de broglie wavelength fits into box
de_broglie_wavelength = hbar / (m * v_vir)
n_wavelengths = Lx / de_broglie_wavelength
assert n_wavelengths > 1

# print some info
print(f"# de Broglie wavelengths in box: {n_wavelengths:.2f}")


# average density of dark matter (set later)
rho_bar = jnp.nan

# external potential (point mass)
# since soliton initial velocity is set by a wave number (k_soliton),
# that must fit cleanly into the box, so we set the halo mass accordingly
# circular orbit velocity: v = sqrt(GM/r)
# v = (1/m)*grad(S) = hbar*k/m ==> k = v * m/hbar
# So: v = sqrt(GM/r) = k_soliton * hbar/m  ==> M = r * k_soliton^2 * hbar^2 / (G * m^2)
# r = 0.25*Lx
M_halo = 0.25 * Lx * k_soliton**2 * hbar**2 / (G * m**2)
print(f"M_halo: {M_halo:.2e} Msun")
assert M_halo > M_soliton * 2.0  # halo should be much more massive than soliton


######
# Mesh

# Domain [0,Lx] x [0,Ly] x [0,Lz]
dx = Lx / nx
vol = dx * dx * dx  # volume of each cell
x_lin = jnp.linspace(0.5 * dx, Lx - 0.5 * dx, nx)
X, Y, Z = jnp.meshgrid(x_lin, x_lin, x_lin, indexing="ij")

# checks
v_resolved = (hbar / m) * jnp.pi / dx
assert v_resolved > v_vir

# Fourier Space Variables
kx_lin = 2.0 * jnp.pi / Lx * jnp.arange(-nx / 2, nx / 2)
kx, ky, kz = jnp.meshgrid(kx_lin, kx_lin, kx_lin, indexing="ij")
kx = jnp.fft.ifftshift(kx)
ky = jnp.fft.ifftshift(ky)
kz = jnp.fft.ifftshift(kz)
k_sq = kx**2 + ky**2 + kz**2

# Time step
dt_fac = 1.0
dt_kin = dt_fac * (m_per_hbar / 6.0) * (dx * dx)
# round up to the nearest multiple of 100
nt = int(jnp.ceil(jnp.ceil(t_end / dt_kin) / 100.0) * 100)
nt_sub = int(jnp.round(nt / 100.0))
dt = t_end / nt


##############
# Checkpointer
options = ocp.CheckpointManagerOptions()
checkpoint_dir = os.path.join(os.getcwd(), f"checkpoints_tidal{args.res}")
path = ocp.test_utils.erase_and_create_empty(checkpoint_dir)
async_checkpoint_manager = ocp.CheckpointManager(path, options=options)


############
# Parameters
params = {}
params["Lx"] = Lx
params["nx"] = nx
params["m_22"] = m_22
params["rho_bar"] = rho_bar
params["M_soliton"] = M_soliton
params["t_end"] = t_end


#########
# Gravity


def get_potential(rho):
    """Solve the Poisson equation."""
    V_hat = -jnp.fft.fftn(4.0 * jnp.pi * G * (rho - rho_bar)) / (k_sq + (k_sq == 0))
    V = jnp.real(jnp.fft.ifftn(V_hat))
    return V


def external_potential():
    """External potential (static)."""
    r = jnp.sqrt((X - 0.5 * Lx) ** 2 + (Y - 0.5 * Lx) ** 2 + (Z - 0.5 * Lx) ** 2)
    V_ext = -G * M_halo / (r + 0.5 * dx)  # softening
    return V_ext


#######################
# Main part of the code


def compute_step(psi, t):
    """Compute the next step in the simulation."""

    # (1/2) kick
    rho_tot = jnp.abs(psi) ** 2
    V = get_potential(rho_tot) + external_potential()
    psi = jnp.exp(-1.0j * m_per_hbar * dt / 2.0 * V) * psi

    # drift
    psi_hat = jnp.fft.fftn(psi)
    psi_hat = jnp.exp(dt * (-1.0j * k_sq / m_per_hbar / 2.0)) * psi_hat
    psi = jnp.fft.ifftn(psi_hat)

    # (1/2) kick
    rho_tot = jnp.abs(psi) ** 2
    V = get_potential(rho_tot) + external_potential()
    psi = jnp.exp(-1.0j * m_per_hbar * dt / 2.0 * V) * psi

    # update time
    t += dt

    return psi, t


@jax.jit
def update(_, state):
    """Update the state of the system by one time step."""
    (
        state["psi"],
        state["t"],
    ) = compute_step(
        state["psi"],
        state["t"],
    )
    return state


def plot_sim(state):
    """Plot the simulation state."""

    # DM projection
    rho_proj_dm = jnp.log10(jnp.mean(jnp.abs(state["psi"]) ** 2, axis=2)).T
    vmin = jnp.log10(rho_bar / 100.0)
    vmax = jnp.log10(rho_bar * 100.0)
    ax = plt.gca()
    ax.imshow(
        rho_proj_dm,
        cmap="inferno",
        origin="lower",
        extent=(0, nx, 0, nx),
        vmin=vmin,
        vmax=vmax,
    )
    # plt.colorbar(im0, ax=axs[0], label="log10(rho_dm)")
    ax.set_aspect("equal")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    plt.tight_layout()


def main():
    """Main physics simulation."""

    # Initial Conditions
    t = 0.0

    # dark matter (soliton)
    r = jnp.sqrt(
        (X - 0.5 * Lx) ** 2 + (Y - 0.5 * Lx - r_separation) ** 2 + (Z - 0.5 * Lx) ** 2
    )
    psi = (
        jnp.sqrt(
            1.9e7 * m_22**-2 * r_soliton**-4 / (1.0 + 0.091 * (r / r_soliton) ** 2) ** 8
        )
        + 0.0j
    )
    # add circular orbit velocity
    psi *= jnp.exp(1.0j * k_soliton * X)

    # re-calculate rho_bar
    global rho_bar
    rho_bar = jnp.mean(jnp.abs(psi) ** 2, axis=(0, 1, 2)) + M_halo / (Lx**3)

    # Construct initial simulation state
    state = {}
    state["t"] = t
    state["psi"] = psi

    # Plot the initial state
    plot_sim(state)
    plt.savefig(os.path.join(checkpoint_dir, "initial.png"), dpi=240)
    plt.clf()

    # Simulation Main Loop
    print("Starting simulation ...")
    with open(os.path.join(checkpoint_dir, "params.json"), "w") as f:
        json.dump(params, f, indent=2)
    t_start_timer = time.time()
    for i in range(100):
        print(f"step {i}")
        state = jax.lax.fori_loop(0, nt_sub, update, init_val=state)
        async_checkpoint_manager.save(i, args=ocp.args.StandardSave(state))
        plot_sim(state)
        plt.savefig(os.path.join(checkpoint_dir, f"snap{i:03d}.png"))
        async_checkpoint_manager.wait_until_finished()
    jax.block_until_ready(state)
    print("Simulation Run Time (s): ", time.time() - t_start_timer)

    # Plot final state
    plot_sim(state)
    plt.savefig(os.path.join(checkpoint_dir, "final.png"), dpi=240)


if __name__ == "__main__":
    main()
