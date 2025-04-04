import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
import numpy as np
import matplotlib.pyplot as plt
import time
import os

"""
A minimal differentiable Schrodinger-Poisson solver written in JAX
to simulate fuzzy dark matter.

Philip Mocz (2025), @pmocz
Flatiron Institute

Simulate the Schrodinger-Poisson system with the spectral method described in:

Mocz, P., et al. (2017)
Galaxy Formation with BECDM: I. Turbulence and relaxation of idealised haloes
Monthly Notices of the Royal Astronomical Society, 471(4), 4559-4570
https://doi.org/10.1093/mnras/stx1887
https://arxiv.org/abs/1705.05845

plus star particles (coupled gravitationally).

"""

# TODO: add distributed support
# TODO: improve redundant calculations
# TODO: improve UI

#############
# Unit System
# [L] = kpc
# [V] = km/s
# [M] = Msun
# ==> [T] = kpc / (km/s) = 0.9778 Gyr


######################################
# Global Simulation Parameters (input)

# resolution
nx = 128
ny = 64
nz = 16

# box dimensions (in units of kpc)
Lx = 128.0
Ly = 64.0
Lz = 16.0

# average density (in units of Msun / kpc^3)
rho_bar = 300.0

# stop time (in units of kpc / (km/s) = 0.9778 Gyr)
t_end = 10.0

# axion mass (in units of 10^-22 eV)
m_22 = 1.0

# stars
M_s = 0.1 * rho_bar * Lx * Ly * Lz  # total mass of stars, in units of Msun
n_s = 400  # number of star particles


##################
# Global Constants

G = 4.30241002e-6  # gravitational constant in kpc (km/s)^2 / Msun  |  [V^2][L]/[M]  |  (G / (km/s)^2 * (mass of sun) / kpc)
hbar = 1.71818134e-87  # in [V][L][M] | (hbar / ((km/s) * kpc * mass of sun))
ev_to_msun = 8.96215334e-67  # mass of electron volt in [M] | (eV/c^2/mass of sun)
m = m_22 * 1.0e-22 * ev_to_msun  # axion mass in [M]
m_per_hbar = m / hbar  # (~0.052 1/([V][M]))
m_s = M_s / n_s  # mass of each star particle


######
# Mesh

# Domain [0,Lx] x [0,Ly] x [0,Lz]
dx = Lx / nx
dy = Ly / ny
dz = Lz / nz
x_lin = jnp.linspace(0.5 * dx, Lx - 0.5 * dx, nx)
y_lin = jnp.linspace(0.5 * dy, Ly - 0.5 * dy, ny)
z_lin = jnp.linspace(0.5 * dz, Lz - 0.5 * dz, nz)
X, Y, Z = jnp.meshgrid(x_lin, y_lin, z_lin, indexing="ij")

# Fourier Space Variables
kx_lin = 2.0 * jnp.pi / Lx * jnp.arange(-nx / 2, nx / 2)
ky_lin = 2.0 * jnp.pi / Ly * jnp.arange(-ny / 2, ny / 2)
kz_lin = 2.0 * jnp.pi / Lz * jnp.arange(-nz / 2, nz / 2)
kx, ky, kz = jnp.meshgrid(kx_lin, ky_lin, kz_lin, indexing="ij")
kx = jnp.fft.ifftshift(kx)
ky = jnp.fft.ifftshift(ky)
kz = jnp.fft.ifftshift(kz)
k_sq = kx**2 + ky**2 + kz**2

# Time step
dt_kin = m_per_hbar / 6.0 * (dx * dy * dz) ** (2.0 / 3.0)
# round up to the nearest multiple of 100
nt = int(jnp.ceil(jnp.ceil(t_end / dt_kin) / 100.0) * 100)
nt_sub = int(jnp.round(nt / 100.0))
dt = t_end / nt


##############
# Checkpointer
path = ocp.test_utils.erase_and_create_empty(os.getcwd() + "/checkpoints")
async_checkpoint_manager = ocp.CheckpointManager(path)


############
# Parameters
params = {}
params["Lx"] = Lx
params["Ly"] = Ly
params["Lz"] = Lz
params["nx"] = nx
params["ny"] = ny
params["nz"] = nz
params["m_22"] = m_22
params["M_s"] = M_s
params["n_s"] = n_s


def get_potential(rho):
    """Solve the Poisson equation."""
    V_hat = -jnp.fft.fftn(4.0 * jnp.pi * G * (rho - rho_bar)) / (k_sq + (k_sq == 0))
    V = jnp.real(jnp.fft.ifftn(V_hat))

    return V


def get_cic_indicies_and_weights(pos):
    """Compute the cloud-in-cell indicies and weights for the star positions."""
    dxs = jnp.array([dx, dy, dz])
    i = jnp.floor((pos - 0.5 * dxs) / dxs)
    ip1 = i + 1.0
    weight_i = ((ip1 + 0.5) * dxs - pos) / dxs
    weight_ip1 = (pos - (i + 0.5) * dxs) / dxs
    i = jnp.mod(i, jnp.array([nx, ny, nz])).astype(int)
    ip1 = jnp.mod(ip1, jnp.array([nx, ny, nz])).astype(int)

    return i, ip1, weight_i, weight_ip1


def bin_stars(pos):
    """Bin the stars into the grid using cloud-in-cell weights."""
    rho = jnp.zeros((nx, ny, nz))
    i, ip1, w_i, w_ip1 = get_cic_indicies_and_weights(pos)

    def deposit_star(s, rho):
        """Deposit the star mass into the grid."""
        fac = m_s / (dx * dy * dz)
        rho = rho.at[i[s, 0], i[s, 1], i[s, 2]].add(
            w_i[s, 0] * w_i[s, 1] * w_i[s, 2] * fac
        )
        rho = rho.at[ip1[s, 0], i[s, 1], i[s, 2]].add(
            w_ip1[s, 0] * w_i[s, 1] * w_i[s, 2] * fac
        )
        rho = rho.at[i[s, 0], ip1[s, 1], i[s, 2]].add(
            w_i[s, 0] * w_ip1[s, 1] * w_i[s, 2] * fac
        )
        rho = rho.at[i[s, 0], i[s, 1], ip1[s, 2]].add(
            w_i[s, 0] * w_i[s, 1] * w_ip1[s, 2] * fac
        )
        rho = rho.at[ip1[s, 0], ip1[s, 1], i[s, 2]].add(
            w_ip1[s, 0] * w_ip1[s, 1] * w_i[s, 2] * fac
        )
        rho = rho.at[ip1[s, 0], i[s, 1], ip1[s, 2]].add(
            w_ip1[s, 0] * w_i[s, 1] * w_ip1[s, 2] * fac
        )
        rho = rho.at[i[s, 0], ip1[s, 1], ip1[s, 2]].add(
            w_i[s, 0] * w_ip1[s, 1] * w_ip1[s, 2] * fac
        )
        rho = rho.at[ip1[s, 0], ip1[s, 1], ip1[s, 2]].add(
            w_ip1[s, 0] * w_ip1[s, 1] * w_ip1[s, 2] * fac
        )
        return rho

    rho = jax.lax.fori_loop(0, n_s, deposit_star, rho)

    return rho


def get_acceleration(pos, rho):
    """Compute the acceleration of the stars."""
    i, ip1, w_i, w_ip1 = get_cic_indicies_and_weights(pos)

    # find accelerations on the grid
    V_hat = -jnp.fft.fftn(4.0 * jnp.pi * G * (rho - rho_bar)) / (k_sq + (k_sq == 0))
    ax = -jnp.real(jnp.fft.ifftn(1.0j * kx * V_hat))
    ay = -jnp.real(jnp.fft.ifftn(1.0j * ky * V_hat))
    az = -jnp.real(jnp.fft.ifftn(1.0j * kz * V_hat))
    a_grid = jnp.stack((ax, ay, az), axis=-1)

    # interpolate the accelerations to the star positions
    acc = jnp.zeros((n_s, 3))
    acc += (w_i[:, 0] * w_i[:, 1] * w_i[:, 2])[:, None] * a_grid[
        i[:, 0], i[:, 1], i[:, 2]
    ]
    acc += (w_ip1[:, 0] * w_i[:, 1] * w_i[:, 2])[:, None] * a_grid[
        ip1[:, 0], i[:, 1], i[:, 2]
    ]
    acc += (w_i[:, 0] * w_ip1[:, 1] * w_i[:, 2])[:, None] * a_grid[
        i[:, 0], ip1[:, 1], i[:, 2]
    ]
    acc += (w_i[:, 0] * w_i[:, 1] * w_ip1[:, 2])[:, None] * a_grid[
        i[:, 0], i[:, 1], ip1[:, 2]
    ]
    acc += (w_ip1[:, 0] * w_ip1[:, 1] * w_i[:, 2])[:, None] * a_grid[
        ip1[:, 0], ip1[:, 1], i[:, 2]
    ]
    acc += (w_ip1[:, 0] * w_i[:, 1] * w_ip1[:, 2])[:, None] * a_grid[
        ip1[:, 0], i[:, 1], ip1[:, 2]
    ]
    acc += (w_i[:, 0] * w_ip1[:, 1] * w_ip1[:, 2])[:, None] * a_grid[
        i[:, 0], ip1[:, 1], ip1[:, 2]
    ]
    acc += (w_ip1[:, 0] * w_ip1[:, 1] * w_ip1[:, 2])[:, None] * a_grid[
        ip1[:, 0], ip1[:, 1], ip1[:, 2]
    ]
    return acc


def compute_step(psi, pos, vel, t):
    """Compute the next step in the simulation."""
    # (1/2) kick
    rho_s = bin_stars(pos)
    rho = jnp.abs(psi) ** 2 + rho_s
    V = get_potential(rho)
    psi = jnp.exp(-1.0j * m_per_hbar * dt / 2.0 * V) * psi

    acc = get_acceleration(pos, rho)
    vel = vel + acc * dt / 2.0

    # drift
    psi_hat = jnp.fft.fftn(psi)
    psi_hat = jnp.exp(dt * (-1.0j * k_sq / m_per_hbar / 2.0)) * psi_hat
    psi = jnp.fft.ifftn(psi_hat)

    pos = pos + vel * dt
    pos = jnp.mod(pos, jnp.array([Lx, Ly, Lz]))

    # (1/2) kick
    rho_s = bin_stars(pos)
    rho = jnp.abs(psi) ** 2 + rho_s
    V = get_potential(rho)
    psi = jnp.exp(-1.0j * m_per_hbar * dt / 2.0 * V) * psi

    acc = get_acceleration(pos, rho)
    vel = vel + acc * dt / 2.0

    # update time
    t += dt

    return psi, pos, vel, t


@jax.jit
def update(_, state):
    """Update the state of the system by one time step."""

    state["psi"], state["pos"], state["vel"], state["t"] = compute_step(
        state["psi"], state["pos"], state["vel"], state["t"]
    )

    return state


def main():
    """Main physics simulation."""

    # Intial Condition
    t = 0.0
    amp = 100.0
    sigma = 4.0
    rho = 10.0
    rho += (
        2.0
        * amp
        * jnp.exp(-((X - 0.5 * Lx) ** 2 + (Y - 0.4 * Ly) ** 2) / 2.0 / sigma**2)
        / (sigma**3 * jnp.sqrt(2.0 * jnp.pi) ** 2)
    )
    rho += (
        1.5
        * amp
        * jnp.exp(-((X - 0.6 * Lx) ** 2 + (Y - 0.5 * Ly) ** 2) / 2.0 / sigma**2)
        / (sigma**3 * jnp.sqrt(2.0 * jnp.pi) ** 2)
    )
    rho += (
        amp
        * jnp.exp(-((X - 0.4 * Lx) ** 2 + (Y - 0.6 * Ly) ** 2) / 2.0 / sigma**2)
        / (sigma**3 * jnp.sqrt(2.0 * jnp.pi) ** 2)
    )
    rho += (
        amp
        * jnp.exp(-((X - 0.6 * Lx) ** 2 + (Y - 0.4 * Ly) ** 2) / 2.0 / sigma**2)
        / (sigma**3 * jnp.sqrt(2.0 * jnp.pi) ** 2)
    )
    rho += (
        amp
        * jnp.exp(-((X - 0.6 * Lx) ** 2 + (Y - 0.6 * Ly) ** 2) / 2.0 / sigma**2)
        / (sigma**3 * jnp.sqrt(2.0 * jnp.pi) ** 2)
    )
    rho += (
        amp
        * jnp.exp(-((X - 0.6 * Lx) ** 2 + (Y - 0.4 * Ly) ** 2) / 2.0 / sigma**2)
        / (sigma**3 * jnp.sqrt(2.0 * jnp.pi) ** 2)
    )
    rho += (
        amp
        * jnp.exp(-((X - 0.5 * Lx) ** 2 + (Y - 0.4 * Ly) ** 2) / 2.0 / sigma**2)
        / (sigma**3 * jnp.sqrt(2.0 * jnp.pi) ** 2)
    )
    rho += (
        amp
        * jnp.exp(-((X - 0.5 * Lx) ** 2 + (Y - 0.4 * Ly) ** 2) / 2.0 / sigma**2)
        / (sigma**3 * jnp.sqrt(2.0 * jnp.pi) ** 2)
    )
    # normalize wavefunction to <|psi|^2>=rho_bar
    rho *= rho_bar / jnp.mean(rho)
    psi = jnp.sqrt(rho) + 0.0j

    # stars have random positions and velocities
    np.random.seed(17)
    pos = np.random.uniform(0.0, 1.0, (n_s, 3))
    pos = pos * np.array([Lx, Ly, Lz])
    vel = np.random.uniform(-1.0, 1.0, (n_s, 3))

    state = {}
    state["t"] = t
    state["psi"] = psi
    state["pos"] = pos
    state["vel"] = vel

    # Simulation Main Loop
    t_start_timer = time.time()
    for i in range(100):
        state = jax.lax.fori_loop(0, nt_sub, update, init_val=state)
        async_checkpoint_manager.save(
            i,
            args=ocp.args.Composite(
                state=ocp.args.StandardSave(state), params=ocp.args.JsonSave(params)
            ),
        )
        # can do other work here in the meantime if you want ...
        async_checkpoint_manager.wait_until_finished()
    jax.block_until_ready(state)
    print("Simulation Run Time (s): ", time.time() - t_start_timer)

    # Plot final state
    fig = plt.figure(figsize=(6, 4), dpi=80)
    ax = fig.add_subplot(111)
    rho_proj = jnp.log10(jnp.mean(jnp.abs(state["psi"]) ** 2, axis=2)).T
    plt.imshow(rho_proj, cmap="inferno", origin="lower", extent=(0, nx, 0, ny))
    # plt.clim(2.45, 2.51)
    sx = jax.lax.slice(state["pos"], (0, 0), (n_s, 1)) / Lx * nx
    sy = jax.lax.slice(state["pos"], (0, 1), (n_s, 2)) / Ly * ny
    plt.plot(sx, sy, color="cyan", marker=".", linestyle="None", markersize=1)
    plt.colorbar(label="log10(|psi|^2)")
    ax.set_aspect("equal")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.savefig("output/quantum.png", dpi=240)
    plt.show()


if __name__ == "__main__":
    main()
