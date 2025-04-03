import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import time

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

# TODO: checkpointing
# TODO: add star particle acceleration

#############
# Unit System
# [L] = kpc
# [V] = km/s
# [M] = Msun
# ==> [T] = kpc / (km/s) = 0.9778 Gyr


######################################
# Global Simulation Parameters (input)

# resolution
nx = 256
ny = 128
nz = 32

# box dimensions (in units of kpc)
Lx = 128.0
Ly = 64.0
Lz = 16.0

# average density (in units of Msun / kpc^3)
rho_bar = 300.0

# stop time (in units of kpc / (km/s) = 0.9778 Gyr)
t_end = 1.0

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
nt = int(jnp.ceil(t_end / dt_kin))
dt = t_end / nt


def get_potential(rho):
    # solve the Poisson equation
    V_hat = -jnp.fft.fftn(4.0 * jnp.pi * G * (rho - rho_bar)) / (k_sq + (k_sq == 0))
    V = jnp.real(jnp.fft.ifftn(V_hat))

    return V


def bin_stars(pos):
    # bin the stars into the grid using cloud-in-cell weights
    rho = jnp.zeros((nx, ny, nz))
    dxs = jnp.array([dx, dy, dz])
    i = jnp.floor((pos - 0.5*dxs) / dxs )
    ip1 = i + 1.0
    weight_i = ((ip1 + 0.5) * dxs - pos) / dxs
    weight_ip1 = (pos - (i + 0.5) * dxs) / dxs
    i = jnp.mod(i, jnp.array([nx, ny, nz])).astype(int)
    ip1 = jnp.mod(ip1, jnp.array([nx, ny, nz])).astype(int)

    def deposit_star(s, rho):
        # deposit the star mass into the grid
        fac = m_s / (dx * dy * dz)
        rho = rho.at[i[s, 0], i[s, 1], i[s, 2]].add(
            weight_i[s, 0] * weight_i[s, 1] * weight_i[s, 2] * fac
        )
        rho = rho.at[ip1[s, 0], i[s, 1], i[s, 2]].add(
            weight_ip1[s, 0] * weight_i[s, 1] * weight_i[s, 2] * fac
        )
        rho = rho.at[i[s, 0], ip1[s, 1], i[s, 2]].add(
            weight_i[s, 0] * weight_ip1[s, 1] * weight_i[s, 2] * fac
        )
        rho = rho.at[i[s, 0], i[s, 1], ip1[s, 2]].add(
            weight_i[s, 0] * weight_i[s, 1] * weight_ip1[s, 2] * fac
        )
        rho = rho.at[ip1[s, 0], ip1[s, 1], i[s, 2]].add(
            weight_ip1[s, 0] * weight_ip1[s, 1] * weight_i[s, 2] * fac
        )
        rho = rho.at[ip1[s, 0], i[s, 1], ip1[s, 2]].add(
            weight_ip1[s, 0] * weight_i[s, 1] * weight_ip1[s, 2] * fac
        )
        rho = rho.at[i[s, 0], ip1[s, 1], ip1[s, 2]].add(
            weight_i[s, 0] * weight_ip1[s, 1] * weight_ip1[s, 2] * fac
        )
        rho = rho.at[ip1[s, 0], ip1[s, 1], ip1[s, 2]].add( 
            weight_ip1[s, 0] * weight_ip1[s, 1] * weight_ip1[s, 2] * fac
        )
        return rho

    rho = jax.lax.fori_loop(0, n_s, deposit_star, rho)

    return rho


def get_acceleration(pos, rho):
    return jnp.zeros_like(pos)
    # compute the acceleration of the stars
    rho = jnp.reshape(rho, (nx, ny, nz))
    V = get_potential(rho)
    V_hat = jnp.fft.fftn(V)
    ax = -jnp.real(jnp.fft.ifftn(-1.0j * kx * V_hat))
    ay = -jnp.real(jnp.fft.ifftn(-1.0j * ky * V_hat))
    az = -jnp.real(jnp.fft.ifftn(-1.0j * kz * V_hat))

    acc = jnp.zeros_like(pos)
    acc[:, 0] = ax[pos[:, 0].astype(int), pos[:, 1].astype(int), pos[:, 2].astype(int)]
    acc[:, 1] = ay[pos[:, 0].astype(int), pos[:, 1].astype(int), pos[:, 2].astype(int)]
    acc[:, 2] = az[pos[:, 0].astype(int), pos[:, 1].astype(int), pos[:, 2].astype(int)]
    # XXX
    return acc


@jax.jit
def update(_, state):
    psi, pos, vel = state

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

    return psi, pos, vel


def main():
    """Physics simulation"""

    # Intial Condition
    amp = 10.0
    sigma = 0.5
    rho = 300.0
    rho += (
        2.0
        * amp
        * jnp.exp(-((X - 0.5 * Lx) ** 2 + (Y - 0.25 * Ly) ** 2) / 2.0 / sigma**2)
        / (sigma**3 * jnp.sqrt(2.0 * jnp.pi) ** 2)
    )
    rho += (
        1.5
        * amp
        * jnp.exp(-((X - 0.2 * Lx) ** 2 + (Y - 0.3 * Ly) ** 2) / 2.0 / sigma**2)
        / (sigma**3 * jnp.sqrt(2.0 * jnp.pi) ** 2)
    )
    rho += (
        amp
        * jnp.exp(-((X - 0.4 * Lx) ** 2 + (Y - 0.6 * Ly) ** 2) / 2.0 / sigma**2)
        / (sigma**3 * jnp.sqrt(2.0 * jnp.pi) ** 2)
    )
    rho += (
        amp
        * jnp.exp(-((X - 0.6 * Lx) ** 2 + (Y - 0.24 * Ly) ** 2) / 2.0 / sigma**2)
        / (sigma**3 * jnp.sqrt(2.0 * jnp.pi) ** 2)
    )
    rho += (
        amp
        * jnp.exp(-((X - 0.8 * Lx) ** 2 + (Y - 0.8 * Ly) ** 2) / 2.0 / sigma**2)
        / (sigma**3 * jnp.sqrt(2.0 * jnp.pi) ** 2)
    )
    rho += (
        amp
        * jnp.exp(-((X - 0.6 * Lx) ** 2 + (Y - 0.27 * Ly) ** 2) / 2.0 / sigma**2)
        / (sigma**3 * jnp.sqrt(2.0 * jnp.pi) ** 2)
    )
    rho += (
        amp
        * jnp.exp(-((X - 0.7 * Lx) ** 2 + (Y - 0.74 * Ly) ** 2) / 2.0 / sigma**2)
        / (sigma**3 * jnp.sqrt(2.0 * jnp.pi) ** 2)
    )
    rho += (
        amp
        * jnp.exp(-((X - 0.3 * Lx) ** 2 + (Y - 0.3 * Ly) ** 2) / 2.0 / sigma**2)
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

    # Simulation Main Loop
    t0 = time.time()
    (psi, pos, vel) = jax.lax.fori_loop(0, nt, update, init_val=(psi, pos, vel))
    jax.block_until_ready(psi)
    print("Simulation Run Time (s): ", time.time() - t0)

    # Plot final state
    fig = plt.figure(figsize=(6, 4), dpi=80)
    ax = fig.add_subplot(111)
    rho_proj = jnp.log10(jnp.mean(jnp.abs(psi) ** 2, axis=2)).T
    plt.imshow(rho_proj, cmap="inferno", origin="lower", extent=(0, nx, 0, ny))
    # plt.clim(2.46, 2.49)
    sx = jax.lax.slice(pos, (0, 0), (n_s, 1)) / Lx * nx
    sy = jax.lax.slice(pos, (0, 1), (n_s, 2)) / Ly * ny
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
