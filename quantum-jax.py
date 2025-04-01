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

"""

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
m22 = 1.0

# stars
M_s = 0.1 * rho_bar * Lx * Ly * Lz  # total mass of stars, in units of Msun
n_s = 10000  # number of star particles


##################
# Global Constants

G = 4.30241002e-6  # gravitational constant in kpc (km/s)^2 / Msun  |  [V^2][L]/[M]  |  (G / (km/s)^2 * (mass of sun) / kpc)
hbar = 1.71818134e-87  # in [V][L][M] | (hbar / ((km/s) * kpc * mass of sun))
ev_to_msun = 8.96215334e-67  # mass of electron volt in [M] | (eV/c^2/mass of sun)
m = m22 * 1.0e-22 * ev_to_msun  # axion mass in [M]
m_per_hbar = m / hbar  # (~0.052 1/([V][M]))
m_s = M_s / n_s  # mass of each star particle


######
# Mesh

# Domain [0,Lx] x [0,Ly] x [0,Lz]
dx = Lx / nx
dy = Ly / ny
dz = Lz / nz
xlin = jnp.linspace(0.5 * dx, Lx - 0.5 * dx, nx)
ylin = jnp.linspace(0.5 * dy, Ly - 0.5 * dy, ny)
zlin = jnp.linspace(0.5 * dz, Lz - 0.5 * dz, nz)
X, Y, Z = jnp.meshgrid(xlin, ylin, zlin, indexing="ij")

# Fourier Space Variables
klinx = 2.0 * jnp.pi / Lx * jnp.arange(-nx / 2, nx / 2)
kliny = 2.0 * jnp.pi / Ly * jnp.arange(-ny / 2, ny / 2)
klinz = 2.0 * jnp.pi / Lz * jnp.arange(-nz / 2, nz / 2)
kx, ky, kz = jnp.meshgrid(klinx, kliny, klinz, indexing="ij")
kx = jnp.fft.ifftshift(kx)
ky = jnp.fft.ifftshift(ky)
kz = jnp.fft.ifftshift(kz)
kSq = kx**2 + ky**2 + kz**2

# Time step
dt_kin = m_per_hbar/6.0*(dx*dy*dz)**(2.0/3.0)
nt = int(jnp.ceil(t_end / dt_kin))
dt = t_end / nt


@jax.jit
def get_potential(psi):
    # solve poisson equation
    Vhat = -jnp.fft.fftn(4.0 * jnp.pi * G * (jnp.abs(psi) ** 2 - rho_bar)) / (
        kSq + (kSq == 0)
    )
    V = jnp.real(jnp.fft.ifftn(Vhat))

    return V


@jax.jit
def update(_, val):

    psi, pos, vel = val

    # (1/2) kick
    V = get_potential(psi)
    psi = jnp.exp(-1.0j * m_per_hbar * dt / 2.0 * V) * psi

    # drift
    psihat = jnp.fft.fftn(psi)
    psihat = jnp.exp(dt * (-1.0j * kSq / m_per_hbar / 2.0)) * psihat
    psi = jnp.fft.ifftn(psihat)

    # (1/2) kick
    V = get_potential(psi)
    psi = jnp.exp(-1.0j * m_per_hbar * dt / 2.0 * V) * psi

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
    vel = np.random.uniform(0.0, 1.0, (n_s, 3))

    # Simulation Main Loop
    t0 = time.time()
    (psi,pos,vel) = jax.lax.fori_loop(0, nt, update, init_val=(psi,pos,vel))
    print("Simulation Run Time (s): ", time.time() - t0)

    # Plot final state
    fig = plt.figure(figsize=(6, 4), dpi=80)
    ax = fig.add_subplot(111)
    plt.imshow(jnp.mean(jnp.log10(jnp.abs(psi) ** 2), axis=2), cmap="inferno")
    # plt.clim(2.46, 2.49)
    plt.colorbar(label="log10(|psi|^2)")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.savefig("output/quantum.png", dpi=240)
    plt.show()


if __name__ == "__main__":
    main()
