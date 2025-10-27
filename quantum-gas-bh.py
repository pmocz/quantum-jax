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
A simple Schrodinger-Poisson + Gas + Black Hole solver written in JAX
to simulate fuzzy dark matter +gas + black hole

Philip Mocz (2025), @pmocz
Flatiron Institute

Simulate the Schrodinger-Poisson system with the spectral method described in:

Mocz, P., et al. (2017)
Galaxy Formation with BECDM: I. Turbulence and relaxation of idealised haloes
Monthly Notices of the Royal Astronomical Society, 471(4), 4559-4570
https://doi.org/10.1093/mnras/stx1887
https://arxiv.org/abs/1705.05845


Example Usage:

python quantum-gas-bh.py --res 1

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
parser.add_argument("--show", action="store_true", help="Show live plots during run")
args = parser.parse_args()

# Enable for double precision
# jax.config.update("jax_enable_x64", True)

# resolution
nx = 64 * args.res

# box dimensions (in units of kpc)
Lx = 10.0

# stop time (in units of kpc / (km/s) = 0.9778 Gyr)
t_end = 1.0

# axion mass (in units of 10^-22 eV)
m_22 = 1.0

# soliton
M_soliton = 1.0e9

# black hole
M_bh = 1.0e7  # mass of black hole (in Msun)

# gas
frac_gas = 0.1
cs = 20.0  # sound speed (km/s)

rho_bar = jnp.nan  # will be set later


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
rho_crit = 3.0 * H0**2 / (8.0 * jnp.pi * G)  # critical density in Msun/kpc^3 (~136)

n_bh = 1  # number of black holes


# Schwarzschild radius
r_s = 2.0 * G * M_bh / c**2  # in kpc

r_soliton = 2.2e8 * m_22**-2 / M_soliton  # in kpc
assert r_soliton < 0.5 * Lx
v_vir = G * M_soliton * m_per_hbar * jnp.sqrt(0.10851)


######
# Mesh

# Domain [0,Lx] x [0,Lx] x [0,Lx]
dx = Lx / nx
vol = dx * dx * dx  # volume of each cell
x_lin = jnp.linspace(0.5 * dx, Lx - 0.5 * dx, nx)
X, Y, Z = jnp.meshgrid(x_lin, x_lin, x_lin, indexing="ij")

# checks
v_resolved = (hbar / m) * jnp.pi / dx
assert r_soliton > dx
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
checkpoint_dir = os.path.join(os.getcwd(), f"checkpoints_gasbh{args.res}")
path = ocp.test_utils.erase_and_create_empty(checkpoint_dir)
async_checkpoint_manager = ocp.CheckpointManager(path, options=options)


############
# Parameters
params = {}
params["Lx"] = Lx
params["nx"] = nx
params["m_22"] = m_22
params["M_bh"] = M_bh
params["t_end"] = t_end
params["M_soliton"] = M_soliton


#########
# Gravity


def get_potential(rho):
    """Solve the Poisson equation."""
    V_hat = -jnp.fft.fftn(4.0 * jnp.pi * G * (rho - rho_bar)) / (k_sq + (k_sq == 0))
    V = jnp.real(jnp.fft.ifftn(V_hat))
    return V


#####
# Gas


def get_conserved(rho, vx, vy, vz, vol):
    Mass = rho * vol
    Momx = rho * vx * vol
    Momy = rho * vy * vol
    Momz = rho * vz * vol

    return Mass, Momx, Momy, Momz


def get_primitive(Mass, Momx, Momy, Momz, vol):
    rho = Mass / vol
    vx = Momx / Mass
    vy = Momy / Mass
    vz = Momz / Mass

    return rho, vx, vy, vz


def get_gradient(f, dx):
    f_dx = (jnp.roll(f, -1, axis=0) - jnp.roll(f, 1, axis=0)) / (2.0 * dx)
    f_dy = (jnp.roll(f, -1, axis=1) - jnp.roll(f, 1, axis=1)) / (2.0 * dx)
    f_dz = (jnp.roll(f, -1, axis=2) - jnp.roll(f, 1, axis=2)) / (2.0 * dx)

    return f_dx, f_dy, f_dz


def extrap_to_face(f, f_dx, f_dy, f_dz, dx):
    f_XL = f + 0.5 * f_dx * dx
    f_XR = f - 0.5 * f_dx * dx
    f_XR = jnp.roll(f_XR, -1, axis=0)

    f_YL = f + 0.5 * f_dy * dx
    f_YR = f - 0.5 * f_dy * dx
    f_YR = jnp.roll(f_YR, -1, axis=1)

    f_ZL = f + 0.5 * f_dz * dx
    f_ZR = f - 0.5 * f_dz * dx
    f_ZR = jnp.roll(f_ZR, -1, axis=2)

    return f_XL, f_XR, f_YL, f_YR, f_ZL, f_ZR


def apply_fluxes(F, flux_F_X, flux_F_Y, flux_F_Z, dx, dt):
    fac = dt * dx * dx
    F += -fac * flux_F_X
    F += fac * jnp.roll(flux_F_X, 1, axis=0)

    F += -fac * flux_F_Y
    F += fac * jnp.roll(flux_F_Y, 1, axis=1)

    F += -fac * flux_F_Z
    F += fac * jnp.roll(flux_F_Z, 1, axis=2)

    return F


def get_flux(rho_L, vx_L, vy_L, vz_L, rho_R, vx_R, vy_R, vz_R):
    # compute star (averaged) states
    rho_star = 0.5 * (rho_L + rho_R)
    momx_star = 0.5 * (rho_L * vx_L + rho_R * vx_R)
    momy_star = 0.5 * (rho_L * vy_L + rho_R * vy_R)
    momz_star = 0.5 * (rho_L * vz_L + rho_R * vz_R)

    P_star = rho_star * cs * cs

    # compute fluxes (local Lax-Friedrichs/Rusanov)
    flux_Mass = momx_star
    flux_Momx = momx_star**2 / rho_star + P_star
    flux_Momy = momx_star * momy_star / rho_star
    flux_Momz = momx_star * momz_star / rho_star

    # find wavespeeds
    C_L = cs + jnp.abs(vx_L)
    C_R = cs + jnp.abs(vx_R)
    C = jnp.maximum(C_L, C_R)

    # add stabilizing diffusive term
    flux_Mass -= C * 0.5 * (rho_R - rho_L)
    flux_Momx -= C * 0.5 * (rho_R * vx_R - rho_L * vx_L)
    flux_Momy -= C * 0.5 * (rho_R * vy_R - rho_L * vy_L)
    flux_Momz -= C * 0.5 * (rho_R * vz_R - rho_L * vz_L)

    return flux_Mass, flux_Momx, flux_Momy, flux_Momz


def apply_grav_accel(rho, vx, vy, vz, dt):
    V_hat = -jnp.fft.fftn(4.0 * jnp.pi * G * (rho - rho_bar)) / (k_sq + (k_sq == 0))

    ax = -jnp.real(jnp.fft.ifftn(1.0j * kx * V_hat))
    ay = -jnp.real(jnp.fft.ifftn(1.0j * ky * V_hat))
    az = -jnp.real(jnp.fft.ifftn(1.0j * kz * V_hat))

    vx += ax * dt
    vy += ay * dt
    vz += az * dt

    return vx, vy, vz


def solve_hydro(rho, vx, vy, vz, dt):
    # calculate gradients
    rho_dx, rho_dy, rho_dz = get_gradient(rho, dx)
    vx_dx, vx_dy, vx_dz = get_gradient(vx, dx)
    vy_dx, vy_dy, vy_dz = get_gradient(vy, dx)
    vz_dx, vz_dy, vz_dz = get_gradient(vz, dx)

    # slope limit gradients
    rho_dx, rho_dy, rho_dz = slope_limiter(rho, dx, rho_dx, rho_dy, rho_dz)
    vx_dx, vx_dy, vx_dz = slope_limiter(vx, dx, vx_dx, vx_dy, vx_dz)
    vy_dx, vy_dy, vy_dz = slope_limiter(vy, dx, vy_dx, vy_dy, vy_dz)
    vz_dx, vz_dy, vz_dz = slope_limiter(vz, dx, vz_dx, vz_dy, vz_dz)

    # extrapolate half-step in time
    rho_prime = rho - 0.5 * dt * (
        vx * rho_dx
        + rho * vx_dx
        + vy * rho_dy
        + rho * vy_dy
        + vz * rho_dz
        + rho * vz_dz
    )
    vx_prime = vx - 0.5 * dt * (
        vx * vx_dx + vy * vx_dy + vz * vx_dz + (1.0 / rho) * rho_dx * cs * cs
    )
    vy_prime = vy - 0.5 * dt * (
        vx * vy_dx + vy * vy_dy + vz * vy_dz + (1.0 / rho) * rho_dy * cs * cs
    )
    vz_prime = vz - 0.5 * dt * (
        vx * vz_dx + vy * vz_dy + vz * vz_dz + (1.0 / rho) * rho_dz * cs * cs
    )

    # extrapolate in space to face centers
    rho_XL, rho_XR, rho_YL, rho_YR, rho_ZL, rho_ZR = extrap_to_face(
        rho_prime, rho_dx, rho_dy, rho_dz, dx
    )
    vx_XL, vx_XR, vx_YL, vx_YR, vx_ZL, vx_ZR = extrap_to_face(
        vx_prime, vx_dx, vx_dy, vx_dz, dx
    )
    vy_XL, vy_XR, vy_YL, vy_YR, vy_ZL, vy_ZR = extrap_to_face(
        vy_prime, vy_dx, vy_dy, vy_dz, dx
    )
    vz_XL, vz_XR, vz_YL, vz_YR, vz_ZL, vz_ZR = extrap_to_face(
        vz_prime, vz_dx, vz_dy, vz_dz, dx
    )

    # compute fluxes (local Lax-Friedrichs/Rusanov)
    flux_Mass_X, flux_Momx_X, flux_Momy_X, flux_Momz_X = get_flux(
        rho_XL, vx_XL, vy_XL, vz_XL, rho_XR, vx_XR, vy_XR, vz_XR
    )
    flux_Mass_Y, flux_Momy_Y, flux_Momz_Y, flux_Momx_Y = get_flux(
        rho_YL, vy_YL, vz_YL, vx_YL, rho_YR, vy_YR, vz_YR, vx_YR
    )
    flux_Mass_Z, flux_Momz_Z, flux_Momx_Z, flux_Momy_Z = get_flux(
        rho_ZL, vz_ZL, vx_ZL, vy_ZL, rho_ZR, vz_ZR, vx_ZR, vy_ZR
    )

    Mass, Momx, Momy, Momz = get_conserved(rho, vx, vy, vz, vol)

    # update solution
    Mass = apply_fluxes(Mass, flux_Mass_X, flux_Mass_Y, flux_Mass_Z, dx, dt)
    Momx = apply_fluxes(Momx, flux_Momx_X, flux_Momx_Y, flux_Momx_Z, dx, dt)
    Momy = apply_fluxes(Momy, flux_Momy_X, flux_Momy_Y, flux_Momy_Z, dx, dt)
    Momz = apply_fluxes(Momz, flux_Momz_X, flux_Momz_Y, flux_Momz_Z, dx, dt)

    # get Primitive variables
    rho, vx, vy, vz = get_primitive(Mass, Momx, Momy, Momz, vol)

    return rho, vx, vy, vz


def slope_limiter(f, dx, f_dx, f_dy, f_dz):
    """
    Apply slope limiter to slopes (minmod)
    """

    eps = 1.0e-12

    # Keep a copy of the original slopes
    orig_f_dx = f_dx
    orig_f_dy = f_dy
    orig_f_dz = f_dz

    # Function to adjust the denominator safely
    def adjust_denominator(denom):
        denom_safe = jnp.where(
            denom > 0, denom + eps, jnp.where(denom < 0, denom - eps, eps)
        )
        return denom_safe

    # For x-direction
    denom = adjust_denominator(orig_f_dx)
    num = (f - jnp.roll(f, 1, axis=0)) / dx
    ratio = num / denom
    limiter = jnp.maximum(0.0, jnp.minimum(1.0, ratio))
    f_dx = limiter * f_dx

    num = -(f - jnp.roll(f, -1, axis=0)) / dx
    ratio = num / denom  # Use the same adjusted denominator
    limiter = jnp.maximum(0.0, jnp.minimum(1.0, ratio))
    f_dx = limiter * f_dx

    # For y-direction
    denom = adjust_denominator(orig_f_dy)
    num = (f - jnp.roll(f, 1, axis=1)) / dx
    ratio = num / denom
    limiter = jnp.maximum(0.0, jnp.minimum(1.0, ratio))
    f_dy = limiter * f_dy

    num = -(f - jnp.roll(f, -1, axis=1)) / dx
    ratio = num / denom
    limiter = jnp.maximum(0.0, jnp.minimum(1.0, ratio))
    f_dy = limiter * f_dy

    # For z-direction
    denom = adjust_denominator(orig_f_dz)
    num = (f - jnp.roll(f, 1, axis=2)) / dx
    ratio = num / denom
    limiter = jnp.maximum(0.0, jnp.minimum(1.0, ratio))
    f_dz = limiter * f_dz

    num = -(f - jnp.roll(f, -1, axis=2)) / dx
    ratio = num / denom
    limiter = jnp.maximum(0.0, jnp.minimum(1.0, ratio))
    f_dz = limiter * f_dz

    return f_dx, f_dy, f_dz


############
# Black Hole


def get_cic_indices_and_weights(pos):
    """Compute the cloud-in-cell indices and weights for the particle positions."""
    dxs = jnp.array([dx, dx, dx])
    i = jnp.floor((pos - 0.5 * dxs) / dxs)
    ip1 = i + 1.0
    weight_i = ((ip1 + 0.5) * dxs - pos) / dxs
    weight_ip1 = (pos - (i + 0.5) * dxs) / dxs
    i = jnp.mod(i, jnp.array([nx, nx, nx])).astype(int)
    ip1 = jnp.mod(ip1, jnp.array([nx, nx, nx])).astype(int)
    return i, ip1, weight_i, weight_ip1


def bin_particles(pos, m_bh):
    """Bin the particles into the grid using cloud-in-cell weights."""
    rho = jnp.zeros((nx, nx, nx))
    i, ip1, w_i, w_ip1 = get_cic_indices_and_weights(pos)

    def deposit_particle(s, rho):
        """Deposit the particle mass into the grid."""
        fac = m_bh[s] / vol
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

    rho = jax.lax.fori_loop(0, n_bh, deposit_particle, rho)
    return rho


def get_acceleration(pos, rho):
    """Compute the acceleration of the particles."""
    i, ip1, w_i, w_ip1 = get_cic_indices_and_weights(pos)

    # find accelerations on the grid
    V_hat = -jnp.fft.fftn(4.0 * jnp.pi * G * (rho - rho_bar)) / (k_sq + (k_sq == 0))
    ax = -jnp.real(jnp.fft.ifftn(1.0j * kx * V_hat))
    ay = -jnp.real(jnp.fft.ifftn(1.0j * ky * V_hat))
    az = -jnp.real(jnp.fft.ifftn(1.0j * kz * V_hat))
    a_grid = jnp.stack((ax, ay, az), axis=-1)
    # a_max = jnp.max(jnp.abs(a_grid))

    # interpolate the accelerations to the particle positions
    acc = jnp.zeros((n_bh, 3))
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


###########
# Accretion


def do_accretion(psi, pos, vel, m_bh):
    """Accrete dark matter onto black hole."""

    # find the cell the BH is in
    dxs = jnp.array([dx, dx, dx])
    i = jnp.floor((pos - 0.5 * dxs) / dxs)
    i = jnp.mod(i, jnp.array([nx, nx, nx])).astype(int)
    s = 0

    psi_at_bh = psi[i[s, 0], i[s, 1], i[s, 2]]
    psi_amp = jnp.abs(psi_at_bh)
    psi_theta = jnp.angle(psi_at_bh)

    v = jnp.sqrt(cs**2 + vel[0, 0] ** 2 + vel[0, 1] ** 2 + vel[0, 2] ** 2)
    # XXX TODO: use alternate velocity estimate?
    xi = 2.0 * jnp.pi * G * m_bh[0] * m_per_hbar / v
    dM_dt = (
        32.0
        * jnp.pi**2
        * (G * m_bh[0]) ** 3
        * m_per_hbar
        * psi_amp**2
        / (c**3 * v * (1.0 - jnp.exp(-xi)))
    )

    # transfer mass from dm to BH
    new_psi_amp = jnp.abs(psi_amp) - jnp.sqrt(dM_dt * dt / vol)
    new_psi = new_psi_amp * jnp.exp(1.0j * psi_theta)

    psi = psi.at[i[s, 0], i[s, 1], i[s, 2]].set(new_psi)

    m_bh += dM_dt * dt

    return psi, m_bh


#######################
# Main part of the code


def compute_step(psi, rho, vx, vy, vz, m_bh, pos, vel, t):
    """Compute the next step in the simulation."""

    # (1/2) kick
    rho_bh = bin_particles(pos, m_bh)
    rho_tot = jnp.abs(psi) ** 2 + rho_bh + rho
    V = get_potential(rho_tot)
    psi = jnp.exp(-1.0j * m_per_hbar * dt / 2.0 * V) * psi

    vx, vy, vz = apply_grav_accel(rho_tot, vx, vy, vz, dt / 2.0)

    acc = get_acceleration(pos, rho_tot)
    vel = vel + acc * dt / 2.0

    # drift
    psi_hat = jnp.fft.fftn(psi)
    psi_hat = jnp.exp(dt * (-1.0j * k_sq / m_per_hbar / 2.0)) * psi_hat
    psi = jnp.fft.ifftn(psi_hat)

    rho, vx, vy, vz = solve_hydro(rho, vx, vy, vz, dt)

    pos = pos + vel * dt
    pos = jnp.mod(pos, jnp.array([Lx, Lx, Lx]))

    # (1/2) kick
    rho_bh = bin_particles(pos, m_bh)
    rho_tot = jnp.abs(psi) ** 2 + rho_bh + rho
    V = get_potential(rho_tot)
    psi = jnp.exp(-1.0j * m_per_hbar * dt / 2.0 * V) * psi

    vx, vy, vz = apply_grav_accel(rho_tot, vx, vy, vz, dt / 2.0)

    acc = get_acceleration(pos, rho_tot)
    vel = vel + acc * dt / 2.0

    # accretion
    psi, m_bh = do_accretion(psi, pos, vel, m_bh)

    # update time
    t += dt

    return psi, rho, vx, vy, vz, m_bh, pos, vel, t


@jax.jit
def update(_, state):
    """Update the state of the system by one time step."""
    (
        state["psi"],
        state["rho"],
        state["vx"],
        state["vy"],
        state["vz"],
        state["m_bh"],
        state["pos"],
        state["vel"],
        state["t"],
    ) = compute_step(
        state["psi"],
        state["rho"],
        state["vx"],
        state["vy"],
        state["vz"],
        state["m_bh"],
        state["pos"],
        state["vel"],
        state["t"],
    )
    return state


def plot_sim(state):
    """Plot the simulation state."""
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), dpi=80)

    # DM projection
    rho_proj_dm = jnp.log10(jnp.mean(jnp.abs(state["psi"]) ** 2, axis=2))
    vmin = jnp.log10(rho_bar / 100.0)
    vmax = jnp.log10(rho_bar * 100.0)
    axs[0].imshow(
        rho_proj_dm,
        cmap="inferno",
        origin="lower",
        extent=(0, nx, 0, nx),
        vmin=vmin,
        vmax=vmax,
    )
    # add bh
    sx = jax.lax.slice(state["pos"], (0, 0), (n_bh, 1)) / Lx * nx
    sy = jax.lax.slice(state["pos"], (0, 1), (n_bh, 2)) / Lx * nx
    axs[0].plot(sx, sy, color="black", marker=".", linestyle="None", markersize=10)
    # plt.colorbar(im0, ax=axs[0], label="log10(rho_dm)")
    axs[0].set_aspect("equal")
    axs[0].get_xaxis().set_visible(False)
    axs[0].get_yaxis().set_visible(False)

    # Gas projection
    rho_proj_gas = jnp.log10(jnp.mean(state["rho"], axis=2))
    vmin = jnp.log10(rho_bar * frac_gas / 100.0)
    vmax = jnp.log10(rho_bar * frac_gas * 100.0)
    axs[1].imshow(
        rho_proj_gas,
        cmap="viridis",
        origin="lower",
        extent=(0, nx, 0, nx),
        vmin=vmin,
        vmax=vmax,
    )
    # add bh
    sx = jax.lax.slice(state["pos"], (0, 0), (n_bh, 1)) / Lx * nx
    sy = jax.lax.slice(state["pos"], (0, 1), (n_bh, 2)) / Lx * nx
    axs[1].plot(sx, sy, color="black", marker=".", linestyle="None", markersize=10)
    # plt.colorbar(im1, ax=axs[1], label="log10(rho_gas)")
    axs[1].set_aspect("equal")
    axs[1].get_xaxis().set_visible(False)
    axs[1].get_yaxis().set_visible(False)

    plt.tight_layout()


def main():
    """Main physics simulation."""

    # Initial Conditions
    t = 0.0

    r = jnp.sqrt((X - 0.5 * Lx) ** 2 + (Y - 0.5 * Lx) ** 2 + (Z - 0.5 * Lx) ** 2)
    psi = (
        jnp.sqrt(
            1.9e7 * m_22**-2 * r_soliton**-4 / (1.0 + 0.091 * (r / r_soliton) ** 2) ** 8
        )
        + 0.0j
    )

    # add gas
    frac_gas = 0.1
    rho_gas = frac_gas * M_soliton / Lx**3
    rho = jnp.ones((nx, nx, nx)) * rho_gas
    vx = jnp.zeros((nx, nx, nx))
    vy = jnp.zeros((nx, nx, nx))
    vz = jnp.zeros((nx, nx, nx))

    # calculate rho_bar
    global rho_bar
    rho_bar = jnp.mean(jnp.abs(psi) ** 2, axis=(0, 1, 2))

    # black hole
    m_bh = M_bh * jnp.ones(1)  # mass of each black hole
    pos = jnp.array([[0.6 * Lx, 0.5 * Lx, 0.5 * Lx]])
    vel = jnp.array([[0.0, 40.0, 0.0]])
    assert n_bh == 1

    # Construct initial simulation state
    state = {}
    state["t"] = t
    state["psi"] = psi
    state["rho"] = rho
    state["vx"] = vx
    state["vy"] = vy
    state["vz"] = vz
    state["m_bh"] = m_bh
    state["pos"] = pos
    state["vel"] = vel

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
        print(f"   m_bh={state['m_bh'][0]:.3e}")
        plot_sim(state)
        plt.savefig(os.path.join(checkpoint_dir, f"snap{i:03d}.png"))
        if args.show:
            plt.pause(0.01)
        plt.clf()
        async_checkpoint_manager.wait_until_finished()
    jax.block_until_ready(state)
    print("Simulation Run Time (s): ", time.time() - t_start_timer)

    # Plot final state
    plot_sim(state)
    plt.savefig(os.path.join(checkpoint_dir, "final.png"), dpi=240)


if __name__ == "__main__":
    main()
