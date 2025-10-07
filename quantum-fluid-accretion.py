#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

import imageio_ffmpeg, matplotlib as mpl

mpl.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

import os, json, time, argparse
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib import animation
import cmasher as cmr
from matplotlib.patches import Circle

parser = argparse.ArgumentParser(description="Simulate the Schrodinger-Poisson system.")
parser.add_argument("--res", type=int, default=1, help="Resolution factor")
try:
    args, _unknown = parser.parse_known_args()
except SystemExit:

    class _Args:
        pass

    args = _Args()
    args.res = 1

# Enable for double precision if you like
jax.config.update("jax_enable_x64", True)

# ----------------------------
# Model toggles
# ----------------------------
GAS_SELF_GRAV = True  # gas sources gravity in Poisson
JEANS_FLOOR_ON = True  # enforce lambda_J >= N_J * dx
N_J = 9.0  # cells per Jeans length

# --- Hydro CFL cap ---
HYDRO_CFL = 0.15

# --- BH toggles/params ---
BH_ON = True
BH_GRAV = True
BH_MOVE = True

# SEED (target) mass after ramp
BH_INIT_M = 1.0e6  # Msun; asymptotic "seed" mass (after ramp completes)

# When to place (at DM peak); mass begins ramp from zero after this time
BH_INJECT_T = 14.5  # internal time units (kpc/(km/s))

# Bondi/sink parameters
R_ACC_MULT = 2.0  # geometric minimum sink radius
LAMBDA_ISO = float(jnp.exp(1.5) / 4.0)  # ≈ 1.12
BH_FMAX = 0.25  # fraction of kernel gas removable per step (None for no cap beyond conservation)

# --- smooth mass ramp controls ---
BH_RAMP_ON = True
BH_RAMP_TAU_MYR = 10.0  # if >0, use this absolute window (Myr); else fallback to fraction of crossing time
BH_RAMP_FRAC_XCROSS = 0.5  # fallback if BH_RAMP_TAU_MYR <= 0; tau = frac * t_cross
BH_RAMP_SHARPNESS = 6.0  # tanh steepness (larger => sharper within same tau)

# ----------------------------
# Unit system and parameters
# ----------------------------
# [L] = kpc, [V] = km/s, [M] = Msun → [T] = kpc/(km/s) ≈ 0.9778 Gyr
T_GYR_PER_UNIT = 0.9778

# Focused movie window after injection (keep same #frames)
FOCUS_POST_WINDOW_GYR = 3.0

nx = 64  # int(32 * args.res)
Lx = 10.0
rho_bar = 1.0e7
t_end = 28.0
m_22 = 1.0

# gas
frac_gas = 0.10
rho_gas = frac_gas * rho_bar
cs_const = 70.0

# dark matter
frac_dm = 1.0 - frac_gas
sigma = 40.0
M_soliton = 1.0e9
r_soliton = 2.2e8 * m_22**-2 / M_soliton  # kpc

# constants
G = 4.30241002e-6
hbar = 1.71818134e-87
ev_to_msun = 8.96215334e-67
ev_to_internal = 8.05478173e-56
c = 299792.458
m = m_22 * 1.0e-22 * ev_to_msun
m_per_hbar = m / hbar
h = 0.7
H0 = 0.1 * h
rho_crit = 3.0 * H0**2 / (8.0 * jnp.pi * G)

# ----------------------------
# Domain / Mesh
# ----------------------------
dx = Lx / nx
vol = dx**3
x_lin = jnp.linspace(0.5 * dx, Lx - 0.5 * dx, nx)
X, Y, Z = jnp.meshgrid(x_lin, x_lin, x_lin, indexing="ij")

# Fourier space
kx_lin = 2.0 * jnp.pi / Lx * jnp.arange(-nx / 2, nx / 2)
kx, ky, kz = jnp.meshgrid(kx_lin, kx_lin, kx_lin, indexing="ij")
kx = jnp.fft.ifftshift(kx)
ky = jnp.fft.ifftshift(ky)
kz = jnp.fft.ifftshift(kz)
k_sq = kx**2 + ky**2 + kz**2

# ----------------------------
# Low-pass filter for BH force sampling only
# ----------------------------
# Cut high-k wiggles that appear as nx increases
K_CUT_FRAC = 0.6
k_cut = K_CUT_FRAC * (jnp.pi / dx)
LP = jnp.exp(-((k_sq / (k_cut**2 + 1e-30)) ** 4))  # smooth roll-off

# ----------------------------
#  Physical blur parameters for peak finding
# ----------------------------
lambda_dB = float(hbar) / (float(m) * float(sigma))  # kpc
INJECT_SMOOTH_ON = True
INJECT_SMOOTH_KPC = 0.4 * lambda_dB

# ----------------------------
# Basic physical sanity checks (use plain floats to avoid JAX bool asserts)
# ----------------------------
de_broglie_wavelength = float(hbar) / (float(m) * float(sigma))
n_wavelengths = float(Lx) / de_broglie_wavelength

jeans_length = float(cs_const) * np.sqrt(np.pi / (float(G) * float(rho_gas)))
n_jeans = float(Lx) / jeans_length
assert n_jeans < 1.0, (
    f"Box smaller than Jeans length requirement failed: Lx/Jeans={n_jeans:.3f}"
)

v_resolved = (float(hbar) / float(m)) * np.pi / float(dx)

# ----------------------------
# Time step control (kinetic split)
# ----------------------------
dt_fac = 0.7
dt_kin = dt_fac * (m_per_hbar / 6.0) * (dx * dx)
nt = int(jnp.ceil(jnp.ceil(t_end / dt_kin) / 100.0) * 100)
nt_sub = int(jnp.round(nt / 100.0))
dt = t_end / nt  # baseline dt from SP kinetics; cap it by hydro CFL per step later

# hard hydro-CFL sanity (keeps dt from being too aggressive for cs_const)
assert dt < 2.0 * dx / cs_const


def cs_floor_from_rho(rho):
    return N_J * dx * jnp.sqrt(G * rho / jnp.pi)


cmax_guess = float(cs_const)
assert r_soliton < 0.5 * Lx


# ----------------------------
# Utilities
# ----------------------------
def get_potential_bg(rho_src, rho_bg):
    V_hat = -jnp.fft.fftn(4.0 * jnp.pi * G * (rho_src - rho_bg)) / (k_sq + (k_sq == 0))
    V = jnp.real(jnp.fft.ifftn(V_hat))
    return V_hat, V


def accel_from_Vhat(V_hat):
    ax = -jnp.real(jnp.fft.ifftn(1.0j * kx * V_hat))
    ay = -jnp.real(jnp.fft.ifftn(1.0j * ky * V_hat))
    az = -jnp.real(jnp.fft.ifftn(1.0j * kz * V_hat))
    return ax, ay, az


def get_conserved(rho, vx, vy, vz, vol):
    Mass = rho * vol
    Momx = rho * vx * vol
    Momy = rho * vy * vol
    Momz = rho * vz * vol
    return Mass, Momx, Momy, Momz


def get_primitive(Mass, Momx, Momy, Momz, vol):
    eps = 1e-30
    rho = Mass / vol
    vx = Momx / (Mass + eps)
    vy = Momy / (Mass + eps)
    vz = Momz / (Mass + eps)
    return rho, vx, vy, vz


def get_gradient(f, dx):
    f_dx = (jnp.roll(f, -1, axis=0) - jnp.roll(f, 1, axis=0)) / (2.0 * dx)
    f_dy = (jnp.roll(f, -1, axis=1) - jnp.roll(f, 1, axis=1)) / (2.0 * dx)
    f_dz = (jnp.roll(f, -1, axis=2) - jnp.roll(f, 1, axis=2)) / (2.0 * dx)
    return f_dx, f_dy, f_dz


def extrap_to_face(f, f_dx, f_dy, f_dz, dx):
    f_XL = f + 0.5 * f_dx * dx
    f_XR = jnp.roll(f - 0.5 * f_dx * dx, -1, axis=0)
    f_YL = f + 0.5 * f_dy * dx
    f_YR = jnp.roll(f - 0.5 * f_dy * dx, -1, axis=1)
    f_ZL = f + 0.5 * f_dz * dx
    f_ZR = jnp.roll(f - 0.5 * f_dz * dx, -1, axis=2)
    return f_XL, f_XR, f_YL, f_YR, f_ZL, f_ZR


def apply_fluxes(F, flux_F_X, flux_F_Y, flux_F_Z, dx, dt):
    fac = dt * dx * dx
    F = F - fac * flux_F_X + fac * jnp.roll(flux_F_X, 1, axis=0)
    F = F - fac * flux_F_Y + fac * jnp.roll(flux_F_Y, 1, axis=1)
    F = F - fac * flux_F_Z + fac * jnp.roll(flux_F_Z, 1, axis=2)
    return F


def slope_limiter(f, dx, f_dx, f_dy, f_dz):
    eps = 1.0e-12

    def adjust_denominator(denom):
        return jnp.where(denom > 0, denom + eps, jnp.where(denom < 0, denom - eps, eps))

    denom = adjust_denominator(f_dx)
    num = (f - jnp.roll(f, 1, axis=0)) / dx
    limiter = jnp.maximum(0.0, jnp.minimum(1.0, num / denom))
    f_dx = limiter * f_dx
    num = -(f - jnp.roll(f, -1, axis=0)) / dx
    limiter = jnp.maximum(0.0, jnp.minimum(1.0, num / denom))
    f_dx = limiter * f_dx
    denom = adjust_denominator(f_dy)
    num = (f - jnp.roll(f, 1, axis=1)) / dx
    limiter = jnp.maximum(0.0, jnp.minimum(1.0, num / denom))
    f_dy = limiter * f_dy
    num = -(f - jnp.roll(f, -1, axis=1)) / dx
    limiter = jnp.maximum(0.0, jnp.minimum(1.0, num / denom))
    f_dy = limiter * f_dy
    denom = adjust_denominator(f_dz)
    num = (f - jnp.roll(f, 1, axis=2)) / dx
    limiter = jnp.maximum(0.0, jnp.minimum(1.0, num / denom))
    f_dz = limiter * f_dz
    num = -(f - jnp.roll(f, -1, axis=2)) / dx
    limiter = jnp.maximum(0.0, jnp.minimum(1.0, num / denom))
    f_dz = limiter * f_dz
    return f_dx, f_dy, f_dz


def cs2_local(rho):
    if not JEANS_FLOOR_ON:
        return jnp.full_like(rho, cs_const**2)
    c_floor = N_J * dx * jnp.sqrt(G * rho / jnp.pi)
    c_eff = jnp.maximum(cs_const, c_floor)
    return c_eff**2


def get_flux_axis(rho_L, vxL, vyL, vzL, rho_R, vxR, vyR, vzR, c2_L, c2_R):
    rho_star = 0.5 * (rho_L + rho_R)
    momx_star = 0.5 * (rho_L * vxL + rho_R * vxR)
    momy_star = 0.5 * (rho_L * vyL + rho_R * vyR)
    momz_star = 0.5 * (rho_L * vzL + rho_R * vzR)
    c2_star = 0.5 * (c2_L + c2_R)
    P_star = rho_star * c2_star
    eps = 1e-30
    flux_M = momx_star
    flux_Mx = momx_star**2 / (rho_star + eps) + P_star
    flux_My = momx_star * momy_star / (rho_star + eps)
    flux_Mz = momx_star * momz_star / (rho_star + eps)
    C_L = jnp.sqrt(c2_L) + jnp.abs(vxL)
    C_R = jnp.sqrt(c2_R) + jnp.abs(vxR)
    C = jnp.maximum(C_L, C_R)
    flux_M -= C * 0.5 * (rho_R - rho_L)
    flux_Mx -= C * 0.5 * (rho_R * vxR - rho_L * vxL)
    flux_My -= C * 0.5 * (rho_R * vyR - rho_L * vyL)
    flux_Mz -= C * 0.5 * (rho_R * vzR - rho_L * vzL)
    return flux_M, flux_Mx, flux_My, flux_Mz


def solve_hydro(rho, vx, vy, vz, dt):
    c2 = cs2_local(rho)
    P = rho * c2
    rho_dx, rho_dy, rho_dz = get_gradient(rho, dx)
    vx_dx, vx_dy, vx_dz = get_gradient(vx, dx)
    vy_dx, vy_dy, vy_dz = get_gradient(vy, dx)
    vz_dx, vz_dy, vz_dz = get_gradient(vz, dx)
    P_dx, P_dy, P_dz = get_gradient(P, dx)
    rho_dx, rho_dy, rho_dz = slope_limiter(rho, dx, rho_dx, rho_dy, rho_dz)
    vx_dx, vx_dy, vx_dz = slope_limiter(vx, dx, vx_dx, vx_dy, vx_dz)
    vy_dx, vy_dy, vy_dz = slope_limiter(vy, dx, vy_dx, vy_dy, vy_dz)
    vz_dx, vz_dy, vz_dz = slope_limiter(vz, dx, vz_dx, vz_dy, vz_dz)
    P_dx, P_dy, P_dz = slope_limiter(P, dx, P_dx, P_dy, P_dz)
    eps = 1e-30
    rho_prime = rho - 0.5 * dt * (
        vx * rho_dx
        + rho * vx_dx
        + vy * rho_dy
        + rho * vy_dy
        + vz * rho_dz
        + rho * vz_dz
    )
    vx_prime = vx - 0.5 * dt * (
        vx * vx_dx + vy * vx_dy + vz * vx_dz + P_dx / (rho + eps)
    )
    vy_prime = vy - 0.5 * dt * (
        vx * vy_dx + vy * vy_dy + vz * vy_dz + P_dy / (rho + eps)
    )
    vz_prime = vz - 0.5 * dt * (
        vx * vz_dx + vy * vz_dy + vz * vz_dz + P_dz / (rho + eps)
    )
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
    c2_Lx, c2_Rx = c2, jnp.roll(c2, -1, axis=0)
    c2_Ly, c2_Ry = c2, jnp.roll(c2, -1, axis=1)
    c2_Lz, c2_Rz = c2, jnp.roll(c2, -1, axis=2)
    FX_M, FX_Mx, FX_My, FX_Mz = get_flux_axis(
        rho_XL, vx_XL, vy_XL, vz_XL, rho_XR, vx_XR, vy_XR, vz_XR, c2_Lx, c2_Rx
    )
    FY_M, FY_My, FY_Mz, FY_Mx = get_flux_axis(
        rho_YL, vy_YL, vz_YL, vx_YL, rho_YR, vy_YR, vz_YR, vx_YR, c2_Ly, c2_Ry
    )
    FZ_M, FZ_Mz, FZ_Mx, FZ_My = get_flux_axis(
        rho_ZL, vz_ZL, vx_ZL, vy_ZL, rho_ZR, vz_ZR, vx_ZR, vy_ZR, c2_Lz, c2_Rz
    )
    Mass, Momx, Momy, Momz = get_conserved(rho, vx, vy, vz, vol)
    Mass = apply_fluxes(Mass, FX_M, FY_M, FZ_M, dx, dt)
    Momx = apply_fluxes(Momx, FX_Mx, FY_Mx, FZ_Mx, dx, dt)
    Momy = apply_fluxes(Momy, FX_My, FY_My, FZ_My, dx, dt)
    Momz = apply_fluxes(Momz, FX_Mz, FY_Mz, FZ_Mz, dx, dt)
    return get_primitive(Mass, Momx, Momy, Momz, vol)


# ----------------------------
# BH helpers
# ----------------------------
def _periodic_delta_1d(xs, x0, L):
    d = jnp.abs(xs - x0)
    return jnp.minimum(d, L - d)


def bh_cic_density(nx, M_bh, x_bh, dx, Lx):
    rho_bh = jnp.zeros((nx, nx, nx))
    xi = x_bh / dx - 0.5
    i0 = jnp.floor(xi).astype(int)
    f = xi - i0

    def mod(i):
        return jnp.mod(i, nx)

    w = jnp.array(
        [
            (1 - f[0]) * (1 - f[1]) * (1 - f[2]),
            f[0] * (1 - f[1]) * (1 - f[2]),
            (1 - f[0]) * f[1] * (1 - f[2]),
            (1 - f[0]) * (1 - f[1]) * f[2],
            f[0] * f[1] * (1 - f[2]),
            f[0] * (1 - f[1]) * f[2],
            (1 - f[0]) * f[1] * f[2],
            f[0] * f[1] * f[2],
        ]
    )
    inds = jnp.array(
        [
            (mod(i0[0]), mod(i0[1]), mod(i0[2])),
            (mod(i0[0] + 1), mod(i0[1]), mod(i0[2])),
            (mod(i0[0]), mod(i0[1] + 1), mod(i0[2])),
            (mod(i0[0]), mod(i0[1]), mod(i0[2] + 1)),
            (mod(i0[0] + 1), mod(i0[1] + 1), mod(i0[2])),
            (mod(i0[0] + 1), mod(i0[1]), mod(i0[2] + 1)),
            (mod(i0[0]), mod(i0[1] + 1), mod(i0[2] + 1)),
            (mod(i0[0] + 1), mod(i0[1] + 1), mod(i0[2] + 1)),
        ]
    )
    for k in range(8):
        i, j, kz = inds[k]
        rho_bh = rho_bh.at[i, j, kz].add(w[k] * M_bh / (dx**3))
    return rho_bh


def _sample_cic(field, x_bh, dx, nx):
    xi = x_bh / dx - 0.5
    i0 = jnp.floor(xi).astype(int)
    f = xi - i0

    def mod(i):
        return jnp.mod(i, nx)

    ii = jnp.array(
        [i0[0], i0[0] + 1, i0[0], i0[0], i0[0] + 1, i0[0] + 1, i0[0], i0[0] + 1]
    )
    jj = jnp.array(
        [i0[1], i0[1], i0[1] + 1, i0[1], i0[1] + 1, i0[1], i0[1] + 1, i0[1] + 1]
    )
    kk = jnp.array(
        [i0[2], i0[2], i0[2], i0[2] + 1, i0[2], i0[2] + 1, i0[2] + 1, i0[2] + 1]
    )
    w = jnp.array(
        [
            (1 - f[0]) * (1 - f[1]) * (1 - f[2]),
            f[0] * (1 - f[1]) * (1 - f[2]),
            (1 - f[0]) * f[1] * (1 - f[2]),
            (1 - f[0]) * (1 - f[1]) * f[2],
            f[0] * f[1] * (1 - f[2]),
            f[0] * (1 - f[1]) * f[2],
            (1 - f[0]) * f[1] * f[2],
            f[0] * f[1] * f[2],
        ]
    )
    vals = field[mod(ii), mod(jj), mod(kk)]
    return jnp.sum(w * vals)


# ----- Bondi (ambient sampling + rB-scaled sink) -----
def bh_bondi_step_ambient(
    rho,
    M_bh_eff,
    x_bh,
    dx,
    Lx,
    cs_iso,
    G,
    dt,
    r_acc_mult=R_ACC_MULT,
    lam=LAMBDA_ISO,
    fmax=BH_FMAX,
    kappa_sink=2.0,
    ann_lo=2.0,
    ann_hi=4.0,
):
    nx = rho.shape[0]
    xs = (jnp.arange(nx) + 0.5) * dx
    DX = _periodic_delta_1d(xs, x_bh[0], Lx)
    DY = _periodic_delta_1d(xs, x_bh[1], Lx)
    DZ = _periodic_delta_1d(xs, x_bh[2], Lx)
    dX, dY, dZ = jnp.meshgrid(DX, DY, DZ, indexing="ij")
    R = jnp.sqrt(dX * dX + dY * dY + dZ * dZ)

    # Bondi radius
    rB = G * M_bh_eff / (cs_iso * cs_iso + 1e-300)

    # sink radius: max of (a few cells) and (kappa_sink * rB)
    r_acc_geom = r_acc_mult * dx  # e.g., 2 cells
    r_acc = jnp.maximum(r_acc_geom, kappa_sink * rB)

    # Gaussian kernel width tied to r_B but not smaller than ~half a cell, nor larger than half r_acc
    rK = jnp.clip(rB, 0.5 * dx, 0.5 * r_acc)
    mask_sink = R <= r_acc
    W_sink = jnp.exp(-0.5 * (R / (rK + 1e-30)) ** 2) * mask_sink

    # ambient density "at infinity": sample an annulus outside the sink
    r_lo = jnp.maximum(ann_lo * rB, 1.5 * dx)
    r_hi = jnp.maximum(ann_hi * rB, r_lo + dx)
    shell = (R >= r_lo) & (R <= r_hi)
    rho_inf = jnp.sum(rho * shell) / (jnp.sum(shell) + 1e-30)

    # Bondi rate (isothermal; v_rel=0)
    dMdt = 4.0 * jnp.pi * lam * (G * M_bh_eff) ** 2 * rho_inf / (cs_iso**3 + 1e-30)
    dM = dMdt * dt

    # Cap removal by available gas and per-step fraction
    WRho = W_sink * rho
    sumWRho = jnp.sum(WRho) + 1e-30
    phi = WRho / sumWRho
    M_avail_kernel = jnp.sum(rho * mask_sink) * (dx**3)

    dM_cap = (
        jnp.minimum(dM, M_avail_kernel)
        if (fmax is None)
        else jnp.minimum(dM, fmax * M_avail_kernel)
    )
    rho_new = jnp.maximum(0.0, rho - (dM_cap / (dx**3)) * phi)
    return rho_new, dMdt, rho_inf, rB, dM_cap


# ----------------------------
# Radial-profile helpers (peak-centered), NumPy side
# ----------------------------
def _periodic_offsets_1d(n, i0):
    idx = np.arange(n)
    d = np.abs(idx - int(i0))
    return np.minimum(d, n - d)


def radial_profile_centered(field_np, i0, j0, k0, dx):
    n = field_np.shape[0]
    nbins = n // 2
    di = _periodic_offsets_1d(n, i0)
    dj = _periodic_offsets_1d(n, j0)
    dk = _periodic_offsets_1d(n, k0)
    di2 = di[:, None, None] ** 2
    dj2 = dj[None, :, None] ** 2
    dk2 = dk[None, None, :] ** 2
    r_over_dx = np.sqrt(di2 + dj2 + dk2)
    bin_idx = np.minimum(r_over_dx.astype(np.int32), nbins - 1)
    vals = field_np.astype(np.float64)
    wsum = np.bincount(bin_idx.ravel(), weights=vals.ravel(), minlength=nbins)
    cnts = np.bincount(bin_idx.ravel(), minlength=nbins)
    prof = np.divide(wsum, cnts, out=np.zeros_like(wsum), where=cnts > 0)
    rc = dx * (np.arange(nbins) + 0.5)
    return rc, prof


# ----------------------------
# Seed mass ramp helper
# ----------------------------
if BH_RAMP_TAU_MYR is not None and BH_RAMP_TAU_MYR > 0.0:
    BH_RAMP_TAU = (BH_RAMP_TAU_MYR / 1000.0) / T_GYR_PER_UNIT  # internal units
else:
    t_cross = 2.0 * r_soliton / sigma  # internal units (kpc/(km/s))
    BH_RAMP_TAU = BH_RAMP_FRAC_XCROSS * t_cross

BH_RAMP_TAU = float(max(BH_RAMP_TAU, 1e-6))
BH_RAMP_W = float(max(BH_RAMP_TAU / BH_RAMP_SHARPNESS, 1e-6))
BH_RAMP_CENTER = float(BH_INJECT_T + 0.5 * BH_RAMP_TAU)


def seed_mass_from_time(t, placed_flag):
    placed = jnp.where(placed_flag > 0.5, 1.0, 0.0)
    if BH_RAMP_ON:
        s = 0.5 * (1.0 + jnp.tanh((t - BH_RAMP_CENTER) / (BH_RAMP_W + 1e-30)))
        s = s * jnp.where(t >= BH_INJECT_T, 1.0, 0.0)
        return BH_INIT_M * s * placed
    else:
        s = jnp.where(t >= BH_INJECT_T, 1.0, 0.0)
        return BH_INIT_M * s * placed


# ----------------------------
# Core step (kick-drift-kick) with BH injection site & smooth seed mass
# ----------------------------
def compute_step(psi, rho, vx, vy, vz, t, M_bh_acc, x_bh, vxbh, vybh, vzbh, bh_placed):
    # ---- Hydro CFL cap (compute dt_eff) ----
    c_eff = jnp.sqrt(cs2_local(rho))
    vmag = jnp.sqrt(vx * vx + vy * vy + vz * vz)
    Cmax = jnp.max(c_eff + vmag)
    dt_h = HYDRO_CFL * dx / (Cmax + 1e-30)
    dt_eff = jnp.minimum(jnp.array(dt), dt_h)

    # Effective BH mass at current time = seed(t) + accreted
    M_seed = seed_mass_from_time(t, bh_placed)
    M_bh_eff = M_seed + M_bh_acc

    # Background density for Poisson solve
    rho_bg = (frac_dm + (frac_gas if GAS_SELF_GRAV else 0.0)) * rho_bar

    # --- BH half-kick from DM+gas only (avoid self-force), BEFORE field half-kick ---
    if BH_ON and BH_MOVE:
        active = jnp.where(bh_placed > 0.5, 1.0, 0.0)
        V_hat_dm_g_pre, _ = get_potential_bg(
            jnp.abs(psi) ** 2 + (rho if GAS_SELF_GRAV else 0.0), rho_bg
        )
        # low-pass only the acceleration used to kick the BH
        axg_pre, ayg_pre, azg_pre = accel_from_Vhat(V_hat_dm_g_pre * LP)
        a_bhx_pre = _sample_cic(axg_pre, x_bh, dx, nx)
        a_bhy_pre = _sample_cic(ayg_pre, x_bh, dx, nx)
        a_bhz_pre = _sample_cic(azg_pre, x_bh, dx, nx)
        vxbh = vxbh + a_bhx_pre * (dt_eff / 2.0) * active
        vybh = vybh + a_bhy_pre * (dt_eff / 2.0) * active
        vzbh = vzbh + a_bhz_pre * (dt_eff / 2.0) * active

    rho_src = jnp.abs(psi) ** 2 + (rho if GAS_SELF_GRAV else 0.0)

    # ---- BH gravity source (mask traced conditions) ----
    if BH_ON and BH_GRAV:
        rho_bh_full = bh_cic_density(nx, M_bh_eff, x_bh, dx, Lx)
        active_bh = jnp.where((bh_placed > 0.5) & (M_bh_eff > 0.0), 1.0, 0.0)
        rho_bh = rho_bh_full * active_bh
    else:
        rho_bh = jnp.zeros((nx, nx, nx), dtype=jnp.abs(psi).dtype)

    rho_src_with_bh = rho_src + (rho_bh if (BH_ON and BH_GRAV) else 0.0)

    # KICK 1 (gravity on ψ and gas)
    V_hat, V = get_potential_bg(rho_src_with_bh, rho_bg)
    psi = jnp.exp(-1.0j * m_per_hbar * dt_eff / 2.0 * V) * psi
    ax, ay, az = accel_from_Vhat(V_hat)
    vx, vy, vz = (
        vx + ax * (dt_eff / 2.0),
        vy + ay * (dt_eff / 2.0),
        vz + az * (dt_eff / 2.0),
    )

    # DRIFT (quantum + hydro)
    psi_hat = jnp.fft.fftn(psi)
    psi_hat = jnp.exp(dt_eff * (-1.0j * k_sq / m_per_hbar / 2.0)) * psi_hat
    psi = jnp.fft.ifftn(psi_hat)
    rho, vx, vy, vz = solve_hydro(rho, vx, vy, vz, dt_eff)

    if BH_ON and BH_MOVE:
        active = jnp.where(bh_placed > 0.5, 1.0, 0.0)
        x_bh = jnp.mod(x_bh + dt_eff * jnp.array([vxbh, vybh, vzbh]) * active, Lx)

    # --- Physical minimum sink radius via r_acc_mult_eff ---
    r_acc_mult_eff = float(R_ACC_MULT)  # just a few cells as a geometric floor

    # --- Bondi accretion, using M_bh_eff ---
    if BH_ON:
        rho_try, dMdt_try, rho_inf_try, r_B_try, dM_cap = bh_bondi_step_ambient(
            rho=rho,
            M_bh_eff=M_bh_eff,
            x_bh=x_bh,
            dx=dx,
            Lx=Lx,
            cs_iso=cs_const,
            G=G,
            dt=dt_eff,
            r_acc_mult=r_acc_mult_eff,
            lam=LAMBDA_ISO,
            fmax=BH_FMAX,
            kappa_sink=2.0,
            ann_lo=2.0,
            ann_hi=4.0,
        )
        active_acc = jnp.where((bh_placed > 0.5) & (M_bh_eff > 0.0), 1.0, 0.0)

        rho = rho * (1.0 - active_acc) + rho_try * active_acc
        dMdt = dMdt_try * active_acc
        rho_inf = rho_inf_try * active_acc + (1.0 - active_acc) * jnp.array(
            rho_gas, dtype=rho_try.dtype
        )
        r_B = r_B_try * active_acc
        M_bh_acc = M_bh_acc + dM_cap * active_acc
    else:
        dMdt = jnp.array(0.0)
        rho_inf = jnp.array(rho_gas)
        r_B = jnp.array(0.0)

    # KICK 2 (gravity)
    M_bh_eff_mid = seed_mass_from_time(t, bh_placed) + M_bh_acc
    rho_src2 = jnp.abs(psi) ** 2 + (rho if GAS_SELF_GRAV else 0.0)

    if BH_ON and BH_GRAV:
        rho_bh2_full = bh_cic_density(nx, M_bh_eff_mid, x_bh, dx, Lx)
        active_bh2 = jnp.where((bh_placed > 0.5) & (M_bh_eff_mid > 0.0), 1.0, 0.0)
        rho_bh2 = rho_bh2_full * active_bh2
    else:
        rho_bh2 = jnp.zeros((nx, nx, nx), dtype=jnp.abs(psi).dtype)

    V_hat2, V2 = get_potential_bg(
        rho_src2 + (rho_bh2 if (BH_ON and BH_GRAV) else 0.0), rho_bg
    )
    psi = jnp.exp(-1.0j * m_per_hbar * dt_eff / 2.0 * V2) * psi
    ax2, ay2, az2 = accel_from_Vhat(V_hat2)
    vx, vy, vz = (
        vx + ax2 * (dt_eff / 2.0),
        vy + ay2 * (dt_eff / 2.0),
        vz + az2 * (dt_eff / 2.0),
    )

    if BH_ON and BH_MOVE:
        active = jnp.where(bh_placed > 0.5, 1.0, 0.0)
        V_hat_dm_g2, _ = get_potential_bg(
            jnp.abs(psi) ** 2 + (rho if GAS_SELF_GRAV else 0.0), rho_bg
        )
        # low-pass on BH force sampling (second half-kick)
        axg2, ayg2, azg2 = accel_from_Vhat(V_hat_dm_g2 * LP)
        a_bhx2 = _sample_cic(axg2, x_bh, dx, nx)
        a_bhy2 = _sample_cic(ayg2, x_bh, dx, nx)
        a_bhz2 = _sample_cic(azg2, x_bh, dx, nx)
        vxbh = vxbh + a_bhx2 * (dt_eff / 2.0) * active
        vybh = vybh + a_bhy2 * (dt_eff / 2.0) * active
        vzbh = vzbh + a_bhz2 * (dt_eff / 2.0) * active

    # advance time
    t = t + dt_eff

    # --- Inject site at DM peak (once), then seed mass begins ramp ---
    inject = (bh_placed < 0.5) & (t >= BH_INJECT_T)

    rho3d_now = jnp.abs(psi) ** 2
    # physical blur in real space (k-space Gaussian)
    if INJECT_SMOOTH_ON:
        sigma_r = float(max(INJECT_SMOOTH_KPC, 1.5 * float(dx)))
        s2 = sigma_r**2
        rho_hat = jnp.fft.fftn(rho3d_now)
        rho_hat = rho_hat * jnp.exp(-0.5 * s2 * k_sq)
        rho3d_s = jnp.real(jnp.fft.ifftn(rho_hat))
    else:
        rho3d_s = rho3d_now

    flat = jnp.argmax(rho3d_s)
    i0, j0, k0 = jnp.unravel_index(flat, rho3d_s.shape)
    x0 = (i0.astype(jnp.float64) + 0.5) * dx
    y0 = (j0.astype(jnp.float64) + 0.5) * dx
    z0 = (k0.astype(jnp.float64) + 0.5) * dx
    peak_xyz = jnp.array([x0, y0, z0])

    x_bh = jnp.where(inject, peak_xyz, x_bh)
    vxbh = jnp.where(inject, 0.0, vxbh)
    vybh = jnp.where(inject, 0.0, vybh)
    vzbh = jnp.where(inject, 0.0, vzbh)
    bh_placed = jnp.where(inject, jnp.array(1.0), bh_placed)

    # Effective mass to report at end-of-step (with new t and potentially placed)
    M_bh_eff_out = seed_mass_from_time(t, bh_placed) + M_bh_acc

    return (
        psi,
        rho,
        vx,
        vy,
        vz,
        t,
        M_bh_eff_out,
        M_bh_acc,
        x_bh,
        vxbh,
        vybh,
        vzbh,
        dMdt,
        rho_inf,
        r_B,
        bh_placed,
    )


@jax.jit
def update(_, state):
    (
        psi,
        rho,
        vx,
        vy,
        vz,
        t,
        M_bh_eff,
        M_bh_acc,
        x_bh,
        vxbh,
        vybh,
        vzbh,
        dMdt,
        rho_inf,
        r_B,
        bh_placed,
    ) = compute_step(
        state["psi"],
        state["rho"],
        state["vx"],
        state["vy"],
        state["vz"],
        state["t"],
        state["M_bh_acc"],
        state["x_bh"],
        state["vxbh"],
        state["vybh"],
        state["vzbh"],
        state["bh_placed"],
    )
    state = {
        "psi": psi,
        "rho": rho,
        "vx": vx,
        "vy": vy,
        "vz": vz,
        "t": t,
        "M_bh": M_bh_eff,
        "M_bh_acc": M_bh_acc,
        "x_bh": x_bh,
        "vxbh": vxbh,
        "vybh": vybh,
        "vzbh": vzbh,
        "bh_dMdt": dMdt,
        "bh_rho_inf": rho_inf,
        "bh_r_B": r_B,
        "bh_placed": bh_placed,
    }
    return state


# ======================================================
#           Frame extraction + video writing
# ======================================================
def _projections_from_state(st, dx, nx):
    """Return dict of 2D arrays for means along axes, plus BH pixel coords."""
    psi_abs2 = np.asarray(np.abs(st["psi"]) ** 2)
    rho_gas = np.asarray(st["rho"])
    # means along axes
    dm_xy = psi_abs2.mean(axis=2).astype(np.float32)  # (x,y)
    dm_xz = psi_abs2.mean(axis=1).astype(np.float32)  # (x,z)
    dm_yz = psi_abs2.mean(axis=0).astype(np.float32)  # (y,z)

    g_xy = rho_gas.mean(axis=2).astype(np.float32)
    g_xz = rho_gas.mean(axis=1).astype(np.float32)
    g_yz = rho_gas.mean(axis=0).astype(np.float32)

    # BH pixel coords (for overlay)
    bx, by, bz = [float(v) for v in st["x_bh"]]
    xpix = bx / float(dx) - 0.5
    ypix = by / float(dx) - 0.5
    zpix = bz / float(dx) - 0.5

    return {
        "dm_xy": dm_xy,
        "dm_xz": dm_xz,
        "dm_yz": dm_yz,
        "g_xy": g_xy,
        "g_xz": g_xz,
        "g_yz": g_yz,
        "bh_xy": (xpix, ypix),
        "bh_xz": (xpix, zpix),
        "bh_yz": (ypix, zpix),
    }


def _append_views(cache, views):
    for k, v in views.items():
        cache.setdefault(k, []).append(v.copy() if isinstance(v, np.ndarray) else v)


def _percentile_limits(frames):
    import numpy as np

    all_vals = np.concatenate([f.T.ravel() for f in frames])
    vmin, vmax = np.percentile(all_vals, [5, 95])
    return float(vmin), float(vmax)


def _write_video_side_by_side(
    frames_left,
    frames_right,
    marks_left,
    marks_right,
    out_path,
    cmap_left,
    cmap_right,
    fps=20,
    dpi=110,
):
    """Write a single MP4 with two panels (left/right) shown simultaneously."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Compute color limits per panel
    vminL, vmaxL = _percentile_limits(frames_left)
    vminR, vmaxR = _percentile_limits(frames_right)

    H, W = frames_left[0].shape
    fig, axs = plt.subplots(
        1, 2, figsize=(10.5, 5.25), dpi=dpi, constrained_layout=True
    )
    for ax in axs:
        ax.set_aspect("equal")
        ax.axis("off")

    imL = axs[0].imshow(
        np.log10(frames_left[0].T + 1e-30),
        origin="lower",
        vmin=np.log10(vminL + 1e-30),
        vmax=np.log10(vmaxL + 1e-30),
        extent=(0, W, 0, H),
        cmap=cmap_left,
    )
    imR = axs[1].imshow(
        np.log10(frames_right[0].T + 1e-30),
        origin="lower",
        vmin=np.log10(vminR + 1e-30),
        vmax=np.log10(vmaxR + 1e-30),
        extent=(0, W, 0, H),
        cmap=cmap_right,
    )

    (dotL,) = axs[0].plot([], [], "o", mfc="none", mec="w", mew=1.5, ms=6)
    (dotR,) = axs[1].plot([], [], "o", mfc="none", mec="w", mew=1.5, ms=6)

    writer = animation.FFMpegWriter(fps=fps)
    with writer.saving(fig, out_path, dpi):
        for fL, fR, mL, mR in zip(frames_left, frames_right, marks_left, marks_right):
            imL.set_data(np.log10(fL.T + 1e-30))
            imR.set_data(np.log10(fR.T + 1e-30))
            if mL is None or any(np.isnan(mL)):
                dotL.set_data([], [])
            else:
                dotL.set_data([mL[0]], [mL[1]])
            if mR is None or any(np.isnan(mR)):
                dotR.set_data([], [])
            else:
                dotR.set_data([mR[0]], [mR[1]])
            writer.grab_frame()
    plt.close(fig)


# ----------------------------
# Main
# ----------------------------
def main():
    # ----- Initial conditions
    t = 0.0
    global rho_bar

    # ============ DM IC: tighter central Gaussian blobs ============
    np.random.seed(17)
    psi_phase = np.exp(1.0j * 2.0 * np.pi * np.random.rand(*k_sq.shape))
    psi_phase = jnp.array(psi_phase)
    psi_phase *= jnp.sqrt(jnp.exp(-k_sq / (2.0 * sigma**2 * m_per_hbar**2)))
    psi_phase = np.fft.ifftn(psi_phase)
    psi_phase /= jnp.abs(psi_phase) + 1e-30

    N_BLOBS = 24
    SIGMA_BLOB = 0.12  # kpc
    DELTA_PEAK = 8.0  # stronger central overdensity
    rng = np.random.default_rng(2024)

    off_std = 0.1  # kpc, centers packed closer to middle
    cx = 0.5 * Lx + rng.normal(0.0, off_std, N_BLOBS)
    cy = 0.5 * Lx + rng.normal(0.0, off_std, N_BLOBS)
    cz = 0.5 * Lx + rng.normal(0.0, off_std, N_BLOBS)

    def dper(a, b):
        d = jnp.abs(a - b)
        return jnp.minimum(d, Lx - d)

    rho_blob = jnp.zeros_like(X)
    for k in range(N_BLOBS):
        dxk = dper(X, cx[k])
        dyk = dper(Y, cy[k])
        dzk = dper(Z, cz[k])
        r2k = dxk * dxk + dyk * dyk + dzk * dzk
        rho_blob = rho_blob + jnp.exp(-0.5 * r2k / (SIGMA_BLOB**2))

    rho_blob = rho_blob / (jnp.max(rho_blob) + 1e-30)
    overdens = 1.0 + DELTA_PEAK * rho_blob

    rho_dm_mean = frac_dm * rho_bar
    rho_dm = rho_dm_mean * overdens
    rho_dm *= rho_dm_mean / (jnp.mean(rho_dm) + 1e-30)

    psi = jnp.sqrt(rho_dm) * psi_phase
    psi *= jnp.sqrt(1.0 + 0.10 * (rng.random(psi.shape) - 0.5))

    rho_bar = jnp.mean(jnp.abs(psi) ** 2, axis=(0, 1, 2))

    # ------------------- gas init -------------------
    rho = jnp.ones((nx, nx, nx)) * rho_gas
    vx = jnp.zeros((nx, nx, nx))
    vy = jnp.zeros((nx, nx, nx))
    vz = jnp.zeros((nx, nx, nx))

    # --- BH state: no mass yet; inject site later; mass ramps smoothly after ---
    x_bh = jnp.array([0.5 * Lx, 0.5 * Lx, 0.5 * Lx])
    M_bh_acc = jnp.array(0.0)  # accreted component only
    vxbh = jnp.array(0.0)
    vybh = jnp.array(0.0)
    vzbh = jnp.array(0.0)
    bh_placed = jnp.array(0.0)  # 0 until site placed at t >= BH_INJECT_T

    state = {
        "t": t,
        "psi": psi,
        "rho": rho,
        "vx": vx,
        "vy": vy,
        "vz": vz,
        "x_bh": x_bh,
        "M_bh": jnp.array(0.0),
        "M_bh_acc": M_bh_acc,
        "vxbh": vxbh,
        "vybh": vybh,
        "vzbh": vzbh,
        "bh_dMdt": jnp.array(0.0),
        "bh_rho_inf": jnp.array(0.0),
        "bh_r_B": jnp.array(0.0),
        "bh_placed": bh_placed,
    }

    # ---------- HISTORY (for diagnostics) ----------
    history = {
        "time": [],
        "M_bh": [],
        "dMdt": [],
        "rho_inf": [],
        "r_B": [],
        "rc": None,
        "prof_dm": [],
        "prof_gas": [],
        "d_bh_peak": [],
        # NEW:
        "vxbh": [],
        "vybh": [],
        "vzbh": [],
        "vbh": [],
    }

    # =========================
    # WARM-UP to injection time
    # =========================
    while float(state["t"]) < BH_INJECT_T:
        state = update(0, state)

    # Recenter to DM peak after warm-up (optional)
    rho3d = np.asarray(np.abs(state["psi"]) ** 2)
    flat = int(np.argmax(rho3d))
    i0c, j0c, k0c = np.unravel_index(flat, rho3d.shape)
    ic = nx // 2
    jc = nx // 2
    kc = nx // 2

    def _roll3(a, sx, sy, sz):
        return np.roll(np.roll(np.roll(a, sx, axis=0), sy, axis=1), sz, axis=2)

    def _best_shift(i, ic, n):
        return int((((ic - i) + n // 2) % n) - n // 2)

    sx = _best_shift(i0c, ic, nx)
    sy = _best_shift(j0c, jc, nx)
    sz = _best_shift(k0c, kc, nx)
    psi_np = _roll3(np.asarray(state["psi"]), sx, sy, sz)
    rho_np = _roll3(np.asarray(state["rho"]), sx, sy, sz)
    vx_np = _roll3(np.asarray(state["vx"]), sx, sy, sz)
    vy_np = _roll3(np.asarray(state["vy"]), sx, sy, sz)
    vz_np = _roll3(np.asarray(state["vz"]), sx, sy, sz)
    state["psi"] = jnp.array(psi_np)
    state["rho"] = jnp.array(rho_np)
    state["vx"] = jnp.array(vx_np)
    state["vy"] = jnp.array(vy_np)
    state["vz"] = jnp.array(vz_np)
    state["x_bh"] = jnp.mod(
        state["x_bh"]
        + jnp.array([sx * dx, sy * dx, sz * dx], dtype=state["x_bh"].dtype),
        Lx,
    )

    # =========================
    # BUILD post-injection grid
    # =========================
    t_start = float(state["t"])  # ~ BH_INJECT_T
    t_stop = t_start + FOCUS_POST_WINDOW_GYR / T_GYR_PER_UNIT
    frames = 300  # snapshots
    fps = 20
    t_targets = np.linspace(t_start, t_stop, frames + 1)[1:]  # per-frame target times

    # --------- SIMULATE ONCE; CACHE VIEWS ----------
    VIEWS = {}  # dict of lists (dm_xy, g_xy, dm_xz, g_xz, dm_yz, g_yz, bh_xy, bh_xz, bh_yz)

    history["time"].append(float(state["t"]))
    history["M_bh"].append(float(state["M_bh"]))
    history["dMdt"].append(float(state["bh_dMdt"]))
    history["rho_inf"].append(float(state["bh_rho_inf"]))
    history["r_B"].append(float(state["bh_r_B"]))
    history["d_bh_peak"].append(np.nan)
    history["vxbh"].append(float(state["vxbh"]))
    history["vybh"].append(float(state["vybh"]))
    history["vzbh"].append(float(state["vzbh"]))
    history["vbh"].append(
        float(np.sqrt(state["vxbh"] ** 2 + state["vybh"] ** 2 + state["vzbh"] ** 2))
    )

    for target_t in t_targets:
        # advance until reaching target time
        st = state
        while float(st["t"]) < target_t:
            st = update(0, st)

        # cache projections and BH pixel positions
        views = _projections_from_state(st, dx=float(dx), nx=nx)
        _append_views(VIEWS, views)

        # --- diagnostics (profiles & distances) ---
        rho3d_now = np.asarray(np.abs(st["psi"]) ** 2)
        gas3d_now = np.asarray(st["rho"])

        # find DM peak with same blur used during injection
        if INJECT_SMOOTH_ON:
            sigma_r = float(max(INJECT_SMOOTH_KPC, 1.5 * float(dx)))
            s2 = sigma_r**2
            rho_hat_np = np.fft.fftn(rho3d_now)
            rho_hat_np *= np.exp(-0.5 * s2 * np.asarray(k_sq))
            rho_for_peak = np.fft.ifftn(rho_hat_np).real
        else:
            rho_for_peak = rho3d_now

        flat = int(np.argmax(rho_for_peak))
        i0, j0, k0 = np.unravel_index(flat, rho_for_peak.shape)
        rc, p_dm = radial_profile_centered(rho3d_now, i0, j0, k0, float(dx))
        _, p_gas = radial_profile_centered(gas3d_now, i0, j0, k0, float(dx))
        if history["rc"] is None:
            history["rc"] = rc
        history["prof_dm"].append(p_dm)
        history["prof_gas"].append(p_gas)

        # BH–peak 3D separation (periodic)
        def per_sep(a, b, L):
            d = abs(float(a) - float(b))
            return min(d, L - d)

        peak_xyz = np.array(
            [(i0 + 0.5) * float(dx), (j0 + 0.5) * float(dx), (k0 + 0.5) * float(dx)],
            dtype=float,
        )
        if float(st["bh_placed"]) > 0.5:
            bx, by, bz = [float(v) for v in st["x_bh"]]
            d = np.sqrt(
                per_sep(bx, peak_xyz[0], Lx) ** 2
                + per_sep(by, peak_xyz[1], Lx) ** 2
                + per_sep(bz, peak_xyz[2], Lx) ** 2
            )
        else:
            d = np.nan
        history["d_bh_peak"].append(d)

        # sync & record history
        state.update(st)
        history["time"].append(float(st["t"]))
        history["M_bh"].append(float(st["M_bh"]))
        history["dMdt"].append(float(st["bh_dMdt"]))
        history["rho_inf"].append(float(st["bh_rho_inf"]))
        history["r_B"].append(float(st["bh_r_B"]))
        # NEW in the loop after existing appends:
        history["vxbh"].append(float(st["vxbh"]))
        history["vybh"].append(float(st["vybh"]))
        history["vzbh"].append(float(st["vzbh"]))
        history["vbh"].append(
            float(np.sqrt(st["vxbh"] ** 2 + st["vybh"] ** 2 + st["vzbh"] ** 2))
        )

    # return both history and cached projections
    return history, VIEWS, fps


# ----------------------------
# Run + write outputs
# ----------------------------
if __name__ == "__main__":
    H, V, fps = main()

    out_dir = "checkpoint_dir"
    os.makedirs(out_dir, exist_ok=True)

    # ---- time conversion: internal time unit -> Gyr
    T_GYR_PER_UNIT = 0.9778  # kpc/(km/s)

    # Full history arrays (post-injection sampling)
    t_all = np.asarray(H["time"], dtype=float)
    t_all_gyr = t_all * T_GYR_PER_UNIT

    M_bh = np.asarray(H["M_bh"], dtype=float)
    dMdt = np.asarray(H["dMdt"], dtype=float)  # Msun / (kpc/(km/s))
    rhoinf = np.asarray(H["rho_inf"], dtype=float)
    rB = np.asarray(H["r_B"], dtype=float)
    dsep = np.asarray(H.get("d_bh_peak", []), dtype=float)

    # --- injection-relative arrays for the top panels ---
    has_bh = M_bh > 0.0
    if np.any(has_bh):
        inj_idx = int(np.argmax(has_bh))
        t_inj_int = t_all[inj_idx]
    else:
        t_inj_int = float(BH_INJECT_T)

    mask = t_all > t_inj_int
    t_gyr = (t_all[mask] - t_inj_int) * T_GYR_PER_UNIT
    M_bh_plot = M_bh[mask]
    dMdt_g = dMdt[mask] / T_GYR_PER_UNIT  # Msun / Gyr
    rhoinf_plot = rhoinf[mask]
    rB_plot = rB[mask]
    dsep_plot = dsep[mask] if dsep.size == t_all.size else np.full_like(t_gyr, np.nan)

    # ---- diagnostics figure (2x2) ----
    fig_ts, axs = plt.subplots(
        2, 2, figsize=(11.5, 7.5), dpi=120, constrained_layout=True
    )

    # (1,1) BH mass growth
    axs[0, 0].plot(t_gyr, M_bh_plot, lw=2)
    axs[0, 0].set_xlabel("t since injection [Gyr]")
    axs[0, 0].set_ylabel(r"$M_\bullet$ [M$_\odot$]")
    axs[0, 0].set_title("BH mass growth (seed ramp + accretion)")
    axs[0, 0].grid(alpha=0.3)

    # (1,2) Dual axis: left = dM/dt, right = rho_inf
    axL = axs[0, 1]
    axR = axL.twinx()
    (l1,) = axL.plot(t_gyr, dMdt_g, lw=2, label=r"$\dot M$ (left)")
    (l2,) = axR.plot(t_gyr, rhoinf_plot, lw=2, ls="--", label=r"$\rho_\infty$ (right)")
    axL.set_xlabel("t since injection [Gyr]")
    axL.set_ylabel(r"$\dot M$ [M$_\odot$/Gyr]")
    axR.set_ylabel(r"$\rho_\infty$ [M$_\odot$/kpc$^3$]")
    axL.set_title("Bondi accretion (left) & Ambient density (right)")
    axL.grid(alpha=0.3)
    # combined legend
    lines, labels = [l1, l2], [l.get_label() for l in (l1, l2)]
    axL.legend(lines, labels, fontsize=8, frameon=False)
    # consistent x-limits
    xmax = t_gyr.max() if t_gyr.size else 1.0
    axL.set_xlim(0.0, xmax)
    axR.set_xlim(0.0, xmax)

    # (2,1) Radial density profiles (log-log)
    ax = axs[1, 0]
    try:
        rc = np.asarray(H["rc"])
        Pdm = np.asarray(H["prof_dm"], dtype=float)  # [nframes, nbins]
        Pgas = np.asarray(H["prof_gas"], dtype=float)  # [nframes, nbins]
        if (
            Pdm.ndim == 2
            and Pgas.ndim == 2
            and Pdm.shape == Pgas.shape
            and rc is not None
            and len(Pdm) > 0
        ):
            n = Pdm.shape[0]
            idxs = np.unique(np.clip([0, n // 2, n - 1], 0, max(n - 1, 0))).astype(int)
            tg = t_all_gyr[1:]  # post-injection sampling times (aligned with profiles)
            for ii in idxs:
                label_dm = f"FDM  t={tg[ii]:.3f} Gyr"
                label_gas = f"Gas  t={tg[ii]:.3f} Gyr"
                ax.plot(rc, Pdm[ii], lw=2, label=label_dm)
                ax.plot(rc, Pgas[ii], lw=1.8, ls="--", label=label_gas)
            ax.set_xlabel("r from DM peak [kpc]")
            ax.set_ylabel(r"⟨ρ(r)⟩ [M$_\odot$/kpc$^3$]")
            ax.set_title("Radial density profiles (peak-centered)")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim(rc[0], rc[-1])
            ax.grid(alpha=0.3, which="both")
            handles, labels = ax.get_legend_handles_labels()
            if len(handles) > 6:
                handles, labels = handles[:6], labels[:6]
            ax.legend(handles, labels, fontsize=8, frameon=False)
        else:
            ax.text(0.5, 0.5, "No profiles recorded", ha="center", va="center")
            ax.axis("off")
    except Exception as e:
        ax.text(0.5, 0.5, f"Profile error:\n{e}", ha="center", va="center")
        ax.axis("off")

    # (2,2) BH–core separation
    axd = axs[1, 1]
    if np.isfinite(dsep_plot).any():
        axd.plot(t_gyr, dsep_plot, lw=2)
        axd.set_xlabel("t since injection [Gyr]")
        axd.set_ylabel("BH–core separation [kpc]")
        axd.set_title("BH distance to smoothed DM peak")
        axd.grid(alpha=0.3)
        axd.set_xlim(0.0, xmax)
    else:
        axd.text(0.5, 0.5, "Separation not available", ha="center", va="center")
        axd.axis("off")

    for ax_ in (axs[0, 0],):
        ax_.set_xlim(0.0, xmax)

    plt.savefig(os.path.join(out_dir, "diagnost6.png"), dpi=240)
    plt.close(fig_ts)

    # --- BH velocity plot (signed vx) in Gyr ---
    vx_all = np.asarray(H["vxbh"], dtype=float)
    vy_all = np.asarray(H["vybh"], dtype=float)  # (optional)
    vz_all = np.asarray(H["vzbh"], dtype=float)  # (optional)
    vbh_all = np.asarray(H["vbh"], dtype=float)  # (optional, magnitude)

    vx_plot = vx_all[mask]  # align with t_gyr
    # Optional extras:
    # vy_plot = vy_all[mask]
    # vz_plot = vz_all[mask]
    # vmag_plot = vbh_all[mask]

    fig_v, axv = plt.subplots(
        1, 1, figsize=(6.6, 4.2), dpi=130, constrained_layout=True
    )
    axv.plot(t_gyr, vx_plot, lw=2)
    axv.set_xlabel("Time since injection [Gyr]")
    axv.set_ylabel(r"$v_x$ (km s$^{-1}$)")
    axv.set_title("BH velocity (x-component)")
    axv.set_xlim(0.0, 0.5)
    axv.grid(alpha=0.35)

    if t_gyr.size:
        for x in np.arange(0.0, t_gyr.max() + 1e-12, 0.05):
            axv.axvline(x, ls="--", color="k", alpha=0.25, lw=0.8)

    fig_v.savefig(os.path.join(out_dir, "bh_velocity_gyr.png"), dpi=200)
    plt.close(fig_v)

    # --- baseline vs boost ---
    rho0 = float(rho_gas)
    lam = float(LAMBDA_ISO)
    Gf = float(G)
    cs = float(cs_const)
    dMdt0_g = (
        4.0 * np.pi * lam * (Gf**2) * (M_bh_plot**2) * rho0 / (cs**3 + 1e-300)
    ) / T_GYR_PER_UNIT
    alpha = np.where(dMdt0_g > 0.0, dMdt_g / dMdt0_g, np.nan)

    fig_boost, axb = plt.subplots(
        1, 1, figsize=(6.5, 4.0), dpi=120, constrained_layout=True
    )
    axb.plot(t_gyr, alpha, lw=2)
    axb.axhline(1.0, ls="--", alpha=0.6)
    axb.set_xlabel("t since injection [Gyr]")
    axb.set_ylabel(r"Boost $\alpha$")
    axb.set_title("Analytic Bondi boost vs uniform-gas baseline")
    axb.grid(alpha=0.3)
    fig_boost.savefig(os.path.join(out_dir, "boost_factor6.png"), dpi=120)
    plt.close(fig_boost)

    # =========================
    # Write side-by-side videos (DM | Gas)
    # =========================
    # Each combined projection: (dm_key, gas_key, mark_key, out_path, cmap_dm, cmap_gas)
    combos = [
        (
            "dm_xy",
            "g_xy",
            "bh_xy",
            os.path.join(out_dir, "combo_xy6.mp4"),
            cmr.bubblegum,
            "viridis",
        ),
        (
            "dm_xz",
            "g_xz",
            "bh_xz",
            os.path.join(out_dir, "combo_xz6.mp4"),
            cmr.bubblegum,
            "viridis",
        ),
        (
            "dm_yz",
            "g_yz",
            "bh_yz",
            os.path.join(out_dir, "combo_yz6.mp4"),
            cmr.bubblegum,
            "viridis",
        ),
    ]

    for dm_key, gas_key, mark_key, path, cmap_dm, cmap_gas in combos:
        frames_dm = V[dm_key]
        frames_gas = V[gas_key]
        marks = V[mark_key]
        n = min(len(frames_dm), len(frames_gas), len(marks))
        _write_video_side_by_side(
            frames_dm[:n],
            frames_gas[:n],
            marks[:n],
            marks[:n],
            path,
            cmap_left=cmap_dm,
            cmap_right=cmap_gas,
            fps=fps,
            dpi=110,
        )
