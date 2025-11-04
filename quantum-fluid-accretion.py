#!/usr/bin/env python
# coding: utf-8

import imageio_ffmpeg, matplotlib as mpl
mpl.rcParams['animation.ffmpeg_path'] = imageio_ffmpeg.get_ffmpeg_exe()
import h5py
import os, json, time, argparse
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib import animation
import cmasher as cmr
from matplotlib.patches import Circle
import matplotlib.patheffects as pe  # <-- NEW (for ring pop)

# ----------------------------
# CLI (safe for Jupyter too)
# ----------------------------
parser = argparse.ArgumentParser(description="Simulate the Schrodinger-Poisson system.")
parser.add_argument("--res", type=int, default=1, help="Resolution factor")
parser.add_argument("--bh-mass", type=float, default=None,
                    help="Override seed BH mass (Msun)")
parser.add_argument("--cs", type=float, default=None,
                    help="Override isothermal sound speed (km/s)")

try:
    args, _unknown = parser.parse_known_args()
except SystemExit:
    class _Args: pass
    args = _Args()
    args.res = 1

# Enable for double precision if you like
jax.config.update("jax_enable_x64", True)

# ----------------------------
# Model toggles
# ----------------------------
# --- Hydro CFL cap (simple) ---
HYDRO_CFL = 0.15  # gentler

# --- BH toggles/params ---
BH_ON        = True
BH_GRAV      = True
BH_MOVE      = True
BH_ACCRETION = True   # <--- turn BH accretion (sink + mass growth) on/off

# SEED (target) mass after ramp
BH_INIT_M    = 1.0e7       # Msun; asymptotic "seed" mass (after ramp completes)
# --- OPTIONAL: override seed mass from CLI or env (no other changes) ---
_env = os.getenv("BH_INIT_M", None)
if _env is not None:
    BH_INIT_M = float(_env)
if getattr(args, "bh_mass", None) is not None:
    BH_INIT_M = float(args.bh_mass)

# When to place (at DM peak); mass begins ramp from zero after this time
BH_INJECT_T  = 13.0        # internal time units (kpc/(km/s))

# Bondi/sink parameters
R_ACC_MULT   = 2.5         # geometric minimum sink radius in *cells*
LAMBDA_ISO   = float(jnp.exp(1.5) / 4.0)  # ≈ 1.12
BH_FMAX      = None       # fraction of kernel gas removable per step (None for no cap beyond conservation)
ann_lo=1.1
ann_hi=1.8

# --- smooth mass ramp controls ---
BH_RAMP_ON          = True
BH_RAMP_TAU_MYR     = 10.0      # if >0, use this absolute window (Myr); else fallback to fraction of crossing time
BH_RAMP_FRAC_XCROSS = 0.5       # fallback if BH_RAMP_TAU_MYR <= 0; tau = frac * t_cross
BH_RAMP_SHARPNESS   = 6.0       # tanh steepness (larger => sharper within same tau)

# ----------------------------
# Unit system and parameters
# ----------------------------
# [L] = kpc, [V] = km/s, [M] = Msun → [T] = kpc/(km/s) ≈ 0.9778 Gyr
T_GYR_PER_UNIT = 0.9778

# Focused movie window after injection (keep same #frames)
FOCUS_POST_WINDOW_GYR = 0.5

nx = 128  # int(32 * args.res)
Lx = 10.0
rho_bar = 1.0e7
t_end = 28.0
m_22 = 1.0

# gas
frac_gas = 0.10
rho_gas = frac_gas * rho_bar

cs_const = 60.0  # default
# Optional: override from environment
_env_cs = os.getenv("CS_CONST", None)
if _env_cs is not None:
    try:
        cs_const = float(_env_cs)
    except ValueError:
        pass
# Optional: override from CLI (wins over env)
if getattr(args, "cs", None) is not None:
    cs_const = float(args.cs)

# dark matter
frac_dm = 1.0 - frac_gas
sigma = 40
M_soliton = 1.0e7
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
LP = jnp.exp(- (k_sq / (k_cut**2 + 1e-30))**4)  # smooth roll-off

# ----------------------------
#  Physical blur parameters for peak finding
# ----------------------------
lambda_dB = float(hbar) / (float(m) * float(sigma))  # kpc
INJECT_SMOOTH_ON = True
INJECT_SMOOTH_KPC = 0.4 * lambda_dB                  # 0.4–0.6 λ_dB is robust

# ----------------------------
# Basic physical sanity checks (use plain floats to avoid JAX bool asserts)
# ----------------------------
de_broglie_wavelength = float(hbar) / (float(m) * float(sigma))
n_wavelengths = float(Lx) / de_broglie_wavelength

jeans_length = float(cs_const) * np.sqrt(np.pi / (float(G) * float(rho_gas)))
n_jeans = float(Lx) / jeans_length
#assert n_jeans < 1.0, f"Box smaller than Jeans length requirement failed: Lx/Jeans={n_jeans:.3f}"

v_resolved = (float(hbar) / float(m)) * np.pi / float(dx)

# ----------------------------
# Time step control (kinetic split)
# ----------------------------
dt_fac = 0.7
dt_kin = dt_fac * (m_per_hbar / 6.0) * (dx * dx)
nt = int(jnp.ceil(jnp.ceil(t_end / dt_kin) / 100.0) * 100)
dt = t_end / nt  # baseline dt from SP kinetics; cap it by hydro CFL per step later

# hard hydro-CFL sanity (keeps dt from being too aggressive for cs_const)
#assert dt < 2.0 * dx / cs_const

cmax_guess = float(cs_const)
#assert r_soliton < 0.5 * Lx
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
    denom = adjust_denominator(f_dx); num = (f - jnp.roll(f, 1, axis=0)) / dx
    limiter = jnp.maximum(0.0, jnp.minimum(1.0, num / denom)); f_dx = limiter * f_dx
    num = -(f - jnp.roll(f, -1, axis=0)) / dx
    limiter = jnp.maximum(0.0, jnp.minimum(1.0, num / denom)); f_dx = limiter * f_dx
    denom = adjust_denominator(f_dy); num = (f - jnp.roll(f, 1, axis=1)) / dx
    limiter = jnp.maximum(0.0, jnp.minimum(1.0, num / denom)); f_dy = limiter * f_dy
    num = -(f - jnp.roll(f, -1, axis=1)) / dx
    limiter = jnp.maximum(0.0, jnp.minimum(1.0, num / denom)); f_dy = limiter * f_dy
    denom = adjust_denominator(f_dz); num = (f - jnp.roll(f, 1, axis=2)) / dx
    limiter = jnp.maximum(0.0, jnp.minimum(1.0, num / denom)); f_dz = limiter * f_dz
    num = -(f - jnp.roll(f, -1, axis=2)) / dx
    limiter = jnp.maximum(0.0, jnp.minimum(1.0, num / denom)); f_dz = limiter * f_dz
    return f_dx, f_dy, f_dz

def cs2_local(rho):
    # Jeans floor removed: always constant sound speed (unchanged by request)
    return jnp.full_like(rho, cs_const**2)

def get_flux_axis(rho_L, vxL, vyL, vzL, rho_R, vxR, vyR, vzR, c2_L, c2_R):
    rho_star  = 0.5 * (rho_L + rho_R)
    momx_star = 0.5 * (rho_L * vxL + rho_R * vxR)
    momy_star = 0.5 * (rho_L * vyL + rho_R * vyR)
    momz_star = 0.5 * (rho_L * vzL + rho_R * vzR)
    c2_star   = 0.5 * (c2_L + c2_R)
    P_star    = rho_star * c2_star
    eps = 1e-30
    flux_M  = momx_star
    flux_Mx = momx_star**2 / (rho_star + eps) + P_star
    flux_My = momx_star * momy_star / (rho_star + eps)
    flux_Mz = momx_star * momz_star / (rho_star + eps)
    C_L = jnp.sqrt(c2_L) + jnp.abs(vxL)
    C_R = jnp.sqrt(c2_R) + jnp.abs(vxR)
    C   = jnp.maximum(C_L, C_R)
    flux_M  -= C * 0.5 * (rho_R - rho_L)
    flux_Mx -= C * 0.5 * (rho_R * vxR - rho_L * vxL)
    flux_My -= C * 0.5 * (rho_R * vyR - rho_L * vyL)
    flux_Mz -= C * 0.5 * (rho_R * vzR - rho_L * vzL)
    return flux_M, flux_Mx, flux_My, flux_Mz

def solve_hydro(rho, vx, vy, vz, dt):
    c2 = cs2_local(rho)
    P  = rho * c2
    rho_dx, rho_dy, rho_dz = get_gradient(rho, dx)
    vx_dx,  vx_dy,  vx_dz  = get_gradient(vx,  dx)
    vy_dx,  vy_dy,  vy_dz  = get_gradient(vy,  dx)
    vz_dx,  vz_dy,  vz_dz  = get_gradient(vz,  dx)
    P_dx,   P_dy,   P_dz   = get_gradient(P,   dx)
    rho_dx, rho_dy, rho_dz = slope_limiter(rho, dx, rho_dx, rho_dy, rho_dz)
    vx_dx,  vx_dy,  vx_dz  = slope_limiter(vx,  dx, vx_dx,  vx_dy,  vx_dz)
    vy_dx,  vy_dy,  vy_dz  = slope_limiter(vy,  dx, vy_dx,  vy_dy,  vy_dz)
    vz_dx,  vz_dy,  vz_dz  = slope_limiter(vz,  dx, vz_dx,  vz_dy,  vz_dz)
    P_dx,   P_dy,   P_dz   = slope_limiter(P,   dx, P_dx,   P_dy,   P_dz)
    eps = 1e-30
    rho_prime = rho - 0.5 * dt * (
        vx * rho_dx + rho * vx_dx +
        vy * rho_dy + rho * vy_dy +
        vz * rho_dz + rho * vz_dz
    )
    vx_prime = vx - 0.5 * dt * (vx*vx_dx + vy*vx_dy + vz*vx_dz + P_dx/(rho + eps))
    vy_prime = vy - 0.5 * dt * (vx*vy_dx + vy*vy_dy + vz*vy_dz + P_dy/(rho + eps))
    vz_prime = vz - 0.5 * dt * (vx*vz_dx + vy*vz_dy + vz*vz_dz + P_dz/(rho + eps))
    rho_XL, rho_XR, rho_YL, rho_YR, rho_ZL, rho_ZR = extrap_to_face(rho_prime, rho_dx, rho_dy, rho_dz, dx)
    vx_XL,  vx_XR,  vx_YL,  vx_YR,  vx_ZL,  vx_ZR  = extrap_to_face(vx_prime,  vx_dx,  vx_dy,  vx_dz,  dx)
    vy_XL,  vy_XR,  vy_YL,  vy_YR,  vy_ZL,  vy_ZR  = extrap_to_face(vy_prime,  vy_dx,  vy_dy,  vy_dz,  dx)
    vz_XL,  vz_XR,  vz_YL,  vz_YR,  vz_ZL,  vz_ZR  = extrap_to_face(vz_prime,  vz_dx,  vz_dy,  vz_dz,  dx)
    c2_Lx, c2_Rx = c2, jnp.roll(c2, -1, axis=0)
    c2_Ly, c2_Ry = c2, jnp.roll(c2, -1, axis=1)
    c2_Lz, c2_Rz = c2, jnp.roll(c2, -1, axis=2)
    FX_M, FX_Mx, FX_My, FX_Mz = get_flux_axis(rho_XL, vx_XL, vy_XL, vz_XL,
                                              rho_XR, vx_XR, vy_XR, vz_XR,
                                              c2_Lx, c2_Rx)
    FY_M, FY_My, FY_Mz, FY_Mx = get_flux_axis(rho_YL, vy_YL, vz_YL, vx_YL,
                                              rho_YR, vy_YR, vz_YR, vx_YR,
                                              c2_Ly, c2_Ry)
    FZ_M, FZ_Mz, FZ_Mx, FZ_My = get_flux_axis(rho_ZL, vz_ZL, vx_ZL, vy_ZL,
                                              rho_ZR, vz_ZR, vx_ZR, vy_ZR,
                                              c2_Lz, c2_Rz)
    Mass, Momx, Momy, Momz = get_conserved(rho, vx, vy, vz, vol)
    Mass = apply_fluxes(Mass, FX_M,  FY_M,  FZ_M,  dx, dt)
    Momx = apply_fluxes(Momx, FX_Mx, FY_Mx, FZ_Mx, dx, dt)
    Momy = apply_fluxes(Momy, FX_My, FY_My, FZ_My, dx, dt)
    Momz = apply_fluxes(Momz, FX_Mz, FY_Mz, FZ_Mz, dx, dt)
    return get_primitive(Mass, Momx, Momy, Momz, vol)

# ---------- Soliton/core fitting & mass helpers ----------
def soliton_rho_model(r, rho0, rc):
    # Standard FDM soliton fit: Schive+ (2014) form
    # ρ(r) = ρ0 [1 + a (r/rc)^2]^(-8), a ≈ 0.091
    a = 0.091
    return rho0 * (1.0 + a * (r / (rc + 1e-30))**2) ** (-8.0)

def _safe_log(y):
    return np.log(np.clip(y, 1e-300, None))

def fit_soliton_core(rc_grid, r_bins, prof_dm, r_max=None):
    """
    Fit (rho0, rc) by scanning rc on rc_grid and solving rho0 by log-least-squares.
    Only use radii r <= r_max to focus on the core. Returns (rho0, rc_best).
    """
    if r_max is None:
        r_max = min(0.25 * float(Lx), 3.0)  # conservative default window (kpc)
    mask = (r_bins > 0.0) & (r_bins <= r_max) & np.isfinite(prof_dm) & (prof_dm > 0.0)
    if not np.any(mask):
        return np.nan, np.nan

    r = r_bins[mask]
    y = prof_dm[mask]
    logy = _safe_log(y)

    best_err = np.inf
    best_rc = np.nan
    best_rho0 = np.nan

    for rc in rc_grid:
        y0 = soliton_rho_model(r, 1.0, rc)          # model with rho0=1
        logy0 = _safe_log(y0)
        # Optimal rho0 in log-space LS: log(rho0) = mean( logy - logy0 )
        log_rho0 = float(np.mean(logy - logy0))
        rho0 = np.exp(log_rho0)
        resid = logy - (log_rho0 + logy0)
        err = float(np.mean(resid * resid))
        if err < best_err:
            best_err = err
            best_rc = float(rc)
            best_rho0 = float(rho0)
    return best_rho0, best_rc

def mass_inside_sphere(rho3d, i0, j0, k0, dx, R):
    """
    Periodic spherical sum of mass inside radius R around (i0,j0,k0).
    rho3d is numpy array (DM density); returns mass in Msun.
    """
    n = rho3d.shape[0]
    # periodic index deltas in cell units
    ii = np.arange(n)[:, None, None]
    jj = np.arange(n)[None, :, None]
    kk = np.arange(n)[None, None, :]
    di = np.minimum(np.abs(ii - i0), n - np.abs(ii - i0))
    dj = np.minimum(np.abs(jj - j0), n - np.abs(jj - j0))
    dk = np.minimum(np.abs(kk - k0), n - np.abs(kk - k0))
    r_cell = np.sqrt(di**2 + dj**2 + dk**2)
    r_phys = r_cell * float(dx)

    mask = (r_phys <= float(R))
    m = np.sum(rho3d[mask]) * (float(dx)**3)
    return float(m)

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
    f  = xi - i0
    def mod(i): return jnp.mod(i, nx)
    w = jnp.array([
        (1-f[0])*(1-f[1])*(1-f[2]),
        f[0]*(1-f[1])*(1-f[2]),
        (1-f[0])*f[1]*(1-f[2]),
        (1-f[0])*(1-f[1])*f[2],
        f[0]*f[1]*(1-f[2]),
        f[0]*(1-f[1])*f[2],
        (1-f[0])*f[1]*f[2],
        f[0]*f[1]*f[2],
    ])
    inds = jnp.array([
        (mod(i0[0]),     mod(i0[1]),     mod(i0[2])),
        (mod(i0[0] + 1), mod(i0[1]),     mod(i0[2])),
        (mod(i0[0]),     mod(i0[1] + 1), mod(i0[2])),
        (mod(i0[0]),     mod(i0[1]),     mod(i0[2] + 1)),
        (mod(i0[0] + 1), mod(i0[1] + 1), mod(i0[2])),
        (mod(i0[0] + 1), mod(i0[1]),     mod(i0[2] + 1)),
        (mod(i0[0]),     mod(i0[1] + 1), mod(i0[2] + 1)),
        (mod(i0[0] + 1), mod(i0[1] + 1), mod(i0[2] + 1)),
    ])
    for k in range(8):
        i,j,kz = inds[k]
        rho_bh = rho_bh.at[i,j,kz].add(w[k] * M_bh / (dx**3))
    return rho_bh

def _sample_cic(field, x_bh, dx, nx):
    xi = x_bh / dx - 0.5
    i0 = jnp.floor(xi).astype(int)
    f  = xi - i0
    def mod(i): return jnp.mod(i, nx)
    ii = jnp.array([i0[0], i0[0]+1, i0[0],   i0[0],   i0[0]+1, i0[0]+1, i0[0],   i0[0]+1])
    jj = jnp.array([i0[1], i0[1],   i0[1]+1, i0[1],   i0[1]+1, i0[1],   i0[1]+1, i0[1]+1])
    kk = jnp.array([i0[2], i0[2],   i0[2],   i0[2]+1, i0[2],   i0[2]+1, i0[2]+1, i0[2]+1])
    w  = jnp.array([
        (1-f[0])*(1-f[1])*(1-f[2]),
        f[0]*(1-f[1])*(1-f[2]),
        (1-f[0])*f[1]*(1-f[2]),
        (1-f[0])*(1-f[1])*f[2],
        f[0]*f[1]*(1-f[2]),
        f[0]*(1-f[1])*f[2],
        (1-f[0])*f[1]*f[2],
        f[0]*f[1]*f[2],
    ])
    vals = field[mod(ii), mod(jj), mod(kk)]
    return jnp.sum(w * vals)

def bh_bondi_step_ambient(
    rho, M_bh_eff, x_bh, dx, Lx, cs_iso, G, dt,
    r_acc_mult=R_ACC_MULT, lam=LAMBDA_ISO, fmax=BH_FMAX,
    kappa_sink=2.0,  # (ignored in fixed-grid mode; kept for API compatibility)
    ann_lo=ann_lo, ann_hi=ann_hi,
    vx=None, vy=None, vz=None, vxbh=0.0, vybh=0.0, vzbh=0.0
):
    """
    Fixed-grid sink + fixed sampling shell (preferred baseline):
      - Sink/control radius r_acc is fixed in *cells*: r_acc = r_acc_mult * dx
      - Ambient density ρ∞ sampled in a thin shell just outside r_acc
      - Optional BHL (v_rel) support via velocity sampling in the same shell
    Returns: rho_new, dMdt_target, rho_inf, rB, dM_cap, r_acc, M_avail_kernel, dM_uncl
    """
    nx = rho.shape[0]
    xs = (jnp.arange(nx) + 0.5) * dx

    # periodic distances to BH
    DX = _periodic_delta_1d(xs, x_bh[0], Lx)
    DY = _periodic_delta_1d(xs, x_bh[1], Lx)
    DZ = _periodic_delta_1d(xs, x_bh[2], Lx)
    dX, dY, dZ = jnp.meshgrid(DX, DY, DZ, indexing="ij")
    R = jnp.sqrt(dX*dX + dY*dY + dZ*dZ)

    # Bondi radius for diagnostics only (unresolved generally)
    rB = G * M_bh_eff / (cs_iso*cs_iso + 1e-30)

    # ---- Fixed control volume (sink) ----
    r_acc = r_acc_mult * dx  # <-- FIXED in cells; no rB scaling
    # kernel width: smooth but never tiny (lets rB inform width if rB > 0.5dx)
    rK = jnp.minimum(0.5 * r_acc, jnp.maximum(0.5*dx, rB))

    mask_sink = (R <= r_acc)
    W_sink = jnp.exp(-0.5 * (R/(rK + 1e-30))**2) * mask_sink

    # ---- Fixed sampling shell (anchored to r_acc) ----
    # small clearance & thickness in *cells* so it stays grid-agnostic
    ANN_DX_CLEAR   = 0.25  # inner clearance beyond r_acc
    ANN_DX_THICK   = 0.50  # shell thickness
    r_lo = jnp.maximum(ann_lo * r_acc, r_acc + ANN_DX_CLEAR * dx)
    r_hi = jnp.maximum(ann_hi * r_acc, r_lo + ANN_DX_THICK * dx)
    shell = (R >= r_lo) & (R <= r_hi)

    rho_inf = jnp.sum(rho * shell) / (jnp.sum(shell) + 1e-30)

    # Optional BHL correction (sample gas velocity in the same shell)
    if (vx is not None) and (vy is not None) and (vz is not None):
        vx_inf = jnp.sum(vx * shell) / (jnp.sum(shell) + 1e-30)
        vy_inf = jnp.sum(vy * shell) / (jnp.sum(shell) + 1e-30)
        vz_inf = jnp.sum(vz * shell) / (jnp.sum(shell) + 1e-30)
        v_rel2 = (vxbh - vx_inf)**2 + (vybh - vy_inf)**2 + (vzbh - vz_inf)**2
        denom = (cs_iso**2 + v_rel2)**1.5
    else:
        denom = cs_iso**3

    # Analytic target (Bondi/BHL)
    dMdt_target = 4.0 * jnp.pi * lam * (G*M_bh_eff)**2 * rho_inf / (denom + 1e-30)
    dM_uncl     = dMdt_target * dt

    # Conservative per-step cap (mass conservation + optional fmax)
    WRho = W_sink * rho
    sumWRho = jnp.sum(WRho) + 1e-30
    phi = WRho / sumWRho
    M_avail_kernel = jnp.sum(rho * mask_sink) * (dx**3)

    dM_cap = jnp.minimum(dM_uncl, M_avail_kernel) if (fmax is None) \
             else jnp.minimum(dM_uncl, fmax * M_avail_kernel)

    rho_new = jnp.maximum(0.0, rho - (dM_cap/(dx**3)) * phi)

    return rho_new, dMdt_target, rho_inf, rB, dM_cap, r_acc, M_avail_kernel, dM_uncl


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
    di2 = di[:, None, None]**2
    dj2 = dj[None, :, None]**2
    dk2 = dk[None, None, :]**2
    r_over_dx = np.sqrt(di2 + dj2 + dk2)
    bin_idx = np.minimum(r_over_dx.astype(np.int32), nbins - 1)
    vals = field_np.astype(np.float64)
    wsum = np.bincount(bin_idx.ravel(), weights=vals.ravel(), minlength=nbins)
    cnts = np.bincount(bin_idx.ravel(), minlength=nbins)
    prof = np.divide(wsum, cnts, out=np.zeros_like(wsum), where=cnts > 0)
    rc = dx * (np.arange(nbins) + 0.5)
    return rc, prof

# ----------------------------
# Triaxiality (reduced-inertia) helpers
# ----------------------------
def _wrap_periodic(dx, L):
    """map displacements to (-L/2, L/2] for periodic domain."""
    return (dx + 0.5*L) % L - 0.5*L

def _ell_radius2(dx, dy, dz, q, s):
    """
    Ellipsoidal radius^2: r_ell^2 = x^2 + (y/q)^2 + (z/s)^2,
    where q=B/A, s=C/A. (A cancels; we use it only to define the cut.)
    """
    q2 = max(q*q, 1e-16)
    s2 = max(s*s, 1e-16)
    return dx*dx + (dy*dy)/q2 + (dz*dz)/s2

def _reduced_shape_tensor(dx, dy, dz, m, q, s, R_sel, eps2):
    """
    Reduced shape tensor S_ij = sum( w * x_i x_j ) / sum(w) with w = m / (r_ell^2 + eps2),
    restricted to the ellipsoid r_ell <= A, where A=R_sel (so axes are A, qA, sA).
    Returns S (3x3) and the boolean mask used.
    """
    r2 = _ell_radius2(dx, dy, dz, q, s)
    mask = r2 <= (R_sel*R_sel)
    if not np.any(mask):
        return np.full((3,3), np.nan), mask

    w = (m / (r2 + eps2))[mask]
    x = dx[mask]; y = dy[mask]; z = dz[mask]
    sw = np.sum(w) + 1e-30

    Sxx = np.sum(w * x * x) / sw
    Syy = np.sum(w * y * y) / sw
    Szz = np.sum(w * z * z) / sw
    Sxy = np.sum(w * x * y) / sw
    Sxz = np.sum(w * x * z) / sw
    Syz = np.sum(w * y * z) / sw
    S = np.array([[Sxx, Sxy, Sxz],
                  [Sxy, Syy, Syz],
                  [Sxz, Syz, Szz]], dtype=float)
    return S, mask

def measure_triax_from_density(rho3d, center_xyz, dx, Lx, rc_hint=None,
                               Rsel_mult=3.0, max_iter=40, tol=1e-4):
    """
    Measure (B/A, C/A, T, E) for the soliton around `center_xyz` using the reduced inertia tensor.
    - rho3d: DM density (numpy array)
    - center_xyz: peak (x0,y0,z0) in kpc
    - rc_hint: if provided, picks selection A=R_sel=Rsel_mult*rc_hint; otherwise falls back to ~min(2.0, 0.3*Lx)
    Returns dict with keys: q (B/A), s (C/A), T, E, A, B, C, evecs (3x3), npts.
    """
    x0, y0, z0 = [float(v) for v in center_xyz]
    # build coordinate grids once per call in NumPy space
    n = rho3d.shape[0]
    xs = (np.arange(n) + 0.5) * float(dx)
    Xg, Yg, Zg = np.meshgrid(xs, xs, xs, indexing="ij")
    dxp = _wrap_periodic(Xg - x0, Lx)
    dyp = _wrap_periodic(Yg - y0, Lx)
    dzp = _wrap_periodic(Zg - z0, Lx)

    vol = float(dx)**3
    m = rho3d.astype(np.float64) * vol

    # selection size: default to ~core scale if known
    if (rc_hint is not None) and np.isfinite(rc_hint) and (rc_hint > 0.0):
        R_sel = float(Rsel_mult) * float(rc_hint)
    else:
        R_sel = float(min(2.0, 0.30 * Lx))  # conservative fallback in kpc

    # reduced-inertia regularization in the very center
    eps2 = (0.30 * float(dx))**2

    # initialize axis ratios
    q, s = 1.0, 1.0

    # ensure we have enough cells; if too few, increase R_sel gently
    def ensure_cells_at_least(min_cells=200):
        nonlocal R_sel
        tries = 0
        while tries < 5:
            r2 = _ell_radius2(dxp, dyp, dzp, q, s)
            if np.count_nonzero(r2 <= R_sel*R_sel) >= min_cells:
                return
            R_sel = min(0.49*Lx, 1.4 * R_sel)  # expand but keep within box
            tries += 1

    ensure_cells_at_least()

    ok = True
    for _ in range(max_iter):
        S, mask = _reduced_shape_tensor(dxp, dyp, dzp, m, q, s, R_sel, eps2)
        if not np.isfinite(S).all() or np.count_nonzero(mask) < 20:
            ok = False
            break
        evals, evecs = np.linalg.eigh(S)
        order = np.argsort(evals)[::-1]   # λ1 >= λ2 >= λ3
        evals = evals[order]; evecs = evecs[:, order]
        if not np.all(np.isfinite(evals)):
            ok = False; break

        lam1, lam2, lam3 = [max(float(x), 1e-30) for x in evals]
        q_new = np.sqrt(lam2 / lam1)
        s_new = np.sqrt(lam3 / lam1)

        if max(abs(q_new - q), abs(s_new - s)) < tol:
            q, s = float(q_new), float(s_new)
            break
        q, s = float(q_new), float(s_new)
    else:
        ok = False

    if not ok or (s <= 0.0) or (q < s) is False:
        # still return something informative
        q = float(np.clip(q, 1e-6, 1.0))
        s = float(np.clip(s, 1e-6, q))

    # define A, B, C using the selection size A=R_sel
    A = float(R_sel); B = q * A; C = s * A
    # Franx+ triaxiality; guard degenerate cases
    denom = (A*A - C*C)
    T = float((A*A - B*B) / denom) if denom > 0 else np.nan
    T = float(np.clip(T, 0.0, 1.0)) if np.isfinite(T) else np.nan
    E = float(1.0 - C / max(A, 1e-30))

    # count contributing cells
    npts = int(np.count_nonzero(_ell_radius2(dxp, dyp, dzp, q, s) <= (R_sel*R_sel)))

    return {"q": q, "s": s, "T": T, "E": E, "A": A, "B": B, "C": C,
            "evecs": evecs if ok else np.eye(3), "npts": npts}

# ----------------------------
# Seed mass ramp helper
# ----------------------------
if BH_RAMP_TAU_MYR is not None and BH_RAMP_TAU_MYR > 0.0:
    BH_RAMP_TAU = (BH_RAMP_TAU_MYR / 1000.0) / T_GYR_PER_UNIT  # internal units
else:
    t_cross = 2.0 * r_soliton / sigma  # internal units (kpc/(km/s))
    BH_RAMP_TAU = BH_RAMP_FRAC_XCROSS * t_cross

BH_RAMP_TAU = float(max(BH_RAMP_TAU, 1e-6))
BH_RAMP_W   = float(max(BH_RAMP_TAU / BH_RAMP_SHARPNESS, 1e-6))
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
    vmag  = jnp.sqrt(vx*vx + vy*vy + vz*vz)
    Cmax  = jnp.max(c_eff + vmag)
    dt_h  = HYDRO_CFL * dx / (Cmax + 1e-30)
    dt_eff = jnp.minimum(jnp.array(dt), dt_h)

    # Effective BH mass at current time = seed(t) + accreted
    M_seed   = seed_mass_from_time(t, bh_placed)
    M_bh_eff = M_seed + M_bh_acc

    # Precompute (diagnostic rB) and FIXED control/softening radius for this step
    rB_now = G * M_bh_eff / (cs_const*cs_const + 1e-30) 
    r_acc_geom = R_ACC_MULT * dx
    r_acc_now  = r_acc_geom       
    r_soft_step = r_acc_now     


    # Background density for Poisson solve (DM + gas mean)
    rho_bg = jnp.mean(jnp.abs(psi)**2) + jnp.mean(rho)

    # --- BH half-kick from DM+gas only (avoid self-force), BEFORE field half-kick ---
    if BH_ON and BH_MOVE:
        active = jnp.where(bh_placed > 0.5, 1.0, 0.0)
        V_hat_dm_g_pre, _ = get_potential_bg(jnp.abs(psi)**2 + rho, rho_bg)
        # low-pass only the acceleration used to kick the BH
        axg_pre, ayg_pre, azg_pre = accel_from_Vhat(V_hat_dm_g_pre * LP)
        a_bhx_pre = _sample_cic(axg_pre, x_bh, dx, nx)
        a_bhy_pre = _sample_cic(ayg_pre, x_bh, dx, nx)
        a_bhz_pre = _sample_cic(azg_pre, x_bh, dx, nx)
        vxbh = vxbh + a_bhx_pre*(dt_eff/2.0)*active
        vybh = vybh + a_bhy_pre*(dt_eff/2.0)*active
        vzbh = vzbh + a_bhz_pre*(dt_eff/2.0)*active

    rho_src = jnp.abs(psi) ** 2 + rho

    # ---- BH gravity source with softening at r_soft = r_acc ----
    if BH_ON and BH_GRAV:
        rho_bh_full = bh_cic_density(nx, M_bh_eff, x_bh, dx, Lx)
        # Gaussian softening in k-space (sigma = r_acc_now)
        rho_bh_hat = jnp.fft.fftn(rho_bh_full)
        rho_bh_hat *= jnp.exp(-0.5 * (r_soft_step**2) * k_sq)
        rho_bh_soft = jnp.real(jnp.fft.ifftn(rho_bh_hat))
        active_bh = jnp.where((bh_placed > 0.5) & (M_bh_eff > 0.0), 1.0, 0.0)
        rho_bh = rho_bh_soft * active_bh
    else:
        rho_bh = jnp.zeros((nx, nx, nx), dtype=jnp.abs(psi).dtype)

    rho_src_with_bh = rho_src + (rho_bh if (BH_ON and BH_GRAV) else 0.0)

    # KICK 1 (gravity on ψ and gas)
    V_hat, V = get_potential_bg(rho_src_with_bh, rho_bg)
    psi = jnp.exp(-1.0j * m_per_hbar * dt_eff / 2.0 * V) * psi
    ax, ay, az = accel_from_Vhat(V_hat)
    vx, vy, vz = vx + ax * (dt_eff/2.0), vy + ay * (dt_eff/2.0), vz + az * (dt_eff/2.0)

    # DRIFT (quantum + hydro)
    psi_hat = jnp.fft.fftn(psi)
    psi_hat = jnp.exp(dt_eff * (-1.0j * k_sq / m_per_hbar / 2.0)) * psi_hat
    psi = jnp.fft.ifftn(psi_hat)
    rho, vx, vy, vz = solve_hydro(rho, vx, vy, vz, dt_eff)

    if BH_ON and BH_MOVE:
        active = jnp.where(bh_placed > 0.5, 1.0, 0.0)
        x_bh = jnp.mod(x_bh + dt_eff * jnp.array([vxbh, vybh, vzbh]) * active, Lx)

    # --- Bondi accretion (gas only), using M_bh_eff ---

        # Gate for whether the BH is placed & has mass
    active_acc = jnp.where((bh_placed > 0.5) & (M_bh_eff > 0.0), 1.0, 0.0)

        # Uniform-gas Bondi baseline (internal units)
    baseline_unif = 4.0 * jnp.pi * jnp.array(LAMBDA_ISO) * (G * M_bh_eff)**2 \
                        * jnp.array(rho_gas, dtype=jnp.float64) / (cs_const**3 + 1e-30)

    dMdt      = jnp.array(0.0)
    dMdt_real = jnp.array(0.0)
    rho_inf   = jnp.array(rho_gas, dtype=jnp.float64)
    r_B       = jnp.array(0.0)
    cap_ratio = jnp.array(0.0)
    r_acc_eff = jnp.maximum(R_ACC_MULT * dx, 2.0 * (G * M_bh_eff / (cs_const**2 + 1e-30)))

    if BH_ON and BH_ACCRETION:
        rho_try, dMdt_try, rho_inf_try, r_B_try, dM_cap, r_acc_used, M_avail, dM_uncl = bh_bondi_step_ambient(
            rho=rho, M_bh_eff=M_bh_eff, x_bh=x_bh, dx=dx, Lx=Lx, cs_iso=cs_const, G=G, dt=dt_eff,
            r_acc_mult=R_ACC_MULT, lam=LAMBDA_ISO, fmax=BH_FMAX, kappa_sink=2.0,  
            ann_lo=ann_lo, ann_hi=ann_hi, vx=vx, vy=vy, vz=vz, vxbh=vxbh, vybh=vybh, vzbh=vzbh
        )

            # Apply only when BH is active/placed
        rho       = rho * (1.0 - active_acc) + rho_try * active_acc
        dMdt      = dMdt_try * active_acc
        rho_inf   = rho_inf_try * active_acc + (1.0 - active_acc) * jnp.array(rho_gas, dtype=rho_try.dtype)
        r_B       = r_B_try * active_acc
        M_bh_acc  = M_bh_acc + dM_cap * active_acc

        dMdt_real = jnp.where(active_acc > 0.5, dM_cap / (dt_eff + 1e-30), 0.0)
        cap_ratio = jnp.where((dM_uncl > 0.0) & (active_acc > 0.5), dM_cap / (dM_uncl + 1e-30), 0.0)
        r_acc_eff = r_acc_used * active_acc + (1.0 - active_acc) * r_acc_eff

        # Boosts (computed once for both branches)
    valid = (baseline_unif > 0.0) & (active_acc > 0.5)
    alpha_target    = jnp.where(valid, dMdt      / (baseline_unif + 1e-30), jnp.nan)
    alpha_real_step = jnp.where(valid, dMdt_real / (baseline_unif + 1e-30), jnp.nan)


    # KICK 2 (gravity)
    M_bh_eff_mid = seed_mass_from_time(t, bh_placed) + M_bh_acc
    rho_src2 = jnp.abs(psi) ** 2 + rho

    if BH_ON and BH_GRAV:
        rho_bh2_full = bh_cic_density(nx, M_bh_eff_mid, x_bh, dx, Lx)
        rho_bh2_hat = jnp.fft.fftn(rho_bh2_full)
        rho_bh2_hat *= jnp.exp(-0.5 * (r_acc_eff**2) * k_sq)
        rho_bh2_soft = jnp.real(jnp.fft.ifftn(rho_bh2_hat))
        active_bh2   = jnp.where((bh_placed > 0.5) & (M_bh_eff_mid > 0.0), 1.0, 0.0)
        rho_bh2 = rho_bh2_soft * active_bh2
    else:
        rho_bh2 = jnp.zeros((nx, nx, nx), dtype=jnp.abs(psi).dtype)

    V_hat2, V2 = get_potential_bg(rho_src2 + (rho_bh2 if (BH_ON and BH_GRAV) else 0.0), rho_bg)
    psi = jnp.exp(-1.0j * m_per_hbar * dt_eff / 2.0 * V2) * psi
    ax2, ay2, az2 = accel_from_Vhat(V_hat2)
    vx, vy, vz = vx + ax2 * (dt_eff/2.0), vy + ay2 * (dt_eff/2.0), vz + az2 * (dt_eff/2.0)

    if BH_ON and BH_MOVE:
        active = jnp.where(bh_placed > 0.5, 1.0, 0.0)
        V_hat_dm_g2, _ = get_potential_bg(jnp.abs(psi)**2 + rho, rho_bg)
        # low-pass on BH force sampling (second half-kick)
        axg2, ayg2, azg2 = accel_from_Vhat(V_hat_dm_g2 * LP)
        a_bhx2 = _sample_cic(axg2, x_bh, dx, nx)
        a_bhy2 = _sample_cic(ayg2, x_bh, dx, nx)
        a_bhz2 = _sample_cic(azg2, x_bh, dx, nx)
        vxbh = vxbh + a_bhx2*(dt_eff/2.0)*active
        vybh = vybh + a_bhy2*(dt_eff/2.0)*active
        vzbh = vzbh + a_bhz2*(dt_eff/2.0)*active

    # advance time
    t = t + dt_eff

    # --- Inject site at DM peak (once), then seed mass begins ramp ---
    inject = (bh_placed < 0.5) & (t >= BH_INJECT_T)

    rho3d_now = jnp.abs(psi)**2
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

    x_bh    = jnp.where(inject, peak_xyz, x_bh)
    vxbh    = jnp.where(inject, 0.0, vxbh)
    vybh    = jnp.where(inject, 0.0, vybh)
    vzbh    = jnp.where(inject, 0.0, vzbh)
    bh_placed = jnp.where(inject, jnp.array(1.0), bh_placed)

    # Effective mass to report at end-of-step (with new t and potentially placed)
    M_bh_eff_out = seed_mass_from_time(t, bh_placed) + M_bh_acc

    return (psi, rho, vx, vy, vz, t,
            M_bh_eff_out, M_bh_acc, x_bh, vxbh, vybh, vzbh,
            dMdt, rho_inf, r_B, bh_placed,
            dMdt_real, cap_ratio, r_acc_eff,         # existing
            alpha_target, alpha_real_step)           # NEW


@jax.jit
def update(_, state):
    (psi, rho, vx, vy, vz, t,
     M_bh_eff, M_bh_acc, x_bh, vxbh, vybh, vzbh,
     dMdt, rho_inf, r_B, bh_placed,
     dMdt_real, cap_ratio, r_acc_eff,
     alpha_target, alpha_real) = compute_step(
        state["psi"], state["rho"], state["vx"], state["vy"], state["vz"], state["t"],
        state["M_bh_acc"], state["x_bh"], state["vxbh"], state["vybh"], state["vzbh"], state["bh_placed"]
    )
    state = {
        "psi": psi, "rho": rho, "vx": vx, "vy": vy, "vz": vz, "t": t,
        "M_bh": M_bh_eff, "M_bh_acc": M_bh_acc,
        "x_bh": x_bh, "vxbh": vxbh, "vybh": vybh, "vzbh": vzbh,
        "bh_dMdt": dMdt, "bh_rho_inf": rho_inf, "bh_r_B": r_B,
        "bh_placed": bh_placed,
        "bh_dMdt_real": dMdt_real, "bh_cap_ratio": cap_ratio, "bh_r_acc": r_acc_eff,
        # NEW:
        "bh_alpha_target": alpha_target, "bh_alpha_real": alpha_real,
    }
    return state


# ======================================================
# Frame extraction + video writing (no re-sim afterwards)
# ======================================================
def _projections_from_state(st, dx, nx):
    """Return dict of 2D arrays for means along axes, plus BH pixel coords."""
    psi_abs2 = np.asarray(np.abs(st["psi"])**2)
    rho_gas  = np.asarray(st["rho"])
    # means along axes
    dm_xy = psi_abs2.mean(axis=2).astype(np.float32)   # (x,y)
    dm_xz = psi_abs2.mean(axis=1).astype(np.float32)   # (x,z)
    dm_yz = psi_abs2.mean(axis=0).astype(np.float32)   # (y,z)

    g_xy  = rho_gas.mean(axis=2).astype(np.float32)
    g_xz  = rho_gas.mean(axis=1).astype(np.float32)
    g_yz  = rho_gas.mean(axis=0).astype(np.float32)

    # BH pixel coords (for overlay)
    bx, by, bz = [float(v) for v in st["x_bh"]]
    xpix = bx / float(dx) - 0.5
    ypix = by / float(dx) - 0.5
    zpix = bz / float(dx) - 0.5

    return {
        "dm_xy": dm_xy, "dm_xz": dm_xz, "dm_yz": dm_yz,
        "g_xy":  g_xy,  "g_xz":  g_xz,  "g_yz":  g_yz,
        "bh_xy": (xpix, ypix),
        "bh_xz": (xpix, zpix),
        "bh_yz": (ypix, zpix),
    }

def _append_views(cache, views):
    for k, v in views.items():
        cache.setdefault(k, []).append(v.copy() if isinstance(v, np.ndarray) else v)

def _write_video_side_by_side(frames_left, frames_right, marks_left, marks_right,
                              out_path, cmap_left, cmap_right, fps=20, dpi=110):
    """Write a single MP4 with two panels (left/right) shown simultaneously."""

    # --- BH marker helpers (NEW) ---
    def make_bh_overlay(ax, core_r=0.9, ring_r=1.25, glow_r=1.9,
                        glow_color="#7aa2ff", ring_alpha=0.95, glow_alpha=0.18, z=50):
        """Black core + white Einstein ring + soft colored glow."""
        glow = Circle((np.nan, np.nan), glow_r, fc='none', ec=glow_color,
                      lw=4, alpha=glow_alpha, zorder=z)
        ring = Circle((np.nan, np.nan), ring_r, fc='none', ec='white',
                      lw=1.2, alpha=ring_alpha, zorder=z+1)
        core = Circle((np.nan, np.nan), core_r, fc='k', ec='k', lw=0.0, zorder=z+2)
        ring.set_path_effects([pe.withStroke(linewidth=2.2, foreground='white', alpha=0.5)])
        for art in (glow, ring, core):
            ax.add_patch(art)
        return {'glow': glow, 'ring': ring, 'core': core}

    def move_bh(bh_artists, xy):
        """Position or hide the BH overlay given (x,y) in data coords."""
        if (xy is None) or np.any(np.isnan(xy)):
            for a in bh_artists.values():
                a.set_visible(False)
        else:
            for a in bh_artists.values():
                a.set_visible(True)
                a.set_center((xy[0], xy[1]))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    H, W = frames_left[0].shape
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 5.25), dpi=dpi, constrained_layout=True)
    for ax in axs:
        ax.set_aspect("equal"); ax.axis("off")

    imL = axs[0].imshow(np.log10(frames_left[0].T + 1e-30), origin="lower",
                        vmin=jnp.log10(rho_bar * frac_dm / 8.0), vmax=jnp.log10(rho_bar * frac_dm * 8.0),
                        extent=(0, W, 0, H), cmap=cmap_left)
    imR = axs[1].imshow(np.log10(frames_right[0].T + 1e-30), origin="lower",
                        vmin=jnp.log10(rho_bar * frac_gas / 2.0), vmax=jnp.log10(rho_bar * frac_gas * 2.0),
                        extent=(0, W, 0, H), cmap=cmap_right)

    baseline_nx = 128.0     # cosmetic BH size
    scale = min(W, H) / baseline_nx  

    bhL = make_bh_overlay(
        axs[0],
        core_r=0.9 * scale,
        ring_r=1.25 * scale,
        glow_r=1.9 * scale,
        glow_color="#7aa2ff"
    )
    bhR = make_bh_overlay(
        axs[1],
        core_r=0.9 * scale,
        ring_r=1.25 * scale,
        glow_r=1.9 * scale,
        glow_color="#ffb86c"
    )


    writer = animation.FFMpegWriter(fps=fps)
    with writer.saving(fig, out_path, dpi):
        for fL, fR, mL, mR in zip(frames_left, frames_right, marks_left, marks_right):
            imL.set_data(np.log10(fL.T + 1e-30))
            imR.set_data(np.log10(fR.T + 1e-30))
            move_bh(bhL, mL)
            move_bh(bhR, mR)
            writer.grab_frame()
    plt.close(fig)

# ----------------------------
# Main
# ----------------------------
def main():
    # ----- Initial conditions
    t = 0.0
    global rho_bar

    # ============ DM IC: low-k–first random phases + Gaussian k-filter ============
    rng = np.random.default_rng(17)

    # sort k-modes so the lowest-k indices come first (stable)
    ksq_np = np.asarray(k_sq)  # host array for sorting
    sid = np.argsort(ksq_np.ravel(), kind="mergesort") 

    phi = np.exp(1.0j * 2.0 * np.pi * rng.random(ksq_np.size))

    # place phases into k-grid in low-k–first order
    psi_k_phase = np.empty(ksq_np.size, dtype=np.complex128)
    psi_k_phase[sid] = phi
    psi_k_phase = psi_k_phase.reshape(ksq_np.shape)

    # apply your Gaussian k-filter (sets amplitude; phases remain random)
    kfilter = jnp.exp(-k_sq / (2.0 * sigma**2 * m_per_hbar**2))
    psi_k = jnp.asarray(psi_k_phase) * jnp.sqrt(kfilter)

    # go to real space and normalize to unit magnitude (pure phase field)
    psi_phase = np.fft.ifftn(np.asarray(psi_k))
    psi_phase = jnp.array(psi_phase)
    psi_phase = psi_phase / (jnp.abs(psi_phase) + 1e-30)

    N_BLOBS     = 12
    SIGMA_BLOB  = 0.12         # kpc
    DELTA_PEAK  = 4.0          # stronger central overdensity
    rng = np.random.default_rng(2024)

    off_std = 0.1              # kpc, centers packed closer to middle
    cx = 0.5*Lx + rng.normal(0.0, off_std, N_BLOBS)
    cy = 0.5*Lx + rng.normal(0.0, off_std, N_BLOBS)
    cz = 0.5*Lx + rng.normal(0.0, off_std, N_BLOBS)

    def dper(a, b):
        d = jnp.abs(a - b)
        return jnp.minimum(d, Lx - d)

    rho_blob = jnp.zeros_like(X)
    for k in range(N_BLOBS):
        dxk = dper(X, cx[k]); dyk = dper(Y, cy[k]); dzk = dper(Z, cz[k])
        r2k = dxk*dxk + dyk*dyk + dzk*dzk
        rho_blob = rho_blob + jnp.exp(-0.5 * r2k / (SIGMA_BLOB**2))

    rho_blob = rho_blob / (jnp.max(rho_blob) + 1e-30)
    overdens = 1.0 + DELTA_PEAK * rho_blob

    rho_dm_mean = frac_dm * rho_bar
    rho_dm = rho_dm_mean * overdens
    rho_dm *= (rho_dm_mean / (jnp.mean(rho_dm) + 1e-30))

    psi = jnp.sqrt(rho_dm) * psi_phase
    psi *= jnp.sqrt(1.0 + 0.10 * (rng.random(psi.shape) - 0.5))

    rho_bar = jnp.mean(jnp.abs(psi) ** 2, axis=(0, 1, 2))

    # ------------------- gas init -------------------
    rho = jnp.ones((nx, nx, nx)) * rho_gas
    vx = jnp.zeros((nx, nx, nx))
    vy = jnp.zeros((nx, nx, nx))
    vz = jnp.zeros((nx, nx, nx))

    # --- BH state: no mass yet; inject site later; mass ramps smoothly after ---
    x_bh  = jnp.array([0.5 * Lx, 0.5 * Lx, 0.5 * Lx])
    M_bh_acc = jnp.array(0.0)      # accreted component only
    vxbh  = jnp.array(0.0)
    vybh  = jnp.array(0.0)
    vzbh  = jnp.array(0.0)
    bh_placed = jnp.array(0.0)     # 0 until site placed at t >= BH_INJECT_T

    state = {
        "t": t, "psi": psi, "rho": rho, "vx": vx, "vy": vy, "vz": vz,
        "x_bh": x_bh, "M_bh": jnp.array(0.0), "M_bh_acc": M_bh_acc,
        "vxbh": vxbh, "vybh": vybh, "vzbh": vzbh,
        "bh_dMdt": jnp.array(0.0), "bh_rho_inf": jnp.array(0.0),
        "bh_r_B": jnp.array(0.0),
        "bh_placed": bh_placed,
    }

    # ---------- HISTORY (for diagnostics) ----------
    history = {
    "time": [], "M_bh": [], "dMdt": [], "rho_inf": [], "r_B": [],
    "rc": None, "prof_dm": [], "prof_gas": [],
    "d_bh_peak": [],
    "vxbh": [], "vybh": [], "vzbh": [], "vbh": [],
    "M_box_dm": [], "M_box_gas": [], "M_box_tot": [],
    "M_soliton": [], "rc_fit": [], "rho0_fit": [],
    "dMdt_real": [], "cap_ratio": [], "r_acc": [],
    "triax_q": [], "triax_s": [], "triax_T": [], "triax_E": [], "triax_npts": [], 
    "mach_rc": None, "mach_profile_last": None,     "rho_center_cell": [], 
    "rho_center_shell": [],  "mach_at_rc": [], "alpha_target": [], "alpha_real": [],

    }

    # =========================
    # WARM-UP to injection time
    # =========================
    while float(state["t"]) < BH_INJECT_T:
        state = update(0, state)

    # Recenter to DM peak after warm-up (optional)
    rho3d = np.asarray(np.abs(state["psi"])**2)
    flat  = int(np.argmax(rho3d))
    i0c, j0c, k0c = np.unravel_index(flat, rho3d.shape)
    ic = nx // 2; jc = nx // 2; kc = nx // 2
    def _roll3(a, sx, sy, sz): return np.roll(np.roll(np.roll(a, sx, axis=0), sy, axis=1), sz, axis=2)
    def _best_shift(i, ic, n): return int((((ic - i) + n//2) % n) - n//2)
    sx = _best_shift(i0c, ic, nx); sy = _best_shift(j0c, jc, nx); sz = _best_shift(k0c, kc, nx)
    psi_np = _roll3(np.asarray(state["psi"]), sx, sy, sz)
    rho_np = _roll3(np.asarray(state["rho"]), sx, sy, sz)
    vx_np  = _roll3(np.asarray(state["vx"]),  sx, sy, sz)
    vy_np  = _roll3(np.asarray(state["vy"]),  sx, sy, sz)
    vz_np  = _roll3(np.asarray(state["vz"]),  sx, sy, sz)
    state["psi"] = jnp.array(psi_np)
    state["rho"] = jnp.array(rho_np)
    state["vx"]  = jnp.array(vx_np)
    state["vy"]  = jnp.array(vy_np)
    state["vz"]  = jnp.array(vz_np)
    state["x_bh"] = jnp.mod(state["x_bh"] + jnp.array([sx*dx, sy*dx, sz*dx], dtype=jnp.float64), Lx)

    # =========================
    # BUILD post-injection grid
    # =========================
    t_start = float(state["t"])  # ~ BH_INJECT_T
    t_stop  = t_start + FOCUS_POST_WINDOW_GYR / T_GYR_PER_UNIT
    frames  = 300                # snapshots
    fps     = 20
    t_targets = np.linspace(t_start, t_stop, frames + 1)[1:]  # per-frame target times

    # --------- SIMULATE ONCE; CACHE VIEWS ----------
    VIEWS = {}  

    history["time"].append(float(state["t"]))
    history["M_bh"].append(float(state["M_bh"]))
    history["dMdt"].append(float(state["bh_dMdt"]))
    history["rho_inf"].append(float(state["bh_rho_inf"]))
    history["r_B"].append(float(state["bh_r_B"]))
    history["d_bh_peak"].append(np.nan)
    history["vxbh"].append(float(state["vxbh"]))
    history["vybh"].append(float(state["vybh"]))
    history["vzbh"].append(float(state["vzbh"]))
    history["vbh"].append(float(np.sqrt(state["vxbh"]**2 + state["vybh"]**2 + state["vzbh"]**2)))
    history["dMdt_real"].append(float(state.get("bh_dMdt_real", 0.0)))
    history["cap_ratio"].append(float(state.get("bh_cap_ratio", 0.0)))
    history["r_acc"].append(float(state.get("bh_r_acc", R_ACC_MULT*dx)))
    history["triax_q"].append(np.nan)
    history["triax_s"].append(np.nan)
    history["triax_T"].append(np.nan)
    history["triax_E"].append(np.nan)
    history["triax_npts"].append(0)
    history["alpha_target"].append(np.nan)
    history["alpha_real"].append(np.nan)


    for target_t in t_targets:
        # advance until reaching target time
        st = state
        while float(st["t"]) < target_t:
            st = update(0, st)

        # cache projections and BH pixel positions
        views = _projections_from_state(st, dx=float(dx), nx=nx)
        _append_views(VIEWS, views)

        # --- diagnostics (profiles & distances) ---
        rho3d_now = np.asarray(np.abs(st["psi"])**2)
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
        rc, p_dm  = radial_profile_centered(rho3d_now, i0, j0, k0, float(dx))
        _,  p_gas = radial_profile_centered(gas3d_now, i0, j0, k0, float(dx))
        if history["rc"] is None:
            history["rc"] = rc
        history["prof_dm"].append(p_dm); history["prof_gas"].append(p_gas)

        rho_c_cell  = float(rho3d_now[i0, j0, k0])
        rho_c_shell = float(p_dm[0]) if p_dm.size else np.nan
        history["rho_center_cell"].append(rho_c_cell)
        history["rho_center_shell"].append(rho_c_shell)


        # --- Mach number radial profile (gas): M(r) = |v|/c_s, keep latest only ---
        vx_now = np.asarray(st["vx"]); vy_now = np.asarray(st["vy"]); vz_now = np.asarray(st["vz"])
        vmag_now = np.sqrt(vx_now**2 + vy_now**2 + vz_now**2)
        rc_m, vmag_prof = radial_profile_centered(vmag_now, i0, j0, k0, float(dx))
        mach_prof = vmag_prof / float(cs_const)
        history["mach_rc"] = rc_m
        history["mach_profile_last"] = mach_prof

        # BH–peak 3D separation (periodic)
        def per_sep(a, b, L):
            d = abs(float(a) - float(b)); return min(d, L - d)
        peak_xyz = np.array([(i0+0.5)*float(dx), (j0+0.5)*float(dx), (k0+0.5)*float(dx)], dtype=float)
        if float(st["bh_placed"]) > 0.5:
            bx, by, bz = [float(v) for v in st["x_bh"]]
            d = np.sqrt(per_sep(bx, peak_xyz[0], Lx)**2 +
                        per_sep(by, peak_xyz[1], Lx)**2 +
                        per_sep(bz, peak_xyz[2], Lx)**2)
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
        history["vxbh"].append(float(st["vxbh"]))
        history["vybh"].append(float(st["vybh"]))
        history["vzbh"].append(float(st["vzbh"]))
        history["vbh"].append(float(np.sqrt(st["vxbh"]**2 + st["vybh"]**2 + st["vzbh"]**2)))
        M_dm_box  = float(np.sum(rho3d_now) * (float(dx)**3))
        M_gas_box = float(np.sum(gas3d_now) * (float(dx)**3))
        M_box_tot = M_dm_box + M_gas_box + float(st["M_bh"])

        history["M_box_dm"].append(M_dm_box)
        history["M_box_gas"].append(M_gas_box)
        history["M_box_tot"].append(M_box_tot)

        # --- fit soliton core and compute M_soliton (within 3 r_c) ---
        try:
            rc_grid = np.geomspace(0.2 * float(dx), 0.5 * float(Lx), 64)
            rho0_fit, rc_fit = fit_soliton_core(rc_grid, rc, p_dm, r_max=None)
            history["rc_fit"].append(float(rc_fit))
            history["rho0_fit"].append(float(rho0_fit))

            if np.isfinite(rc_fit) and rc_fit > 0.0 and np.isfinite(rho0_fit) and rho0_fit > 0.0:
                rmax = 3.0 * rc_fit
                r_int = np.linspace(0.0, rmax, 512)
                rho_int = soliton_rho_model(r_int, rho0_fit, rc_fit)
                M_sol = float(4.0 * np.pi * np.trapz(rho_int * r_int**2, r_int))
            else:
                M_sol = np.nan
        except Exception:
            rc_fit, rho0_fit, M_sol = np.nan, np.nan, np.nan
            history["rc_fit"].append(rc_fit)
            history["rho0_fit"].append(rho0_fit)

        history["M_soliton"].append(M_sol)

        try:
            if (np.isfinite(rc_fit) and rc_fit > 0.0 and mach_prof.size and rc_m.size):
                mach_at_rc_val = float(np.interp(rc_fit, rc_m, mach_prof))
            else:
                mach_at_rc_val = np.nan
        except Exception:
            mach_at_rc_val = np.nan
        history["mach_at_rc"].append(mach_at_rc_val)


        # --- NEW: store realized accretion diagnostics from state ---
        history["dMdt_real"].append(float(st.get("bh_dMdt_real", 0.0)))
        history["cap_ratio"].append(float(st.get("bh_cap_ratio", 0.0)))
        history["r_acc"].append(float(st.get("bh_r_acc", R_ACC_MULT*dx)))

        history["alpha_target"].append(float(st.get("bh_alpha_target", np.nan)))
        history["alpha_real"].append(float(st.get("bh_alpha_real", np.nan)))

                # --- Triaxiality of the DM soliton (reduced inertia) ---
        try:
            center_xyz = peak_xyz  # use DM peak as center
            rc_hint = rc_fit if (np.isfinite(rc_fit) and rc_fit > 0.0) else None
            tri = measure_triax_from_density(
                rho3d=rho3d_now, center_xyz=center_xyz,
                dx=float(dx), Lx=float(Lx),
                rc_hint=rc_hint, Rsel_mult=3.0,  # measure inside ~3 rc
                max_iter=40, tol=1e-4
            )
            history["triax_q"].append(float(tri["q"]))
            history["triax_s"].append(float(tri["s"]))
            history["triax_T"].append(float(tri["T"]))
            history["triax_E"].append(float(tri["E"]))
            history["triax_npts"].append(int(tri["npts"]))
        except Exception:
            history["triax_q"].append(np.nan)
            history["triax_s"].append(np.nan)
            history["triax_T"].append(np.nan)
            history["triax_E"].append(np.nan)
            history["triax_npts"].append(0)

    return history, VIEWS, fps

def _stack_list(arr_list, dtype=None):
    """Safely stack a list of arrays into a single ndarray."""
    if not arr_list:
        return None
    a = [np.asarray(a) for a in arr_list]
    try:
        a = np.stack(a, axis=0)
    except Exception:
        a = np.array(a, dtype=object)
    if dtype is not None:
        a = a.astype(dtype, copy=False)
    return a

def save_run_to_hdf5(h5_path, H, V, fps):
    """
    Save everything needed to recreate your plots + the XY projections for the movie.

    Contents:
      /params/*            → model/domain/BH/gas/DM settings
      /history/*           → time series (M_bh, dMdt, rho_inf, r_B, sep, v_bh, etc.)
      /profiles/rc         → radial bins (peak-centered)
      /profiles/prof_dm    → DM radial profiles [nframes, nbins]
      /profiles/prof_gas   → Gas radial profiles [nframes, nbins]
      /mach/r              → r-array used for the Mach profile (final frame)
      /mach/profile_last   → Mach(r) for the final frame
      /views_xy/dm         → <ρ_DM>_z XY frames [nframes, nx, nx]
      /views_xy/gas        → <ρ_gas>_z XY frames [nframes, nx, nx]
      /views_xy/bh_pixels  → BH marker pixels per frame [nframes, 2]
      attrs: fps, created
    """
    os.makedirs(os.path.dirname(h5_path), exist_ok=True)

    with h5py.File(h5_path, "w") as f:
        # ---------------- Params (as attributes)
        gpar = f.create_group("params")
        # domain / numerics
        gpar.attrs["nx"]                = int(nx)
        gpar.attrs["Lx_kpc"]            = float(Lx)
        gpar.attrs["dx_kpc"]            = float(dx)
        gpar.attrs["t_end_units"]       = float(t_end)           # kpc/(km/s)
        gpar.attrs["T_GYR_PER_UNIT"]    = float(T_GYR_PER_UNIT)
        gpar.attrs["K_CUT_FRAC"]        = float(K_CUT_FRAC)
        gpar.attrs["INJECT_SMOOTH_ON"]  = bool(INJECT_SMOOTH_ON)
        gpar.attrs["INJECT_SMOOTH_KPC"] = float(INJECT_SMOOTH_KPC)

        # gas / DM
        gpar.attrs["rho_bar"]   = float(rho_bar)
        gpar.attrs["frac_gas"]  = float(frac_gas)
        gpar.attrs["frac_dm"]   = float(frac_dm)
        gpar.attrs["rho_gas"]   = float(rho_gas)
        gpar.attrs["cs_const"]  = float(cs_const)
        gpar.attrs["sigma"]     = float(sigma)
        gpar.attrs["m_22"]      = float(m_22)
        gpar.attrs["r_soliton"] = float(r_soliton)

        # BH toggles/params
        gpar.attrs["BH_ON"]          = bool(BH_ON)
        gpar.attrs["BH_GRAV"]        = bool(BH_GRAV)
        gpar.attrs["BH_MOVE"]        = bool(BH_MOVE)
        gpar.attrs["BH_ACCRETION"]   = bool(BH_ACCRETION)
        gpar.attrs["BH_INIT_M"]      = float(BH_INIT_M)
        gpar.attrs["BH_INJECT_T"]    = float(BH_INJECT_T)
        gpar.attrs["BH_RAMP_ON"]     = bool(BH_RAMP_ON)
        gpar.attrs["BH_RAMP_TAU_MYR"]= float(BH_RAMP_TAU_MYR)
        gpar.attrs["BH_RAMP_FRAC_XCROSS"] = float(BH_RAMP_FRAC_XCROSS)
        gpar.attrs["BH_RAMP_SHARPNESS"]   = float(BH_RAMP_SHARPNESS)
        gpar.attrs["R_ACC_MULT_cells"]    = float(R_ACC_MULT)
        gpar.attrs["LAMBDA_ISO"]          = float(LAMBDA_ISO)
        gpar.attrs["BH_FMAX"]             = -1.0 if (BH_FMAX is None) else float(BH_FMAX)
        gpar.attrs["ANN_LO"]              = float(ann_lo)
        gpar.attrs["ANN_HI"]              = float(ann_hi)

        # ---------------- History (time series)
        gh = f.create_group("history")
        def _d(name, data):
            if data is None:
                return
            arr = np.asarray(data)
            gh.create_dataset(name, data=arr, compression="gzip", compression_opts=4)

        _d("time",          H.get("time"))             # internal units
        _d("M_bh",          H.get("M_bh"))
        _d("dMdt",          H.get("dMdt"))             # Msun / (kpc/(km/s))
        _d("rho_inf",       H.get("rho_inf"))
        _d("r_B",           H.get("r_B"))
        _d("d_bh_peak",     H.get("d_bh_peak"))
        _d("vxbh",          H.get("vxbh"))
        _d("vybh",          H.get("vybh"))
        _d("vzbh",          H.get("vzbh"))
        _d("vbh",           H.get("vbh"))
        _d("M_box_dm",      H.get("M_box_dm"))
        _d("M_box_gas",     H.get("M_box_gas"))
        _d("M_box_tot",     H.get("M_box_tot"))
        _d("M_soliton",     H.get("M_soliton"))
        _d("rc_fit",        H.get("rc_fit"))
        _d("rho0_fit",      H.get("rho0_fit"))
        _d("dMdt_real",     H.get("dMdt_real"))
        _d("cap_ratio",     H.get("cap_ratio"))
        _d("r_acc",         H.get("r_acc"))
        _d("triax_q",       H.get("triax_q"))
        _d("triax_s",       H.get("triax_s"))
        _d("triax_T",       H.get("triax_T"))
        _d("triax_E",       H.get("triax_E"))
        _d("triax_npts",    H.get("triax_npts"))
        _d("rho_center_cell",  H.get("rho_center_cell"))
        _d("rho_center_shell", H.get("rho_center_shell"))
        _d("mach_at_rc",       H.get("mach_at_rc"))
        _d("alpha_target",  H.get("alpha_target"))
        _d("alpha_real",    H.get("alpha_real"))

        # ---------------- Profiles
        gp = f.create_group("profiles")
        rc = H.get("rc")
        if rc is not None:
            gp.create_dataset("rc", data=np.asarray(rc), compression="gzip", compression_opts=4)

        prof_dm  = _stack_list(H.get("prof_dm", []), dtype=np.float32)
        prof_gas = _stack_list(H.get("prof_gas", []), dtype=np.float32)
        if prof_dm is not None:
            gp.create_dataset("prof_dm", data=prof_dm, compression="gzip", compression_opts=4)
        if prof_gas is not None:
            gp.create_dataset("prof_gas", data=prof_gas, compression="gzip", compression_opts=4)

        # ---------------- Mach profile (final frame)
        gm = f.create_group("mach")
        mach_r   = np.asarray(H.get("mach_rc", []))
        mach_prof= np.asarray(H.get("mach_profile_last", []))
        if mach_r.size:
            gm.create_dataset("r", data=mach_r, compression="gzip", compression_opts=4)
        if mach_prof.size:
            gm.create_dataset("profile_last", data=mach_prof, compression="gzip", compression_opts=4)

        # ---------------- XY movie projections
        gv = f.create_group("views_xy")
        dm_xy  = _stack_list(V.get("dm_xy", []), dtype=np.float32)   # [nframes, nx, nx]
        gas_xy = _stack_list(V.get("g_xy",  []), dtype=np.float32)
        bh_xy  = np.asarray(V.get("bh_xy", []), dtype=np.float32)    # [nframes, 2]

        if dm_xy is not None:
            gv.create_dataset("dm",  data=dm_xy,  compression="gzip", compression_opts=4)
        if gas_xy is not None:
            gv.create_dataset("gas", data=gas_xy, compression="gzip", compression_opts=4)
        if bh_xy.size:
            gv.create_dataset("bh_pixels", data=bh_xy, compression="gzip", compression_opts=4)

        # some handy attrs
        f.attrs["fps"]     = int(fps)
        f.attrs["created"] = time.strftime("%Y-%m-%d %H:%M:%S")


# ----------------------------
# Run + write outputs  (HDF5 only; no plots/videos here)
# ----------------------------
if __name__ == "__main__":
    H, V, fps = main()

    def _run_tag(cs_kms, M_seed):
        cs_tag = f"cs{int(round(float(cs_kms)))}"

        # BH exponent: BH6 for 1e6, BH7 for 1e7, etc.
        M_seed = float(M_seed)
        if M_seed > 0:
            exp = int(np.floor(np.log10(M_seed)))
            if np.isclose(M_seed, 10.0**exp, rtol=0, atol=0):
                bh_tag = f"BH{exp}"
            else:
                # Fallback for non–power-of-ten seeds (keeps things filesystem-safe)
                mant = M_seed / (10.0**exp)           # e.g., 3.2
                bh_tag = f"BH{mant:.2f}e{exp}".replace(".", "p")
        else:
            bh_tag = "BH0"

        # Final tag: cs70BH6, cs60BH8, etc.
        return f"{cs_tag}{bh_tag}"

    tag = _run_tag(cs_const, BH_INIT_M)

    out_dir = "checkpoint_rawdata"
    os.makedirs(out_dir, exist_ok=True)

    h5_path = os.path.join(out_dir, f"{tag}.h5")
    # Avoid accidental overwrite: append a timestamp if file exists
    if os.path.exists(h5_path):
        stamp = time.strftime("%Y%m%d_%H%M%S")
        h5_path = os.path.join(out_dir, f"{tag}_{stamp}.h5")

    save_run_to_hdf5(h5_path, H, V, fps)
    print(f"[ok] wrote {h5_path}  (contains params, history, profiles, and XY projections)")

