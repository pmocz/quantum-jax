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
A simple Schrodinger-Poisson + Black Hole (sink particle) solver written in JAX
to simulate fuzzy dark matter + black hole

Philip Mocz (2025), @pmocz
Flatiron Institute

Simulate the Schrodinger-Poisson system with the spectral method described in:

Mocz, P., et al. (2017)
Galaxy Formation with BECDM: I. Turbulence and relaxation of idealised haloes
Monthly Notices of the Royal Astronomical Society, 471(4), 4559-4570
https://doi.org/10.1093/mnras/stx1887
https://arxiv.org/abs/1705.05845

For BH model, see:

Davies, E. & Mocz, P. (2020)
Fuzzy Dark Matter Soliton Cores around Supermassive Black Holes
Monthly Notices of the Royal Astronomical Society, 492(4), 5721-5729
https://academic.oup.com/mnras/article/492/4/5721/5714762?login=true
https://arxiv.org/pdf/1908.04790


Example Usage:

python quantum-bh.py --res 1

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
parser.add_argument(
    "--soliton", action="store_true", help="Run with soliton initial conditions"
)
args = parser.parse_args()

# Enable for double precision
# jax.config.update("jax_enable_x64", True)

# resolution
nx = 32 * args.res

# box dimensions (in units of kpc)
Lx = 1.0

# average density of dark matter in the simulation (in units of Msun / kpc^3)
rho_bar = 1.0e8

# stop time (in units of kpc / (km/s) = 0.9778 Gyr)
t_end = 10.0

# axion mass (in units of 10^-22 eV)
m_22 = 1.0

# black hole
M_bh = 1.0e8  # mass of black hole (in Msun)

# dark matter
sigma = 100.0  # velocity dispersion of dm (in km/s)

# soliton (optional)
use_soliton_ics = False
if args.soliton:
    use_soliton_ics = True
M_soliton = 3.0e9


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

# check that de broglie wavelength fits into box
de_broglie_wavelength = hbar / (m * sigma)
n_wavelengths = Lx / de_broglie_wavelength
assert n_wavelengths > 3, f"{n_wavelengths}"

# check the Schwarzschild radius, bondi radius, jeans length
r_s = 2.0 * G * M_bh / c**2  # in kpc
r_bondi_est = G * M_bh / (sigma**2)
jeans_length = sigma * jnp.sqrt(1.0 / (G * rho_bar))
n_jeans = Lx / jeans_length
assert n_jeans < 0.8, f"{n_jeans}"

r_soliton = 2.2e8 * m_22**-2 / M_soliton  # in kpc
assert r_soliton < 0.5 * Lx
v_vir = G * M_soliton * m_per_hbar * jnp.sqrt(0.10851)

# print some info
if use_soliton_ics:
    print(f"M_bh/M_soliton: {M_bh / M_soliton:.2f}")
    print(f"# r_soliton in box: {Lx / r_soliton:.2f}")
else:
    print(f"# de Broglie wavelengths in box: {n_wavelengths:.2f}")
    print(f"# Jeans lengths in box: {n_jeans:.2f}")
    print(f"# r_s in box: {Lx / r_s:.2f}")
    print(f"# r_bondi_est in box: {Lx / r_bondi_est:.2f}")
    print(f"M_bh/M_dm: {M_bh / (Lx * Lx * Lx * rho_bar):.2f}")
    print(f"<rho>/rho_crit: {rho_bar / rho_crit:.2f}")


######
# Mesh

# Domain [0,Lx] x [0,Lx] x [0,Lx]
dx = Lx / nx
vol = dx * dx * dx  # volume of each cell
x_lin = jnp.linspace(0.5 * dx, Lx - 0.5 * dx, nx)
X, Y, Z = jnp.meshgrid(x_lin, x_lin, x_lin, indexing="ij")

# checks
v_resolved = (hbar / m) * jnp.pi / dx
assert v_resolved > sigma
assert r_soliton > 2.0 * dx
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


# check that estimated bondi radius fits in box
assert r_bondi_est < 0.5 * Lx, f"{r_bondi_est} {Lx}"

##############
# Checkpointer
options = ocp.CheckpointManagerOptions()
checkpoint_dir = os.path.join(os.getcwd(), f"checkpoints_bh{args.res}")
path = ocp.test_utils.erase_and_create_empty(checkpoint_dir)
async_checkpoint_manager = ocp.CheckpointManager(path, options=options)


############
# Parameters
params = {}
params["Lx"] = Lx
params["nx"] = nx
params["m_22"] = m_22
params["rho_bar"] = rho_bar
params["sigma"] = sigma
params["M_bh"] = M_bh
params["t_end"] = t_end
params["use_soliton_ics"] = use_soliton_ics
params["M_soliton"] = M_soliton


#########
# Gravity


def get_potential(rho):
    """Solve the Poisson equation."""
    V_hat = -jnp.fft.fftn(4.0 * jnp.pi * G * (rho - rho_bar)) / (k_sq + (k_sq == 0))
    V = jnp.real(jnp.fft.ifftn(V_hat))
    return V


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

    v = jnp.sqrt(sigma**2 + vel[0, 0] ** 2 + vel[0, 1] ** 2 + vel[0, 2] ** 2)
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


def compute_step(psi, m_bh, pos, vel, t):
    """Compute the next step in the simulation."""

    # (1/2) kick
    rho_bh = bin_particles(pos, m_bh)
    rho_tot = jnp.abs(psi) ** 2 + rho_bh
    V = get_potential(rho_tot)
    psi = jnp.exp(-1.0j * m_per_hbar * dt / 2.0 * V) * psi

    acc = get_acceleration(pos, rho_tot)
    vel = vel + acc * dt / 2.0

    # drift
    psi_hat = jnp.fft.fftn(psi)
    psi_hat = jnp.exp(dt * (-1.0j * k_sq / m_per_hbar / 2.0)) * psi_hat
    psi = jnp.fft.ifftn(psi_hat)

    pos = pos + vel * dt
    pos = jnp.mod(pos, jnp.array([Lx, Lx, Lx]))

    # (1/2) kick
    rho_bh = bin_particles(pos, m_bh)
    rho_tot = jnp.abs(psi) ** 2 + rho_bh
    V = get_potential(rho_tot)
    psi = jnp.exp(-1.0j * m_per_hbar * dt / 2.0 * V) * psi

    acc = get_acceleration(pos, rho_tot)
    vel = vel + acc * dt / 2.0

    # accretion
    psi, m_bh = do_accretion(psi, pos, vel, m_bh)

    # update time
    t += dt

    return psi, m_bh, pos, vel, t


@jax.jit
def update(_, state):
    """Update the state of the system by one time step."""
    (
        state["psi"],
        state["m_bh"],
        state["pos"],
        state["vel"],
        state["t"],
    ) = compute_step(
        state["psi"],
        state["m_bh"],
        state["pos"],
        state["vel"],
        state["t"],
    )
    return state


def plot_sim(state):
    """Plot the simulation state."""
    # DM projection
    rho_proj_dm = jnp.log10(jnp.mean(jnp.abs(state["psi"]) ** 2, axis=2)).T
    vmin = jnp.log10(rho_bar / 10.0)
    vmax = jnp.log10(rho_bar * 10.0)
    plt.imshow(
        rho_proj_dm,
        cmap="plasma",
        origin="lower",
        vmin=vmin,
        vmax=vmax,
        extent=(0, nx, 0, nx),
    )
    # add bh
    sx = jax.lax.slice(state["pos"], (0, 0), (n_bh, 1)) / Lx * nx
    sy = jax.lax.slice(state["pos"], (0, 1), (n_bh, 2)) / Lx * nx
    plt.plot(sx, sy, color="black", marker=".", linestyle="None", markersize=10)
    plt.colorbar(label="log10(rho_dm)")
    ax = plt.gca()
    ax.set_aspect("equal")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.tight_layout()


def main():
    """Main physics simulation."""

    # Initial Conditions
    t = 0.0

    # dark matter
    if use_soliton_ics:
        # soliton initial condition
        r = jnp.sqrt((X - 0.5 * Lx) ** 2 + (Y - 0.5 * Lx) ** 2 + (Z - 0.5 * Lx) ** 2)
        psi = (
            jnp.sqrt(
                1.9e7
                * m_22**-2
                * r_soliton**-4
                / (1.0 + 0.091 * (r / r_soliton) ** 2) ** 8
            )
            + 0.0j
        )
        # re-calculate rho_bar
        global rho_bar
        rho_bar = jnp.mean(jnp.abs(psi) ** 2, axis=(0, 1, 2))
    else:
        # turbulent initial condition
        # we initialize a random field with a gaussian power spectrum
        # construct in fourier space according to Eq (27) of our paper [https://arxiv.org/abs/1801.03507]
        np.random.seed(17)
        # initialize random phases
        psi = np.exp(1.0j * 2.0 * np.pi * np.random.rand(*k_sq.shape))
        psi = jnp.array(psi)
        psi *= np.sqrt(np.exp(-k_sq / (2.0 * sigma**2 * m_per_hbar**2)))
        psi = np.fft.ifftn(psi)
        # re-normalize it
        psi *= jnp.sqrt(rho_bar / jnp.mean(jnp.abs(psi) ** 2))

    # black hole
    m_bh = M_bh * jnp.ones(n_bh)  # mass of each black hole
    pos = 0.5 * Lx * jnp.ones((n_bh, 3))
    vel = jnp.zeros((n_bh, 3))

    # Construct initial simulation state
    state = {}
    state["t"] = t
    state["psi"] = psi
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
