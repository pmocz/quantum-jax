import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

"""
A differentiable Schrodinger-Poisson solver written in JAX

Philip Mocz, @pmocz
Flatiron Institute
2025

Simulate the Schrodinger-Poisson system with the spectral method based on:

TODO: cite Mocz paper

"""


def main():
    """Quantum simulation"""

    # Simulation parameters
    nx = 512
    ny = 512
    nz = 1
    t = 0
    t_end = 0.03
    dt = 0.0001
    t_out = 0.0001
    G = 4000
    plot_real_time = True

    # Domain [0,1] x [0,1]
    Lx = 1.0
    Ly = 1.0
    Lz = 1.0
    dx = Lx / nx
    dy = Ly / ny
    dz = Lz / nz
    xlin = jnp.linspace(0.5 * dx, Lx - 0.5 * dx, nx)
    ylin = jnp.linspace(0.5 * dy, Ly - 0.5 * dy, ny)
    zlin = jnp.linspace(0.5 * dz, Lz - 0.5 * dz, nz)
    X, Y = jnp.meshgrid(xlin, ylin, indexing="ij")

    # Intial Condition
    amp = 0.01
    sigma = 0.03
    rho = 0.9
    rho += (
        2.0
        * amp
        * jnp.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / 2.0 / sigma**2)
        / (sigma**3 * jnp.sqrt(2.0 * jnp.pi) ** 2)
    )
    rho += (
        1.5
        * amp
        * jnp.exp(-((X - 0.2) ** 2 + (Y - 0.7) ** 2) / 2.0 / sigma**2)
        / (sigma**3 * jnp.sqrt(2.0 * jnp.pi) ** 2)
    )
    rho += (
        amp
        * jnp.exp(-((X - 0.4) ** 2 + (Y - 0.6) ** 2) / 2.0 / sigma**2)
        / (sigma**3 * jnp.sqrt(2.0 * jnp.pi) ** 2)
    )
    rho += (
        amp
        * jnp.exp(-((X - 0.6) ** 2 + (Y - 0.8) ** 2) / 2.0 / sigma**2)
        / (sigma**3 * jnp.sqrt(2.0 * jnp.pi) ** 2)
    )
    rho += (
        amp
        * jnp.exp(-((X - 0.8) ** 2 + (Y - 0.2) ** 2) / 2.0 / sigma**2)
        / (sigma**3 * jnp.sqrt(2.0 * jnp.pi) ** 2)
    )
    rho += (
        amp
        * jnp.exp(-((X - 0.6) ** 2 + (Y - 0.7) ** 2) / 2.0 / sigma**2)
        / (sigma**3 * jnp.sqrt(2.0 * jnp.pi) ** 2)
    )
    rho += (
        amp
        * jnp.exp(-((X - 0.7) ** 2 + (Y - 0.4) ** 2) / 2.0 / sigma**2)
        / (sigma**3 * jnp.sqrt(2.0 * jnp.pi) ** 2)
    )
    rho += (
        amp
        * jnp.exp(-((X - 0.3) ** 2 + (Y - 0.3) ** 2) / 2.0 / sigma**2)
        / (sigma**3 * jnp.sqrt(2.0 * jnp.pi) ** 2)
    )
    # normalize wavefunction to <|psi|^2>=1
    rhobar = jnp.mean(rho)
    rho /= rhobar
    psi = jnp.sqrt(rho)

    # Fourier Space Variables
    klinx = 2.0 * jnp.pi / Lx * jnp.arange(-nx / 2, nx / 2)
    kliny = 2.0 * jnp.pi / Ly * jnp.arange(-ny / 2, ny / 2)
    kx, ky = jnp.meshgrid(klinx, kliny, indexing="ij")
    kx = jnp.fft.ifftshift(kx)
    ky = jnp.fft.ifftshift(ky)
    kSq = kx**2 + ky**2

    # Potential
    Vhat = -jnp.fft.fftn(4.0 * jnp.pi * G * (jnp.abs(psi) ** 2 - 1.0)) / (kSq + (kSq == 0))
    V = jnp.real(jnp.fft.ifftn(Vhat))

    # number of timesteps
    nt = int(jnp.ceil(t_end / dt))

    # prep figure
    fig = plt.figure(figsize=(6, 4), dpi=80)
    grid = plt.GridSpec(1, 2, wspace=0.0, hspace=0.0)
    ax1 = plt.subplot(grid[0, 0])
    ax2 = plt.subplot(grid[0, 1])
    outputCount = 1

    # Simulation Main Loop
    for i in range(nt):
        # (1/2) kick
        psi = jnp.exp(-1.0j * dt / 2.0 * V) * psi

        # drift
        psihat = jnp.fft.fftn(psi)
        psihat = jnp.exp(dt * (-1.0j * kSq / 2.0)) * psihat
        psi = jnp.fft.ifftn(psihat)

        # update potential
        Vhat = -jnp.fft.fftn(4.0 * jnp.pi * G * (jnp.abs(psi) ** 2 - 1.0)) / (
            kSq + (kSq == 0)
        )
        V = jnp.real(jnp.fft.ifftn(Vhat))

        # (1/2) kick
        psi = jnp.exp(-1.0j * dt / 2.0 * V) * psi

        # update time
        t += dt

        # plot in real time
        plot_this_turn = False
        if t + dt > outputCount * t_out:
            plot_this_turn = True
        if (plot_real_time and plot_this_turn) or (i == nt - 1):
            plt.sca(ax1)
            plt.cla()

            plt.imshow(jnp.log10(jnp.abs(psi) ** 2), cmap="inferno")
            plt.clim(-1, 2)
            ax1.get_xaxis().set_visible(False)
            ax1.get_yaxis().set_visible(False)
            ax1.set_aspect("equal")

            plt.sca(ax2)
            plt.cla()
            plt.imshow(jnp.angle(psi), cmap="bwr")
            plt.clim(-jnp.pi, jnp.pi)
            ax2.get_xaxis().set_visible(False)
            ax2.get_yaxis().set_visible(False)
            ax2.set_aspect("equal")

            plt.pause(0.001)
            outputCount += 1

    # Save figure
    plt.sca(ax1)
    plt.title(r"$\log_{10}(|\psi|^2)$")
    plt.sca(ax2)
    plt.title(r"${\rm angle}(\psi)$")
    plt.savefig("quantum.png", dpi=240)
    plt.show()


if __name__ == "__main__":
    main()
