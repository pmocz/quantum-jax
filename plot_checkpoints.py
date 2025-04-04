import jax
import orbax.checkpoint as ocp
import numpy as np
import matplotlib.pyplot as plt
import os

"""
Plot checkpointed data.

Philip Mocz (2025), @pmocz
Flatiron Institute
"""

##############
# Checkpointer
path = os.path.join(os.path.dirname(__file__), "checkpoints")
async_checkpoint_manager = ocp.CheckpointManager(path)


def main():
    all_rho = np.array([])
    sx_all = np.array([])
    sy_all = np.array([])
    grid_i = 10
    grid_j = 10

    for i in range(grid_i):
        row_rho = np.array([])
        for j in range(grid_j):
            # Load the checkpoint
            restored = async_checkpoint_manager.restore(i * grid_j + j)
            rho = np.mean(np.abs(restored.state["psi"]) ** 2, axis=2).T
            if row_rho.size == 0:
                row_rho = rho
            else:
                row_rho = np.hstack((row_rho, rho))
            nx = restored.params["nx"]
            ny = restored.params["ny"]
            Lx = restored.params["Lx"]
            Ly = restored.params["Ly"]
            n_s = restored.params["n_s"]
            pos = restored.state["pos"]
            sx = jax.lax.slice(pos, (0, 0), (n_s, 1)) / Lx * nx
            sy = jax.lax.slice(pos, (0, 1), (n_s, 2)) / Ly * ny
            sx_all = np.append(sx_all, sx + i * nx)
            sy_all = np.append(sy_all, sy + j * ny)
        if all_rho.size == 0:
            all_rho = row_rho
        else:
            all_rho = np.vstack((all_rho, row_rho))

    # imshow all_psi
    fig = plt.figure(figsize=(16, 8), dpi=80)
    ax = fig.add_subplot(111)
    plt.imshow(
        np.log10(all_rho),
        cmap="inferno",
        origin="lower",
        extent=(0, grid_i * nx, 0, grid_j * ny),
    )
    plt.plot(sx_all, sy_all, color="cyan", marker=".", linestyle="None", markersize=1)
    plt.colorbar(label="log10(|psi|^2)")
    ax.set_aspect("equal")
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
