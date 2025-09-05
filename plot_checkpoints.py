import jax
import orbax.checkpoint as ocp
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import json

"""
Plot checkpointed data.

Philip Mocz (2025), @pmocz
Flatiron Institute

Example Usage:

python plot_checkpoints.py

python plot_checkpoints.py --res_factor 8

"""

# Command Line Input
parser = argparse.ArgumentParser(description="Plot checkpoints.")
parser.add_argument("--res_factor", type=int, default=1, help="Resolution factor")
args = parser.parse_args()


# Checkpointer
path = os.path.join(os.path.dirname(__file__), f"checkpoints{args.res_factor}")
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
            # Load the parameters
            params_path = os.path.join(path, "params.json")
            if not os.path.exists(params_path):
                raise FileNotFoundError(f"Parameters file not found: {params_path}")
            with open(params_path, "r") as f:
                params = json.load(f)
            # Load the checkpoint
            i_checkpoint = i * grid_j + j
            restored = async_checkpoint_manager.restore(i_checkpoint)
            rho = np.mean(np.abs(restored["psi"]) ** 2, axis=2).T
            if row_rho.size == 0:
                row_rho = rho
            else:
                row_rho = np.hstack((row_rho, rho))
            nx = params["nx"]
            ny = params["ny"]
            Lx = params["Lx"]
            Ly = params["Ly"]
            n_s = params["n_s"]
            pos = restored["pos"]
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
