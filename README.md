# quantum-jax

Authors: [Philip Mocz (@pmocz)](https://github.com/pmocz/), [Eric Ludwig (@TheRealEKL)](https://github.com/TheRealEKL)

(c) 2025

A simple differentiable Schrodinger-Poisson solver written in JAX

* `quantum-stars.py` includes star particles and self-interaction
* `quantum-fluid.py` includes isothermal gas
* `quantum-bh.py` includes a black hole sink particle
* `quantum-cosmo.py` includes cosmological comoving coordinates


## How to Run (Locally)

Check out the code: 

```console
git clone git@github.com:pmocz/quantum-jax.git
```

Go into the folder:

```console
cd quantum-jax
```

Install dependencies:

```console
pip install -r requirements.txt
```

Run the simulation:

```console
python quantum-stars.py --show
```

## How to Run (On Rusty)

Check out the code: 

```console
git clone git@github.com:pmocz/quantum-jax.git
```

Go into the folder:

```console
cd quantum-jax
```

Make a virtual environment (first make sure your `~/.bashrc` file contains `export VENVDIR=/mnt/home/USER_NAME/FOLDER_TO_YOUR_ENVIRONMENTS`) -- this is a one-time setup:

```console
./venv_setup.sh
```

Submit a job:

```console
sbatch sbatch_rusty.sh
```

## Plot checkpoints

Plot the results by running the script:

```console
python plot_checkpoints.py
```

Below is an example of a snapshot from a simulation:

![Example output figure](example_output.png)
