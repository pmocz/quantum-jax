#!/usr/bin/bash
#SBATCH --job-name=quantum-jax
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --partition gpu
#SBATCH --constraint=h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=80G
#SBATCH --time=00-00:40

module purge
module load cuda
module load python

export PYTHONUNBUFFERED=TRUE

source $VENVDIR/quantum-jax-venv/bin/activate

srun python quantum-jax.py --res_factor=2
