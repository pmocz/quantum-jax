#!/bin/bash

# Create virtual environment on rusty
rm -fr $VENVDIR/quantum-jax-venv

module purge
module load python/3.11
python -m venv --system-site-packages $VENVDIR/quantum-jax-venv
source $VENVDIR/quantum-jax-venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
