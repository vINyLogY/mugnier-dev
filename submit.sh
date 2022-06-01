#!/bin/bash
#SBATCH --partition=femto
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=4-00:00:00    # Time limit hrs:min:sec
pwd; hostname; date

if [ "$1" != "" ]; then
    echo "Running $1"
    ~/.conda/envs/mkl/bin/python ./tests/test_brownian.py --width $1
else
    echo "Parameter 1 required."
fi

date

