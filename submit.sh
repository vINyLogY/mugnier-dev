#!/bin/bash
#SBATCH --partition=femto
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=1-00:00:00
pwd; hostname; date

~/.conda/envs/mkl/bin/python ./tests/test_dnb.py --width_b $1 --out $2

date

