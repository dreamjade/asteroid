#!/bin/bash

#SBATCH --job-name=knn
#SBATCH --time=30:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=94
#SBATCH --account=douglase
#SBATCH --partition=high_priority

module add python/3.9/3.9.10
export OMP_NUM_THREADS=94
source ~/YU_Python3/bin/activate
python3 final.py
