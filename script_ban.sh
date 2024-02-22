#!/bin/bash

## BEGIN SBATCH directives
#SBATCH --job-name=batch1
#SBATCH --output="eoles/outputs/batch1.txt"
#SBATCH --error="eoles/outputs/error_batch1.txt"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --time=100:00:00
#SBATCH --partition=cpu_shared
#SBATCH --account=gennn
#SBATCH --nodelist=node020
#SBATCH --mem=MaxMemPerNode
## END SBATCH directives

## load modules
module purge
module load anaconda3/2020.11 #cuda/10.2
module load gurobi
export GRB_LICENSE_FILE="/mnt/beegfs/softs/opt/core/gurobi/9.5.2/linux64/gurobi.20.lic"
conda activate envCoupling

python main_ban.py --cpu 20 --N 100 --montecarlo True