#!/bin/bash

## BEGIN SBATCH directives
#SBATCH --job-name=batch4
#SBATCH --output="eoles/outputs/batch4.txt"
#SBATCH --error="eoles/outputs/error_batch4.txt"
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --time=100:00:00
#SBATCH --partition=cpu_shared
#SBATCH --account=gennn
#SBATCH --nodelist=node019
#SBATCH --mem=MaxMemPerNode
## END SBATCH directives

## load modules
module purge
module load anaconda3/2020.11 #cuda/10.2
module load gurobi
export GRB_LICENSE_FILE="/mnt/beegfs/softs/opt/core/gurobi/9.5.2/linux64/gurobi.19.lic"
conda activate envCoupling

python main_coupling_resirf.py --cpu 6 --configdir "eoles/inputs/xps/20231212"