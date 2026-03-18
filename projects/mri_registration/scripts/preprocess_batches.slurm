#!/bin/bash

#Submit this script with: sbatch preprocess_batches.slurm

#SBATCH --time=32:00:00   # job time limit
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=1   # number of tasks per node
#SBATCH --cpus-per-task=64   # number of CPU cores per task
#SBATCH --partition=standard   # partition
#SBATCH --mem=96G   # memory
#SBATCH -J "Preprocess Coregistered" #job name (change when running for other directories)
#SBATCH --mail-user=mnn7bx@virginia.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --account=Gam_ai_group   # allocation name


# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE
module purge
module load miniforge
conda activate antspy_registration_env

# Config file path — set this or pass as first argument to sbatch
CONFIG="${1:-../configs/default.yaml}"

cd "$(dirname "$0")/.."
# python src/preprocessing.py --config "$CONFIG"  # legacy direct invocation
python scripts/run_preprocessing.py --config "$CONFIG"
