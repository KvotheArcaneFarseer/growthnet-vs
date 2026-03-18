#!/bin/bash

#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --mem=64G
#SBATCH -J "HD-BET"
#SBATCH --mail-user=mnn7bx@virginia.edu
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --account=Gam_ai_group

module purge
module load miniforge
conda activate nguyen_ants_env

# Config file path — set this or pass as first argument to sbatch
CONFIG="${1:-../configs/default.yaml}"

cd "$(dirname "$0")/.."
# python src/hd_bet.py --config "$CONFIG"  # legacy direct invocation
python scripts/run_hd_bet.py --config "$CONFIG"
