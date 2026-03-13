#!/bin/bash
#SBATCH --job-name=geoenc_smoke_val
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output logs/job-%x-%j.out
#SBATCH --error logs/job-%x-%j.err
#SBATCH --mail-user=rdinnage@fiu.edu
#SBATCH --mail-type=FAIL,END
#SBATCH --account=rdinnage.fiu
#SBATCH --qos=rdinnage.fiu

module load conda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate rstudio-gpu

mkdir -p logs

echo "=== Smoke Validation Job ==="
date
hostname
nvidia-smi
echo "============================"

stdbuf -oL -eL Rscript run.R validate_geoencoder.R \
  device=cuda:0 \
  'live_dir="output/checkpoints/geoencoder/live"'

echo "=== Smoke Validation Complete ==="
date
