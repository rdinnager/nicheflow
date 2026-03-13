#!/bin/bash
#SBATCH --job-name=bench_logdensity
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

echo "=== Job Info ==="
date
hostname
nvidia-smi
echo "================"

Rscript claude_notes_and_utils/2026-03-12_benchmark_log_density.R

echo "=== Job Complete ==="
date
