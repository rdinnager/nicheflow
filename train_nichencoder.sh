#!/bin/bash
#SBATCH --job-name=nichencoder_train
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=36:00:00
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
pwd
nvidia-smi
echo "================"

Rscript -e 'targets::tar_make(names = nichencoder_training)'

echo "=== Job Complete ==="
date
nvidia-smi
