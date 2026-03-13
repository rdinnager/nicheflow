#!/bin/bash
#SBATCH --job-name=nicheflow_train
#SBATCH --partition=default
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=124G
#SBATCH --time=96:00:00
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

# Launch both training scripts in parallel on separate GPUs
Rscript run.R train_env_vae.R device=cuda:0 2>&1 | tee logs/env_vae_train.log &
VAE_PID=$!

Rscript run.R train_geode.R device=cuda:1 2>&1 | tee logs/geode_train.log &
GEODE_PID=$!

echo "VAE PID: $VAE_PID"
echo "GeODE PID: $GEODE_PID"

# Wait for both to finish
wait $VAE_PID
VAE_EXIT=$?
echo "VAE exit code: $VAE_EXIT"

wait $GEODE_PID
GEODE_EXIT=$?
echo "GeODE exit code: $GEODE_EXIT"

echo "=== Job Complete ==="
date
nvidia-smi

exit $(( VAE_EXIT + GEODE_EXIT ))
