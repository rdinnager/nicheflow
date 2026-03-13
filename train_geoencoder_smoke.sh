#!/bin/bash
#SBATCH --job-name=geoenc_smoke
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
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
pwd
nvidia-smi
echo "================"

# Smoke test: 5 epochs of 1M samples each, validate on last epoch
stdbuf -oL -eL Rscript run.R train_geoencoder.R \
  num_epochs=5 \
  val_every=5 \
  checkpoint_every=5 \
  batch_size=64 \
  samples_per_epoch=1000000 \
  device=cuda:0 \
  val_device=cuda:1 \
  clear_checkpoints=TRUE \
  'downstream_coords_parquet="data/processed/geoencoder_downstream_coords.parquet"' \
  'downstream_env_parquet="data/processed/geoencoder_downstream_env.parquet"' \
  'vae_checkpoint="output/checkpoints/env_vae/gamma_-2/epoch_0500_model.pt"'

echo "=== Job Complete ==="
date
nvidia-smi
