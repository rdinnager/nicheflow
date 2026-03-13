#!/bin/bash
#SBATCH --job-name=geoenc_train_b200
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:b200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output logs/job-%x-%j.out
#SBATCH --error logs/job-%x-%j.err
#SBATCH --mail-user=rdinnage@fiu.edu
#SBATCH --mail-type=FAIL,END
#SBATCH --account=rdinnage.fiu
#SBATCH --qos=rdinnage.fiu

module load conda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate rstudio-gpu2

mkdir -p logs

echo "=== GeoEncoder Training (B200) ==="
date
hostname
nvidia-smi
echo "===================================="

stdbuf -oL -eL Rscript run.R train_geoencoder.R \
  device=cuda:0 \
  num_epochs=250 \
  batch_size=512 \
  samples_per_epoch=1000000 \
  checkpoint_every=10 \
  val_every=2 \
  clear_checkpoints=TRUE \
  'live_dir="output/checkpoints/geoencoder/live"' \
  'downstream_coords_parquet="data/processed/geoencoder_downstream_coords.parquet"' \
  'downstream_env_parquet="data/processed/geoencoder_downstream_env.parquet"' \
  'vae_checkpoint="output/checkpoints/env_vae/gamma_-2/epoch_0500_model.pt"'

echo "=== Training Complete ==="
date
