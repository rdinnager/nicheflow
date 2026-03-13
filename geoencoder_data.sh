#!/bin/bash
#SBATCH --job-name=geoencoder_data
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=312G
#SBATCH --time=48:00:00
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
echo "================"

Rscript -e 'targets::tar_make(names = c(
  geoencoder_corrupted_coords,
  nichencoder_species_embeddings,
  geoencoder_dataset,
  geoencoder_val_downstream
))'

echo "=== Job Complete ==="
date
