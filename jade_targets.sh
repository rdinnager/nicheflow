#!/bin/bash
#SBATCH --job-name=nicheflow_jade31
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=72:00:00
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
  jade_bioclim_files_31, bioclim_sds_31, abc_rasters_31,
  jacobian_raster_path_31, jade_samples_31, jade_samples_clean_31,
  jade_samples_clean_31_filled, jade_species_counts_31,
  jade_split_assignments_31, jade_train_val_test_31,
  jade_train_parquet_31, jade_val_parquet_31, jade_test_parquet_31,
  jade_split_summary_31
))'

echo "=== Job Complete ==="
date
