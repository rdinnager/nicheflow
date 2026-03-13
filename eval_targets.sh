#!/bin/bash
#SBATCH --job-name=nicheflow_eval
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:l4:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=96G
#SBATCH --time=3-00:00:00
#SBATCH --output logs/job-%x-%j.out
#SBATCH --error logs/job-%x-%j.err
#SBATCH --mail-user=rdinnage@fiu.edu
#SBATCH --mail-type=FAIL,END
#SBATCH --account=rdinnage.fiu
#SBATCH --qos=rdinnage.fiu

# ---------------------------------------------------------------------------
# NicheFlow Evaluation Pipeline
# ---------------------------------------------------------------------------
# GPU0 (cuda:0): AUC NicheFlow scoring (KDE + LL) + Geographic EMD
# GPU1 (cuda:1): SWD evaluation (runs in parallel with GPU0 tasks)
# CPU (16 crew workers): MaxEnt + RF per species, data prep, disdat
#
# ~600 species stratified by taxon, latitude, range size
# Estimated runtime: 24-48 hours
# ---------------------------------------------------------------------------

module load conda
source $(conda info --base)/etc/profile.d/conda.sh
conda activate rstudio-gpu

mkdir -p logs output/evaluation

echo "=== Evaluation Job Info ==="
date
hostname
pwd
nvidia-smi
echo "==========================="

# Run all evaluation + disdat targets
# nichencoder_training is protected with cue="never" so it won't re-run
# even though it shows as outdated from depend-hash drift.
Rscript -e '
  targets::tar_make(
    names = c(
      eval_swd_parquet,
      eval_auc_parquet,
      eval_emd_parquet,
      disdat_parquets
    )
  )
'

echo "=== Evaluation Complete ==="
date
nvidia-smi
