#!/bin/bash
# train_and_validate_geoencoder.sh
# Submits two SLURM jobs: geoencoder training (B200) and validation (L4)
# Training runs uninterrupted; validation monitors for new checkpoints.

set -e

LIVE_DIR="output/checkpoints/geoencoder/live"

# Clean previous IPC state
rm -f "${LIVE_DIR}/TRAINER_READY" "${LIVE_DIR}/VALIDATOR_READY" "${LIVE_DIR}/TRAINING_DONE"
rm -f "${LIVE_DIR}/latest_checkpoint.pt" "${LIVE_DIR}/latest_meta.rds"
rm -f "${LIVE_DIR}/config.rds"
mkdir -p "${LIVE_DIR}/val_results" logs

echo "=== Submitting Geoencoder Training + Validation ==="
echo "Live dir: ${LIVE_DIR}"
date
echo ""

# Submit training job (B200)
TRAIN_JOB=$(sbatch --parsable train_geoencoder_b200.sh)
echo "Training job submitted: ${TRAIN_JOB}"

# Submit validation job (L4)
VAL_JOB=$(sbatch --parsable validate_geoencoder_l4.sh)
echo "Validation job submitted: ${VAL_JOB}"

echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f logs/job-geoenc_train-${TRAIN_JOB}.err"
echo "  tail -f logs/job-geoenc_val-${VAL_JOB}.err"
echo "  cat ${LIVE_DIR}/val_results/val_metrics.log"
