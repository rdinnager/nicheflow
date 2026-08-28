#!/bin/bash
#SBATCH --job-name=debug_emd
#SBATCH --partition=hpg-turin
#SBATCH --gres=gpu:l4:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=1:00:00
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

echo "=== Debug EMD ==="
date
nvidia-smi

stdbuf -oL -eL Rscript -e '
library(torch)
library(dagnn)
library(zeallot)
library(dplyr)
library(purrr)
library(stringr)
library(sf)
library(terra)
library(arrow)
library(targets)

# Source all functions
lapply(list.files("./R", pattern = "^functions_.*\\.R$", full.names = TRUE), source)
source("R/utils.R")

# Load data and models
cat("Loading data...\n")
jade_test_data <- tar_read(eval_test_data)
jade_train_data <- tar_read(eval_train_data)
species_map <- tar_read(nichencoder_species_map)
chelsa_var_meta <- tar_read(chelsa_var_meta)
env_mean_sd <- tar_read(env_mean_sd)
xy_mean_sd <- tar_read(xy_mean_sd)
active_dims <- tar_read(vae_active_dims)
chelsa_bio_dir <- tar_read(chelsa_bio_dir)

env_cols <- setdiff(names(jade_test_data),
                    c("X", "Y", "jacobian", "species", "taxon", "split_type"))

device <- "cuda:0"

cat("Loading models...\n")
vae_model <- env_vae_mod(31L, 16L)
load_model_checkpoint(vae_model, "output/checkpoints/env_vae/gamma_-2/epoch_0500_model.pt")
vae_model <- vae_model$to(device = device)
vae_model$eval()

flow_ckpt <- find_latest_checkpoint("output/checkpoints/nichencoder")
flow_model <- nichencoder_traj_net(
  coord_dim = length(active_dims), n_species = length(species_map),
  spec_embed_dim = 64L, breadths = c(512L, 256L, 128L))
load_model_checkpoint(flow_model, flow_ckpt$path)
flow_model <- flow_model$to(device = device)
flow_model$eval()

geode_model <- load_geode_model("output/checkpoints/geode/epoch_1000_model.pt", device = device)

cat("Models loaded. Testing EMD on one species...\n")

# Pick a species
sp <- names(species_map)[1]
sp_id <- species_map[sp]
cat("Species:", sp, "ID:", sp_id, "\n")

sp_test <- jade_test_data |> filter(species == sp)
sp_train <- jade_train_data |> filter(species == sp)
cat("Test rows:", nrow(sp_test), "Train rows:", nrow(sp_train), "\n")

truth_xy <- cbind(sp_test$X, sp_test$Y)

# Step 1: Create prediction grid
cat("\n--- Step 1: Prediction grid ---\n")
grid <- tryCatch({
  g <- create_prediction_grid(truth_xy, chelsa_var_meta, env_mean_sd, chelsa_bio_dir)
  cat("Grid cells:", nrow(g$grid_xy), "\n")
  cat("Grid env dims:", dim(g$grid_env_std), "\n")
  g
}, error = \(e) { cat("GRID ERROR:", e$message, "\n"); NULL })
if (is.null(grid)) quit("no", 1)

# Step 2: NicheFlow KDE
cat("\n--- Step 2: NicheFlow KDE ---\n")
tryCatch({
  gen_xy <- generate_geo_samples(sp_id, 1000L, flow_model, vae_model, geode_model,
                                  active_dims, xy_mean_sd, 16L, device)
  cat("Generated geo points:", nrow(gen_xy), "\n")
  cat("Range lon:", range(gen_xy[,1]), "lat:", range(gen_xy[,2]), "\n")
  kde_scores <- score_geographic_kde(gen_xy, grid$grid_xy)
  cat("KDE scores range:", range(kde_scores, na.rm=TRUE), "\n")
  emd_kde <- compute_geographic_emd(kde_scores, grid$grid_xy, truth_xy)
  cat("EMD (KDE):", emd_kde, "\n")
}, error = \(e) cat("KDE ERROR:", e$message, "\n", conditionCall(e), "\n"))

# Step 3: NicheFlow LL
cat("\n--- Step 3: NicheFlow LL ---\n")
tryCatch({
  ll_scores <- compute_log_density(grid$grid_env_std, as.integer(sp_id),
                                    vae_model, flow_model, active_dims,
                                    K = 5L, ode_steps = 50L, batch_size = 500L, device = device)
  cat("LL scores range:", range(ll_scores, na.rm=TRUE), "\n")
  ll_weights <- exp(ll_scores - max(ll_scores))
  emd_ll <- compute_geographic_emd(ll_weights, grid$grid_xy, truth_xy)
  cat("EMD (LL):", emd_ll, "\n")
}, error = \(e) cat("LL ERROR:", e$message, "\n"))

# Step 4: MaxEnt on grid
cat("\n--- Step 4: MaxEnt ---\n")
tryCatch({
  train_env <- standardize_env(sp_train[, env_cols], env_mean_sd)
  cat("train_env dims:", dim(train_env), "\n")
  all_pres_xy <- rbind(cbind(sp_train$X, sp_train$Y), truth_xy)
  bg_xy <- generate_background_points(all_pres_xy, 5000L)
  rast_stack <- build_chelsa_rast_stack(chelsa_var_meta, chelsa_bio_dir)
  pts <- terra::vect(bg_xy, crs = "EPSG:4326")
  bg_raw <- as.matrix(terra::extract(rast_stack, pts)[, -1])
  for (j in seq_len(ncol(bg_raw))) {
    na_m <- is.na(bg_raw[, j])
    if (any(na_m)) bg_raw[na_m, j] <- env_mean_sd$mean[j]
  }
  bg_env <- standardize_env(bg_raw, env_mean_sd)
  cat("bg_env dims:", dim(bg_env), "\n")
  mx_train_env <- rbind(train_env, bg_env)
  mx_labels <- c(rep(1, nrow(train_env)), rep(0, nrow(bg_env)))
  mx_grid_scores <- run_maxnet_predict(mx_train_env, mx_labels, grid$grid_env_std)
  cat("MaxEnt grid scores range:", range(mx_grid_scores, na.rm=TRUE), "\n")
  emd_mx <- compute_geographic_emd(mx_grid_scores, grid$grid_xy, truth_xy)
  cat("EMD (MaxEnt):", emd_mx, "\n")
}, error = \(e) cat("MAXENT ERROR:", e$message, "\n"))

cat("\n=== Debug complete ===\n")
'

echo "=== Done ==="
date
