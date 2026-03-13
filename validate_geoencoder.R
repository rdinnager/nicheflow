# validate_geoencoder.R
# Standalone geo-encoder validation monitor for decoupled SLURM execution.
# Monitors for new checkpoints written by the training job and runs validation.
#
# Usage:
#   Rscript run.R validate_geoencoder.R live_dir="output/checkpoints/geoencoder/live"
#   Rscript run.R validate_geoencoder.R device=cuda:0

library(torch)
library(dagnn)
library(purrr)
library(dplyr)
options(torch.serialization_version = 3)

source("R/models.R")
source("R/functions_env_vae_model.R")
source("R/functions_geoencoder_model.R")
source("R/functions_vae_validation.R")
source("R/functions_nichencoder.R")
source("R/functions_ipc.R")

# ===========================================================================
# Parameters
# ===========================================================================

#| param
live_dir <- "output/checkpoints/geoencoder/live"
#| param
device <- "cuda:0"
#| param
startup_timeout <- 0
#| param
poll_interval <- 5
#| param
val_batch_size <- 128L

# ===========================================================================
# Data keys (captured by run_script)
# ===========================================================================

#| data val_epoch val_mse val_cosine val_loss
#| data val_energy val_swd val_centroid_mse

# ===========================================================================
# Wait for trainer
# ===========================================================================

message("Validation monitor starting")
message("Live directory: ", live_dir)
message("Device: ", device)
flush(stderr())

trainer_ready_path <- file.path(live_dir, "TRAINER_READY")
message("Waiting for TRAINER_READY...")
flush(stderr())

if (!wait_for_file(trainer_ready_path, timeout = startup_timeout)) {
  stop("Training job did not start within ", startup_timeout, "s. Exiting.")
}

# ===========================================================================
# Load config and data
# ===========================================================================

config <- readRDS(file.path(live_dir, "config.rds"))
message("Config loaded. Model: embed_dim=", config$embed_dim,
        ", n_blocks=", config$n_blocks, ", output_dim=", config$output_dim)
flush(stderr())

# Load target embeddings
emb_data <- readRDS(config$embeddings_file)
target_embeddings <- emb_data$embedding_matrix
species_map <- emb_data$species_map
message("Target embeddings: ", nrow(target_embeddings), " x ",
        ncol(target_embeddings))

# Load validation data
message("Loading validation data from: ", config$val_parquet)
val_raw <- arrow::read_parquet(config$val_parquet)
val_groups_list <- val_raw |>
  group_by(species, version_id, set_id) |>
  group_split() |>
  map(\(g) {
    coords <- as.matrix(g[, c("X_norm", "Y_norm")])
    if (nrow(coords) > config$max_points) {
      coords <- coords[sample.int(nrow(coords), config$max_points), , drop = FALSE]
    }
    list(species = g$species[1], coords = coords)
  })
message("Validation groups: ", format(length(val_groups_list), big.mark = ","))
rm(val_raw)

# Flatten for GC efficiency
message("Flattening validation groups...")
val_flat <- flatten_groups(val_groups_list)
rm(val_groups_list)
gc(verbose = FALSE)
message("Val flat coords: ", format(nrow(val_flat$coords_all), big.mark = ","), " rows")

# Load downstream validation data
downstream_flat <- NULL
downstream_env <- NULL
nichencoder_model <- NULL
vae_model <- NULL

if (config$downstream_coords_parquet != "") {
  message("Loading downstream validation data...")

  downstream_raw <- arrow::read_parquet(config$downstream_coords_parquet)
  downstream_groups_list <- downstream_raw |>
    group_by(species, version_id, set_id) |>
    group_split() |>
    map(\(g) {
      coords <- as.matrix(g[, c("X_norm", "Y_norm")])
      if (nrow(coords) > config$max_points) {
        coords <- coords[sample.int(nrow(coords), config$max_points), , drop = FALSE]
      }
      list(species = g$species[1], coords = coords)
    })
  message("Downstream groups: ", length(downstream_groups_list))
  rm(downstream_raw)

  # Flatten for GC efficiency
  downstream_flat <- flatten_groups(downstream_groups_list)
  rm(downstream_groups_list)
  gc(verbose = FALSE)

  downstream_env <- arrow::read_parquet(config$downstream_env_parquet)
  message("Downstream env samples: ", format(nrow(downstream_env), big.mark = ","))

  # Load NichEncoder model
  message("Loading NichEncoder model...")
  nichencoder_model <- nichencoder_traj_net(
    coord_dim = config$nichencoder_coord_dim,
    n_species = config$nichencoder_n_species,
    spec_embed_dim = config$nichencoder_spec_embed_dim,
    breadths = config$nichencoder_breadths
  )
  nc_ckpt <- find_latest_checkpoint(config$nichencoder_checkpoint_dir)
  load_model_checkpoint(nichencoder_model, nc_ckpt$path)
  nichencoder_model <- nichencoder_model$to(device = device)
  nichencoder_model$eval()
  message("NichEncoder loaded from epoch ", nc_ckpt$epoch, " on ", device)

  # Load VAE model
  message("Loading VAE model...")
  vae_model <- env_vae_mod(config$vae_input_dim, config$vae_latent_dim)
  load_model_checkpoint(vae_model, config$vae_checkpoint)
  vae_model <- vae_model$to(device = device)
  vae_model$eval()
  message("VAE loaded from: ", config$vae_checkpoint, " on ", device)
}

gc(verbose = FALSE)

# Init geoencoder model (empty weights — will load from checkpoints)
message("Initializing GeoEncoderTransformer (empty weights)...")
model <- geoencoder_transformer(
  input_dim = 2L,
  embed_dim = config$embed_dim,
  output_dim = config$output_dim,
  n_blocks = config$n_blocks,
  num_heads = config$num_heads,
  dropout_prob = config$dropout_prob
)
model <- model$to(device = device)
model$eval()
message("Model ready on ", device)
flush(stderr())

# ===========================================================================
# Signal ready
# ===========================================================================

dir.create(file.path(live_dir, "val_results"), recursive = TRUE, showWarnings = FALSE)
file.create(file.path(live_dir, "VALIDATOR_READY"))
message("VALIDATOR_READY written. Entering monitoring loop.")
flush(stderr())

# ===========================================================================
# Monitoring loop
# ===========================================================================

last_meta_mtime <- as.POSIXct("1970-01-01")
meta_path <- file.path(live_dir, "latest_meta.rds")
ckpt_path <- file.path(live_dir, "latest_checkpoint.pt")
done_path <- file.path(live_dir, "TRAINING_DONE")
metrics_log <- file.path(live_dir, "val_results", "val_metrics.log")

while (TRUE) {
  # Check for new checkpoint
  if (file.exists(meta_path)) {
    current_mtime <- file.mtime(meta_path)
    if (current_mtime > last_meta_mtime) {
      meta <- tryCatch(readRDS(meta_path), error = \(e) {
        message("  Warning: could not read meta file, will retry: ", e$message)
        NULL
      })

      if (!is.null(meta)) {
        message("\n=== New checkpoint: epoch ", meta$epoch,
                " (train_loss=", round(meta$train_loss, 6), ") ===")
        flush(stderr())

        tryCatch({
          # Load weights
          state_dict <- torch_load(ckpt_path)
          model$load_state_dict(state_dict)
          model$eval()

          # Run validation
          t_val <- Sys.time()
          val_result <- validate_geoencoder(
            model, target_embeddings = target_embeddings,
            species_map = species_map,
            device = device, batch_size = val_batch_size,
            val_flat = val_flat,
            val_subsample = 20000L,
            downstream_flat = downstream_flat,
            downstream_env = downstream_env,
            nichencoder_model = nichencoder_model,
            vae_model = vae_model,
            n_plot_species = config$n_plot_species,
            n_downstream_plot_species = config$n_downstream_plot_species,
            n_metric_species = config$n_metric_species,
            fixed_plot_species = config$fixed_plot_species,
            ode_steps = config$ode_steps,
            vae_active_dims = config$vae_active_dims,
            vae_latent_dim = config$vae_latent_dim,
            checkpoint_dir = config$checkpoint_dir, epoch = meta$epoch
          )
          val_time <- round(as.numeric(Sys.time() - t_val, units = "secs"), 1)

          # Output metrics (captured by run_script)
          cat("val_epoch:", meta$epoch, "\n")
          cat("val_mse:", round(val_result$val_mse, 8), "\n")
          cat("val_cosine:", round(val_result$val_cosine, 8), "\n")
          cat("val_loss:", round(val_result$val_loss, 8), "\n")
          if (!is.null(val_result$val_energy)) {
            cat("val_energy:", round(val_result$val_energy, 8), "\n")
            cat("val_swd:", round(val_result$val_swd, 8), "\n")
            cat("val_centroid_mse:", round(val_result$val_centroid_mse, 8), "\n")
          }

          # Save detailed results
          write_atomic_rds(
            c(val_result, list(epoch = meta$epoch, val_time = val_time)),
            file.path(live_dir, "val_results",
                      sprintf("val_epoch_%04d.rds", meta$epoch))
          )

          # Append to metrics log
          log_lines <- sprintf(
            "epoch=%d mse=%.6f cosine=%.6f loss=%.6f%s time=%.1fs\n",
            meta$epoch, val_result$val_mse, val_result$val_cosine, val_result$val_loss,
            if (!is.null(val_result$val_energy))
              sprintf(" energy=%.6f swd=%.6f centroid_mse=%.6f",
                      val_result$val_energy, val_result$val_swd, val_result$val_centroid_mse)
            else "",
            val_time
          )
          cat(log_lines, file = metrics_log, append = TRUE)

          message("Validation for epoch ", meta$epoch, " complete in ", val_time, "s")
          message("  MSE: ", round(val_result$val_mse, 6),
                  " | Cosine: ", round(val_result$val_cosine, 6),
                  " | Loss: ", round(val_result$val_loss, 6))
          if (!is.null(val_result$val_energy)) {
            message("  Energy: ", round(val_result$val_energy, 6),
                    " | SWD: ", round(val_result$val_swd, 6),
                    " | Centroid MSE: ", round(val_result$val_centroid_mse, 6))
          }
          flush(stderr())

        }, error = \(e) {
          message("  Validation error at epoch ", meta$epoch, ": ", e$message)
          flush(stderr())
        })

        last_meta_mtime <- current_mtime
        gc(verbose = FALSE)
        cuda_empty_cache()
      }
    }
  }

  # Check if training is done
  if (file.exists(done_path)) {
    # Process any final checkpoint we haven't seen yet
    if (file.exists(meta_path)) {
      final_mtime <- file.mtime(meta_path)
      if (final_mtime > last_meta_mtime) {
        message("Training done but final checkpoint not yet processed. Will process on next iteration.")
        flush(stderr())
        Sys.sleep(poll_interval)
        next
      }
    }
    message("\n=== Training complete. All checkpoints processed. Exiting. ===")
    flush(stderr())
    break
  }

  Sys.sleep(poll_interval)
}

message("Validation monitor finished.")
