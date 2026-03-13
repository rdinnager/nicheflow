# train_geoencoder.R
# Geo-Encoder training script
#
# Trains a transformer to predict NichEncoder species embeddings from
# variable-length sets of (lon,lat) occurrence coordinates.
# Maps: (lon,lat) points -> 64-dim niche embedding
#
# Usage:
#   Rscript run.R train_geoencoder.R device=cuda:0 num_epochs=500
#   Rscript run.R train_geoencoder.R num_epochs=3 batch_size=32  # smoke test

library(torch)
library(dagnn)
library(purrr)
library(dplyr)
options(torch.serialization_version = 3)

source("R/models.R")               # for transformer_block
source("R/functions_env_vae_model.R")  # for env_vae_mod (unconditional VAE decoder)
source("R/functions_geoencoder_model.R")
source("R/functions_vae_validation.R") # for checkpoint functions
source("R/functions_nichencoder.R")    # for nichencoder_traj_net + metrics
source("R/functions_ipc.R")             # for atomic file writes + wait_for_file

# ===========================================================================
# Parameters
# ===========================================================================

#| param
train_parquet <- "data/processed/geoencoder_train.parquet"
#| param
val_parquet <- "data/processed/geoencoder_val.parquet"
#| param
embeddings_file <- "output/geoencoder_config/target_embeddings.rds"
#| param
species_map_file <- "output/nichencoder_config/species_map.rds"
#| param
device <- "cuda:0"
#| param
val_device <- ""
#| param
embed_dim <- 256L
#| param
n_blocks <- 8L
#| param
num_heads <- 8L
#| param
output_dim <- 64L
#| param
max_points <- 500L
#| param
batch_size <- 256L
#| param
num_epochs <- 500L
#| param
lr <- 0.0005
#| param
loss_type <- "mse_cosine"
#| param
cosine_weight <- 0.5
#| param
n_cycles <- 2L
#| param
cycle_1_fraction <- 0.5
#| param
cycle_2_lr_factor <- 0.1
#| param
checkpoint_dir <- "output/checkpoints/geoencoder"
#| param
checkpoint_every <- 25L
#| param
val_every <- 10L
#| param
jitter_sd <- 0.001
#| param
dropout_frac <- 0.05
#| param
clear_checkpoints <- FALSE
#| param
dropout_prob <- 0.1
#| param
samples_per_epoch <- 0L
#| param
downstream_coords_parquet <- ""
#| param
downstream_env_parquet <- ""
#| param
nichencoder_checkpoint_dir <- "output/checkpoints/nichencoder"
#| param
nichencoder_coord_dim <- 6L
#| param
nichencoder_n_species <- 18121L
#| param
nichencoder_spec_embed_dim <- 64L
#| param
nichencoder_breadths <- c(512L, 256L, 128L)
#| param
vae_checkpoint <- ""
#| param
vae_input_dim <- 31L
#| param
vae_latent_dim <- 16L
#| param
vae_active_dims <- c(7L, 9L, 11L, 13L, 15L, 16L)
#| param
n_plot_species <- 5L
#| param
n_downstream_plot_species <- 6L
#| param
n_metric_species <- 50L
#| param
fixed_plot_species <- c("Sclerophrys kerinyagae", "Cycloramphus dubius")
#| param
ode_steps <- 200L
#| param
live_dir <- ""
#| param
validator_timeout <- 900L

# ===========================================================================
# Data keys (captured by run_script)
# ===========================================================================

#| data train_epoch train_loss
#| data val_epoch val_mse val_cosine val_loss
#| data val_energy val_swd val_centroid_mse

# ===========================================================================
# Output
# ===========================================================================

#| output
checkpoint_dir_out <- checkpoint_dir

# ===========================================================================
# Data Loading
# ===========================================================================

# Resolve val_device: empty string means same as training device
if (val_device == "") val_device <- device
message("Training device: ", device, " | Validation device: ", val_device)

message("Loading training data from: ", train_parquet)
t_load <- Sys.time()

train_raw <- arrow::read_parquet(train_parquet)
message("Training rows: ", format(nrow(train_raw), big.mark = ","))

# Load target embeddings
emb_data <- readRDS(embeddings_file)
target_embeddings <- emb_data$embedding_matrix
species_map <- emb_data$species_map
message("Target embeddings: ", nrow(target_embeddings), " x ",
        ncol(target_embeddings))

# Group training data into coordinate sets
# Each group: species + version_id + set_id -> one variable-length coord set
message("Grouping into coordinate sets...")
train_groups_list <- train_raw |>
  group_by(species, version_id, set_id) |>
  group_split() |>
  map(\(g) {
    coords <- as.matrix(g[, c("X_norm", "Y_norm")])
    # Truncate to max_points
    if (nrow(coords) > max_points) {
      coords <- coords[sample.int(nrow(coords), max_points), , drop = FALSE]
    }
    list(species = g$species[1], coords = coords)
  })
message("Training groups: ", format(length(train_groups_list), big.mark = ","))

# Flatten into compact representation to reduce R GC pressure
# (1.7M list elements = ~10M R objects; flat = ~5 R objects)
message("Flattening group data for GC efficiency...")
train_flat <- flatten_groups(train_groups_list)
rm(train_groups_list)
gc(verbose = FALSE)
message("Flat coords: ", format(nrow(train_flat$coords_all), big.mark = ","),
        " rows in single matrix")

# Load validation data and downstream models (only for inline validation mode)
val_flat <- NULL
downstream_flat <- NULL
downstream_env <- NULL
nichencoder_model <- NULL
vae_model <- NULL

if (live_dir == "") {
  message("Loading validation data from: ", val_parquet)
  val_raw <- arrow::read_parquet(val_parquet)

  val_groups_list <- val_raw |>
    group_by(species, version_id, set_id) |>
    group_split() |>
    map(\(g) {
      coords <- as.matrix(g[, c("X_norm", "Y_norm")])
      if (nrow(coords) > max_points) {
        coords <- coords[sample.int(nrow(coords), max_points), , drop = FALSE]
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

  if (downstream_coords_parquet != "") {
    message("Loading downstream validation data...")

    downstream_raw <- arrow::read_parquet(downstream_coords_parquet)
    downstream_groups_list <- downstream_raw |>
      group_by(species, version_id, set_id) |>
      group_split() |>
      map(\(g) {
        coords <- as.matrix(g[, c("X_norm", "Y_norm")])
        if (nrow(coords) > max_points) {
          coords <- coords[sample.int(nrow(coords), max_points), , drop = FALSE]
        }
        list(species = g$species[1], coords = coords)
      })
    message("Downstream groups: ", length(downstream_groups_list))
    rm(downstream_raw)

    # Flatten for GC efficiency
    downstream_flat <- flatten_groups(downstream_groups_list)
    rm(downstream_groups_list)
    gc(verbose = FALSE)

    downstream_env <- arrow::read_parquet(downstream_env_parquet)
    message("Downstream env samples: ", format(nrow(downstream_env), big.mark = ","))

    # Load NichEncoder model
    message("Loading NichEncoder model...")
    nichencoder_model <- nichencoder_traj_net(
      coord_dim = nichencoder_coord_dim,
      n_species = nichencoder_n_species,
      spec_embed_dim = nichencoder_spec_embed_dim,
      breadths = nichencoder_breadths
    )
    nc_ckpt <- find_latest_checkpoint(nichencoder_checkpoint_dir)
    load_model_checkpoint(nichencoder_model, nc_ckpt$path)
    nichencoder_model <- nichencoder_model$to(device = val_device)
    nichencoder_model$eval()
    message("NichEncoder loaded from epoch ", nc_ckpt$epoch, " on ", val_device)

    # Load VAE model (simple version, no species conditioning)
    message("Loading VAE model...")
    vae_model <- env_vae_mod(vae_input_dim, vae_latent_dim)
    load_model_checkpoint(vae_model, vae_checkpoint)
    vae_model <- vae_model$to(device = val_device)
    vae_model$eval()
    message("VAE loaded from: ", vae_checkpoint, " on ", val_device)

    gc(verbose = FALSE)
  }
} else {
  message("Decoupled validation mode: skipping val/downstream data loading")
}

rm(train_raw)
gc(verbose = FALSE)

load_time <- round(as.numeric(Sys.time() - t_load, units = "secs"))
message("Data load time: ", load_time, "s")

# ===========================================================================
# Batch Index Vectors
# ===========================================================================

n_train <- train_flat$n_groups
use_epoch_sampling <- samples_per_epoch > 0L && samples_per_epoch < n_train
epoch_n <- if (use_epoch_sampling) samples_per_epoch else n_train
n_batches <- ceiling(epoch_n / batch_size)

if (use_epoch_sampling) {
  message("Epoch sampling: ", format(samples_per_epoch, big.mark = ","),
          " of ", format(n_train, big.mark = ","),
          " groups per epoch (", n_batches, " batches)")
} else {
  message("Full dataset: ", format(n_train, big.mark = ","),
          " groups per epoch (", n_batches, " batches)")
}

# ===========================================================================
# Model Init / Checkpoint Resume
# ===========================================================================

dir.create(checkpoint_dir, recursive = TRUE, showWarnings = FALSE)

if (clear_checkpoints && dir.exists(checkpoint_dir)) {
  message("Clearing checkpoint directory: ", checkpoint_dir)
  unlink(list.files(checkpoint_dir, full.names = TRUE))
}

checkpoint <- find_latest_checkpoint(checkpoint_dir)
start_epoch <- 0L

message("Initializing GeoEncoderTransformer: embed_dim=", embed_dim,
        ", n_blocks=", n_blocks, ", output_dim=", output_dim)
model <- geoencoder_transformer(
  input_dim = 2L,
  embed_dim = embed_dim,
  output_dim = output_dim,
  n_blocks = n_blocks,
  num_heads = num_heads,
  dropout_prob = dropout_prob
)
model <- model$to(device = device)

if (!is.null(checkpoint)) {
  message("Loading weights from checkpoint: ", checkpoint$path,
          " (epoch ", checkpoint$epoch, ")")
  load_model_checkpoint(model, checkpoint$path)
  start_epoch <- checkpoint$epoch
}

n_params <- sum(map_dbl(model$parameters, ~ .x$numel()))
message("Model parameters: ", format(n_params, big.mark = ","))

# ===========================================================================
# Optimizer & Scheduler (Two-Cycle LR with Optimizer Reset)
# ===========================================================================

remaining_epochs <- num_epochs - start_epoch
cycle_1_end <- as.integer(num_epochs * cycle_1_fraction)

if (start_epoch < cycle_1_end) {
  current_cycle <- 1L
} else {
  current_cycle <- 2L
}

if (current_cycle == 1L) {
  cycle_1_remaining <- cycle_1_end - start_epoch
  optimizer <- optim_adamw(model$parameters, lr = lr, weight_decay = 0.01)
  scheduler <- lr_one_cycle(
    optimizer, max_lr = lr,
    epochs = cycle_1_remaining,
    steps_per_epoch = n_batches,
    cycle_momentum = FALSE
  )
  message("Cycle 1: epochs ", start_epoch + 1, "-", cycle_1_end,
          " | lr=", lr, " | ", cycle_1_remaining, " epochs remaining")
} else {
  cycle_2_remaining <- num_epochs - start_epoch
  cycle_2_lr <- lr * cycle_2_lr_factor
  optimizer <- optim_adamw(model$parameters, lr = cycle_2_lr, weight_decay = 0.01)
  scheduler <- lr_one_cycle(
    optimizer, max_lr = cycle_2_lr,
    epochs = cycle_2_remaining,
    steps_per_epoch = n_batches,
    cycle_momentum = FALSE
  )
  message("Cycle 2 (resumed): epochs ", start_epoch + 1, "-", num_epochs,
          " | lr=", cycle_2_lr, " | ", cycle_2_remaining, " epochs remaining")
}

message("Batches per epoch: ", n_batches,
        " | Remaining epochs: ", remaining_epochs,
        " | Cycle boundary: epoch ", cycle_1_end)

# ===========================================================================
# IPC Setup (decoupled validation mode)
# ===========================================================================

if (live_dir != "") {
  dir.create(file.path(live_dir, "val_results"), recursive = TRUE, showWarnings = FALSE)

  # Write config for the validator
  validator_config <- list(
    embed_dim = embed_dim, n_blocks = n_blocks, num_heads = num_heads,
    output_dim = output_dim, dropout_prob = dropout_prob,
    val_parquet = val_parquet, embeddings_file = embeddings_file,
    species_map_file = species_map_file, max_points = max_points,
    batch_size = batch_size, loss_type = loss_type, cosine_weight = cosine_weight,
    downstream_coords_parquet = downstream_coords_parquet,
    downstream_env_parquet = downstream_env_parquet,
    nichencoder_checkpoint_dir = nichencoder_checkpoint_dir,
    nichencoder_coord_dim = nichencoder_coord_dim,
    nichencoder_n_species = nichencoder_n_species,
    nichencoder_spec_embed_dim = nichencoder_spec_embed_dim,
    nichencoder_breadths = nichencoder_breadths,
    vae_checkpoint = vae_checkpoint, vae_input_dim = vae_input_dim,
    vae_latent_dim = vae_latent_dim, vae_active_dims = vae_active_dims,
    n_plot_species = n_plot_species,
    n_downstream_plot_species = n_downstream_plot_species,
    n_metric_species = n_metric_species,
    fixed_plot_species = fixed_plot_species,
    ode_steps = ode_steps,
    checkpoint_dir = checkpoint_dir
  )
  write_atomic_rds(validator_config, file.path(live_dir, "config.rds"))
  file.create(file.path(live_dir, "TRAINER_READY"))
  message("IPC: Wrote config.rds and TRAINER_READY")
  flush(stderr())

  # Wait for validator (with timeout)
  validator_ready_path <- file.path(live_dir, "VALIDATOR_READY")
  if (wait_for_file(validator_ready_path, timeout = validator_timeout)) {
    message("IPC: Validator ready. Starting training.")
  } else {
    message("WARNING: Validator did not start within ", validator_timeout,
            "s. Proceeding without external validation.")
  }
  flush(stderr())
}

# ===========================================================================
# Training Loop
# ===========================================================================

message("\n=== Starting training ===")
message("Device: ", device, " | Epochs: ", start_epoch + 1, "-", num_epochs)

for (epoch in seq(start_epoch + 1, num_epochs)) {
  epoch_start <- Sys.time()
  epoch_loss <- 0
  model$train()

  # Sample or shuffle indices for this epoch
  if (use_epoch_sampling) {
    epoch_idx <- sample.int(n_train, epoch_n)
  } else {
    epoch_idx <- sample.int(n_train)
  }

  for (bi in seq_len(n_batches)) {
    optimizer$zero_grad()

    b_start <- (bi - 1) * batch_size + 1
    b_end <- min(bi * batch_size, epoch_n)
    idx <- epoch_idx[b_start:b_end]

    # Collate with on-the-fly augmentations (flat representation)
    batch <- collate_geoencoder_batch_flat(
      idx, train_flat, target_embeddings, species_map, device,
      jitter_sd = jitter_sd, dropout_frac = dropout_frac
    )

    # Forward
    predicted <- model(batch$coords, batch$mask)
    loss <- geoencoder_loss(predicted, batch$target_emb,
                            loss_type = loss_type,
                            cosine_weight = cosine_weight)

    loss$backward()
    optimizer$step()
    scheduler$step()

    batch_loss <- as.numeric(loss$cpu())
    epoch_loss <- epoch_loss + batch_loss

    # Periodically clear CUDA cache to prevent allocator fragmentation
    # (variable sequence lengths cause different-sized tensor allocations
    #  that fragment the caching allocator, leading to multi-second stalls)
    if (bi %% 50 == 0) cuda_empty_cache()

    if (bi %% 500 == 0 || bi == n_batches) {
      message("  Epoch ", epoch, " batch ", bi, "/", n_batches,
              " | loss: ", round(batch_loss, 6))
      flush(stderr())
    }
  }

  mean_epoch_loss <- epoch_loss / n_batches
  epoch_time <- round(as.numeric(Sys.time() - epoch_start, units = "secs"), 1)

  cat("train_epoch:", epoch, "\n")
  cat("train_loss:", round(mean_epoch_loss, 8), "\n")

  message("Epoch ", epoch, " complete in ", epoch_time, "s",
          " | mean loss: ", round(mean_epoch_loss, 6))
  flush(stderr())

  # Cycle transition
  if (n_cycles >= 2L && epoch == cycle_1_end) {
    message("\n=== Cycle 1 complete, resetting optimizer for Cycle 2 ===")
    cycle_2_lr <- lr * cycle_2_lr_factor
    cycle_2_epochs <- num_epochs - cycle_1_end
    message("Cycle 2: epochs ", cycle_1_end + 1, "-", num_epochs,
            " | lr=", cycle_2_lr, " | ", cycle_2_epochs, " epochs")
    optimizer <- optim_adamw(model$parameters, lr = cycle_2_lr, weight_decay = 0.01)
    scheduler <- lr_one_cycle(
      optimizer, max_lr = cycle_2_lr,
      epochs = cycle_2_epochs,
      steps_per_epoch = n_batches,
      cycle_momentum = FALSE
    )
  }

  # Checkpoint
  if (epoch %% checkpoint_every == 0 || epoch == num_epochs) {
    message("  Saving checkpoint...")
    save_model_checkpoint(model, checkpoint_dir, epoch)
  }

  # Validation
  if (epoch %% val_every == 0 || epoch == num_epochs) {
    if (live_dir != "") {
      # Decoupled mode: write checkpoint for external validator
      message("  Writing validation checkpoint for epoch ", epoch, "...")
      write_atomic_pt(model$state_dict(), file.path(live_dir, "latest_checkpoint.pt"))
      write_atomic_rds(
        list(epoch = epoch, train_loss = mean_epoch_loss, timestamp = Sys.time()),
        file.path(live_dir, "latest_meta.rds")
      )
      flush(stderr())
    } else {
      # Inline mode: run validation directly
      message("  Running validation...")
      tryCatch({
        val_result <- validate_geoencoder(
          model, target_embeddings = target_embeddings,
          species_map = species_map,
          device = val_device, batch_size = batch_size,
          val_flat = val_flat,
          val_subsample = 20000L,
          downstream_flat = downstream_flat,
          downstream_env = downstream_env,
          nichencoder_model = nichencoder_model,
          vae_model = vae_model,
          n_plot_species = n_plot_species,
          n_downstream_plot_species = n_downstream_plot_species,
          n_metric_species = n_metric_species,
          fixed_plot_species = fixed_plot_species,
          ode_steps = ode_steps,
          vae_active_dims = vae_active_dims,
          vae_latent_dim = vae_latent_dim,
          checkpoint_dir = checkpoint_dir, epoch = epoch
        )
        cat("val_epoch:", epoch, "\n")
        cat("val_mse:", round(val_result$val_mse, 8), "\n")
        cat("val_cosine:", round(val_result$val_cosine, 8), "\n")
        cat("val_loss:", round(val_result$val_loss, 8), "\n")
        if (!is.null(val_result$val_energy)) {
          cat("val_energy:", round(val_result$val_energy, 8), "\n")
          cat("val_swd:", round(val_result$val_swd, 8), "\n")
          cat("val_centroid_mse:", round(val_result$val_centroid_mse, 8), "\n")
        }
      }, error = \(e) {
        message("  Validation error: ", e$message)
      })
      gc(); cuda_empty_cache()
    }
  }

  gc(); cuda_empty_cache()
}

# Signal training complete for decoupled validator
if (live_dir != "") {
  file.create(file.path(live_dir, "TRAINING_DONE"))
  message("IPC: Wrote TRAINING_DONE")
  flush(stderr())
}

message("\n=== Training complete ===")
message("Final checkpoint directory: ", checkpoint_dir)
