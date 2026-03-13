# train_nichencoder.R
# NichEncoder Rectified Flow training script
#
# Learns species-specific environmental distributions in VAE latent space
# via a conditional rectified flow (U-Net trajectory network with species
# embeddings). GPU-native Euler ODE integration.
#
# Usage:
#   Rscript run.R train_nichencoder.R device=cuda:0 num_epochs=3000
#   Rscript run.R train_nichencoder.R num_epochs=3 batch_size=10000  # smoke test

library(torch)
library(dagnn)
library(zeallot)
options(torch.serialization_version = 3)

source("R/functions_vae_validation.R")   # for checkpoint functions
source("R/functions_nichencoder.R")

# ===========================================================================
# Parameters
# ===========================================================================

#| param
encoded_train_parquet <- "data/processed/jade_encoded_train.parquet"
#| param
encoded_val_parquet <- "data/processed/jade_encoded_val.parquet"
#| param
species_map_file <- "output/nichencoder_config/species_map.rds"
#| param
device <- "cuda:0"
#| param
spec_embed_dim <- 64L
#| param
breadths <- c(512L, 256L, 128L)
#| param
batch_size <- 500000L
#| param
num_epochs <- 3000L
#| param
lr <- 0.001
#| param
latent_noise_scale <- 0.01
#| param
time_sampling <- "logit_normal"
#| param
time_logit_normal_mean <- 0
#| param
time_logit_normal_sd <- 1
#| param
checkpoint_dir <- "output/checkpoints/nichencoder"
#| param
checkpoint_every <- 50L
#| param
val_every <- 50L
#| param
loss_type <- "pseudo_huber"
#| param
ode_steps <- 500L
#| param
n_cycles <- 2L
#| param
cycle_2_lr_factor <- 0.1
#| param
cycle_1_fraction <- 0.5
#| param
clear_checkpoints <- FALSE
#| param
n_fixed_val_species <- 5L
#| param
n_random_val_species <- 5L
#| param
n_metric_species <- 100L

# ===========================================================================
# Data keys (captured by run_script)
# ===========================================================================

#| data train_epoch train_loss
#| data val_epoch val_loss val_swd val_centroid_mse

# ===========================================================================
# Output
# ===========================================================================

#| output
checkpoint_dir_out <- checkpoint_dir

# ===========================================================================
# Data Loading
# ===========================================================================

message("Loading training data from: ", encoded_train_parquet)
t_load <- Sys.time()

train_data <- arrow::read_parquet(encoded_train_parquet)
latent_cols <- grep("^latent_", names(train_data), value = TRUE)
coord_dim <- length(latent_cols)
message("Latent columns: ", paste(latent_cols, collapse = ", "))
message("Coord dim (active latent dims): ", coord_dim)

# Load species map
species_map <- readRDS(species_map_file)
n_species <- length(species_map)
message("Species map: ", n_species, " species")

# Convert to tensors
train_latent <- torch_tensor(
  as.matrix(train_data[, latent_cols]),
  dtype = torch_float32()
)
train_species_ids <- torch_tensor(
  as.integer(species_map[train_data$species]),
  dtype = torch_long()
)
n_train <- train_latent$shape[1]
message("Training samples: ", format(n_train, big.mark = ","))

# Validation data
message("Loading validation data from: ", encoded_val_parquet)
val_data <- arrow::read_parquet(encoded_val_parquet)
val_latent <- torch_tensor(
  as.matrix(val_data[, latent_cols]),
  dtype = torch_float32()
)
# Map val species; unknown species get NA -> filter them out
val_species_raw <- species_map[val_data$species]
val_known_mask <- !is.na(val_species_raw)
n_unknown <- sum(!val_known_mask)
if (n_unknown > 0) {
  message("Filtering ", n_unknown, " validation samples with unknown species")
  val_latent <- val_latent[val_known_mask, ]
  val_species_raw <- val_species_raw[val_known_mask]
}
val_species_ids <- torch_tensor(
  as.integer(val_species_raw),
  dtype = torch_long()
)
message("Validation samples: ", format(val_latent$shape[1], big.mark = ","),
        " (", length(unique(as.integer(val_species_raw))), " known species)")

# Select fixed validation species (most samples in val set, tracked across epochs)
val_species_counts <- table(as.integer(val_species_raw))
top_species <- as.integer(names(sort(val_species_counts, decreasing = TRUE)))
n_fix <- min(n_fixed_val_species, length(top_species))
fixed_val_species <- top_species[seq_len(n_fix)]
fixed_names <- names(species_map)[match(fixed_val_species, species_map)]
message("Fixed validation species (", n_fix, "): ",
        paste(fixed_names, collapse = ", "))

rm(train_data, val_data, val_species_raw, val_species_counts, top_species)
gc(verbose = FALSE)

load_time <- round(as.numeric(Sys.time() - t_load, units = "secs"))
message("Data load time: ", load_time, "s")

# ===========================================================================
# Batch Index Vectors (memory-efficient, from GeODE pattern)
# ===========================================================================

message("Shuffling and creating batch index vectors...")
shuffled_idx <- sample.int(n_train)
n_batches <- ceiling(n_train / batch_size)

batch_indices <- vector("list", n_batches)
for (b in seq_len(n_batches)) {
  b_start <- (b - 1) * batch_size + 1
  b_end <- min(b * batch_size, n_train)
  batch_indices[[b]] <- shuffled_idx[b_start:b_end]
}
rm(shuffled_idx)
gc(verbose = FALSE)
message("Created ", n_batches, " batch index vectors of ~",
        format(batch_size, big.mark = ","), " samples")

# ===========================================================================
# Model Init / Checkpoint Resume
# ===========================================================================

dir.create(checkpoint_dir, recursive = TRUE, showWarnings = FALSE)

# Clear checkpoints if requested (fresh start with new architecture)
if (clear_checkpoints && dir.exists(checkpoint_dir)) {
  message("Clearing checkpoint directory: ", checkpoint_dir)
  unlink(list.files(checkpoint_dir, full.names = TRUE))
}

# Two copies: model (for checkpointing/ODE) and model_jit (for training)
checkpoint <- find_latest_checkpoint(checkpoint_dir)
start_epoch <- 0L

message("Initializing NichEncoderTrajNet: coord_dim=", coord_dim,
        ", n_species=", n_species, ", spec_embed_dim=", spec_embed_dim)
model <- nichencoder_traj_net(
  coord_dim = coord_dim, n_species = n_species,
  spec_embed_dim = spec_embed_dim, breadths = breadths,
  model_device = device, loss_type = loss_type
)
model_jit <- nichencoder_traj_net(
  coord_dim = coord_dim, n_species = n_species,
  spec_embed_dim = spec_embed_dim, breadths = breadths,
  model_device = device, loss_type = loss_type
)
model <- model$to(device = device)
model_jit <- model_jit$to(device = device)

if (!is.null(checkpoint)) {
  message("Loading weights from checkpoint: ", checkpoint$path,
          " (epoch ", checkpoint$epoch, ")")
  load_model_checkpoint(model, checkpoint$path)
  model_jit$load_state_dict(model$state_dict())
  start_epoch <- checkpoint$epoch
}

# JIT trace components for training speed
# species_embedding stays outside JIT (integer input), encode_spec gets
# the embedded float tensor
message("JIT tracing model components...")
trace_idx <- batch_indices[[1]][seq_len(min(1000L, length(batch_indices[[1]])))]
test_batch <- generate_nichencoder_training_data(
  train_latent[trace_idx, ], train_species_ids[trace_idx],
  device = device, noise_scale = latent_noise_scale,
  time_sampling = time_sampling,
  time_logit_normal_mean = time_logit_normal_mean,
  time_logit_normal_sd = time_logit_normal_sd
)

# Embed species for JIT trace inputs
test_spec_emb <- model_jit$species_embedding(test_batch$spec_ids)
test_spec_enc <- model_jit$encode_spec(test_spec_emb)

model_jit$encode_t <- jit_trace(model_jit$encode_t, test_batch$t)
model_jit$encode_spec <- jit_trace(model_jit$encode_spec, test_spec_emb)
model_jit$unet <- jit_trace(
  model_jit$unet, test_batch$coords,
  model_jit$encode_t(test_batch$t),
  test_spec_enc
)
message("JIT tracing complete")

rm(test_batch, test_spec_emb, test_spec_enc)
gc(verbose = FALSE)

# ===========================================================================
# Optimizer & Scheduler (Two-Cycle LR with Optimizer Reset)
# ===========================================================================

remaining_epochs <- num_epochs - start_epoch

# Compute cycle boundaries
cycle_1_end <- as.integer(num_epochs * cycle_1_fraction)

# Determine which cycle we are in (for checkpoint resume)
if (start_epoch < cycle_1_end) {
  current_cycle <- 1L
} else {
  current_cycle <- 2L
}

if (current_cycle == 1L) {
  cycle_1_remaining <- cycle_1_end - start_epoch
  optimizer <- optim_adamw(model_jit$parameters, lr = lr, weight_decay = 0.01)
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
  optimizer <- optim_adamw(model_jit$parameters, lr = cycle_2_lr, weight_decay = 0.01)
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
# Training Loop
# ===========================================================================

message("\n=== Starting training ===")
message("Device: ", device, " | Epochs: ", start_epoch + 1, "-", num_epochs)

for (epoch in seq(start_epoch + 1, num_epochs)) {
  epoch_start <- Sys.time()
  epoch_loss <- 0

  for (b in seq_len(n_batches)) {
    optimizer$zero_grad()

    idx <- batch_indices[[b]]
    batch <- generate_nichencoder_training_data(
      train_latent[idx, ], train_species_ids[idx],
      device = device, noise_scale = latent_noise_scale,
      time_sampling = time_sampling,
      time_logit_normal_mean = time_logit_normal_mean,
      time_logit_normal_sd = time_logit_normal_sd
    )

    # Embed species (outside JIT) then encode
    spec_emb <- model_jit$species_embedding(batch$spec_ids)
    spec_enc <- model_jit$encode_spec(spec_emb)

    # Forward through JIT-traced model
    output <- model_jit(batch$coords, batch$t, spec_enc)
    loss <- model_jit$loss_function(output, batch$target)

    loss$backward()
    optimizer$step()
    scheduler$step()

    batch_loss <- as.numeric(loss$cpu())
    epoch_loss <- epoch_loss + batch_loss

    if (b %% 10 == 0 || b == n_batches) {
      message("  Epoch ", epoch, " batch ", b, "/", n_batches,
              " | loss: ", round(batch_loss, 6))
    }
  }

  mean_epoch_loss <- epoch_loss / n_batches
  epoch_time <- round(as.numeric(Sys.time() - epoch_start, units = "secs"), 1)

  # Print train data (captured by #| data)
  cat("train_epoch:", epoch, "\n")
  cat("train_loss:", round(mean_epoch_loss, 8), "\n")

  message("Epoch ", epoch, " complete in ", epoch_time, "s",
          " | mean loss: ", round(mean_epoch_loss, 6))

  # Cycle transition: reset optimizer at cycle boundary
  if (n_cycles >= 2L && epoch == cycle_1_end) {
    message("\n=== Cycle 1 complete, resetting optimizer for Cycle 2 ===")
    cycle_2_lr <- lr * cycle_2_lr_factor
    cycle_2_epochs <- num_epochs - cycle_1_end
    message("Cycle 2: epochs ", cycle_1_end + 1, "-", num_epochs,
            " | lr=", cycle_2_lr, " | ", cycle_2_epochs, " epochs")
    optimizer <- optim_adamw(model_jit$parameters, lr = cycle_2_lr, weight_decay = 0.01)
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
    model$load_state_dict(model_jit$state_dict())
    save_model_checkpoint(model, checkpoint_dir, epoch)
  }

  # Validation
  if (epoch %% val_every == 0 || epoch == num_epochs) {
    message("  Running validation...")
    model$load_state_dict(model_jit$state_dict())
    tryCatch({
      val_result <- validate_nichencoder(
        model, val_latent, val_species_ids, species_map,
        fixed_species_ids = fixed_val_species,
        n_random_plot_species = n_random_val_species,
        n_metric_species = n_metric_species,
        device = device, ode_steps = ode_steps,
        checkpoint_dir = checkpoint_dir, epoch = epoch
      )
      cat("val_epoch:", epoch, "\n")
      cat("val_loss:", round(val_result$val_loss, 8), "\n")
      cat("val_swd:", round(val_result$val_swd, 8), "\n")
      cat("val_centroid_mse:", round(val_result$val_centroid_mse, 8), "\n")
    }, error = \(e) {
      message("  Validation error: ", e$message)
    })
    gc(); cuda_empty_cache()
  }

  gc(); cuda_empty_cache()
}

message("\n=== Training complete ===")
message("Final checkpoint directory: ", checkpoint_dir)
