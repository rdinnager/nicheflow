# train_geode.R
# GeODE Rectified Flow training script
#
# Learns to map environment -> geographic coordinates via a U-Net trajectory
# network trained with rectified flow (straight-line ODE paths from noise
# to coordinates, conditioned on environment).
#
# Usage:
#   Rscript run.R train_geode.R device=cuda:1 num_epochs=2000
#   Rscript run.R train_geode.R num_epochs=3 batch_size=100000  # smoke test

library(torch)
library(dagnn)
library(zeallot)
options(torch.serialization_version = 3)

source("R/functions_chelsa_data.R")
source("R/functions_vae_validation.R")   # for find_latest_checkpoint()
source("R/functions_geode_validation.R")

# ===========================================================================
# Parameters
# ===========================================================================

#| param
env_dir <- "output/chelsa_tensors/env_standardized"
#| param
xy_dir <- "output/chelsa_tensors/xy_standardized"
#| param
xy_raw_dir <- "output/chelsa_tensors/xy_coords"
#| param
xy_mean_sd_file <- NULL
#| param
device <- "cuda:1"
#| param
batch_size <- 300000L
#| param
num_epochs <- 400L
#| param
lr <- 0.001
#| param
checkpoint_dir <- "output/checkpoints/geode"
#| param
checkpoint_every <- 25L
#| param
val_every <- 50L
#| param
n_fixed_ecoregions <- 2L
#| param
n_random_ecoregions <- 3L
#| param
ecoregion_path <- "/blue/rdinnage.fiu/rdinnage.fiu/Data/SDM/maps/ecoregions_valid.rds"
#| param
xy_noise_scale <- 0.0001890600
#| param
env_noise_scale <- 0.01
#| param
restart_schedule <- TRUE

# ===========================================================================
# Data keys (captured by run_script)
# ===========================================================================

#| data train_epoch train_loss

# ===========================================================================
# Output
# ===========================================================================

#| output
checkpoint_dir_out <- checkpoint_dir

# ===========================================================================
# Model Definition
# ===========================================================================

traj_net <- nn_module("TrajNet",
  initialize = function(coord_dim, env_dim, breadths = c(512, 256, 128),
                        t_encode = 32L, env_encode = 64L,
                        model_device = "cuda:1") {
    if (length(breadths) != 3) stop("breadths should be length 3!")

    self$coord_dim <- coord_dim
    self$model_device <- model_device

    self$encode_t <- nn_linear(1L, t_encode)
    self$encode_env <- nn_linear(env_dim, env_encode)

    self$unet <- nndag(
      input = ~ coord_dim,
      t_encoded = ~ t_encode,
      env_encoded = ~ env_encode,
      e_1 = input + t_encoded + env_encoded ~ breadths[1],
      e_2 = e_1 + t_encoded + env_encoded ~ breadths[2],
      e_3 = e_2 + t_encoded + env_encoded ~ breadths[3],
      d_1 = e_3 + t_encoded + env_encoded ~ breadths[3],
      d_2 = d_1 + e_3 + t_encoded + env_encoded ~ breadths[2],
      d_3 = d_2 + e_2 + t_encoded + env_encoded ~ breadths[1],
      output = d_3 + e_1 + t_encoded + env_encoded ~ coord_dim,
      .act = list(nn_relu, output = nn_identity)
    )

    self$loss_function <- function(input, target) {
      torch_mean((target - input)^2)
    }

    # GPU-native Euler integration (much faster than deSolve for rectified flows)
    self$sample_trajectory <- function(initial_vals, env_vals, steps = 200L) {
      with_no_grad({
        dt <- 1.0 / steps
        n <- initial_vals$shape[1]
        y <- initial_vals$detach()

        # Precompute env encoding once (same for all steps)
        env_enc <- self$encode_env(env_vals$detach())

        for (s in seq_len(steps)) {
          t_val <- (s - 1) * dt
          t_tensor <- torch_full(c(n, 1L), t_val,
                                 device = y$device)
          t_enc <- self$encode_t(t_tensor)
          velocity <- self$unet(
            input = y, t_encoded = t_enc,
            env_encoded = env_enc
          )
          y <- y + velocity * dt
        }

        # Return final coordinates (single GPU→CPU transfer)
        y$cpu()
      })
    }
  },

  forward = function(coords, t, env) {
    t_encoded <- self$encode_t(t)
    env_encoded <- self$encode_env(env)
    self$unet(input = coords, t_encoded = t_encoded, env_encoded = env_encoded)
  }
)


#' Generate training data for rectified flow
#'
#' @param xy Batch of standardized XY coordinates (CPU tensor)
#' @param env Batch of standardized env variables (CPU tensor)
#' @param xy_noise Noise scale for XY (half the grid resolution in std units)
#' @param env_noise Noise scale for env
#' @param device Target device
#' @param x_sample_fun Function to sample x noise
#' @param y_sample_fun Function to sample y noise
#' @param ... Additional args passed to sample functions
generate_training_data <- function(xy, env,
                                   xy_noise = 0.0001890600,
                                   env_noise = 0.01,
                                   device = "cuda:1",
                                   x_sample_fun = rnorm,
                                   y_sample_fun = rnorm, ...) {
  with_no_grad({
    n <- xy$size()[1]
    t <- torch_rand(n, 1, device = device)
    z <- torch_tensor(cbind(x_sample_fun(n, ...), y_sample_fun(n, ...)),
                      device = device)

    # Add grid noise to xy to avoid overfitting to discrete raster cells
    target <- (xy$to(device = device) +
                 (torch_rand_like(xy, device = device) * xy_noise)) - z
    coords <- z + t * target
  })

  # Add noise to env for robustness
  list(
    coords = coords,
    t = t,
    env = env$to(device = device) +
      torch_randn_like(env, device = device) * env_noise,
    target = target
  )
}

# ===========================================================================
# Data Loading
# ===========================================================================

message("Loading environment tensor from: ", env_dir)
t_load <- Sys.time()
env_tensor <- load_large_tensor(env_dir)
env_dim <- env_tensor$shape[2]
n_total <- env_tensor$shape[1]
message("Env tensor: ", n_total, " x ", env_dim)

message("Loading XY tensor from: ", xy_dir)
xy_tensor <- load_large_tensor(xy_dir)
message("XY tensor: ", paste(xy_tensor$shape, collapse = " x "))

message("Loading raw XY tensor from: ", xy_raw_dir)
xy_raw_tensor <- load_large_tensor(xy_raw_dir)
# Extract as R vectors for spatial operations
xy_raw_lon <- as.numeric(xy_raw_tensor[, 1]$cpu())
xy_raw_lat <- as.numeric(xy_raw_tensor[, 2]$cpu())
rm(xy_raw_tensor)
gc(verbose = FALSE)
message("Raw XY loaded as vectors: lon range [",
        round(min(xy_raw_lon), 2), ", ", round(max(xy_raw_lon), 2), "]")

message("Load time: ", round(as.numeric(Sys.time() - t_load, units = "secs")), "s")

# Load XY mean/sd for un-standardizing predictions
if (!is.null(xy_mean_sd_file) && file.exists(xy_mean_sd_file)) {
  xy_mean_sd <- readRDS(xy_mean_sd_file)
  message("Loaded XY mean/sd from: ", xy_mean_sd_file)
} else {
  # Compute from raw if file not provided
  message("Computing XY mean/sd from raw coordinates...")
  xy_mean_sd <- list(
    mean = c(mean(xy_raw_lon), mean(xy_raw_lat)),
    sd = c(sd(xy_raw_lon), sd(xy_raw_lat))
  )
}

# Noise sampling functions scaled to standardized coordinate SDs
x_sd <- 1.0  # standardized, so SD ~1
y_sd <- 1.0
x_samp <- function(n) rnorm(n, sd = x_sd * 1.1)
y_samp <- function(n) rnorm(n, sd = y_sd * 1.1)

# ===========================================================================
# Pre-split into batch index vectors (memory-efficient)
# ===========================================================================
# Instead of copying tensor slices into batches (~40 GB), we store only
# integer index vectors (~2.4 GB) and index into the original tensors
# during training. This halves peak RAM usage.

message("Shuffling and creating batch index vectors...")
shuffled_idx <- sample.int(n_total)
n_batches <- ceiling(n_total / batch_size)

batch_indices <- vector("list", n_batches)
for (b in seq_len(n_batches)) {
  b_start <- (b - 1) * batch_size + 1
  b_end <- min(b * batch_size, n_total)
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

# We keep two copies: trajnet (for checkpointing/ODE) and trajnet_jit (for training)
checkpoint <- find_latest_checkpoint(checkpoint_dir)
start_epoch <- 0L

# Always create fresh models (avoids serializing closure environments with large tensors)
message("Initializing TrajNet: coord_dim=2, env_dim=", env_dim)
trajnet <- traj_net(coord_dim = 2L, env_dim = env_dim, model_device = device)
trajnet_jit <- traj_net(coord_dim = 2L, env_dim = env_dim, model_device = device)
trajnet <- trajnet$to(device = device)
trajnet_jit <- trajnet_jit$to(device = device)

if (!is.null(checkpoint)) {
  message("Loading weights from checkpoint: ", checkpoint$path,
          " (epoch ", checkpoint$epoch, ")")
  load_model_checkpoint(trajnet, checkpoint$path)
  trajnet_jit$load_state_dict(trajnet$state_dict())
  start_epoch <- checkpoint$epoch
}

# JIT trace the components for training speed (use small sample to save VRAM)
message("JIT tracing model components...")
trace_idx <- batch_indices[[1]][seq_len(min(1000L, length(batch_indices[[1]])))]
test_batch <- generate_training_data(
  xy_tensor[trace_idx, ], env_tensor[trace_idx, ],
  xy_noise = xy_noise_scale, env_noise = env_noise_scale,
  device = device, x_sample_fun = x_samp, y_sample_fun = y_samp
)
trajnet_jit$encode_env <- jit_trace(trajnet_jit$encode_env, test_batch$env)
trajnet_jit$encode_t <- jit_trace(trajnet_jit$encode_t, test_batch$t)
trajnet_jit$unet <- jit_trace(
  trajnet_jit$unet, test_batch$coords,
  trajnet_jit$encode_t(test_batch$t),
  trajnet_jit$encode_env(test_batch$env)
)
message("JIT tracing complete")

# ===========================================================================
# Optimizer & Scheduler
# ===========================================================================

remaining_epochs <- num_epochs - start_epoch

optimizer <- optim_adamw(trajnet_jit$parameters, lr = lr, weight_decay = 0.01)

if (start_epoch > 0 && restart_schedule) {
  # Fresh one-cycle schedule over remaining epochs only
  message("Restarting lr schedule for ", remaining_epochs, " remaining epochs")
  scheduler <- lr_one_cycle(
    optimizer, max_lr = lr,
    epochs = remaining_epochs,
    steps_per_epoch = n_batches,
    cycle_momentum = FALSE
  )
} else {
  # Full schedule over all epochs
  scheduler <- lr_one_cycle(
    optimizer, max_lr = lr,
    epochs = num_epochs,
    steps_per_epoch = n_batches,
    cycle_momentum = FALSE
  )
  # Step forward to resume position
  if (start_epoch > 0) {
    message("Stepping scheduler forward ",
            start_epoch * n_batches, " steps...")
    for (s in seq_len(start_epoch * n_batches)) scheduler$step()
  }
}

message("Batches per epoch: ", n_batches,
        " | Remaining epochs: ", remaining_epochs)

# ===========================================================================
# Ecoregion Validation Setup
# ===========================================================================

ecoregions_loaded <- FALSE
fixed_ecoregions <- NULL
all_ecoregions <- NULL

if (file.exists(ecoregion_path) && val_every <= num_epochs) {
  tryCatch({
    message("Loading ecoregions from: ", ecoregion_path)
    all_ecoregions <- load_ecoregions(ecoregion_path)
    message("Loaded ", nrow(all_ecoregions), " ecoregions")

    # Fixed ecoregions: Southwest Australia woodlands + Hispaniolan pine forests
    # (diverse biomes, moderate size, tracked across all validation rounds)
    fixed_ids <- c(206L, 555L)
    fixed_ecoregions <- all_ecoregions[
      all_ecoregions$ECO_ID %in% fixed_ids,
    ]
    message("Fixed validation ecoregions:")
    for (i in seq_len(nrow(fixed_ecoregions))) {
      message("  ", i, ": ", fixed_ecoregions$ECO_NAME[i],
              " (biome ", fixed_ecoregions$BIOME_NUM[i], ")")
    }
    ecoregions_loaded <- TRUE
  }, error = \(e) {
    message("Warning: Could not load ecoregions: ", e$message)
    message("Ecoregion validation will be skipped.")
  })
}

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
    batch <- generate_training_data(
      xy_tensor[idx, ], env_tensor[idx, ],
      xy_noise = xy_noise_scale, env_noise = env_noise_scale,
      device = device,
      x_sample_fun = x_samp, y_sample_fun = y_samp
    )

    output <- trajnet_jit(batch$coords, batch$t, batch$env)
    loss <- trajnet_jit$loss_function(output, batch$target)

    loss$backward()
    optimizer$step()
    scheduler$step()

    batch_loss <- as.numeric(loss$cpu())
    epoch_loss <- epoch_loss + batch_loss

    # Progress to stderr (not captured)
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

  # Checkpoint
  if (epoch %% checkpoint_every == 0 || epoch == num_epochs) {
    message("  Saving checkpoint...")
    trajnet$load_state_dict(trajnet_jit$state_dict())
    save_model_checkpoint(trajnet, checkpoint_dir, epoch)
  }

  # Ecoregion validation
  if (ecoregions_loaded &&
      (epoch %% val_every == 0 || epoch == num_epochs)) {
    message("  Running ecoregion validation...")

    # Use non-JIT model for validation (avoids JIT VRAM accumulation)
    trajnet$load_state_dict(trajnet_jit$state_dict())
    tryCatch({
      generate_ecoregion_zoom_plots(
        model = trajnet,
        fixed_ecoregions = fixed_ecoregions,
        all_ecoregions = all_ecoregions,
        env_tensor = env_tensor,
        xy_raw_lon = xy_raw_lon,
        xy_raw_lat = xy_raw_lat,
        checkpoint_dir = checkpoint_dir,
        epoch = epoch,
        xy_mean_sd = xy_mean_sd,
        device = device,
        x_sd = x_sd, y_sd = y_sd,
        n_random = n_random_ecoregions,
        n_sample = 5000,
        ode_steps = 200
      )
    }, error = \(e) {
      message("  Ecoregion validation error: ", e$message)
    })
    gc(); cuda_empty_cache()
  }

  gc(); cuda_empty_cache()
}

message("\n=== Training complete ===")
message("Final checkpoint directory: ", checkpoint_dir)
