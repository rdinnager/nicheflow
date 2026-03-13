# train_env_vae.R
# Environmental VAE training script
#
# Learns the intrinsic dimensionality of environmental space via trainable
# loggamma. Unconditional VAE (no species conditioning).
#
# Usage:
#   Rscript run.R train_env_vae.R device=cuda:0 num_epochs=500
#   Rscript run.R train_env_vae.R num_epochs=3 batch_size=100000  # smoke test

library(torch)
library(dagnn)
library(zeallot)
if (requireNamespace("conflicted", quietly = TRUE)) {
  conflicted::conflicts_prefer(zeallot::`%<-%`)
  conflicted::conflicts_prefer(base::intersect)
  conflicted::conflicts_prefer(base::setdiff)
  conflicted::conflicts_prefer(base::union)
}
options(torch.serialization_version = 3)

source("R/functions_chelsa_data.R")
source("R/functions_vae_validation.R")

# ===========================================================================
# Parameters
# ===========================================================================

#| param
env_dir <- "output/chelsa_tensors/env_standardized"
#| param
device <- "cuda:0"
#| param
latent_dim <- 16L
#| param
batch_size <- 1000000L
#| param
num_epochs <- 500L
#| param
lr <- 0.0025
#| param
loggamma_init <- -3
#| param
checkpoint_dir <- "output/checkpoints/env_vae"
#| param
checkpoint_every <- 25L
#| param
val_fraction <- 0.05
#| param
val_every <- 25L
#| param
restart_schedule <- TRUE

# ===========================================================================
# Data keys (captured by run_script)
# ===========================================================================

#| data train_epoch train_loss
#| data val_epoch val_loss val_recon val_kl val_loggamma val_active_dims

# ===========================================================================
# Output
# ===========================================================================

#| output
checkpoint_dir_out <- checkpoint_dir

# ===========================================================================
# Model Definition (sourced from shared file)
# ===========================================================================

source("R/functions_env_vae_model.R")

# ===========================================================================
# Data Loading
# ===========================================================================

message("Loading environment tensor from: ", env_dir)
t_load_start <- Sys.time()
env_tensor <- load_large_tensor(env_dir)
input_dim <- env_tensor$shape[2]
n_total <- env_tensor$shape[1]
message("Tensor shape: ", n_total, " x ", input_dim)
message("Load time: ", round(as.numeric(Sys.time() - t_load_start, units = "secs")), "s")

# Train/val split (index-based, no data copy for train)
set.seed(42)
n_val <- as.integer(n_total * val_fraction)
n_train <- n_total - n_val
val_idx <- sort(sample.int(n_total, n_val))
train_idx <- setdiff(seq_len(n_total), val_idx)

message("Train: ", format(n_train, big.mark = ","),
        " | Val: ", format(n_val, big.mark = ","))

# Pre-extract validation data as a separate tensor
message("Extracting validation tensor...")
val_data <- env_tensor[val_idx, ]
message("Validation tensor: ", paste(val_data$shape, collapse = " x "))

# ===========================================================================
# Model Init / Checkpoint Resume
# ===========================================================================

dir.create(checkpoint_dir, recursive = TRUE, showWarnings = FALSE)

checkpoint <- find_latest_checkpoint(checkpoint_dir)
start_epoch <- 0L

# Always create a fresh model first (avoids serializing closure environments)
message("Initializing model: input_dim=", input_dim,
        " latent_dim=", latent_dim, " loggamma_init=", loggamma_init)
vae <- env_vae_mod(input_dim, latent_dim, loggamma_init = loggamma_init)
vae <- vae$to(device = device)

if (!is.null(checkpoint)) {
  message("Loading weights from checkpoint: ", checkpoint$path,
          " (epoch ", checkpoint$epoch, ")")
  load_model_checkpoint(vae, checkpoint$path)
  start_epoch <- checkpoint$epoch
}

# ===========================================================================
# Optimizer & Scheduler
# ===========================================================================

n_batches_per_epoch <- ceiling(n_train / batch_size)
remaining_epochs <- num_epochs - start_epoch

optimizer <- optim_adam(vae$parameters, lr = lr)
scheduler <- lr_one_cycle(
  optimizer, max_lr = lr,
  epochs = remaining_epochs,
  steps_per_epoch = n_batches_per_epoch,
  cycle_momentum = FALSE
)

message("Batches per epoch: ", n_batches_per_epoch,
        " | Remaining epochs: ", remaining_epochs)

# ===========================================================================
# Training Loop
# ===========================================================================

message("\n=== Starting training ===")
message("Device: ", device, " | Epochs: ", start_epoch + 1, "-", num_epochs)

for (epoch in seq(start_epoch + 1, num_epochs)) {
  epoch_start <- Sys.time()
  epoch_loss <- 0
  epoch_samples <- 0

  # Shuffle training indices each epoch
  shuffled <- train_idx[sample.int(n_train)]

  for (b in seq_len(n_batches_per_epoch)) {
    b_start <- (b - 1) * batch_size + 1
    b_end <- min(b * batch_size, n_train)
    batch_idx <- shuffled[b_start:b_end]

    input <- env_tensor[batch_idx, ]$to(device = device)
    n <- input$shape[1]

    optimizer$zero_grad()
    c(reconstruction, input_out, means, log_vars) %<-% vae(input)
    c(loss, recon_loss, kl_loss) %<-% vae$loss_function(
      reconstruction, input_out, means, log_vars
    )

    loss$backward()
    optimizer$step()
    scheduler$step()

    batch_loss <- as.numeric(loss$cpu())
    epoch_loss <- epoch_loss + batch_loss * n
    epoch_samples <- epoch_samples + n

    # Progress to stderr (not captured)
    if (b %% 50 == 0 || b == n_batches_per_epoch) {
      active_dims <- as.numeric(
        (torch_exp(log_vars)$mean(dim = 1L) < 0.5)$sum()$cpu()
      )
      message("  Epoch ", epoch, " batch ", b, "/", n_batches_per_epoch,
              " | loss: ", round(batch_loss, 4),
              " | recon: ", round(as.numeric(recon_loss$cpu()), 5),
              " | KL: ", round(as.numeric(kl_loss$cpu()), 3),
              " | loggamma: ", round(as.numeric(vae$loggamma$cpu()), 3),
              " | active: ", active_dims)
    }
  }

  mean_epoch_loss <- epoch_loss / epoch_samples
  epoch_time <- round(as.numeric(Sys.time() - epoch_start, units = "secs"), 1)

  # Print train data (captured by #| data)
  cat("train_epoch:", epoch, "\n")
  cat("train_loss:", round(mean_epoch_loss, 6), "\n")

  message("Epoch ", epoch, " complete in ", epoch_time, "s",
          " | mean loss: ", round(mean_epoch_loss, 6))

  # Validation
  if (epoch %% val_every == 0 || epoch == num_epochs) {
    message("  Running validation...")
    val_metrics <- compute_vae_validation(
      vae, val_data, device, batch_size, latent_dim
    )

    cat("val_epoch:", epoch, "\n")
    cat("val_loss:", round(val_metrics$val_loss, 6), "\n")
    cat("val_recon:", round(val_metrics$val_recon, 6), "\n")
    cat("val_kl:", round(val_metrics$val_kl, 6), "\n")
    cat("val_loggamma:", round(val_metrics$val_loggamma, 4), "\n")
    cat("val_active_dims:", val_metrics$active_dims, "\n")

    message("  Val loss: ", round(val_metrics$val_loss, 4),
            " | recon: ", round(val_metrics$val_recon, 5),
            " | KL: ", round(val_metrics$val_kl, 3),
            " | loggamma: ", round(val_metrics$val_loggamma, 3),
            " | active dims: ", val_metrics$active_dims)
  }

  # Checkpoint
  if (epoch %% checkpoint_every == 0 || epoch == num_epochs) {
    save_vae_checkpoint(vae, checkpoint_dir, epoch)
  }

  cuda_empty_cache()
}

message("\n=== Training complete ===")
message("Final checkpoint directory: ", checkpoint_dir)
