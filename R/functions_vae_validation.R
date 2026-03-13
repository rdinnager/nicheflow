#' VAE Validation and Checkpoint Functions
#'
#' Validation metrics and checkpoint management for the Environmental VAE.
#' Used by train_env_vae.R.

#' Compute VAE validation metrics
#'
#' Iterates through validation data in batches under no_grad(),
#' computing loss, reconstruction, KL, and active dimensions.
#'
#' @param vae The VAE model (on device)
#' @param val_data Validation tensor (CPU)
#' @param device torch device string
#' @param batch_size Batch size for validation
#' @param latent_dim Latent dimension size
#' @return Named list: val_loss, val_recon, val_kl, val_loggamma, active_dims
compute_vae_validation <- function(vae, val_data, device, batch_size, latent_dim) {
  n_val <- val_data$shape[1]
  n_batches <- ceiling(n_val / batch_size)

  total_loss <- 0
  total_recon <- 0
  total_kl <- 0
  total_active <- 0
  total_samples <- 0

  torch::with_no_grad({
    for (b in seq_len(n_batches)) {
      start_idx <- (b - 1) * batch_size + 1
      end_idx <- min(b * batch_size, n_val)
      batch <- val_data[start_idx:end_idx, ]$to(device = device)
      n <- batch$shape[1]

      result <- vae(batch)
      reconstruction <- result[[1]]
      input <- result[[2]]
      means <- result[[3]]
      log_vars <- result[[4]]

      loss_parts <- vae$loss_function(reconstruction, input, means, log_vars)
      loss <- loss_parts[[1]]
      recon <- loss_parts[[2]]
      kl <- loss_parts[[3]]

      # Active dims: where mean posterior variance < 0.5
      active <- as.numeric(
        (torch::torch_exp(log_vars)$mean(dim = 1L) < 0.5)$sum()$cpu()
      )

      total_loss <- total_loss + as.numeric(loss$cpu()) * n
      total_recon <- total_recon + as.numeric(recon$cpu()) * n
      total_kl <- total_kl + as.numeric(kl$cpu()) * n
      total_active <- total_active + active * n
      total_samples <- total_samples + n
    }
  })
  torch::cuda_empty_cache()

  list(
    val_loss = total_loss / total_samples,
    val_recon = total_recon / total_samples,
    val_kl = total_kl / total_samples,
    val_loggamma = as.numeric(vae$loggamma$cpu()),
    active_dims = round(total_active / total_samples)
  )
}


#' Save a model checkpoint via state_dict
#'
#' Saves the model's state_dict to checkpoint_dir/epoch_{N}_model.pt.
#' Uses state_dict instead of whole module to avoid serializing large
#' closure environments (which triggers "long vectors not supported" errors
#' when the training environment holds tensors with >2^31 elements).
#'
#' @param model The nn_module to save
#' @param checkpoint_dir Directory for checkpoints
#' @param epoch Current epoch number
save_model_checkpoint <- function(model, checkpoint_dir, epoch) {
  dir.create(checkpoint_dir, recursive = TRUE, showWarnings = FALSE)
  path <- file.path(checkpoint_dir, sprintf("epoch_%04d_model.pt", epoch))
  torch::torch_save(model$state_dict(), path)
  message("  Checkpoint saved: ", path)
}

# Keep old name as alias for backwards compatibility
save_vae_checkpoint <- save_model_checkpoint


#' Load a model checkpoint from state_dict
#'
#' Loads a state_dict from disk and applies it to a fresh model instance.
#'
#' @param model A fresh nn_module instance (same architecture as saved)
#' @param checkpoint_path Path to the saved state_dict .pt file
#' @return The model with loaded weights (same object, modified in place)
load_model_checkpoint <- function(model, checkpoint_path) {
  state <- torch::torch_load(checkpoint_path)
  model$load_state_dict(state)
  model
}


#' Find latest checkpoint in a directory
#'
#' Scans for files matching pattern and returns the most recent by epoch number.
#'
#' @param checkpoint_dir Directory to scan
#' @param pattern Regex pattern for checkpoint files
#' @return Named list with path and epoch, or NULL if no checkpoints found
find_latest_checkpoint <- function(checkpoint_dir,
                                   pattern = "epoch_(\\d+)_model\\.pt$") {
  if (!dir.exists(checkpoint_dir)) return(NULL)

  files <- list.files(checkpoint_dir, pattern = pattern, full.names = TRUE)
  if (length(files) == 0) return(NULL)

  epochs <- as.integer(
    sub(".*epoch_(\\d+)_model\\.pt$", "\\1", basename(files))
  )
  best <- which.max(epochs)

  list(path = files[best], epoch = epochs[best])
}
