#' VAE Encoding Functions
#'
#' Functions for encoding JADE samples through the trained EnvVAE,
#' detecting active latent dimensions, and building species ID maps.


#' Detect active latent dimensions from the VAE's original training data
#'
#' Encodes a random subset of the global standardized environmental tensor
#' through the VAE encoder and determines which latent dimensions are "active"
#' (posterior variance significantly below prior variance of 1).
#'
#' @param checkpoint_path Path to the VAE checkpoint (.pt state_dict)
#' @param env_tensor_dir Directory containing the standardized env tensor
#'   (per-column .pt files from the CHELSA pipeline)
#' @param latent_dim VAE latent dimension size (e.g. 16)
#' @param device Torch device string (e.g. "cuda:0")
#' @param batch_size Batch size for encoding
#' @param n_samples Number of random rows to encode (default 1M)
#' @param threshold Active dimension threshold: dims where
#'   mean(exp(logvar)) < threshold are considered active (default 0.5)
#' @return Integer vector of active dimension indices (1-based)
detect_active_dims <- function(checkpoint_path, env_tensor_dir,
                               latent_dim = 16L, device = "cuda:0",
                               batch_size = 500000L, n_samples = 1000000L,
                               threshold = 0.5) {

  message("=== Detecting active latent dimensions ===")
  message("Checkpoint: ", checkpoint_path)
  message("Env tensor: ", env_tensor_dir)
  message("Samples: ", format(n_samples, big.mark = ","),
          " | Threshold: ", threshold)

  # Load only the subset we need (column-wise to avoid full 38GB load)
  meta <- readRDS(file.path(env_tensor_dir, "meta.rds"))
  input_dim <- meta$n_cols
  n_total <- meta$shape[1]

  set.seed(42)
  n_use <- min(n_samples, n_total)
  sample_idx <- sort(sample.int(n_total, n_use))
  message("Using ", format(n_use, big.mark = ","), " of ",
          format(n_total, big.mark = ","), " rows")

  # Load subset column-by-column (memory efficient: ~240MB for 1M x 31 float32)
  message("Loading ", input_dim, " columns for sampled rows...")
  env_subset <- torch::torch_zeros(n_use, input_dim,
                                    dtype = torch::torch_float32())
  for (i in seq_len(input_dim)) {
    col_path <- file.path(env_tensor_dir, sprintf("col_%03d.pt", i))
    col <- torch::torch_load(col_path)
    env_subset[, i] <- col[sample_idx]
    rm(col)
  }
  gc(verbose = FALSE)
  message("Subset tensor: ", paste(env_subset$shape, collapse = " x "))

  # Initialize and load model
  vae <- env_vae_mod(input_dim, latent_dim)
  vae <- vae$to(device = device)
  load_model_checkpoint(vae, checkpoint_path)
  vae$eval()

  # Encode in batches, accumulate logvar statistics
  n_batches <- ceiling(n_use / batch_size)
  logvar_sum <- torch::torch_zeros(latent_dim, device = device)
  total_encoded <- 0L

  torch::with_no_grad({
    for (b in seq_len(n_batches)) {
      b_start <- (b - 1) * batch_size + 1
      b_end <- min(b * batch_size, n_use)
      batch <- env_subset[b_start:b_end, ]$to(device = device)
      n <- batch$shape[1]

      c(means, logvars) %<-% vae$encoder(batch)
      logvar_sum <- logvar_sum + torch::torch_exp(logvars)$sum(dim = 1L)
      total_encoded <- total_encoded + n

      if (b %% 5 == 0 || b == n_batches) {
        message("  Batch ", b, "/", n_batches, " (", total_encoded, " encoded)")
      }
    }
  })

  rm(env_subset)

  # Compute per-dimension mean of exp(logvar)
  mean_exp_logvar <- logvar_sum / total_encoded
  mean_exp_logvar_cpu <- as.numeric(mean_exp_logvar$cpu())

  message("\nPer-dimension mean(exp(logvar)):")
  for (d in seq_len(latent_dim)) {
    active_flag <- if (mean_exp_logvar_cpu[d] < threshold) " ** ACTIVE **" else ""
    message("  Dim ", d, ": ", round(mean_exp_logvar_cpu[d], 6), active_flag)
  }

  active_dims <- which(mean_exp_logvar_cpu < threshold)
  message("\nActive dimensions: ", length(active_dims), " of ", latent_dim)
  message("Indices: ", paste(active_dims, collapse = ", "))

  # Cleanup
  rm(vae, logvar_sum, mean_exp_logvar)
  gc(verbose = FALSE)
  torch::cuda_empty_cache()

  active_dims
}


#' Encode JADE samples through the VAE into latent space
#'
#' Loads a JADE parquet file, standardizes the environmental columns using the
#' same statistics as the global CHELSA tensor, encodes through the VAE encoder,
#' and returns a tibble with species metadata and latent codes on active dims.
#'
#' @param parquet_path Path to JADE parquet file (jade_31_train.parquet etc.)
#' @param checkpoint_path Path to VAE checkpoint (.pt state_dict)
#' @param env_mean_sd List with `mean` and `sd` numeric vectors (from
#'   compute_standardization on the global CHELSA tensor)
#' @param chelsa_var_meta Data frame with `chelsa_name` column defining variable
#'   order (matches env_mean_sd ordering)
#' @param active_dims Integer vector of active latent dimension indices
#' @param latent_dim VAE latent dimension size (e.g. 16)
#' @param device Torch device string
#' @param batch_size Batch size for encoding
#' @return Tibble with columns: species, taxon, split_type, latent_1..latent_D
encode_jade_through_vae <- function(parquet_path, checkpoint_path,
                                     env_mean_sd, chelsa_var_meta,
                                     active_dims, latent_dim = 16L,
                                     device = "cuda:0",
                                     batch_size = 500000L) {

  message("=== Encoding JADE samples through VAE ===")
  message("Parquet: ", parquet_path)
  message("Active dims: ", paste(active_dims, collapse = ", "))

  # Load parquet
  jade <- arrow::read_parquet(parquet_path)
  n_samples <- nrow(jade)
  message("Loaded ", format(n_samples, big.mark = ","), " samples")

  # Match CHELSA columns in parquet to chelsa_var_meta ordering
  chelsa_names <- chelsa_var_meta$chelsa_name
  parquet_cols <- names(jade)

  # Find parquet column for each chelsa_name (e.g. "bio1" -> "CHELSA_bio1_1981-2010_V.2.1")
  env_col_idx <- purrr::map_int(chelsa_names, \(cn) {
    matches <- which(stringr::str_detect(parquet_cols, paste0("_", cn, "_")))
    if (length(matches) == 0) {
      stop("Could not find parquet column matching chelsa_name: ", cn)
    }
    if (length(matches) > 1) {
      stop("Multiple parquet columns match chelsa_name: ", cn,
           " -> ", paste(parquet_cols[matches], collapse = ", "))
    }
    matches
  })

  env_cols <- parquet_cols[env_col_idx]
  message("Matched ", length(env_cols), " environmental columns")

  # Extract env matrix in correct order
  env_mat <- as.matrix(jade[, env_cols])

  # Standardize using same stats as global CHELSA tensor
  n_vars <- length(env_mean_sd$mean)
  if (ncol(env_mat) != n_vars) {
    stop("Column count mismatch: parquet has ", ncol(env_mat),
         " env columns but env_mean_sd has ", n_vars)
  }

  for (i in seq_len(n_vars)) {
    env_mat[, i] <- (env_mat[, i] - env_mean_sd$mean[i]) / env_mean_sd$sd[i]
  }

  # Replace NAs with 0 (same convention as VAE training)
  n_na <- sum(is.na(env_mat))
  if (n_na > 0) {
    message("Replacing ", format(n_na, big.mark = ","), " NAs with 0")
    env_mat[is.na(env_mat)] <- 0
  }

  # Convert to tensor
  env_tensor <- torch::torch_tensor(env_mat, dtype = torch::torch_float32())
  rm(env_mat)
  gc(verbose = FALSE)

  # Load VAE
  input_dim <- length(env_mean_sd$mean)
  vae <- env_vae_mod(input_dim, latent_dim)
  vae <- vae$to(device = device)
  load_model_checkpoint(vae, checkpoint_path)
  vae$eval()

  # Encode in batches
  n_batches <- ceiling(n_samples / batch_size)
  n_active <- length(active_dims)
  latent_matrix <- matrix(0, nrow = n_samples, ncol = n_active)

  torch::with_no_grad({
    for (b in seq_len(n_batches)) {
      b_start <- (b - 1) * batch_size + 1
      b_end <- min(b * batch_size, n_samples)
      batch <- env_tensor[b_start:b_end, ]$to(device = device)

      c(means, logvars) %<-% vae$encoder(batch)
      # Extract only active dimensions
      latent_matrix[b_start:b_end, ] <- as.matrix(means[, active_dims]$cpu())

      if (b %% 5 == 0 || b == n_batches) {
        message("  Batch ", b, "/", n_batches)
      }
    }
  })

  # Build output tibble
  latent_names <- paste0("latent_", seq_len(n_active))
  latent_df <- tibble::as_tibble(latent_matrix, .name_repair = "minimal")
  names(latent_df) <- latent_names

  result <- dplyr::bind_cols(
    jade |> dplyr::select(dplyr::any_of(c("species", "taxon", "split_type"))),
    latent_df
  )

  message("Encoding complete: ", nrow(result), " samples x ",
          n_active, " active latent dims")

  # Cleanup
  rm(env_tensor, vae, latent_matrix)
  gc(verbose = FALSE)
  torch::cuda_empty_cache()

  result
}


#' Build species-to-integer-ID mapping from encoded data
#'
#' Creates a named integer vector mapping species names to 1-based IDs
#' for use with nn_embedding in the NichEncoder.
#'
#' @param encoded_data Tibble with a `species` column
#' @return Named integer vector (species_name -> integer_id)
build_species_id_map <- function(encoded_data) {
  species_names <- sort(unique(encoded_data$species))
  species_map <- seq_along(species_names)
  names(species_map) <- species_names

  message("Species ID map: ", length(species_map), " species")
  message("  First: ", names(species_map)[1], " -> ", species_map[1])
  message("  Last: ", names(species_map)[length(species_map)],
          " -> ", species_map[length(species_map)])

  species_map
}
