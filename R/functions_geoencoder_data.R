#' Geo-Encoder Data Functions
#'
#' Corruption/augmentation functions for generating training data, embedding
#' extraction from trained NichEncoder, and dataset assembly for the
#' geo-encoder model.
#'
#' @author rdinnage


#' Extract species embeddings from trained NichEncoder checkpoint
#'
#' Loads the NichEncoder state_dict and extracts the learned species
#' embedding weight matrix (n_species x spec_embed_dim).
#'
#' @param checkpoint_dir Path to NichEncoder checkpoint directory
#' @param species_map_file Path to species_map.rds
#' @param coord_dim Latent coordinate dimension (must match trained model)
#' @param n_species Number of species (must match trained model)
#' @param spec_embed_dim Species embedding dimension (default 64)
#' @param breadths U-Net breadth vector (must match trained model)
#' @return List with embedding_matrix (n_species x spec_embed_dim matrix)
#'   and species_map (named integer vector)
#' @export
extract_nichencoder_embeddings <- function(checkpoint_dir,
                                           species_map_file,
                                           coord_dim = 6L,
                                           n_species = 18121L,
                                           spec_embed_dim = 64L,
                                           breadths = c(512L, 256L, 128L)) {
  source("R/functions_vae_validation.R")
  source("R/functions_nichencoder.R")

  species_map <- readRDS(species_map_file)

  checkpoint <- find_latest_checkpoint(checkpoint_dir)
  if (is.null(checkpoint)) {
    stop("No checkpoint found in: ", checkpoint_dir)
  }
  message("Loading NichEncoder checkpoint: ", checkpoint$path,
          " (epoch ", checkpoint$epoch, ")")

  # Build model with identical architecture to load state_dict
  model <- nichencoder_traj_net(
    coord_dim = coord_dim, n_species = n_species,
    spec_embed_dim = spec_embed_dim, breadths = breadths,
    model_device = "cpu", loss_type = "pseudo_huber"
  )

  load_model_checkpoint(model, checkpoint$path)

  # Extract embedding weight matrix
  emb_weight <- as.matrix(model$species_embedding$weight$cpu())
  rownames(emb_weight) <- names(species_map)[order(species_map)]

  message("Extracted embedding matrix: ", nrow(emb_weight), " x ",
          ncol(emb_weight))

  list(
    embedding_matrix = emb_weight,
    species_map = species_map
  )
}


#' Sample corrupted coordinate sets for one species
#'
#' Resamples from existing JADE-sampled coordinates with GBIF observation bias
#' weighting, spatial block removal, and realistic chunk sizing (Beta
#' distribution matching empirical GBIF per-species record counts).
#'
#' @param species_coords data.frame with X, Y, species, taxon columns
#' @param bias_xy matrix of GBIF bias point coordinates (columns X, Y)
#' @param n_versions Number of corrupted versions to generate (default 20)
#' @param min_n Minimum dataset size when chopping (default 4)
#' @param max_n Maximum dataset size when chopping (default 1000)
#' @param beta_alpha Alpha parameter for Beta chunk size distribution
#' @param beta_beta Beta parameter for Beta chunk size distribution
#' @param bias_levels Vector of bias proportions to sample from
#' @param no_block_frac Fraction of versions with no block removal (default 0.3)
#' @param max_blocks Maximum number of blocks per version (default 3)
#' @param min_remaining_points Minimum points guaranteed after block removal
#' @return Tibble with species, version_id, set_id, bias_level,
#'   block_removed, n_blocks, frac_removed, X, Y
#' @export
sample_corrupted_coords <- function(species_coords, bias_xy,
                                    n_versions = 20L,
                                    min_n = 4L, max_n = 1000L,
                                    beta_alpha = 1.3, beta_beta = 20,
                                    bias_levels = c(0, 0.05, 0.25, 0.5, 0.75, 0.95),
                                    no_block_frac = 0.3,
                                    max_blocks = 3L,
                                    min_remaining_points = 10L) {

  spec_name <- species_coords$species[1]
  pool_coords <- species_coords[, c("X", "Y"), drop = FALSE]
  n_actual <- nrow(pool_coords)
  if (n_actual == 0) return(tibble())

  # --- Compute equal-area projection from bounding box ---
  xr <- range(pool_coords$X)
  yr <- range(pool_coords$Y)
  bb_poly <- st_as_sfc(st_bbox(c(xmin = xr[1], ymin = yr[1],
                                   xmax = xr[2], ymax = yr[2]),
                                 crs = 4326))
  projection <- find_equal_area_projection(bb_poly)
  bb_proj <- st_transform(bb_poly, crs = projection)
  bb_m <- st_bbox(bb_proj)
  range_extent <- sqrt((bb_m$xmax - bb_m$xmin)^2 +
                        (bb_m$ymax - bb_m$ymin)^2)

  # --- Compute KDE weights from GBIF bias points ---
  kde_weights <- rep(1, n_actual)
  has_kde <- FALSE

  # Filter bias points by expanded bounding box (bias_xy is a plain matrix)
  x_expand <- (xr[2] - xr[1]) * 0.05
  y_expand <- (yr[2] - yr[1]) * 0.05
  in_bbox <- bias_xy[, 1] >= (xr[1] - x_expand) &
             bias_xy[, 1] <= (xr[2] + x_expand) &
             bias_xy[, 2] >= (yr[1] - y_expand) &
             bias_xy[, 2] <= (yr[2] + y_expand)
  bias_local <- bias_xy[in_bbox, , drop = FALSE]

  if (nrow(bias_local) > 10) {
    # Project bias points and species coords to equal-area CRS
    bias_sf <- st_as_sf(as.data.frame(bias_local),
                         coords = c("X", "Y"), crs = 4326)
    bias_proj <- st_coordinates(st_transform(bias_sf, crs = projection))

    pool_sf <- st_as_sf(pool_coords, coords = c("X", "Y"), crs = 4326)
    pool_proj <- st_coordinates(st_transform(pool_sf, crs = projection))

    bxr <- range(bias_proj[, 1])
    byr <- range(bias_proj[, 2])
    bx_exp <- (bxr[2] - bxr[1]) * 0.01
    by_exp <- (byr[2] - byr[1]) * 0.01

    dens <- possibly(MASS::kde2d)(
      bias_proj[, 1], bias_proj[, 2], n = 100,
      lims = c(bxr + c(-bx_exp, bx_exp), byr + c(-by_exp, by_exp))
    )
    if (!is.null(dens)) {
      dens_rast <- rast(dens)
      pool_vect <- terra::vect(pool_proj, type = "points")
      terra::crs(pool_vect) <- terra::crs(dens_rast)
      kde_vals <- terra::extract(dens_rast, pool_vect)
      kde_weights <- kde_vals$lyr.1
      kde_weights[is.na(kde_weights)] <- 0
      if (sum(kde_weights > 0) > 0) {
        has_kde <- TRUE
      } else {
        kde_weights <- rep(1, n_actual)
      }
    }
  }

  # --- Generate versions by subsampling the pool ---
  all_results <- vector("list", n_versions)

  for (v in seq_len(n_versions)) {
    prop_biased <- sample(bias_levels, 1)
    if (!has_kde) prop_biased <- 0

    # Compute per-point sampling weights: mix of uniform and KDE
    uniform_w <- rep(1 / n_actual, n_actual)
    if (prop_biased > 0) {
      kde_w <- kde_weights / sum(kde_weights)
      mix_w <- (1 - prop_biased) * uniform_w + prop_biased * kde_w
    } else {
      mix_w <- uniform_w
    }

    # Subsample from pool (with replacement to allow varied subsets)
    n_sub <- sample(ceiling(n_actual * 0.3):n_actual, 1)
    idx <- sample.int(n_actual, n_sub, replace = TRUE, prob = mix_w)
    coords_df <- pool_coords[idx, , drop = FALSE]

    if (nrow(coords_df) == 0) next

    # Apply spatial block removal (70% of versions)
    do_blocks <- runif(1) > no_block_frac
    n_blocks_applied <- 0L
    frac_removed <- 0

    if (do_blocks && nrow(coords_df) > min_remaining_points) {
      n_blocks_applied <- sample.int(max_blocks, 1)
      target_frac <- runif(1, 0.05, 0.5)

      block_result <- apply_spatial_block_removal(
        coords_df, n_blocks_applied, target_frac,
        min_remaining_points, range_extent
      )
      coords_df <- block_result$coords
      frac_removed <- block_result$frac_removed
    }

    # Chop into random-sized subsets using Beta distribution
    # Beta(1.3, 20) on [min_n, max_n] matches empirical GBIF per-species counts
    set_id <- 0L
    remaining <- coords_df
    version_sets <- list()

    while (nrow(remaining) > 0) {
      set_id <- set_id + 1L
      n_grab <- floor(qbeta(runif(1), beta_alpha, beta_beta) *
                         (max_n - min_n) + min_n)
      n_grab <- max(n_grab, min_n)
      n_grab <- min(n_grab, nrow(remaining))

      version_sets[[set_id]] <- remaining[seq_len(n_grab), , drop = FALSE] |>
        mutate(
          species = spec_name,
          version_id = v,
          set_id = set_id,
          bias_level = prop_biased,
          block_removed = do_blocks,
          n_blocks = n_blocks_applied,
          frac_removed = frac_removed
        )

      remaining <- remaining[-seq_len(n_grab), , drop = FALSE]
    }

    all_results[[v]] <- bind_rows(version_sets)
  }

  bind_rows(compact(all_results))
}


#' Apply spatial block removal with adaptive sizing
#'
#' Removes circular or rectangular spatial blocks from a point set.
#' Block size scales with range extent to ensure proportional removal
#' regardless of species range size.
#'
#' @param coords_df Data frame with X, Y columns (lon/lat)
#' @param n_blocks Number of blocks to remove
#' @param target_removal_frac Target fraction of points to remove
#' @param min_remaining Minimum points to keep (restored if needed)
#' @param range_extent Bbox diagonal of species polygon in projected coords
#' @return List with coords (filtered df) and frac_removed (actual fraction)
apply_spatial_block_removal <- function(coords_df, n_blocks,
                                        target_removal_frac,
                                        min_remaining = 10L,
                                        range_extent = 1) {
  n_orig <- nrow(coords_df)
  if (n_orig <= min_remaining) {
    return(list(coords = coords_df, frac_removed = 0))
  }

  # Compute per-block target: total removal spread across blocks
  per_block_frac <- target_removal_frac / n_blocks

  # Convert range_extent to approximate degrees for lon/lat comparison
  # range_extent is in projected meters, convert to degrees (~111km per degree)
  range_deg <- range_extent / 111320

  removed_idx <- logical(n_orig)

  for (b in seq_len(n_blocks)) {
    # Pick a center from existing (non-removed) points
    available <- which(!removed_idx)
    if (length(available) < min_remaining) break

    center_idx <- sample(available, 1)
    cx <- coords_df$X[center_idx]
    cy <- coords_df$Y[center_idx]

    # Randomly choose circular or rectangular
    is_circular <- runif(1) > 0.5

    if (is_circular) {
      # radius so expected area fraction ~ per_block_frac
      radius <- range_deg * sqrt(per_block_frac / pi)
      # Add some randomness to radius (0.5x to 1.5x)
      radius <- radius * runif(1, 0.5, 1.5)

      # Euclidean distance in degrees (approximate)
      dists <- sqrt((coords_df$X - cx)^2 + (coords_df$Y - cy)^2)
      in_block <- dists <= radius
    } else {
      # half-side so expected area fraction ~ per_block_frac
      half_side <- range_deg * sqrt(per_block_frac)
      half_side <- half_side * runif(1, 0.5, 1.5)

      in_block <- abs(coords_df$X - cx) <= half_side &
                  abs(coords_df$Y - cy) <= half_side
    }

    removed_idx <- removed_idx | in_block
  }

  # Safety check: restore points if too many removed
  n_remaining <- sum(!removed_idx)
  if (n_remaining < min_remaining) {
    removed_indices <- which(removed_idx)
    n_restore <- min_remaining - n_remaining
    restore_idx <- sample(removed_indices, min(n_restore, length(removed_indices)))
    removed_idx[restore_idx] <- FALSE
  }

  frac_removed <- sum(removed_idx) / n_orig

  list(
    coords = coords_df[!removed_idx, , drop = FALSE],
    frac_removed = frac_removed
  )
}


#' Error-protected corrupted coordinate sampling
#'
#' Wraps sample_corrupted_coords() with error handling and logging.
#' Returns NULL on failure so targets can continue with other species.
#'
#' @inheritParams sample_corrupted_coords
#' @param log_file Path to log file
#' @return Tibble of corrupted coordinates, or NULL on failure
#' @export
sample_corrupted_coords_safe <- function(species_coords, bias_xy,
                                          n_versions = 20L,
                                          min_n = 4L, max_n = 1000L,
                                          beta_alpha = 1.3, beta_beta = 20,
                                          bias_levels = c(0, 0.05, 0.25, 0.5, 0.75, 0.95),
                                          no_block_frac = 0.3,
                                          max_blocks = 3L,
                                          min_remaining_points = 10L,
                                          log_file = "logs/geoencoder_corruption.log") {
  sp_name <- tryCatch(species_coords$species[1], error = \(e) "unknown")
  dir.create(dirname(log_file), showWarnings = FALSE, recursive = TRUE)

  tryCatch(
    {
      result <- sample_corrupted_coords(
        species_coords, bias_xy,
        n_versions = n_versions,
        min_n = min_n, max_n = max_n,
        beta_alpha = beta_alpha, beta_beta = beta_beta,
        bias_levels = bias_levels,
        no_block_frac = no_block_frac,
        max_blocks = max_blocks,
        min_remaining_points = min_remaining_points
      )
      n_rows <- if (is.null(result)) 0L else nrow(result)
      cat(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), " OK       ", sp_name,
          " (", n_rows, " rows)\n", sep = "",
          file = log_file, append = TRUE)
      result
    },
    error = \(e) {
      cat(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), " FAIL     ", sp_name,
          ": ", conditionMessage(e), "\n", sep = "",
          file = log_file, append = TRUE)
      warning("Corruption sampling failed for ", sp_name, ": ",
              conditionMessage(e))
      NULL
    }
  )
}


#' Build geo-encoder training dataset from corrupted samples
#'
#' Aggregates corrupted samples, matches to NichEncoder species embeddings,
#' normalizes coordinates, splits by species, and exports parquet files.
#'
#' @param corrupted_list List of tibbles from dynamic branching
#' @param nichencoder_embeddings Output from extract_nichencoder_embeddings()
#' @param xy_mean_sd Named list with mean and sd for lon/lat normalization
#' @param train_frac Species-level train fraction (default 0.85)
#' @param val_frac Species-level validation fraction (default 0.10)
#' @param seed Random seed for species-level splitting
#' @return List with file paths: train_parquet, val_parquet, test_parquet,
#'   embeddings_file
#' @export
build_geoencoder_dataset <- function(corrupted_list,
                                      nichencoder_embeddings,
                                      xy_mean_sd,
                                      train_frac = 0.85,
                                      val_frac = 0.10,
                                      seed = 42) {

  embedding_matrix <- nichencoder_embeddings$embedding_matrix
  species_map <- nichencoder_embeddings$species_map

  # Combine all corrupted samples
  message("Combining corrupted samples from ", length(corrupted_list), " species...")
  all_data <- bind_rows(compact(corrupted_list))
  message("Total rows: ", format(nrow(all_data), big.mark = ","))

  # Filter to species present in species_map
  known_species <- names(species_map)
  all_data <- all_data |>
    filter(species %in% known_species)
  message("After filtering to known species: ",
          format(nrow(all_data), big.mark = ","),
          " (", n_distinct(all_data$species), " species)")

  # Z-score normalize coordinates
  x_mean <- xy_mean_sd$mean[1]
  x_sd <- xy_mean_sd$sd[1]
  y_mean <- xy_mean_sd$mean[2]
  y_sd <- xy_mean_sd$sd[2]

  all_data <- all_data |>
    mutate(
      X_norm = (X - x_mean) / x_sd,
      Y_norm = (Y - y_mean) / y_sd
    )

  # Species-level train/val/test split
  set.seed(seed)
  unique_species <- unique(all_data$species)
  n_sp <- length(unique_species)
  sp_order <- sample(unique_species)

  n_train <- floor(n_sp * train_frac)
  n_val <- floor(n_sp * val_frac)

  train_species <- sp_order[seq_len(n_train)]
  val_species <- sp_order[(n_train + 1):(n_train + n_val)]
  test_species <- sp_order[(n_train + n_val + 1):n_sp]

  message("Species split: train=", length(train_species),
          " val=", length(val_species),
          " test=", length(test_species))

  train_data <- all_data |> filter(species %in% train_species)
  val_data <- all_data |> filter(species %in% val_species)
  test_data <- all_data |> filter(species %in% test_species)

  message("Row counts: train=", format(nrow(train_data), big.mark = ","),
          " val=", format(nrow(val_data), big.mark = ","),
          " test=", format(nrow(test_data), big.mark = ","))

  # Write parquet files
  out_dir <- "data/processed"
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

  train_path <- file.path(out_dir, "geoencoder_train.parquet")
  val_path <- file.path(out_dir, "geoencoder_val.parquet")
  test_path <- file.path(out_dir, "geoencoder_test.parquet")

  arrow::write_parquet(train_data, train_path)
  arrow::write_parquet(val_data, val_path)
  arrow::write_parquet(test_data, test_path)

  # Save target embeddings
  emb_dir <- "output/geoencoder_config"
  dir.create(emb_dir, showWarnings = FALSE, recursive = TRUE)
  emb_path <- file.path(emb_dir, "target_embeddings.rds")
  saveRDS(nichencoder_embeddings, emb_path)

  message("Saved: ", train_path, ", ", val_path, ", ", test_path,
          ", ", emb_path)

  list(
    train_parquet = train_path,
    val_parquet = val_path,
    test_parquet = test_path,
    embeddings_file = emb_path
  )
}


#' Build geo-encoder downstream validation data
#'
#' Prepares validation data for zero-shot species: one corrupted coordinate
#' set per species (for geo-encoder prediction) and raw environmental data
#' (for ground truth comparison after full pipeline decoding).
#'
#' @param corrupted_list List of tibbles from dynamic branching
#' @param test_parquet Path to jade test parquet (raw env data)
#' @param split_assignments Tibble from assign_species_splits()
#' @param xy_mean_sd Named list with mean and sd for lon/lat normalization
#' @param chelsa_var_meta Tibble with chelsa_name column for env variable ordering
#' @param env_mean_sd Named list with mean and sd for env standardization
#' @param seed Random seed for version selection
#' @return List with downstream_coords_parquet and downstream_env_parquet paths
build_geoencoder_val_downstream <- function(corrupted_list,
                                             test_parquet,
                                             split_assignments,
                                             xy_mean_sd,
                                             chelsa_var_meta,
                                             env_mean_sd,
                                             seed = 42) {

  # Identify zero-shot species
  zeroshot_species <- split_assignments |>
    filter(split_role == "zeroshot") |>
    pull(species)
  message("Zero-shot species: ", length(zeroshot_species))

  # Combine corrupted samples and filter to zero-shot species
  message("Combining corrupted samples...")
  all_corrupted <- bind_rows(compact(corrupted_list))
  zs_corrupted <- all_corrupted |>
    filter(species %in% zeroshot_species)
  message("Zero-shot corrupted rows: ", format(nrow(zs_corrupted), big.mark = ","),
          " (", n_distinct(zs_corrupted$species), " species)")

  # Select one corrupted version per species
  set.seed(seed)
  selected_versions <- zs_corrupted |>
    distinct(species, version_id, set_id) |>
    slice_sample(n = 1, by = species)
  message("Selected one version per species: ", nrow(selected_versions), " groups")

  zs_selected <- zs_corrupted |>
    semi_join(selected_versions, by = c("species", "version_id", "set_id"))

  # Normalize coordinates
  x_mean <- xy_mean_sd$mean[1]
  x_sd <- xy_mean_sd$sd[1]
  y_mean <- xy_mean_sd$mean[2]
  y_sd <- xy_mean_sd$sd[2]

  zs_selected <- zs_selected |>
    mutate(
      X_norm = (X - x_mean) / x_sd,
      Y_norm = (Y - y_mean) / y_sd
    )

  # Write downstream coords parquet
  out_dir <- "data/processed"
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  coords_path <- file.path(out_dir, "geoencoder_downstream_coords.parquet")
  arrow::write_parquet(zs_selected, coords_path)
  message("Saved downstream coords: ", coords_path,
          " (", format(nrow(zs_selected), big.mark = ","), " rows)")

  # Load raw environmental test data for zero-shot species
  message("Loading test parquet for environmental ground truth...")
  test_data <- arrow::read_parquet(test_parquet)

  zs_env <- test_data |>
    filter(species %in% zeroshot_species)
  message("Zero-shot env samples: ", format(nrow(zs_env), big.mark = ","),
          " (", n_distinct(zs_env$species), " species)")

  # Extract environmental columns in correct order
  chelsa_names <- chelsa_var_meta$chelsa_name
  parquet_cols <- names(zs_env)
  env_col_idx <- purrr::map_int(chelsa_names, \(cn) {
    matches <- which(stringr::str_detect(parquet_cols, paste0("_", cn, "_")))
    if (length(matches) == 0) {
      stop("Could not find parquet column matching chelsa_name: ", cn)
    }
    matches[1]
  })
  env_cols <- parquet_cols[env_col_idx]

  # Standardize environmental data
  env_mat <- as.matrix(zs_env[, env_cols])
  for (i in seq_along(env_mean_sd$mean)) {
    env_mat[, i] <- (env_mat[, i] - env_mean_sd$mean[i]) / env_mean_sd$sd[i]
  }
  env_mat[is.na(env_mat)] <- 0

  # Build output dataframe with species + standardized env
  env_df <- tibble::as_tibble(env_mat, .name_repair = "minimal")
  colnames(env_df) <- paste0("env_", seq_len(ncol(env_df)))
  env_df <- bind_cols(
    tibble::tibble(species = zs_env$species),
    env_df
  )

  env_path <- file.path(out_dir, "geoencoder_downstream_env.parquet")
  arrow::write_parquet(env_df, env_path)
  message("Saved downstream env: ", env_path,
          " (", format(nrow(env_df), big.mark = ","), " rows)")

  list(
    downstream_coords_parquet = coords_path,
    downstream_env_parquet = env_path
  )
}
