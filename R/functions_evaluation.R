#' NicheFlow Evaluation Pipeline Functions
#'
#' Functions for evaluating NicheFlow against baseline SDM methods (MaxEnt,
#' Random Forest) using AUC, TSS, SWD, and geographic EMD metrics.
#'
#' @author rdinnage

library(zeallot)
# RANN, MASS, maxnet, randomForest, transport loaded via namespaced calls
# to avoid polluting the global namespace and invalidating upstream targets


# ===========================================================================
# Species Metadata
# ===========================================================================

#' Build species metadata for evaluation
#'
#' Computes taxonomic group, range size, median latitude, and sample counts
#' for all species in the JADE data. Used to stratify evaluation sampling
#' and enrich metric outputs.
#'
#' @param jade_samples Combined train + test JADE data with columns:
#'   species, taxon, X, Y
#' @param species_map Named integer vector (species_name -> id)
#' @return Tibble with species, species_id, taxon, range_size_km2,
#'   median_lat, n_train, n_test
build_species_metadata <- function(jade_samples, species_map) {
  # Per-species geographic stats
  geo_stats <- jade_samples |>
    summarise(
      median_lat = median(Y),
      lon_range = max(X) - min(X),
      lat_range = max(Y) - min(Y),
      n_points = n(),
      .by = c(species, taxon)
    ) |>
    mutate(
      # Approximate range size from bounding box (lon_range * lat_range * cos(lat))
      # Convert degrees to km (1 deg lat ≈ 111 km)
      range_size_km2 = lon_range * 111 * cos(median_lat * pi / 180) *
        lat_range * 111
    )

  # Determine shot_type per species from split_type
  # A species is "fewshot" if any of its samples have split_type == "fewshot",
  # "zeroshot" if only zeroshot, otherwise "full"
  shot_types <- jade_samples |>
    summarise(
      has_fewshot = any(split_type == "fewshot", na.rm = TRUE),
      has_zeroshot = any(split_type == "zeroshot", na.rm = TRUE),
      has_within = any(split_type == "within_species", na.rm = TRUE),
      .by = species
    ) |>
    mutate(
      shot_type = case_when(
        has_fewshot ~ "fewshot",
        has_zeroshot & !has_within ~ "zeroshot",
        TRUE ~ "full"
      )
    ) |>
    select(species, shot_type)

  # Match species IDs
  species_ids <- tibble(
    species = names(species_map),
    species_id = as.integer(species_map)
  )

  geo_stats |>
    left_join(species_ids, by = "species") |>
    left_join(shot_types, by = "species") |>
    select(species, species_id, taxon, shot_type,
           range_size_km2, median_lat, n_points) |>
    filter(!is.na(species_id))
}


# ===========================================================================
# Background Point Generation
# ===========================================================================

#' Generate background points via circular noise + KD-tree density compensation
#'
#' Adds uniform circular noise to presence points, then compensates for
#' edge effects using inverse neighbor-count weighting via RANN KD-tree.
#' This avoids sf polygon operations entirely.
#'
#' @param presence_xy Matrix [N, 2] of (lon, lat) presence coordinates
#' @param n_background Number of background points to return
#' @param radius_multiplier Fraction of bounding box diagonal for noise radius
#' @param oversample_factor Generate this many times n_background candidates
#' @return Matrix [n_background, 2] of (lon, lat) background coordinates
generate_background_points <- function(presence_xy,
                                       n_background,
                                       radius_multiplier = 0.25,
                                       oversample_factor = 5L) {
  n_pres <- nrow(presence_xy)

  # Compute noise radius from bounding box diagonal
  bbox_diag <- sqrt(diff(range(presence_xy[, 1]))^2 +
                      diff(range(presence_xy[, 2]))^2)
  noise_radius <- bbox_diag * radius_multiplier

  # Generate candidate points: random presence + circular uniform noise
  n_candidates <- n_background * oversample_factor
  source_idx <- sample.int(n_pres, n_candidates, replace = TRUE)

  # Circular uniform noise: angle ~ U(0, 2*pi), r ~ sqrt(U) * radius
  angles <- runif(n_candidates, 0, 2 * pi)
  radii <- sqrt(runif(n_candidates)) * noise_radius
  noise_x <- radii * cos(angles)
  noise_y <- radii * sin(angles)

  candidates <- cbind(
    presence_xy[source_idx, 1] + noise_x,
    presence_xy[source_idx, 2] + noise_y
  )

  # KD-tree: count presence points within noise_radius of each candidate
  nn_result <- RANN::nn2(presence_xy, candidates,
                         k = min(50L, n_pres),
                         searchtype = "radius",
                         radius = noise_radius)
  # Count how many neighbors are within radius (nn.dists == 0 means no match
  # in radius search mode, but RANN returns Inf for no match)
  neighbor_counts <- rowSums(nn_result$nn.dists < noise_radius)
  neighbor_counts[neighbor_counts == 0] <- 1  # avoid division by zero

  # Weight inversely by neighbor count (edge compensation)
  weights <- 1 / neighbor_counts
  weights <- weights / sum(weights)

  # Weighted subsample without replacement
  selected <- sample.int(n_candidates, n_background,
                         replace = FALSE, prob = weights)

  candidates[selected, , drop = FALSE]
}


# ===========================================================================
# CHELSA Extraction at Arbitrary Coordinates
# ===========================================================================

#' Build CHELSA raster stack in chelsa_var_meta order
#'
#' Matches all 31 variables to their raster files, ordered consistently
#' with the VAE training data and standardization stats.
#'
#' @param chelsa_var_meta Variable metadata tibble (must have chelsa_name col)
#' @param chelsa_bio_dir Path to CHELSA bio raster directory
#' @return SpatRast stack with 31 layers in chelsa_var_meta order
build_chelsa_rast_stack <- function(chelsa_var_meta, chelsa_bio_dir) {
  all_files <- list.files(chelsa_bio_dir, full.names = TRUE)
  rast_files <- purrr::map_chr(chelsa_var_meta$chelsa_name, \(cn) {
    matched <- stringr::str_subset(all_files, paste0("CHELSA_", cn, "_"))
    if (length(matched) == 0) stop("No raster found for ", cn)
    matched[1]
  })
  terra::rast(rast_files)
}


#' Extract CHELSA-BIOCLIM+ variables at arbitrary coordinates
#'
#' Loads CHELSA rasters and extracts environmental values at given (lon, lat)
#' coordinates. Returns standardized values using the same mean/sd as VAE.
#'
#' @param xy Matrix [N, 2] of (lon, lat) coordinates
#' @param chelsa_var_meta Tibble with file paths and variable info
#' @param env_mean_sd List with mean and sd vectors for standardization
#' @param chelsa_bio_dir Path to CHELSA bio directory
#' @return Tibble with X, Y, and 31 standardized env columns
extract_chelsa_at_coords <- function(xy, chelsa_var_meta, env_mean_sd,
                                     chelsa_bio_dir) {
  rast_stack <- build_chelsa_rast_stack(chelsa_var_meta, chelsa_bio_dir)

  # Extract values at coordinates
  pts <- terra::vect(xy, crs = "EPSG:4326")
  extracted <- terra::extract(rast_stack, pts)

  # Drop the ID column from terra::extract
  env_raw <- as.matrix(extracted[, -1, drop = FALSE])

  # Apply documented CHELSA NA fills (e.g., snow/frost = 0 in tropics)
  env_raw <- apply_chelsa_na_fills(env_raw, chelsa_var_meta)

  # Standardize
  env_std <- sweep(env_raw, 2, env_mean_sd$mean, "-")
  env_std <- sweep(env_std, 2, env_mean_sd$sd, "/")

  # Return as tibble with coordinates
  result <- as_tibble(env_std)
  result$X <- xy[, 1]
  result$Y <- xy[, 2]
  result <- relocate(result, X, Y)
  result
}


# ===========================================================================
# NicheFlow Generation Pipeline
# ===========================================================================

#' Generate environmental samples for a species (NichEncoder → VAE decode)
#'
#' @param species_id Integer species ID (1-based)
#' @param n_samples Number of samples to generate
#' @param flow_model nichencoder_traj_net on device
#' @param vae_model env_vae_mod on device
#' @param active_dims Integer vector of active latent dimensions (1-based)
#' @param vae_latent_dim Full VAE latent dimension
#' @param device Torch device string
#' @param ode_steps ODE integration steps for NichEncoder
#' @return Matrix [n_samples, n_env_dims] of standardized env values
generate_env_samples <- function(species_id, n_samples,
                                 flow_model, vae_model,
                                 active_dims,
                                 vae_latent_dim = 16L,
                                 device = "cuda:0",
                                 ode_steps = 200L) {
  active_dim <- length(active_dims)

  # 1. Sample noise in active latent space
  z0 <- torch_randn(n_samples, active_dim, device = device)
  sp_ids <- torch_full(c(n_samples), species_id,
                       dtype = torch_long(), device = device)

  # 2. Forward ODE: noise → latent codes
  z_active <- flow_model$sample_trajectory(z0, sp_ids, steps = ode_steps)
  # z_active is on CPU after sample_trajectory

  # 3. Pad to full latent dimension (inactive dims = 0)
  z_full <- torch_zeros(n_samples, vae_latent_dim)
  z_full[, active_dims] <- z_active

  # 4. Decode through VAE
  z_full_dev <- z_full$to(device = device)
  with_no_grad({
    env_std <- vae_model$decoder(z_full_dev)
  })

  as.matrix(env_std$cpu())
}


#' Generate geographic samples (full pipeline: NichEncoder → VAE → GeODE)
#'
#' @param species_id Integer species ID
#' @param n_samples Number of samples
#' @param flow_model NichEncoder on device
#' @param vae_model VAE on device
#' @param geode_model GeODE on device
#' @param active_dims Active latent dimension indices
#' @param xy_mean_sd List with mean and sd for XY un-standardization
#' @param vae_latent_dim Full latent dimension
#' @param device Torch device
#' @param flow_ode_steps NichEncoder ODE steps
#' @param geode_ode_steps GeODE ODE steps
#' @return Matrix [n_samples, 2] of raw (lon, lat) coordinates
generate_geo_samples <- function(species_id, n_samples,
                                 flow_model, vae_model, geode_model,
                                 active_dims, xy_mean_sd,
                                 vae_latent_dim = 16L,
                                 device = "cuda:0",
                                 flow_ode_steps = 200L,
                                 geode_ode_steps = 200L) {
  # Generate standardized env samples
  env_std <- generate_env_samples(
    species_id, n_samples, flow_model, vae_model,
    active_dims, vae_latent_dim, device, flow_ode_steps
  )

  # Convert to tensor on device for GeODE

  env_tensor <- torch_tensor(env_std, device = device)

  # Initial XY noise (standardized space)
  xy_init <- torch_randn(n_samples, 2L, device = device)

  # GeODE: env → geographic coordinates (standardized)
  xy_std <- geode_model$sample_trajectory(xy_init, env_tensor,
                                          steps = geode_ode_steps)
  # xy_std is on CPU

  # Un-standardize to raw lon/lat
  xy_raw <- as.matrix(xy_std)
  xy_raw[, 1] <- xy_raw[, 1] * xy_mean_sd$sd[1] + xy_mean_sd$mean[1]
  xy_raw[, 2] <- xy_raw[, 2] * xy_mean_sd$sd[2] + xy_mean_sd$mean[2]

  xy_raw
}


# ===========================================================================
# Geographic KDE Scoring
# ===========================================================================

#' Score points using 2D geographic KDE from generated samples
#'
#' Projects generated and evaluation points to equal-area LAEA projection
#' centered on the generated points' centroid, then fits a 2D KDE and
#' evaluates at each evaluation point.
#'
#' @param generated_xy Matrix [M, 2] of generated (lon, lat)
#' @param eval_xy Matrix [N, 2] of evaluation (lon, lat)
#' @param n_grid KDE grid resolution for MASS::kde2d
#' @return Numeric vector [N] of KDE density scores at eval points
score_geographic_kde <- function(generated_xy, eval_xy, n_grid = 200L) {
  # Restrict generated points to the convex hull of eval points + buffer
  # (GeODE generates globally, but we only care about the focal area)
  eval_hull <- grDevices::chull(eval_xy)
  hull_poly <- eval_xy[c(eval_hull, eval_hull[1]), ]
  # Keep generated points within expanded bounding box of eval points
  eval_bbox <- apply(eval_xy, 2, range)
  bbox_buf <- (eval_bbox[2, ] - eval_bbox[1, ]) * 0.5
  in_bbox <- generated_xy[, 1] >= (eval_bbox[1, 1] - bbox_buf[1]) &
    generated_xy[, 1] <= (eval_bbox[2, 1] + bbox_buf[1]) &
    generated_xy[, 2] >= (eval_bbox[1, 2] - bbox_buf[2]) &
    generated_xy[, 2] <= (eval_bbox[2, 2] + bbox_buf[2])
  generated_xy <- generated_xy[in_bbox, , drop = FALSE]

  # Need at least some points for KDE
  if (nrow(generated_xy) < 10) {
    return(rep(0, nrow(eval_xy)))
  }

  # Center point for LAEA projection
  center_lon <- mean(eval_xy[, 1])
  center_lat <- mean(eval_xy[, 2])

  # Define LAEA projection centered on species
  laea_crs <- sprintf(
    "+proj=laea +lat_0=%f +lon_0=%f +datum=WGS84 +units=m",
    center_lat, center_lon
  )

  # Project generated points
  gen_df <- as.data.frame(generated_xy)
  names(gen_df) <- c("lon", "lat")
  gen_sf <- st_as_sf(gen_df, coords = c("lon", "lat"), crs = 4326) |>
    st_transform(laea_crs)
  gen_coords <- st_coordinates(gen_sf)

  # Project evaluation points
  eval_df <- as.data.frame(eval_xy)
  names(eval_df) <- c("lon", "lat")
  eval_sf <- st_as_sf(eval_df, coords = c("lon", "lat"), crs = 4326) |>
    st_transform(laea_crs)
  eval_coords <- st_coordinates(eval_sf)

  # Fit 2D KDE on generated points
  # Extend limits slightly beyond data range to cover eval points
  all_x <- c(gen_coords[, 1], eval_coords[, 1])
  all_y <- c(gen_coords[, 2], eval_coords[, 2])
  xlim <- range(all_x) + c(-1, 1) * diff(range(all_x)) * 0.1
  ylim <- range(all_y) + c(-1, 1) * diff(range(all_y)) * 0.1

  kde_fit <- MASS::kde2d(gen_coords[, 1], gen_coords[, 2],
                         n = n_grid, lims = c(xlim, ylim))

  # Evaluate KDE at each eval point via bilinear interpolation
  scores <- numeric(nrow(eval_coords))
  for (i in seq_len(nrow(eval_coords))) {
    # Find nearest grid cell indices
    ix <- findInterval(eval_coords[i, 1], kde_fit$x)
    iy <- findInterval(eval_coords[i, 2], kde_fit$y)
    ix <- max(1, min(ix, length(kde_fit$x) - 1))
    iy <- max(1, min(iy, length(kde_fit$y) - 1))

    # Bilinear interpolation
    dx <- (eval_coords[i, 1] - kde_fit$x[ix]) /
      (kde_fit$x[ix + 1] - kde_fit$x[ix])
    dy <- (eval_coords[i, 2] - kde_fit$y[iy]) /
      (kde_fit$y[iy + 1] - kde_fit$y[iy])
    dx <- max(0, min(1, dx))
    dy <- max(0, min(1, dy))

    scores[i] <- kde_fit$z[ix, iy] * (1 - dx) * (1 - dy) +
      kde_fit$z[ix + 1, iy] * dx * (1 - dy) +
      kde_fit$z[ix, iy + 1] * (1 - dx) * dy +
      kde_fit$z[ix + 1, iy + 1] * dx * dy
  }

  scores
}


# ===========================================================================
# MaxEnt and Random Forest Wrappers
# ===========================================================================

#' Train and predict with MaxEnt (maxnet) for one species
#'
#' @param train_env_std Matrix/df of standardized env vars for training
#' @param train_labels Numeric vector 0/1 (background/presence)
#' @param test_env_std Matrix/df of standardized env vars for test
#' @return Numeric vector of cloglog predictions on test data
run_maxnet_predict <- function(train_env_std, train_labels, test_env_std) {
  set.seed(32639)
  X_train <- as.data.frame(train_env_std)
  X_test <- as.data.frame(test_env_std)
  # Clean column names: CHELSA names have hyphens/dots that break formulas
  clean_names <- paste0("v", seq_len(ncol(X_train)))
  names(X_train) <- clean_names
  names(X_test) <- clean_names

  mxnet <- maxnet::maxnet(
    p = train_labels,
    data = X_train,
    regmult = 1,
    maxnet::maxnet.formula(train_labels, X_train, classes = "lqp")
  )

  as.numeric(predict(mxnet, X_test, type = "cloglog"))
}


#' Train and predict with balanced Random Forest for one species
#'
#' @param train_env_std Matrix/df of standardized env vars for training
#' @param train_labels Factor with levels c("no", "yes")
#' @param test_env_std Matrix/df of standardized env vars for test
#' @return Numeric vector of presence probability predictions
run_rf_predict <- function(train_env_std, train_labels, test_env_std) {
  set.seed(32639)
  train_df <- as.data.frame(train_env_std)
  # Clean column names: CHELSA names have hyphens/dots that break formulas
  clean_names <- paste0("v", seq_len(ncol(train_df)))
  names(train_df) <- clean_names
  train_df$occ <- train_labels

  n_pres <- sum(train_labels == "yes")
  smpsize <- c("no" = n_pres, "yes" = n_pres)

  rf <- randomForest::randomForest(
    formula = occ ~ .,
    data = train_df,
    ntree = 1000,
    sampsize = smpsize,
    replace = TRUE
  )

  X_test <- as.data.frame(test_env_std)
  names(X_test) <- clean_names
  preds <- predict(rf, X_test, type = "prob")
  as.numeric(preds[, "yes"])
}


# ===========================================================================
# Evaluation Metrics
# ===========================================================================

#' Compute AUC, PR-AUC, and TSS from truth labels and scores
#'
#' @param truth Factor with levels c("no", "yes") or numeric 0/1
#' @param scores Numeric prediction scores (higher = more likely presence)
#' @return Tibble with roc_auc, pr_auc, tss
compute_eval_metrics <- function(truth, scores) {
  # Ensure factor format for yardstick
  if (is.numeric(truth)) {
    truth <- factor(ifelse(truth == 1, "yes", "no"),
                    levels = c("no", "yes"))
  }

  dat <- tibble(truth = truth, score = scores)

  roc_auc <- tryCatch(
    yardstick::roc_auc_vec(truth, scores, event_level = "second"),
    error = \(e) NA_real_
  )

  pr_auc <- tryCatch(
    yardstick::pr_auc_vec(truth, scores, event_level = "second"),
    error = \(e) NA_real_
  )

  # TSS: optimize threshold via Youden's J
  tss <- tryCatch({
    thresholds <- seq(
      min(scores, na.rm = TRUE),
      max(scores, na.rm = TRUE),
      length.out = 200
    )
    perf <- yardstick::threshold_perf(
      dat, truth, score,
      thresholds = thresholds,
      event_level = "second",
      metrics = yardstick::metric_set(yardstick::j_index)
    )
    max(perf$.estimate, na.rm = TRUE)
  }, error = \(e) NA_real_)

  tibble(roc_auc = roc_auc, pr_auc = pr_auc, tss = tss)
}


# ===========================================================================
# SWD Batch Evaluation
# ===========================================================================

#' Apply documented CHELSA NA fill values
#'
#' Fills known NA values (e.g., snow/frost variables = 0 in tropics)
#' using the fill values specified in chelsa_var_meta.
#'
#' @param env_raw Matrix [N, 31] of raw CHELSA values
#' @param chelsa_var_meta Variable metadata with na_fill column
#' @return Matrix with documented NAs filled
apply_chelsa_na_fills <- function(env_raw, chelsa_var_meta) {
  for (j in seq_len(nrow(chelsa_var_meta))) {
    fill_val <- chelsa_var_meta$na_fill[j]
    if (!is.na(fill_val)) {
      na_mask <- is.na(env_raw[, j])
      if (any(na_mask)) env_raw[na_mask, j] <- fill_val
    }
  }
  env_raw
}


#' Standardize raw JADE env data using env_mean_sd
#'
#' @param env_raw Matrix or tibble of raw env values (31 cols)
#' @param env_mean_sd List with mean and sd vectors
#' @return Matrix of standardized values
standardize_env <- function(env_raw, env_mean_sd) {
  m <- as.matrix(env_raw)
  m <- sweep(m, 2, env_mean_sd$mean, "-")
  sweep(m, 2, env_mean_sd$sd, "/")
}


#' Evaluate SWD for a batch of species
#'
#' Loads models once, generates samples, computes SWD in env and latent space.
#'
#' @param species_batch Character vector of species names
#' @param jade_test_data Test data tibble
#' @param species_map Named integer vector
#' @param vae_checkpoint Path to VAE checkpoint
#' @param flow_checkpoint_dir NichEncoder checkpoint directory
#' @param active_dims Active latent dimensions
#' @param vae_latent_dim Full latent dim
#' @param env_mean_sd Standardization stats
#' @param device Torch device
#' @param ode_steps ODE steps for generation
#' @param n_swd_slices Number of SWD random directions
#' @return Tibble(species, swd_env, swd_latent, n_test)
evaluate_swd_batch <- function(species_batch, jade_test_data,
                               species_map, vae_checkpoint,
                               flow_checkpoint_dir, active_dims,
                               env_mean_sd,
                               vae_latent_dim = 16L,
                               device = "cuda:0",
                               ode_steps = 200L,
                               n_swd_slices = 100L) {
  # Load models once
  vae_model <- env_vae_mod(31L, vae_latent_dim)
  load_model_checkpoint(vae_model, vae_checkpoint)
  vae_model <- vae_model$to(device = device)
  vae_model$eval()

  flow_ckpt <- find_latest_checkpoint(flow_checkpoint_dir)
  flow_model <- nichencoder_traj_net(
    coord_dim = length(active_dims),
    n_species = length(species_map),
    spec_embed_dim = 64L,
    breadths = c(512L, 256L, 128L)
  )
  load_model_checkpoint(flow_model, flow_ckpt$path)
  flow_model <- flow_model$to(device = device)
  flow_model$eval()

  # Get env column names (everything except X, Y, jacobian, species, taxon, split_type)
  env_cols <- setdiff(names(jade_test_data),
                      c("X", "Y", "jacobian", "species", "taxon", "split_type"))

  results <- vector("list", length(species_batch))

  for (i in seq_along(species_batch)) {
    sp <- species_batch[i]
    sp_id <- species_map[sp]
    if (is.na(sp_id)) next

    # Get test data for this species
    sp_test <- jade_test_data |> filter(species == sp)
    n_test <- nrow(sp_test)
    if (n_test < 10) next

    # Standardize test env
    test_env_std <- standardize_env(sp_test[, env_cols], env_mean_sd)

    # Generate same number of env samples
    gen_env_std <- generate_env_samples(
      sp_id, n_test, flow_model, vae_model,
      active_dims, vae_latent_dim, device, ode_steps
    )

    # SWD in environmental space (31-dim)
    swd_env <- compute_swd(test_env_std, gen_env_std,
                           n_slices = n_swd_slices)

    # Encode both through VAE for latent SWD
    test_tensor <- torch_tensor(test_env_std, device = device)
    gen_tensor <- torch_tensor(gen_env_std, device = device)
    with_no_grad({
      c(test_mu, .) %<-% vae_model$encoder(test_tensor)
      c(gen_mu, .) %<-% vae_model$encoder(gen_tensor)
    })
    test_latent <- as.matrix(test_mu$cpu())[, active_dims, drop = FALSE]
    gen_latent <- as.matrix(gen_mu$cpu())[, active_dims, drop = FALSE]

    swd_latent <- compute_swd(test_latent, gen_latent,
                              n_slices = n_swd_slices)

    results[[i]] <- tibble(
      species = sp, swd_env = swd_env,
      swd_latent = swd_latent, n_test = n_test
    )

    if (i %% 10 == 0) {
      message("  SWD: ", i, "/", length(species_batch), " species")
    }

    gc(verbose = FALSE)
    if (grepl("cuda", device)) cuda_empty_cache()
  }

  list_rbind(compact(results))
}


# ===========================================================================
# AUC Evaluation Pipeline
# ===========================================================================

#' Prepare evaluation data for a batch of species
#'
#' For each species in the batch: extract presence env data, generate
#' shared background, extract CHELSA at background points.
#'
#' @param species_batch Character vector of species names
#' @param jade_train_data Training parquet tibble
#' @param jade_test_data Test parquet tibble
#' @param chelsa_var_meta Variable metadata
#' @param env_mean_sd Standardization stats
#' @param chelsa_bio_dir CHELSA raster directory
#' @param n_background Background points per species
#' @return List of per-species data (list of lists)
prepare_eval_batch <- function(species_batch, jade_train_data,
                               jade_test_data, chelsa_var_meta,
                               env_mean_sd, chelsa_bio_dir,
                               n_background = 5000L,
                               n_test_bg = 500L,
                               max_test = 100L) {
  env_cols <- setdiff(names(jade_test_data),
                      c("X", "Y", "jacobian", "species", "taxon", "split_type"))

  # Load CHELSA raster stack once for background extraction (all 31 vars)
  rast_stack <- build_chelsa_rast_stack(chelsa_var_meta, chelsa_bio_dir)

  results <- vector("list", length(species_batch))

  for (i in seq_along(species_batch)) {
    sp <- species_batch[i]

    sp_train <- jade_train_data |> filter(species == sp)
    sp_test <- jade_test_data |> filter(species == sp)

    if (nrow(sp_train) == 0 || nrow(sp_test) == 0) next

    # Subsample test points to max_test for comparability across species
    if (nrow(sp_test) > max_test) {
      sp_test <- sp_test |> slice_sample(n = max_test)
    }

    # All presence XY for background generation
    all_pres_xy <- rbind(
      cbind(sp_train$X, sp_train$Y),
      cbind(sp_test$X, sp_test$Y)
    )

    # Generate shared background
    bg_xy <- generate_background_points(all_pres_xy, n_background)

    # Extract CHELSA at background points
    pts <- terra::vect(bg_xy, crs = "EPSG:4326")
    bg_extracted <- terra::extract(rast_stack, pts)
    bg_env_raw <- as.matrix(bg_extracted[, -1, drop = FALSE])

    # Apply documented CHELSA NA fills, drop remaining ocean points
    bg_env_raw <- apply_chelsa_na_fills(bg_env_raw, chelsa_var_meta)
    land <- complete.cases(bg_env_raw)
    bg_env_raw <- bg_env_raw[land, , drop = FALSE]
    bg_xy <- bg_xy[land, , drop = FALSE]

    # Standardize
    bg_env_std <- standardize_env(bg_env_raw, env_mean_sd)

    # Standardize presence env
    train_env_std <- standardize_env(sp_train[, env_cols], env_mean_sd)
    test_env_std <- standardize_env(sp_test[, env_cols], env_mean_sd)

    # Subsample background for test scoring (all methods use same subset)
    # Training still uses full background pool
    if (n_background > n_test_bg) {
      test_bg_idx <- sample.int(n_background, n_test_bg)
    } else {
      test_bg_idx <- seq_len(n_background)
    }

    results[[i]] <- list(
      species = sp,
      train_env_std = train_env_std,
      test_env_std = test_env_std,
      train_bg_env_std = bg_env_std,          # full 5000 for training MaxEnt/RF
      test_bg_env_std = bg_env_std[test_bg_idx, , drop = FALSE],  # 500 for scoring
      train_xy = cbind(sp_train$X, sp_train$Y),
      test_xy = cbind(sp_test$X, sp_test$Y),
      train_bg_xy = bg_xy,                    # full 5000
      test_bg_xy = bg_xy[test_bg_idx, , drop = FALSE],  # 500
      n_train = nrow(sp_train),
      n_test = nrow(sp_test)
    )

    if (i %% 10 == 0) {
      message("  Data prep: ", i, "/", length(species_batch), " species")
    }
  }

  compact(results)
}


# ===========================================================================
# 20-variable per-species datasets with held-out splits (random + spatial)
# ===========================================================================

#' Column-subset env matrices in a prepared batch to a CHELSA variable subset
subset_batch_to_vars <- function(batch, var_pattern) {
  lapply(batch, \(sp) {
    if (is.null(sp)) return(NULL)
    keep <- grepl(var_pattern, colnames(sp$train_env_std))
    sp$train_env_std    <- sp$train_env_std[,    keep, drop = FALSE]
    sp$test_env_std     <- sp$test_env_std[,     keep, drop = FALSE]
    sp$train_bg_env_std <- sp$train_bg_env_std[, keep, drop = FALSE]
    sp$test_bg_env_std  <- sp$test_bg_env_std[,  keep, drop = FALSE]
    sp
  })
}


# vectorized isTRUE; backgrounds with NA stay excluded
isTRUE_safe <- function(x) !is.na(x) & x


#' Add a jacobian_20 column to a species dataset by extracting from a raster
#'
#' Extracts the 20-var Jacobian at each (X, Y) and appends the value as a
#' jacobian_20 column. Used downstream of build_species_dataset() so the
#' column is available in the on-disk parquet for analysis.
add_jacobian_column <- function(ds, jac_rast, col_name = "jacobian_20") {
  if (is.null(ds) || nrow(ds) == 0L) return(ds)
  pts <- terra::vect(as.matrix(ds[, c("X", "Y")]), crs = "EPSG:4326")
  ds[[col_name]] <- terra::extract(jac_rast, pts)[, 2]
  ds
}


#' Sample uniform presence points from species polygons (per batch)
#'
#' For each species in species_batch: filter polygons to extant + native
#' (presence in {1,2}, origin == 1), project to an equal-area CRS (Mollweide,
#' ESRI:54009), draw n_per_species uniform random points, transform back to
#' WGS84, extract CHELSA at the points, apply documented NA fills, drop ocean
#' points, standardize with env_mean_sd, and subset columns to var_pattern.
#'
#' Equal-area sampling avoids the lat-lon area distortion that pure
#' st_sample(...) on WGS84 polygons would introduce. Loads the CHELSA raster
#' stack once per batch (called inside a worker via dynamic branching).
#'
#' Returns a named list (by species) of per-species lists with xy (matrix)
#' and env_std (matrix). Species whose polygons all fail the filter are dropped.
sample_uniform_polygon_presences <- function(species_batch,
                                              all_taxa_polygons,
                                              jade_polygon_metadata,
                                              chelsa_var_meta, env_mean_sd,
                                              chelsa_bio_dir, var_pattern,
                                              n_per_species = 1100L,
                                              equal_area_crs = "ESRI:54009") {
  set.seed(31337L)

  meta <- jade_polygon_metadata |>
    dplyr::mutate(.row_idx = dplyr::row_number()) |>
    dplyr::filter(presence %in% c(1L, 2L), origin == 1L,
                  species %in% species_batch)

  rast_stack <- build_chelsa_rast_stack(chelsa_var_meta, chelsa_bio_dir)

  result <- list()
  for (i in seq_along(species_batch)) {
    sp <- species_batch[i]
    sp_meta <- meta |> dplyr::filter(species == sp)
    if (nrow(sp_meta) == 0L) next

    sp_poly <- all_taxa_polygons[sp_meta$.row_idx, ]
    sp_union <- sf::st_union(sp_poly)
    sp_eq <- sf::st_transform(sp_union, equal_area_crs)

    pts_eq <- sf::st_sample(sp_eq, size = n_per_species, type = "random")
    pts_wgs <- sf::st_transform(pts_eq, 4326)
    coords <- sf::st_coordinates(pts_wgs)

    pts_vect <- terra::vect(coords, crs = "EPSG:4326")
    extracted <- terra::extract(rast_stack, pts_vect)[, -1, drop = FALSE]
    extracted <- as.matrix(extracted)
    extracted <- apply_chelsa_na_fills(extracted, chelsa_var_meta)

    land <- complete.cases(extracted)
    extracted <- extracted[land, , drop = FALSE]
    coords    <- coords[land, , drop = FALSE]
    if (nrow(extracted) == 0L) next

    env_std <- standardize_env(extracted, env_mean_sd)
    keep <- grepl(var_pattern, colnames(env_std))
    env_std <- env_std[, keep, drop = FALSE]

    result[[sp]] <- list(xy = coords, env_std = env_std)

    if (i %% 10L == 0L) {
      message("  Uniform polygon sampling: ", i, "/", length(species_batch))
    }
  }
  result
}


#' Pool, standardize, and 20-var-subset presences for selected species
#'
#' Reads jade_31_train/val/test parquets, filters to species_vec, standardizes
#' env vars with env_mean_sd, subsets columns to var_pattern. Returns a named
#' list (by species) of per-species lists with xy and env_std elements.
build_pooled_presences_20 <- function(species_vec, train_path, val_path,
                                       test_path, env_mean_sd, var_pattern) {
  species_vec <- unique(species_vec)
  read_filt <- function(path) {
    arrow::open_dataset(path) |>
      dplyr::filter(species %in% !!species_vec) |>
      dplyr::collect()
  }
  pooled <- dplyr::bind_rows(
    read_filt(train_path), read_filt(val_path), read_filt(test_path)
  )

  env_cols_31 <- grep("^CHELSA_", names(pooled), value = TRUE)
  env_raw <- as.matrix(pooled[, env_cols_31])
  env_std <- standardize_env(env_raw, env_mean_sd)
  keep <- grepl(var_pattern, colnames(env_std))
  env_std <- env_std[, keep, drop = FALSE]

  by_sp <- split(seq_len(nrow(pooled)), pooled$species)
  lapply(by_sp, \(idx) list(
    xy      = as.matrix(pooled[idx, c("X", "Y")]),
    env_std = env_std[idx, , drop = FALSE]
  ))
}


#' Build one species' long-form dataset with both split-assignment columns
#'
#' Combines presences with backgrounds (sp$train_bg_env_std + train_bg_xy from
#' cached eval_batch_data) into one tibble. Adds:
#'   - split_random:  "train"/"test" for presences only (NA for bg)
#'   - split_spatial: "train"/"test" for both presences and bg (class-symmetric
#'                    block CV: blocks computed over all points, test blocks
#'                    contribute their pres + bg to the test set)
#'
#' Spatial blocks are adaptive per species: block_size = sqrt(bbox_area /
#' target_n_blocks) so each species gets ~target_n_blocks blocks regardless
#' of range size. Block size is recorded in block_size_deg for diagnostics.
#'
#' Returns NULL if presences are missing, the bbox is degenerate, or either
#' split would leave a class with no train or no test points.
build_species_dataset <- function(sp, pooled = NULL, test_frac = 0.20,
                                  target_n_blocks = 25L,
                                  n_test_bg = 500L) {
  if (is.null(sp)) return(NULL)
  if (!is.null(pooled)) {
    if (nrow(pooled$env_std) == 0L) return(NULL)
    sp$train_env_std <- pooled$env_std
    sp$train_xy      <- pooled$xy
  }
  n_pres <- nrow(sp$train_env_std)
  n_bg   <- nrow(sp$train_bg_env_std)

  set.seed(sum(utf8ToInt(sp$species)))

  # Random split (presences only — bg gets NA)
  is_test_random <- runif(n_pres) < test_frac

  # Class-symmetric spatial-block split: blocks computed over presences + bg,
  # test blocks contribute their points (both classes) to the test set.
  all_xy <- rbind(sp$train_xy, sp$train_bg_xy)
  xrange <- diff(range(all_xy[, 1]))
  yrange <- diff(range(all_xy[, 2]))
  bbox_area <- xrange * yrange
  if (!is.finite(bbox_area) || bbox_area <= 0) return(NULL)
  block_size <- sqrt(bbox_area / target_n_blocks)
  bx <- floor(all_xy[, 1] / block_size)
  by <- floor(all_xy[, 2] / block_size)
  all_block_id  <- paste(bx, by, sep = "_")
  pres_block_id <- all_block_id[seq_len(n_pres)]
  bg_block_id   <- all_block_id[(n_pres + 1L):length(all_block_id)]
  block_counts  <- table(all_block_id)
  target_test_n <- round(test_frac * length(all_block_id))

  is_test_pres_spatial <- NULL
  is_test_bg_spatial   <- NULL
  for (attempt in seq_len(5L)) {
    set.seed(sum(utf8ToInt(sp$species)) + 13L * attempt)
    shuffled <- sample(names(block_counts))
    cum <- cumsum(block_counts[shuffled])
    # pick the prefix length whose cumulative count is closest to target_test_n
    # (avoids systematic overshoot when one block crosses the threshold)
    k_over  <- which(cum >= target_test_n)[1]
    if (is.na(k_over)) k_over <- length(shuffled)
    k_under <- max(1L, k_over - 1L)
    k <- if (abs(cum[k_over]  - target_test_n) <
             abs(cum[k_under] - target_test_n)) k_over else k_under
    test_blocks <- shuffled[seq_len(k)]

    cand_pres <- pres_block_id %in% test_blocks
    cand_bg   <- bg_block_id   %in% test_blocks

    if (sum(cand_pres) >= 1L && sum(!cand_pres) >= 2L &&
        sum(cand_bg)   >= 1L && sum(!cand_bg)   >= 2L) {
      is_test_pres_spatial <- cand_pres
      is_test_bg_spatial   <- cand_bg
      break
    }
  }
  if (is.null(is_test_pres_spatial)) return(NULL)

  if (sum(!is_test_random) < 2L || sum(is_test_random) < 1L) return(NULL)

  # Subsample bg-test for split_spatial: of all bg in test blocks, keep
  # n_test_bg as the AUC scoring set; mark the rest as NA (unused). Same
  # per-species seed used by random split's bg-test subsample.
  bg_test_idx <- which(is_test_bg_spatial)
  bg_spatial_label <- ifelse(is_test_bg_spatial, "test", "train")
  if (length(bg_test_idx) > n_test_bg) {
    set.seed(sum(utf8ToInt(sp$species)) + 19L)
    keep <- sample.int(length(bg_test_idx), n_test_bg)
    drop <- bg_test_idx[-keep]
    bg_spatial_label[drop] <- NA_character_
  }

  pres <- tibble::as_tibble(sp$train_env_std) |>
    tibble::add_column(species = sp$species, .before = 1) |>
    tibble::add_column(point_type = "presence",
                       X = sp$train_xy[, 1], Y = sp$train_xy[, 2],
                       .after = "species") |>
    tibble::add_column(
      split_random   = ifelse(is_test_random,       "test", "train"),
      split_spatial  = ifelse(is_test_pres_spatial, "test", "train"),
      block_size_deg = block_size
    )

  bg <- tibble::as_tibble(sp$train_bg_env_std) |>
    tibble::add_column(species = sp$species, .before = 1) |>
    tibble::add_column(point_type = "background",
                       X = sp$train_bg_xy[, 1], Y = sp$train_bg_xy[, 2],
                       .after = "species") |>
    tibble::add_column(
      split_random   = NA_character_,
      split_spatial  = bg_spatial_label,
      block_size_deg = block_size
    )

  dplyr::bind_rows(pres, bg)
}


#' Run MaxEnt on one species' dataset (presence-only split column)
#'
#' Used for split_random: split_col splits presences only; backgrounds are all
#' used as training negatives, and a per-species deterministic 500-pt bg
#' subset (seed = sum(utf8ToInt(species)) + 19L, independent of n_subsample)
#' is held out for AUC scoring.
run_maxnet_on_dataset <- function(sp_dataset, split_col, n_subsample,
                                  n_test_bg = 500L) {
  if (is.null(sp_dataset) || nrow(sp_dataset) == 0) return(NULL)
  sp_name  <- sp_dataset$species[1]
  env_cols <- grep("^CHELSA_", names(sp_dataset), value = TRUE)

  pres <- sp_dataset[sp_dataset$point_type == "presence", ]
  bg   <- sp_dataset[sp_dataset$point_type == "background", ]

  pres_train <- pres[pres[[split_col]] == "train", ]
  pres_test  <- pres[pres[[split_col]] == "test",  ]
  if (nrow(pres_train) < 2L || nrow(pres_test) < 1L) return(NULL)

  set.seed(sum(utf8ToInt(sp_name)) + 1L)
  perm <- sample(seq_len(nrow(pres_train)))
  keep <- perm[seq_len(min(n_subsample, nrow(pres_train)))]
  pres_train <- pres_train[keep, ]

  if (nrow(bg) > n_test_bg) {
    set.seed(sum(utf8ToInt(sp_name)) + 19L)
    bg_test <- bg[sample.int(nrow(bg), n_test_bg), , drop = FALSE]
  } else {
    bg_test <- bg
  }

  train_env <- as.matrix(rbind(pres_train[, env_cols], bg[, env_cols]))
  train_lab <- c(rep(1, nrow(pres_train)), rep(0, nrow(bg)))
  test_env  <- as.matrix(rbind(pres_test[, env_cols],  bg_test[, env_cols]))

  scores <- run_maxnet_predict(train_env, train_lab, test_env)

  tibble::tibble(
    species      = sp_name,
    split_type   = sub("^split_", "", split_col),
    n_subsample  = n_subsample,
    n_train_used = nrow(pres_train),
    n_test_held  = nrow(pres_test),
    point_type   = c(rep("presence",  nrow(pres_test)),
                     rep("background", nrow(bg_test))),
    score        = scores
  )
}


#' Run MaxEnt with a class-symmetric split (split_col defined for both classes)
#'
#' Used for split_spatial (redesigned) and split_envblock: split_col is
#' "train" / "test" / NA for both presences and backgrounds. The dataset
#' itself has bg-test already subsampled to ~500 per species (other test-pool
#' bg are NA = unused). The scorer reads splits as-is — no internal
#' subsampling, so splits match the parquet exactly.
run_maxnet_on_class_symmetric_dataset <- function(sp_dataset, split_col,
                                                   n_subsample) {
  if (is.null(sp_dataset) || nrow(sp_dataset) == 0) return(NULL)
  sp_name  <- sp_dataset$species[1]
  env_cols <- grep("^CHELSA_", names(sp_dataset), value = TRUE)

  pres <- sp_dataset[sp_dataset$point_type == "presence",  ]
  bg   <- sp_dataset[sp_dataset$point_type == "background", ]

  isTRUE_str <- function(x, val) !is.na(x) & x == val
  pres_train <- pres[isTRUE_str(pres[[split_col]], "train"), ]
  pres_test  <- pres[isTRUE_str(pres[[split_col]], "test"),  ]
  bg_train   <- bg[isTRUE_str(bg[[split_col]],   "train"), ]
  bg_test    <- bg[isTRUE_str(bg[[split_col]],   "test"),  ]
  if (nrow(pres_train) < 2L || nrow(pres_test) < 1L) return(NULL)
  if (nrow(bg_train)   < 2L || nrow(bg_test)   < 1L) return(NULL)

  set.seed(sum(utf8ToInt(sp_name)) + 1L)
  perm <- sample(seq_len(nrow(pres_train)))
  keep <- perm[seq_len(min(n_subsample, nrow(pres_train)))]
  pres_train <- pres_train[keep, ]

  train_env <- as.matrix(rbind(pres_train[, env_cols], bg_train[, env_cols]))
  train_lab <- c(rep(1, nrow(pres_train)), rep(0, nrow(bg_train)))
  test_env  <- as.matrix(rbind(pres_test[, env_cols],  bg_test[, env_cols]))

  scores <- run_maxnet_predict(train_env, train_lab, test_env)

  tibble::tibble(
    species      = sp_name,
    split_type   = sub("^split_", "", split_col),
    n_subsample  = n_subsample,
    n_train_used = nrow(pres_train),
    n_test_held  = nrow(pres_test),
    point_type   = c(rep("presence",  nrow(pres_test)),
                     rep("background", nrow(bg_test))),
    score        = scores
  )
}


#' Run MaxEnt on a *batch* of species datasets (presence-only split column)
#'
#' Wrapper that iterates run_maxnet_on_dataset() over a list of per-species
#' datasets and returns a single concatenated tibble. Lets the targets pattern
#' run one batch (~50 species) per branch instead of one species per branch,
#' which dramatically reduces dispatch / serialization overhead.
run_maxnet_batch <- function(sp_dataset_batch, split_col, n_subsample) {
  purrr::map(sp_dataset_batch, \(sp) {
    run_maxnet_on_dataset(sp, split_col, n_subsample)
  }) |> purrr::compact() |> purrr::list_rbind()
}


#' Run MaxEnt on a *batch* of species datasets (class-symmetric split column)
run_maxnet_class_symmetric_batch <- function(sp_dataset_batch, split_col,
                                              n_subsample) {
  purrr::map(sp_dataset_batch, \(sp) {
    run_maxnet_on_class_symmetric_dataset(sp, split_col, n_subsample)
  }) |> purrr::compact() |> purrr::list_rbind()
}


#' Add a class-symmetric `split_random` column to the long-form parquet
#'
#' Presences: 80/20 random with seed sum(utf8ToInt(species)) -> "train"/"test".
#' Background: 500 sampled with seed sum + 19L -> "test"; rest -> NA.
#' Seeds match build_species_dataset() and run_maxnet_on_dataset() so the
#' "test" bg here is exactly the 500 used by the cached random_* MaxEnt
#' targets.
add_random_split_classes <- function(df, n_test_bg = 500L) {
  df$split_random <- NA_character_
  by_sp <- split(seq_len(nrow(df)), df$species)
  for (sp_name in names(by_sp)) {
    idx <- by_sp[[sp_name]]
    is_pres <- df$point_type[idx] == "presence"
    pres_idx <- idx[is_pres]
    bg_idx   <- idx[!is_pres]

    if (length(pres_idx) > 0L) {
      set.seed(sum(utf8ToInt(sp_name)))
      is_test <- runif(length(pres_idx)) < 0.20
      df$split_random[pres_idx] <- ifelse(is_test, "test", "train")
    }

    if (length(bg_idx) > 0L) {
      n_keep <- min(n_test_bg, length(bg_idx))
      set.seed(sum(utf8ToInt(sp_name)) + 19L)
      test_pos <- sample.int(length(bg_idx), n_keep)
      df$split_random[bg_idx[test_pos]] <- "test"
    }
  }
  df
}


#' Add a per-row `in_range_polygon` boolean to the long-form parquet
#'
#' TRUE iff (X, Y) lies inside any of the row's species's range polygons
#' (filtered to presence in {1, 2}, origin == 1). NA when the species has no
#' qualifying polygons.
add_in_range_polygon <- function(df, all_taxa_polygons, jade_polygon_metadata) {
  # Disable s2 for this operation: GEOS planar handles malformed IUCN
  # polygons (degenerate edges, self-intersections) more forgivingly.
  s2_was <- sf::sf_use_s2()
  sf::sf_use_s2(FALSE)
  on.exit(sf::sf_use_s2(s2_was), add = TRUE)

  meta <- jade_polygon_metadata |>
    dplyr::mutate(.row_idx = dplyr::row_number()) |>
    dplyr::filter(presence %in% c(1L, 2L), origin == 1L)

  df$in_range_polygon <- NA
  by_sp <- split(seq_len(nrow(df)), df$species)
  sp_names <- names(by_sp)
  n_sp <- length(sp_names)
  n_failed <- 0L
  for (i in seq_along(sp_names)) {
    sp_name <- sp_names[i]
    idx <- by_sp[[sp_name]]
    sp_meta <- meta |> dplyr::filter(species == sp_name)
    if (nrow(sp_meta) == 0L) next

    inside <- tryCatch({
      polys <- suppressWarnings(
        sf::st_make_valid(all_taxa_polygons[sp_meta$.row_idx, ]) |>
          sf::st_union()
      )
      # Use polys' CRS object directly to avoid a PROJ-database EPSG lookup
      # (which can fail in some conda/PROJ installs and silently leave
      # st_as_sf with crs = NA, then st_intersects errors on CRS mismatch).
      pts <- sf::st_as_sf(df[idx, c("X", "Y"), drop = FALSE],
                          coords = c("X", "Y"), crs = sf::st_crs(polys))
      lengths(suppressMessages(sf::st_intersects(pts, polys))) > 0L
    }, error = \(e) {
      n_failed <<- n_failed + 1L
      message("  in_range_polygon FAILED for ", sp_name, ": ",
              conditionMessage(e))
      rep(NA, length(idx))
    })
    df$in_range_polygon[idx] <- inside
    if (i %% 50L == 0L)
      message("  in_range_polygon: ", i, "/", n_sp,
              " (", n_failed, " failed)")
  }
  message("  in_range_polygon done: ", n_failed, " species failed (NA)")
  df
}


#' Add a Mahalanobis-ellipsoid environmental in-filling split column
#'
#' Adds split_envblock (and diagnostic envblock_md2_thresh, envblock_attempt)
#' to the long-form per-species dataset. Holes are species-specific
#' Mahalanobis ellipsoids (using the bg covariance) centered on n_holes_pres
#' presences + n_holes_bg backgrounds. Threshold is the (target_test_frac)
#' quantile of each point's minimum Mahalanobis-squared distance to any
#' center, so exactly ~target_test_frac of points are held out. Up to
#' max_attempts retries with distinct seeds if the draw degenerates.
add_envblock_split <- function(sp_dataset, n_holes = 30L,
                                target_test_frac = 0.20,
                                max_attempts = 5L,
                                n_test_bg = 500L) {
  if (is.null(sp_dataset) || nrow(sp_dataset) == 0L) return(NULL)
  sp_name  <- sp_dataset$species[1]
  env_cols <- grep("^CHELSA_", names(sp_dataset), value = TRUE)
  pres <- sp_dataset[sp_dataset$point_type == "presence",  ]
  bg   <- sp_dataset[sp_dataset$point_type == "background", ]
  if (nrow(bg) < (n_holes + 50L)) return(NULL)

  bg_env   <- as.matrix(bg[,   env_cols])
  pres_env <- as.matrix(pres[, env_cols])
  all_env  <- as.matrix(sp_dataset[, env_cols])

  cov_bg <- stats::cov(bg_env)
  prec   <- tryCatch(solve(cov_bg),
                     error = \(e) solve(cov_bg + 0.01 * diag(ncol(cov_bg))))

  n_holes_pres <- floor(n_holes / 2L)
  n_holes_bg   <- n_holes - n_holes_pres
  if (nrow(pres) < n_holes_pres) {
    n_holes_pres <- nrow(pres)
    n_holes_bg   <- n_holes - n_holes_pres
  }

  for (attempt in seq_len(max_attempts)) {
    set.seed(sum(utf8ToInt(sp_name)) + 7L * attempt)
    centers_pres <- if (n_holes_pres > 0L) {
      pres_env[sample.int(nrow(pres_env), n_holes_pres), , drop = FALSE]
    } else {
      pres_env[integer(0), , drop = FALSE]
    }
    centers_bg <- bg_env[sample.int(nrow(bg_env), n_holes_bg), , drop = FALSE]
    centers <- rbind(centers_pres, centers_bg)

    min_md2 <- rep(Inf, nrow(all_env))
    for (i in seq_len(nrow(centers))) {
      d <- sweep(all_env, 2, centers[i, ], FUN = "-")
      md2 <- rowSums((d %*% prec) * d)
      min_md2 <- pmin(min_md2, md2)
    }

    k <- max(1L, round(target_test_frac * nrow(sp_dataset)))
    thresh <- sort(min_md2)[k]
    is_test <- min_md2 <= thresh

    n_pres_test <- sum(is_test  & sp_dataset$point_type == "presence")
    n_pres_tr   <- sum(!is_test & sp_dataset$point_type == "presence")
    n_bg_test   <- sum(is_test  & sp_dataset$point_type == "background")
    n_bg_tr     <- sum(!is_test & sp_dataset$point_type == "background")

    if (n_pres_test >= 1L && n_pres_tr >= 2L &&
        n_bg_test   >= 1L && n_bg_tr   >= 2L) {
      label <- ifelse(is_test, "test", "train")

      # Subsample bg-test for split_envblock: of all bg in holes, keep
      # n_test_bg as the AUC scoring set; mark the rest as NA (unused).
      is_bg <- sp_dataset$point_type == "background"
      bg_test_idx <- which(is_bg & is_test)
      if (length(bg_test_idx) > n_test_bg) {
        set.seed(sum(utf8ToInt(sp_name)) + 19L)
        keep <- sample.int(length(bg_test_idx), n_test_bg)
        drop <- bg_test_idx[-keep]
        label[drop] <- NA_character_
      }

      sp_dataset$split_envblock      <- label
      sp_dataset$envblock_md2_thresh <- thresh
      sp_dataset$envblock_attempt    <- attempt
      return(sp_dataset)
    }
  }
  NULL
}


#' Flatten batch data into per-species list for dynamic branching
#'
#' Takes the output of prepare_eval_batch (list of batches, each a list of
#' species) and flattens to a single list with one entry per species.
#'
#' @param batch_data List of lists from prepare_eval_batch
#' @return Flat list, one element per species
flatten_batch_to_species <- function(batch_data) {
  list_flatten(batch_data)
}


# ===========================================================================
# Uniform-polygon training/test datasets (20-var) helpers
# ===========================================================================

#' Compute per-species AUC across a list of MaxEnt score lists
#'
#' Takes any number of MaxEnt score targets (each a list of per-species
#' tibbles with columns species, split_type, n_subsample, n_train_used,
#' n_test_held, point_type, score) and returns a single tibble with one
#' row per (species, split_type, n_subsample) and an `auc` column.
combine_maxent_auc <- function(...) {
  all <- purrr::list_rbind(purrr::map(list(...), purrr::list_rbind))
  if (nrow(all) == 0L) return(tibble::tibble())
  all |>
    dplyr::group_by(species, split_type, n_subsample,
                    n_train_used, n_test_held) |>
    dplyr::summarise(
      auc = {
        labels <- ifelse(point_type == "presence", 1, 0)
        if (length(unique(labels)) < 2L) NA_real_
        else tryCatch(
          as.numeric(pROC::auc(labels, score, quiet = TRUE,
                               direction = "<")),
          error = \(e) NA_real_
        )
      },
      .groups = "drop"
    )
}


#' List of species with at least one polygon passing the JADE clean filters
#' (presence in {1,2}, origin == 1).
build_eligible_uniform_species <- function(jade_polygon_metadata) {
  jade_polygon_metadata |>
    dplyr::filter(presence %in% c(1L, 2L), origin == 1L) |>
    dplyr::pull(species) |>
    unique()
}


#' Per-species metadata for stratification: taxon, range_size_km2, median_lat
#'
#' range_size_km2 = sum of qualifying polygon areas.
#' median_lat = area-weighted median of polygon centroid latitudes.
build_uniform_species_metadata <- function(jade_polygon_metadata,
                                            all_taxa_polygons) {
  meta <- jade_polygon_metadata |>
    dplyr::mutate(.row_idx = dplyr::row_number()) |>
    dplyr::filter(presence %in% c(1L, 2L), origin == 1L)

  # Centroid latitude per polygon (suppress geometry warnings)
  cents <- suppressWarnings(
    sf::st_centroid(all_taxa_polygons[meta$.row_idx, ]) |>
      sf::st_coordinates()
  )
  meta$cent_lat <- cents[, 2]

  meta |>
    dplyr::group_by(species, taxon) |>
    dplyr::summarise(
      range_size_km2 = sum(area_km2, na.rm = TRUE),
      median_lat     = matrixStats::weightedMedian(cent_lat, w = area_km2,
                                                    na.rm = TRUE),
      n_polys        = dplyr::n(),
      .groups = "drop"
    )
}


#' Stratified species selection (taxon x lat_band x range_size_band)
#'
#' Mirrors the eval_selected_species block but parameterized and without the
#' shot_type stratum.
select_stratified_species <- function(species_metadata, eligible_species,
                                       target_n = 2500L, seed = 42L) {
  set.seed(seed)
  meta <- species_metadata |>
    dplyr::filter(species %in% eligible_species) |>
    dplyr::mutate(
      lat_band = cut(abs(median_lat),
                     breaks = c(0, 15, 35, 90),
                     labels = c("low", "mid", "high"),
                     include.lowest = TRUE),
      size_band = cut(range_size_km2,
                      breaks = quantile(range_size_km2,
                                        c(0, 1/3, 2/3, 1), na.rm = TRUE),
                      labels = c("small", "mid", "large"),
                      include.lowest = TRUE),
      stratum = paste(taxon, lat_band, size_band, sep = "_")
    )

  n_strata <- length(unique(meta$stratum))
  per_stratum <- ceiling(target_n / n_strata)

  meta <- meta |>
    dplyr::mutate(max_n = ifelse(size_band == "large",
                                  ceiling(per_stratum * 0.5),
                                  per_stratum))

  sampled <- meta |>
    dplyr::group_by(stratum) |>
    dplyr::mutate(.rand = runif(dplyr::n())) |>
    dplyr::arrange(.rand, .by_group = TRUE) |>
    dplyr::filter(dplyr::row_number() <= max_n[1]) |>
    dplyr::ungroup() |>
    dplyr::select(-.rand)

  if (nrow(sampled) > target_n) {
    sampled <- dplyr::slice_sample(sampled, n = target_n)
  }

  message("Selected ", nrow(sampled), " species across ", n_strata, " strata")
  sampled$species
}


#' Split a character vector of species into batches of a given size
split_into_batches <- function(species_vec, batch_size = 50L) {
  split(species_vec, ceiling(seq_along(species_vec) / batch_size))
}


#' Write a uniform-samples list to a parquet, optionally adding jacobian_20
#'
#' @param uniform_samples Named list (by species) of list(xy, env_std)
#' @param species_keep   Character vector of species to include
#' @param path           Output parquet path
#' @param species_metadata Tibble with species + taxon (joined as a column)
#' @param add_jacobian   Logical, append jacobian_20 column
#' @param jacobian_raster_path Path to the 20-var Jacobian raster (used iff
#'   add_jacobian = TRUE)
write_uniform_samples_parquet <- function(uniform_samples, species_keep, path,
                                           species_metadata,
                                           add_jacobian = FALSE,
                                           jacobian_raster_path = NULL) {
  jac <- if (add_jacobian) terra::rast(jacobian_raster_path) else NULL
  taxon_lookup <- setNames(species_metadata$taxon, species_metadata$species)

  rows <- purrr::map(species_keep, \(sp) {
    s <- uniform_samples[[sp]]
    if (is.null(s) || nrow(s$env_std) == 0L) return(NULL)
    df <- tibble::as_tibble(s$env_std) |>
      tibble::add_column(species = sp, taxon = unname(taxon_lookup[sp]),
                         .before = 1) |>
      tibble::add_column(X = s$xy[, 1], Y = s$xy[, 2], .after = "taxon")
    if (add_jacobian) {
      pts <- terra::vect(s$xy, crs = "EPSG:4326")
      df$jacobian_20 <- terra::extract(jac, pts)[, 2]
    }
    df
  }) |> purrr::compact() |> purrr::list_rbind()

  dir.create(dirname(path), showWarnings = FALSE, recursive = TRUE)
  arrow::write_parquet(rows, path)
  path
}


#' Prepare an evaluation batch using uniform-polygon presences as the source
#'
#' Mirror of prepare_eval_batch() but takes per-species presences from a
#' named-list of uniform samples (output of sample_uniform_polygon_presences).
#' Generates n_background backgrounds + n_test_bg-pt scoring subset per species.
#' Returns the same per-species list shape as eval_batch_data so downstream
#' build_species_dataset() works unchanged.
prepare_uniform_eval_batch <- function(species_batch, uniform_samples,
                                        chelsa_var_meta, env_mean_sd,
                                        chelsa_bio_dir, var_pattern,
                                        n_background = 5000L,
                                        n_test_bg = 500L,
                                        radius_multiplier = 0.25) {
  rast_stack <- build_chelsa_rast_stack(chelsa_var_meta, chelsa_bio_dir)

  results <- vector("list", length(species_batch))
  for (i in seq_along(species_batch)) {
    sp <- species_batch[i]
    sp_data <- uniform_samples[[sp]]
    if (is.null(sp_data) || nrow(sp_data$xy) == 0L) next

    bg_xy <- generate_background_points(sp_data$xy, n_background,
                                         radius_multiplier = radius_multiplier)
    pts <- terra::vect(bg_xy, crs = "EPSG:4326")
    bg_extracted <- terra::extract(rast_stack, pts)[, -1, drop = FALSE]
    bg_env_raw <- as.matrix(bg_extracted)
    bg_env_raw <- apply_chelsa_na_fills(bg_env_raw, chelsa_var_meta)

    land <- complete.cases(bg_env_raw)
    bg_env_raw <- bg_env_raw[land, , drop = FALSE]
    bg_xy <- bg_xy[land, , drop = FALSE]
    if (nrow(bg_env_raw) == 0L) next

    bg_env_std <- standardize_env(bg_env_raw, env_mean_sd)
    keep <- grepl(var_pattern, colnames(bg_env_std))
    bg_env_std <- bg_env_std[, keep, drop = FALSE]

    n_bg_avail <- nrow(bg_env_std)
    test_bg_idx <- if (n_bg_avail > n_test_bg) {
      sample.int(n_bg_avail, n_test_bg)
    } else {
      seq_len(n_bg_avail)
    }

    results[[i]] <- list(
      species = sp,
      train_env_std    = sp_data$env_std,
      test_env_std     = sp_data$env_std,
      train_bg_env_std = bg_env_std,
      test_bg_env_std  = bg_env_std[test_bg_idx, , drop = FALSE],
      train_xy         = sp_data$xy,
      test_xy          = sp_data$xy,
      train_bg_xy      = bg_xy,
      test_bg_xy       = bg_xy[test_bg_idx, , drop = FALSE],
      n_train          = nrow(sp_data$xy),
      n_test           = nrow(sp_data$xy)
    )

    if (i %% 10L == 0L) {
      message("  Uniform eval batch: ", i, "/", length(species_batch))
    }
  }
  compact(results)
}


#' Read the species->bin assignment from the existing 4-bin JADE chunks
#'
#' Returns a named list (names = "0".."3") of character vectors of species
#' in each bin. A species may appear in multiple bins (rare — happens for
#' a handful of species when the upstream splitter created duplicates).
read_jade_bin_species_map <- function(jade_bins_dir, n_bins = 4L) {
  result <- list()
  for (b in 0:(n_bins - 1L)) {
    path <- file.path(jade_bins_dir,
                      sprintf("jade_samples_clean_%02d.parquet", b))
    ds <- arrow::open_dataset(path)
    sp <- ds |> dplyr::distinct(species) |> dplyr::collect() |>
      dplyr::pull(species)
    result[[as.character(b)]] <- sp
  }
  result
}


#' Sample a batch of species uniformly within polygons, n_per_species rows,
#' with raw CHELSA env values + jacobian per row, routed into per-bin
#' sub-data-frames so the downstream parquet write can stitch by bin
#' without ever holding the full ~99M-row table in memory.
#'
#' Returns a named list (names = "0".."3") of data.frames. Each df has
#' the schema X, Y, 20 alphabetically-sorted CHELSA cols, jacobian,
#' species, taxon — matching jade_samples_clean_*.parquet exactly.
sample_uniform_train_batch <- function(species_batch, bin_species_map,
                                        all_taxa_polygons,
                                        jade_polygon_metadata,
                                        chelsa_var_meta, chelsa_bio_dir,
                                        jacobian_raster_path,
                                        bioclim_include_pattern,
                                        n_per_species = 5000L,
                                        equal_area_crs = "ESRI:54009") {
  rast_stack <- build_chelsa_rast_stack(chelsa_var_meta, chelsa_bio_dir)
  jac_rast   <- terra::rast(jacobian_raster_path)

  # Inverted lookup: species -> character vector of bin keys it belongs to
  sp_to_bins <- list()
  for (bk in names(bin_species_map)) {
    for (sp in bin_species_map[[bk]]) {
      sp_to_bins[[sp]] <- c(sp_to_bins[[sp]], bk)
    }
  }

  meta <- jade_polygon_metadata |>
    dplyr::mutate(.row_idx = dplyr::row_number()) |>
    dplyr::filter(presence %in% c(1L, 2L), origin == 1L,
                  species %in% species_batch)

  bin_dfs <- setNames(vector("list", length(bin_species_map)),
                      names(bin_species_map))
  for (b in names(bin_dfs)) bin_dfs[[b]] <- list()

  set.seed(31337L)
  for (i in seq_along(species_batch)) {
    sp <- species_batch[i]
    if (is.null(sp_to_bins[[sp]])) next  # not in any bin (shouldn't happen)

    sp_meta <- meta |> dplyr::filter(species == sp)
    if (nrow(sp_meta) == 0L) next

    sp_poly <- all_taxa_polygons[sp_meta$.row_idx, ]
    sp_union <- suppressWarnings(sf::st_union(sp_poly))
    sp_eq <- suppressWarnings(sf::st_transform(sp_union, equal_area_crs))

    pts_eq <- suppressMessages(
      sf::st_sample(sp_eq, size = n_per_species, type = "random")
    )
    pts_wgs <- sf::st_transform(pts_eq, sf::st_crs(all_taxa_polygons))
    coords <- sf::st_coordinates(pts_wgs)

    pts_vect <- terra::vect(coords, crs = sf::st_crs(all_taxa_polygons)$wkt)
    extracted <- terra::extract(rast_stack, pts_vect)[, -1, drop = FALSE]
    extracted <- as.matrix(extracted)
    extracted <- apply_chelsa_na_fills(extracted, chelsa_var_meta)

    land <- complete.cases(extracted)
    extracted <- extracted[land, , drop = FALSE]
    coords    <- coords[land, , drop = FALSE]
    if (nrow(extracted) == 0L) next

    keep <- grepl(bioclim_include_pattern, colnames(extracted))
    extracted <- extracted[, keep, drop = FALSE]

    jac_vals <- terra::extract(jac_rast,
                                terra::vect(coords,
                                            crs = sf::st_crs(all_taxa_polygons)$wkt))[, 2]

    n_have <- nrow(extracted)
    if (n_have < n_per_species) {
      idx <- sample.int(n_have, n_per_species, replace = TRUE)
      extracted <- extracted[idx, , drop = FALSE]
      coords    <- coords[idx, , drop = FALSE]
      jac_vals  <- jac_vals[idx]
    }

    env_cols_sorted <- sort(colnames(extracted))
    df <- data.frame(X = coords[, 1], Y = coords[, 2],
                     check.names = FALSE)
    for (col in env_cols_sorted) df[[col]] <- extracted[, col]
    df$jacobian <- jac_vals
    df$species <- sp
    df$taxon   <- sp_meta$taxon[1]

    for (bk in sp_to_bins[[sp]]) {
      bin_dfs[[bk]][[length(bin_dfs[[bk]]) + 1L]] <- df
    }

    if (i %% 10L == 0L) {
      message("  uniform_train_batch: ", i, "/", length(species_batch))
    }
  }

  purrr::map(bin_dfs, \(dfs) if (length(dfs) == 0L) NULL else
                              purrr::list_rbind(dfs))
}


#' Aggregate per-batch lists-of-bin-dfs into 4 final parquets at out_dir.
#' Each iteration holds only one bin's rows in memory at a time.
write_uniform_clean_4bins <- function(batch_data_list, out_dir,
                                       n_bins = 4L) {
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  paths <- character(n_bins)
  for (b in 0:(n_bins - 1L)) {
    bin_key <- as.character(b)
    bin_df <- purrr::map(batch_data_list,
                         \(branch) branch[[bin_key]]) |>
      purrr::compact() |>
      purrr::list_rbind()
    out <- file.path(out_dir, sprintf("uniform_clean_%02d.parquet", b))
    arrow::write_parquet(bin_df, out)
    message("  wrote ", out, " (", nrow(bin_df), " rows, ",
            dplyr::n_distinct(bin_df$species), " species)")
    paths[b + 1L] <- out
    rm(bin_df); gc(verbose = FALSE)
  }
  paths
}


#' Build a per-species JADE-resampled presence list for the eval pipeline
#'
#' Reads the cached `data/processed/jade_samples_clean.parquet` (the 5000-pt
#' JADE accept-reject samples for ~18k species), filters to the supplied
#' eval species, caps each species to `n_per_species` rows, and z-scores
#' the 20 bioclim columns by matching `chelsa_var_meta$chelsa_name` to the
#' parquet column names (positional alignment via meta index → env_mean_sd
#' mean/sd). Returns a named list (one element per species) with `xy` and
#' `env_std` matrices — identical shape to `uniform_samples_20`, so it
#' plugs straight into `prepare_uniform_eval_batch()` as a drop-in.
build_jade_samples_for_eval <- function(jade_parquet_path,
                                         eval_species,
                                         env_mean_sd,
                                         chelsa_var_meta,
                                         bioclim_include_pattern,
                                         n_per_species = 1100L,
                                         seed = 31337L) {
  message("Reading JADE samples for ", length(eval_species), " species...")
  ds <- arrow::open_dataset(jade_parquet_path)
  raw <- ds |>
    dplyr::filter(species %in% eval_species) |>
    dplyr::collect()
  message("  loaded ", nrow(raw), " rows from ",
          dplyr::n_distinct(raw$species), " species")

  env_cols <- grep(bioclim_include_pattern, names(raw), value = TRUE)

  # Map each parquet env column to its position in env_mean_sd (which is
  # ordered by chelsa_var_meta). Column "CHELSA_bio1_1981-2010_V.2.1" -> "bio1".
  short_names <- sub("^CHELSA_([^_]+)_.*", "\\1", env_cols)
  meta_idx <- match(short_names, chelsa_var_meta$chelsa_name)
  if (any(is.na(meta_idx))) {
    stop("Cannot match these JADE columns to chelsa_var_meta$chelsa_name: ",
         paste(env_cols[is.na(meta_idx)], collapse = ", "))
  }
  env_mean_vec <- env_mean_sd$mean[meta_idx]
  env_sd_vec   <- env_mean_sd$sd[meta_idx]

  by_sp <- split(raw, raw$species)
  set.seed(seed)
  result <- purrr::map(by_sp, \(df_sp) {
    n <- nrow(df_sp)
    if (n > n_per_species) {
      df_sp <- df_sp[sample.int(n, n_per_species), , drop = FALSE]
    }
    xy <- as.matrix(df_sp[, c("X", "Y")])
    env_raw <- as.matrix(df_sp[, env_cols])
    env_std <- sweep(env_raw, 2, env_mean_vec, "-")
    env_std <- sweep(env_std, 2, env_sd_vec, "/")
    colnames(env_std) <- env_cols
    list(xy = xy, env_std = env_std)
  })

  message("  built per-species list for ", length(result), " species")
  result
}


#' Sample background points uniformly within the union of all RESOLVE 2017
#' ecoregions touched by any presence point of the species
#'
#' Returns an XY matrix in WGS84 of size n_background * oversample_factor.
#' Caller drops ocean/NA rows via complete.cases() and truncates to
#' n_background after CHELSA extraction.
generate_ecoregion_union_bg <- function(presence_xy, ecoregions_sf,
                                         n_background,
                                         oversample_factor = 5L,
                                         equal_area_crs = "ESRI:54009") {
  s2_was <- sf::sf_use_s2()
  sf::sf_use_s2(FALSE)
  on.exit(sf::sf_use_s2(s2_was), add = TRUE)

  pts <- sf::st_as_sf(as.data.frame(presence_xy),
                      coords = c(1L, 2L), crs = "EPSG:4326")
  hit_idx <- unique(unlist(sf::st_intersects(pts, ecoregions_sf)))
  if (length(hit_idx) == 0L) return(matrix(numeric(0), ncol = 2))

  union_wgs <- suppressWarnings(
    sf::st_union(sf::st_make_valid(ecoregions_sf[hit_idx, ]))
  )
  union_eq  <- sf::st_transform(union_wgs, equal_area_crs)

  pts_eq <- suppressMessages(
    sf::st_sample(union_eq, size = n_background * oversample_factor,
                  type = "random")
  )
  pts_wgs <- sf::st_transform(pts_eq, 4326)
  sf::st_coordinates(pts_wgs)
}


#' Wide-bg variant of prepare_uniform_eval_batch(): bg sampled uniformly in
#' the ecoregion union touched by presences, instead of presence-anchored noise.
prepare_uniform_eval_batch_widebg <- function(species_batch, uniform_samples,
                                               ecoregions_sf,
                                               chelsa_var_meta, env_mean_sd,
                                               chelsa_bio_dir, var_pattern,
                                               n_background = 5000L,
                                               n_test_bg = 500L) {
  rast_stack <- build_chelsa_rast_stack(chelsa_var_meta, chelsa_bio_dir)

  results <- vector("list", length(species_batch))
  for (i in seq_along(species_batch)) {
    sp <- species_batch[i]
    sp_data <- uniform_samples[[sp]]
    if (is.null(sp_data) || nrow(sp_data$xy) == 0L) next

    one <- tryCatch({
      bg_xy <- generate_ecoregion_union_bg(sp_data$xy, ecoregions_sf,
                                            n_background)
      if (!is.matrix(bg_xy) || nrow(bg_xy) < 50L) return(NULL)
      bg_xy <- as.matrix(bg_xy)

      pts <- terra::vect(bg_xy, crs = "EPSG:4326")
      bg_extracted <- terra::extract(rast_stack, pts)[, -1, drop = FALSE]
      bg_env_raw <- as.matrix(bg_extracted)
      bg_env_raw <- apply_chelsa_na_fills(bg_env_raw, chelsa_var_meta)

      land <- complete.cases(bg_env_raw)
      bg_env_raw <- bg_env_raw[land, , drop = FALSE]
      bg_xy      <- bg_xy[land, , drop = FALSE]
      if (nrow(bg_env_raw) < 50L) return(NULL)

      if (nrow(bg_env_raw) > n_background) {
        keep_idx <- sample.int(nrow(bg_env_raw), n_background)
        bg_env_raw <- bg_env_raw[keep_idx, , drop = FALSE]
        bg_xy      <- bg_xy[keep_idx, , drop = FALSE]
      }
      n_keep <- nrow(bg_env_raw)

      bg_env_std <- standardize_env(bg_env_raw, env_mean_sd)
      keep <- grepl(var_pattern, colnames(bg_env_std))
      bg_env_std <- bg_env_std[, keep, drop = FALSE]

      test_bg_idx <- if (n_keep > n_test_bg) {
        sample.int(n_keep, n_test_bg)
      } else {
        seq_len(n_keep)
      }

      list(
        species = sp,
        train_env_std    = sp_data$env_std,
        test_env_std     = sp_data$env_std,
        train_bg_env_std = bg_env_std,
        test_bg_env_std  = bg_env_std[test_bg_idx, , drop = FALSE],
        train_xy         = sp_data$xy,
        test_xy          = sp_data$xy,
        train_bg_xy      = bg_xy,
        test_bg_xy       = bg_xy[test_bg_idx, , drop = FALSE],
        n_train          = nrow(sp_data$xy),
        n_test           = nrow(sp_data$xy)
      )
    }, error = \(e) {
      message("  widebg FAILED for ", sp, ": ", conditionMessage(e))
      NULL
    })

    if (!is.null(one)) results[[i]] <- one

    if (i %% 10L == 0L) {
      message("  Uniform widebg eval batch: ", i, "/", length(species_batch))
    }
  }
  compact(results)
}


#' Score a batch of species with NicheFlow methods (GPU)
#'
#' Loads all three models once, scores each species with generative KDE
#' and approximate LL methods.
#'
#' @param batch_species_data List of per-species data from prepare_eval_batch
#' @param species_map Named integer vector
#' @param vae_checkpoint Path to VAE checkpoint
#' @param flow_checkpoint_dir NichEncoder checkpoint dir
#' @param geode_checkpoint Path to GeODE checkpoint
#' @param xy_mean_sd XY standardization stats
#' @param active_dims Active latent dimensions
#' @param vae_latent_dim Full latent dim
#' @param device Torch device
#' @param kde_n_gen Number of geographic points to generate for KDE
#' @param ll_K IWAE importance samples
#' @param ll_ode_steps LL reverse ODE steps
#' @return List of per-species score tibbles
score_nicheflow_batch <- function(batch_species_data, species_map,
                                  vae_checkpoint, flow_checkpoint_dir,
                                  active_dims,
                                  vae_latent_dim = 16L,
                                  device = "cuda:0",
                                  ll_K = 5L,
                                  ll_ode_steps = 50L) {
  # Load VAE + NichEncoder (no GeODE needed for LL scoring)
  vae_model <- env_vae_mod(31L, vae_latent_dim)
  load_model_checkpoint(vae_model, vae_checkpoint)
  vae_model <- vae_model$to(device = device)
  vae_model$eval()

  flow_ckpt <- find_latest_checkpoint(flow_checkpoint_dir)
  flow_model <- nichencoder_traj_net(
    coord_dim = length(active_dims),
    n_species = length(species_map),
    spec_embed_dim = 64L,
    breadths = c(512L, 256L, 128L)
  )
  load_model_checkpoint(flow_model, flow_ckpt$path)
  flow_model <- flow_model$to(device = device)
  flow_model$eval()

  # Per-batch log file for tracking progress
  log_file <- file.path("logs", paste0("nicheflow_scores_batch_",
                                        format(Sys.time(), "%Y%m%d_%H%M%S"),
                                        ".log"))
  dir.create(dirname(log_file), showWarnings = FALSE, recursive = TRUE)
  log_msg <- function(...) {
    msg <- paste0("[", format(Sys.time(), "%H:%M:%S"), "] ", ...)
    cat(msg, "\n", file = log_file, append = TRUE)
    message(msg)
  }

  log_msg("Starting NicheFlow scoring batch: ", length(batch_species_data),
          " species, K=", ll_K, ", steps=", ll_ode_steps)

  results <- vector("list", length(batch_species_data))

  for (i in seq_along(batch_species_data)) {
    sp_data <- batch_species_data[[i]]
    sp <- sp_data$species
    sp_id <- species_map[sp]
    if (is.na(sp_id)) next

    n_test <- sp_data$n_test
    n_bg <- nrow(sp_data$test_bg_env_std)
    t0 <- Sys.time()

    log_msg(i, "/", length(batch_species_data), " ", sp, " (id=", sp_id,
            ", n_test=", n_test, ", n_bg=", n_bg, ") starting...")

    results[[i]] <- tryCatch({
      # -- LL method (approximate log-likelihood) --
      log_msg("  computing LL...")
      all_env <- rbind(sp_data$test_env_std, sp_data$test_bg_env_std)
      ll_scores <- compute_log_density(
        all_env, as.integer(sp_id),
        vae_model, flow_model, active_dims,
        K = ll_K, ode_steps = ll_ode_steps,
        batch_size = 600L, device = device
      )

      elapsed <- round(as.numeric(Sys.time() - t0, units = "secs"), 1)
      log_msg("  done ", elapsed, "s",
              " LL_range=[", round(min(ll_scores), 1), ",",
              round(max(ll_scores), 1), "]")

      tibble(
        species = sp,
        point_type = c(rep("presence", n_test),
                       rep("background", n_bg)),
        score_ll = ll_scores
      )
    }, error = \(e) {
      log_msg("  ERROR: ", e$message)
      tibble(
        species = sp,
        point_type = c(rep("presence", n_test),
                       rep("background", n_bg)),
        score_ll = rep(NA_real_, n_test + n_bg)
      )
    })

    gc(verbose = FALSE)
    if (grepl("cuda", device)) cuda_empty_cache()
  }

  compact(results)
}


#' Run MaxEnt for one species (CPU, for dynamic branching)
#'
#' @param sp_data Single species data list from prepare_eval_batch
#' @return Tibble with species, point_type, score_maxent
run_maxnet_species <- function(sp_data) {
  sp <- sp_data$species

  # Training: presences + full background (5000)
  n_train_bg <- min(nrow(sp_data$train_bg_env_std), sp_data$n_train * 5)
  bg_idx <- sample.int(nrow(sp_data$train_bg_env_std), n_train_bg)
  train_env <- rbind(sp_data$train_env_std, sp_data$train_bg_env_std[bg_idx, ])
  train_labels <- c(rep(1, sp_data$n_train), rep(0, n_train_bg))

  # Test scoring: presences + test background (500 subsample)
  test_env <- rbind(sp_data$test_env_std, sp_data$test_bg_env_std)

  scores <- run_maxnet_predict(train_env, train_labels, test_env)

  tibble(
    species = sp,
    point_type = c(rep("presence", sp_data$n_test),
                   rep("background", nrow(sp_data$test_bg_env_std))),
    score_maxent = scores
  )
}


#' Run balanced Random Forest for one species (CPU, for dynamic branching)
#'
#' @param sp_data Single species data list from prepare_eval_batch
#' @return Tibble with species, point_type, score_rf
run_rf_species <- function(sp_data) {
  sp <- sp_data$species

  # Training: presences + full background (5000)
  n_train_bg <- min(nrow(sp_data$train_bg_env_std), sp_data$n_train * 5)
  bg_idx <- sample.int(nrow(sp_data$train_bg_env_std), n_train_bg)
  train_env <- rbind(sp_data$train_env_std, sp_data$train_bg_env_std[bg_idx, ])
  train_labels <- factor(
    c(rep("yes", sp_data$n_train), rep("no", n_train_bg)),
    levels = c("no", "yes")
  )

  # Test scoring: presences + test background (500 subsample)
  test_env <- rbind(sp_data$test_env_std, sp_data$test_bg_env_std)

  scores <- run_rf_predict(train_env, train_labels, test_env)

  tibble(
    species = sp,
    point_type = c(rep("presence", sp_data$n_test),
                   rep("background", nrow(sp_data$test_bg_env_std))),
    score_rf = scores
  )
}


#' Combine all method scores and compute metrics
#'
#' Joins NicheFlow, MaxEnt, and RF scores, computes AUC/TSS/PR-AUC
#' per species per method, and attaches species metadata.
#'
#' @param nicheflow_scores List of per-batch score tibbles (from score_nicheflow_batch)
#' @param maxent_scores List of per-species maxent tibbles
#' @param rf_scores List of per-species RF tibbles
#' @param species_metadata Tibble from build_species_metadata
#' @return Tibble with species, method, roc_auc, pr_auc, tss, + metadata cols
combine_and_compute_metrics <- function(nicheflow_scores,
                                        maxent_scores,
                                        rf_scores,
                                        species_metadata) {
  # Helper: compute metrics for one method's score table
  compute_method_metrics <- function(score_list, score_col, method_label) {
    score_df <- list_rbind(compact(score_list))
    if (nrow(score_df) == 0) return(tibble())

    score_df |>
      group_by(species) |>
      group_map(\(dat, key) {
        truth <- ifelse(dat$point_type == "presence", 1, 0)
        scores <- dat[[score_col]]
        if (all(is.na(scores))) {
          return(tibble(species = key$species, method = method_label,
                        roc_auc = NA_real_, pr_auc = NA_real_, tss = NA_real_))
        }
        metrics <- compute_eval_metrics(truth, scores)
        tibble(species = key$species, method = method_label,
               roc_auc = metrics$roc_auc, pr_auc = metrics$pr_auc,
               tss = metrics$tss)
      }) |> list_rbind()
  }

  # Flatten NicheFlow scores (batched → flat list)
  nf_flat <- if (is.list(nicheflow_scores[[1]]) && !is.data.frame(nicheflow_scores[[1]]))
    list_flatten(nicheflow_scores) else nicheflow_scores

  # Compute metrics independently per method (no join needed)
  metric_results <- list_rbind(list(
    compute_method_metrics(nf_flat, "score_ll", "nicheflow_ll"),
    compute_method_metrics(maxent_scores, "score_maxent", "maxent"),
    compute_method_metrics(rf_scores, "score_rf", "rf")
  ))

  # Attach metadata
  metric_results |>
    left_join(species_metadata, by = "species")
}


# ===========================================================================
# Geographic EMD Evaluation
# ===========================================================================

#' Create geographic prediction grid for a species
#'
#' @param presence_xy Matrix [N, 2] of presence (lon, lat)
#' @param chelsa_var_meta Variable metadata
#' @param env_mean_sd Standardization stats
#' @param chelsa_bio_dir CHELSA raster directory
#' @param buffer_frac Fractional buffer around bbox
#' @param grid_resolution Grid cell size in degrees
#' @param max_cells Max grid cells
#' @return List with grid_xy [M, 2], grid_env_std [M, 31]
create_prediction_grid <- function(presence_xy, chelsa_var_meta, env_mean_sd,
                                   chelsa_bio_dir,
                                   buffer_frac = 0.2,
                                   grid_resolution = 0.1,
                                   max_cells = 50000L) {
  # Bounding box with buffer
  lon_range <- range(presence_xy[, 1])
  lat_range <- range(presence_xy[, 2])
  lon_buf <- diff(lon_range) * buffer_frac
  lat_buf <- diff(lat_range) * buffer_frac

  lon_seq <- seq(lon_range[1] - lon_buf, lon_range[2] + lon_buf,
                 by = grid_resolution)
  lat_seq <- seq(lat_range[1] - lat_buf, lat_range[2] + lat_buf,
                 by = grid_resolution)

  # Coarsen if too many cells
  while (length(lon_seq) * length(lat_seq) > max_cells) {
    grid_resolution <- grid_resolution * 1.5
    lon_seq <- seq(lon_range[1] - lon_buf, lon_range[2] + lon_buf,
                   by = grid_resolution)
    lat_seq <- seq(lat_range[1] - lat_buf, lat_range[2] + lat_buf,
                   by = grid_resolution)
  }

  grid_xy <- expand.grid(lon = lon_seq, lat = lat_seq)
  grid_xy <- as.matrix(grid_xy)

  # Extract CHELSA at grid cells (all 31 vars in correct order)
  rast_stack <- build_chelsa_rast_stack(chelsa_var_meta, chelsa_bio_dir)
  pts <- terra::vect(grid_xy, crs = "EPSG:4326")
  extracted <- terra::extract(rast_stack, pts)
  env_raw <- as.matrix(extracted[, -1, drop = FALSE])

  # Apply documented NA fill values (e.g., snow/frost = 0 in tropics)
  for (j in seq_len(nrow(chelsa_var_meta))) {
    fill_val <- chelsa_var_meta$na_fill[j]
    if (!is.na(fill_val)) {
      na_mask <- is.na(env_raw[, j])
      if (any(na_mask)) env_raw[na_mask, j] <- fill_val
    }
  }

  # Filter to land cells (remaining NAs are true ocean/missing)
  land_mask <- complete.cases(env_raw)
  grid_xy <- grid_xy[land_mask, , drop = FALSE]
  env_raw <- env_raw[land_mask, , drop = FALSE]
  env_std <- standardize_env(env_raw, env_mean_sd)

  list(grid_xy = grid_xy, grid_env_std = env_std)
}


#' Compute geographic EMD between predictions and truth
#'
#' @param pred_weights Numeric vector of prediction scores on grid
#' @param grid_xy Matrix [M, 2] of grid (lon, lat)
#' @param truth_xy Matrix [N, 2] of true presence (lon, lat)
#' @param p Wasserstein-p distance
#' @param max_points Max points for EMD computation
#' @return Scalar EMD
compute_geographic_emd <- function(pred_weights, grid_xy, truth_xy,
                                   p = 1, max_points = 1000L) {
  # Normalize prediction weights to sum to 1
  pred_weights[pred_weights < 0] <- 0
  pred_weights <- pred_weights / sum(pred_weights)

  # Subsample if needed
  n_grid <- nrow(grid_xy)
  n_truth <- nrow(truth_xy)

  if (n_grid > max_points) {
    idx <- sample.int(n_grid, max_points, replace = FALSE,
                      prob = pred_weights)
    grid_xy <- grid_xy[idx, , drop = FALSE]
    pred_weights <- pred_weights[idx]
    pred_weights <- pred_weights / sum(pred_weights)
  }

  if (n_truth > max_points) {
    idx <- sample.int(n_truth, max_points, replace = FALSE)
    truth_xy <- truth_xy[idx, , drop = FALSE]
  }

  # Truth weights: uniform
  truth_weights <- rep(1 / nrow(truth_xy), nrow(truth_xy))

  # Create weighted point patterns for transport package
  a <- transport::wpp(grid_xy, pred_weights)
  b <- transport::wpp(truth_xy, truth_weights)

  transport::wasserstein(a, b, p = p)
}


#' Evaluate geographic EMD for a batch of species
#'
#' @param species_batch Character vector of species
#' @param jade_test_data Test data
#' @param jade_train_data Train data (for MaxEnt/RF)
#' @param species_map Named integer vector
#' @param chelsa_var_meta Variable metadata
#' @param env_mean_sd Standardization stats
#' @param chelsa_bio_dir CHELSA raster dir
#' @param vae_checkpoint VAE checkpoint path
#' @param flow_checkpoint_dir NichEncoder checkpoint dir
#' @param geode_checkpoint GeODE checkpoint path
#' @param xy_mean_sd XY stats
#' @param active_dims Active dims
#' @param vae_latent_dim Full latent dim
#' @param device Torch device
#' @return Tibble(species, method, emd)
evaluate_emd_batch <- function(species_batch, jade_test_data, jade_train_data,
                               species_map, chelsa_var_meta, env_mean_sd,
                               chelsa_bio_dir,
                               vae_checkpoint, flow_checkpoint_dir,
                               geode_checkpoint, xy_mean_sd,
                               active_dims, vae_latent_dim = 16L,
                               device = "cuda:0") {
  env_cols <- setdiff(names(jade_test_data),
                      c("X", "Y", "jacobian", "species", "taxon", "split_type"))

  # Load models
  vae_model <- env_vae_mod(31L, vae_latent_dim)
  load_model_checkpoint(vae_model, vae_checkpoint)
  vae_model <- vae_model$to(device = device)
  vae_model$eval()

  flow_ckpt <- find_latest_checkpoint(flow_checkpoint_dir)
  flow_model <- nichencoder_traj_net(
    coord_dim = length(active_dims),
    n_species = length(species_map),
    spec_embed_dim = 64L,
    breadths = c(512L, 256L, 128L)
  )
  load_model_checkpoint(flow_model, flow_ckpt$path)
  flow_model <- flow_model$to(device = device)
  flow_model$eval()

  geode_model <- load_geode_model(geode_checkpoint, device = device)

  # Load CHELSA raster stack once for all species in batch
  emd_rast <- build_chelsa_rast_stack(chelsa_var_meta, chelsa_bio_dir)

  results <- vector("list", length(species_batch))

  for (i in seq_along(species_batch)) {
    sp <- species_batch[i]
    sp_id <- species_map[sp]
    if (is.na(sp_id)) next

    # Wrap per-species EMD in tryCatch so one failure doesn't kill the batch
    results[[i]] <- tryCatch({
      sp_test <- jade_test_data |> filter(species == sp)
      sp_train <- jade_train_data |> filter(species == sp)
      if (nrow(sp_test) == 0) return(NULL)

      truth_xy <- cbind(sp_test$X, sp_test$Y)

      # Build grid over the full background area (all presences + buffer)
      all_pres_xy <- rbind(
        cbind(sp_train$X, sp_train$Y), truth_xy)
      grid <- create_prediction_grid(
        all_pres_xy, chelsa_var_meta, env_mean_sd,
        chelsa_bio_dir, buffer_frac = 0.25)
      if (nrow(grid$grid_xy) < 10) {
        message("  Skipping ", sp, ": grid too small (",
                nrow(grid$grid_xy), " cells)")
        return(NULL)
      }

      # -- NicheFlow: generate geographic points --
      gen_xy <- generate_geo_samples(
        sp_id, 10000L, flow_model, vae_model, geode_model,
        active_dims, xy_mean_sd, vae_latent_dim, device
      )

      # Direct point-to-point EMD
      n_gen <- nrow(gen_xy)
      n_truth <- nrow(truth_xy)
      gen_sub <- if (n_gen > 1000L) {
        gen_xy[sample.int(n_gen, 1000L), , drop = FALSE]
      } else gen_xy
      truth_sub <- if (n_truth > 1000L) {
        truth_xy[sample.int(n_truth, 1000L), , drop = FALSE]
      } else truth_xy
      gen_w <- rep(1 / nrow(gen_sub), nrow(gen_sub))
      truth_w <- rep(1 / nrow(truth_sub), nrow(truth_sub))
      emd_direct <- transport::wasserstein(
        transport::wpp(gen_sub, gen_w),
        transport::wpp(truth_sub, truth_w), p = 1
      )

      # KDE-weighted grid EMD
      kde_scores <- score_geographic_kde(gen_xy, grid$grid_xy)
      emd_kde <- compute_geographic_emd(
        kde_scores, grid$grid_xy, truth_xy)

      # -- MaxEnt + RF on grid --
      train_env <- standardize_env(sp_train[, env_cols], env_mean_sd)
      bg_xy <- generate_background_points(all_pres_xy, 5000L)

      pts <- terra::vect(bg_xy, crs = "EPSG:4326")
      bg_raw <- as.matrix(
        terra::extract(emd_rast, pts)[, -1])
      bg_raw <- apply_chelsa_na_fills(bg_raw, chelsa_var_meta)
      bg_land <- complete.cases(bg_raw)
      bg_raw <- bg_raw[bg_land, , drop = FALSE]
      bg_env <- standardize_env(bg_raw, env_mean_sd)

      mx_train <- rbind(train_env, bg_env)
      mx_labels <- c(rep(1, nrow(train_env)),
                      rep(0, nrow(bg_env)))
      mx_scores <- run_maxnet_predict(
        mx_train, mx_labels, grid$grid_env_std)
      emd_mx <- compute_geographic_emd(
        mx_scores, grid$grid_xy, truth_xy)

      rf_labels <- factor(
        c(rep("yes", nrow(train_env)),
          rep("no", nrow(bg_env))),
        levels = c("no", "yes"))
      rf_scores <- run_rf_predict(
        mx_train, rf_labels, grid$grid_env_std)
      emd_rf <- compute_geographic_emd(
        rf_scores, grid$grid_xy, truth_xy)

      tibble(
        species = sp,
        method = c("nicheflow_direct", "nicheflow_kde",
                    "maxent", "rf"),
        emd = c(emd_direct, emd_kde, emd_mx, emd_rf)
      )
    }, error = \(e) {
      message("  EMD error for ", sp, ": ", e$message)
      tibble(
        species = sp,
        method = c("nicheflow_direct", "nicheflow_kde",
                    "maxent", "rf"),
        emd = rep(NA_real_, 4)
      )
    })

    if (i %% 5 == 0) {
      message("  EMD: ", i, "/", length(species_batch))
    }

    gc(verbose = FALSE)
    if (grepl("cuda", device)) cuda_empty_cache()
  }

  list_rbind(compact(results))
}
