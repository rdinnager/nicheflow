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

#' Extract CHELSA-BIOCLIM+ variables at arbitrary coordinates
#'
#' Loads CHELSA rasters and extracts environmental values at given (lon, lat)
#' coordinates. Returns standardized values using the same mean/sd as VAE.
#'
#' @param xy Matrix [N, 2] of (lon, lat) coordinates
#' @param chelsa_var_meta Tibble with file paths and variable info
#' @param env_mean_sd List with mean and sd vectors for standardization
#' @param chelsa_bio_dir Path to CHELSA bio directory
#' @param bioclim_pattern Regex pattern for filtering bioclim files
#' @return Tibble with X, Y, and 31 standardized env columns
extract_chelsa_at_coords <- function(xy, chelsa_var_meta, env_mean_sd,
                                     chelsa_bio_dir, bioclim_pattern) {
  # Build raster stack in the same variable order as chelsa_var_meta
  rast_files <- list.files(chelsa_bio_dir, full.names = TRUE) |>
    str_subset(bioclim_pattern)

  # Match file order to chelsa_var_meta
  var_names <- chelsa_var_meta$variable
  rast_stack <- terra::rast(rast_files)

  # Extract values at coordinates
  pts <- terra::vect(xy, crs = "EPSG:4326")
  extracted <- terra::extract(rast_stack, pts)

  # Drop the ID column from terra::extract
  env_raw <- as.matrix(extracted[, -1, drop = FALSE])

  # Handle NAs: fill with column means (same as VAE training)
  for (j in seq_len(ncol(env_raw))) {
    na_mask <- is.na(env_raw[, j])
    if (any(na_mask)) {
      env_raw[na_mask, j] <- env_mean_sd$mean[j]  # fill with training mean
    }
  }

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
  # Center point for LAEA projection
  center_lon <- mean(generated_xy[, 1])
  center_lat <- mean(generated_xy[, 2])

  # Define LAEA projection centered on species
  laea_crs <- sprintf(
    "+proj=laea +lat_0=%f +lon_0=%f +datum=WGS84 +units=m",
    center_lat, center_lon
  )

  # Project generated points
  gen_sf <- st_as_sf(as.data.frame(generated_xy),
                     coords = c("V1", "V2"), crs = 4326) |>
    st_transform(laea_crs)
  gen_coords <- st_coordinates(gen_sf)

  # Project evaluation points
  eval_sf <- st_as_sf(as.data.frame(eval_xy),
                      coords = c("V1", "V2"), crs = 4326) |>
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

  mxnet <- maxnet::maxnet(
    p = train_labels,
    data = X_train,
    regmult = 1,
    maxnet::maxnet.formula(train_labels, X_train, classes = "default")
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

  preds <- predict(rf, as.data.frame(test_env_std), type = "prob")
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
#' @param bioclim_pattern Regex for bioclim file filtering
#' @param n_background Background points per species
#' @return List of per-species data (list of lists)
prepare_eval_batch <- function(species_batch, jade_train_data,
                               jade_test_data, chelsa_var_meta,
                               env_mean_sd, chelsa_bio_dir,
                               bioclim_pattern, n_background = 5000L) {
  env_cols <- setdiff(names(jade_test_data),
                      c("X", "Y", "jacobian", "species", "taxon", "split_type"))

  # Load CHELSA raster stack once for background extraction
  rast_files <- list.files(chelsa_bio_dir, full.names = TRUE) |>
    str_subset(bioclim_pattern)
  rast_stack <- terra::rast(rast_files)

  results <- vector("list", length(species_batch))

  for (i in seq_along(species_batch)) {
    sp <- species_batch[i]

    sp_train <- jade_train_data |> filter(species == sp)
    sp_test <- jade_test_data |> filter(species == sp)

    if (nrow(sp_train) == 0 || nrow(sp_test) == 0) next

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

    # Fill NAs with training mean
    for (j in seq_len(ncol(bg_env_raw))) {
      na_mask <- is.na(bg_env_raw[, j])
      if (any(na_mask)) bg_env_raw[na_mask, j] <- env_mean_sd$mean[j]
    }

    # Standardize
    bg_env_std <- standardize_env(bg_env_raw, env_mean_sd)

    # Standardize presence env
    train_env_std <- standardize_env(sp_train[, env_cols], env_mean_sd)
    test_env_std <- standardize_env(sp_test[, env_cols], env_mean_sd)

    results[[i]] <- list(
      species = sp,
      train_env_std = train_env_std,
      test_env_std = test_env_std,
      bg_env_std = bg_env_std,
      train_xy = cbind(sp_train$X, sp_train$Y),
      test_xy = cbind(sp_test$X, sp_test$Y),
      bg_xy = bg_xy,
      n_train = nrow(sp_train),
      n_test = nrow(sp_test)
    )

    if (i %% 10 == 0) {
      message("  Data prep: ", i, "/", length(species_batch), " species")
    }
  }

  compact(results)
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
                                  geode_checkpoint, xy_mean_sd,
                                  active_dims,
                                  vae_latent_dim = 16L,
                                  device = "cuda:0",
                                  kde_n_gen = 10000L,
                                  ll_K = 5L,
                                  ll_ode_steps = 50L) {
  # Load all three models
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

  results <- vector("list", length(batch_species_data))

  for (i in seq_along(batch_species_data)) {
    sp_data <- batch_species_data[[i]]
    sp <- sp_data$species
    sp_id <- species_map[sp]
    if (is.na(sp_id)) next

    n_test <- sp_data$n_test
    n_bg <- nrow(sp_data$bg_env_std)

    # -- KDE method --
    tryCatch({
      gen_xy <- generate_geo_samples(
        sp_id, kde_n_gen, flow_model, vae_model, geode_model,
        active_dims, xy_mean_sd, vae_latent_dim, device
      )

      eval_xy <- rbind(sp_data$test_xy, sp_data$bg_xy)
      kde_scores <- score_geographic_kde(gen_xy, eval_xy)
    }, error = \(e) {
      message("  KDE error for ", sp, ": ", e$message)
      kde_scores <<- rep(NA_real_, n_test + n_bg)
    })

    # -- LL method --
    tryCatch({
      all_env <- rbind(sp_data$test_env_std, sp_data$bg_env_std)
      ll_scores <- compute_log_density(
        all_env, as.integer(sp_id),
        vae_model, flow_model, active_dims,
        K = ll_K, ode_steps = ll_ode_steps,
        batch_size = 500L, device = device
      )
    }, error = \(e) {
      message("  LL error for ", sp, ": ", e$message)
      ll_scores <<- rep(NA_real_, n_test + n_bg)
    })

    results[[i]] <- tibble(
      species = sp,
      point_type = c(rep("presence", n_test), rep("background", n_bg)),
      score_kde = kde_scores,
      score_ll = ll_scores
    )

    if (i %% 5 == 0) {
      message("  NicheFlow scores: ", i, "/", length(batch_species_data))
    }

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

  # Training data: presences + background subsample
  n_train_bg <- min(nrow(sp_data$bg_env_std), sp_data$n_train * 5)
  bg_idx <- sample.int(nrow(sp_data$bg_env_std), n_train_bg)
  train_env <- rbind(sp_data$train_env_std, sp_data$bg_env_std[bg_idx, ])
  train_labels <- c(rep(1, sp_data$n_train), rep(0, n_train_bg))

  # Test data: presences + all background
  test_env <- rbind(sp_data$test_env_std, sp_data$bg_env_std)

  scores <- tryCatch(
    run_maxnet_predict(train_env, train_labels, test_env),
    error = \(e) {
      message("  MaxEnt error for ", sp, ": ", e$message)
      rep(NA_real_, nrow(test_env))
    }
  )

  tibble(
    species = sp,
    point_type = c(rep("presence", sp_data$n_test),
                   rep("background", nrow(sp_data$bg_env_std))),
    score_maxent = scores
  )
}


#' Run balanced Random Forest for one species (CPU, for dynamic branching)
#'
#' @param sp_data Single species data list from prepare_eval_batch
#' @return Tibble with species, point_type, score_rf
run_rf_species <- function(sp_data) {
  sp <- sp_data$species

  # Training data: presences + background subsample
  n_train_bg <- min(nrow(sp_data$bg_env_std), sp_data$n_train * 5)
  bg_idx <- sample.int(nrow(sp_data$bg_env_std), n_train_bg)
  train_env <- rbind(sp_data$train_env_std, sp_data$bg_env_std[bg_idx, ])
  train_labels <- factor(
    c(rep("yes", sp_data$n_train), rep("no", n_train_bg)),
    levels = c("no", "yes")
  )

  # Test data
  test_env <- rbind(sp_data$test_env_std, sp_data$bg_env_std)

  scores <- tryCatch(
    run_rf_predict(train_env, train_labels, test_env),
    error = \(e) {
      message("  RF error for ", sp, ": ", e$message)
      rep(NA_real_, nrow(test_env))
    }
  )

  tibble(
    species = sp,
    point_type = c(rep("presence", sp_data$n_test),
                   rep("background", nrow(sp_data$bg_env_std))),
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
  # Flatten and combine NicheFlow scores
  nf_all <- list_rbind(
    if (is.list(nicheflow_scores[[1]]) && !is.data.frame(nicheflow_scores[[1]]))
      list_flatten(nicheflow_scores)
    else
      nicheflow_scores
  )

  mx_all <- list_rbind(maxent_scores)
  rf_all <- list_rbind(rf_scores)

  # Join all scores
  all_scores <- nf_all |>
    left_join(mx_all |> select(species, point_type, score_maxent),
              by = c("species", "point_type")) |>
    left_join(rf_all |> select(species, point_type, score_rf),
              by = c("species", "point_type"))

  # Compute metrics per species per method
  species_list <- unique(all_scores$species)
  methods <- c("score_kde", "score_ll", "score_maxent", "score_rf")
  method_labels <- c("nicheflow_kde", "nicheflow_ll", "maxent", "rf")

  metric_results <- map(species_list, \(sp) {
    sp_scores <- all_scores |> filter(species == sp)
    truth <- ifelse(sp_scores$point_type == "presence", 1, 0)

    map2(methods, method_labels, \(col, label) {
      scores <- sp_scores[[col]]
      if (all(is.na(scores))) {
        return(tibble(species = sp, method = label,
                      roc_auc = NA_real_, pr_auc = NA_real_, tss = NA_real_))
      }
      metrics <- compute_eval_metrics(truth, scores)
      tibble(species = sp, method = label,
             roc_auc = metrics$roc_auc, pr_auc = metrics$pr_auc,
             tss = metrics$tss)
    }) |> list_rbind()
  }) |> list_rbind()

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
#' @param bioclim_pattern Regex for bioclim files
#' @param buffer_frac Fractional buffer around bbox
#' @param grid_resolution Grid cell size in degrees
#' @param max_cells Max grid cells
#' @return List with grid_xy [M, 2], grid_env_std [M, 31]
create_prediction_grid <- function(presence_xy, chelsa_var_meta, env_mean_sd,
                                   chelsa_bio_dir, bioclim_pattern,
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

  # Extract CHELSA at grid cells
  rast_files <- list.files(chelsa_bio_dir, full.names = TRUE) |>
    str_subset(bioclim_pattern)
  rast_stack <- terra::rast(rast_files)
  pts <- terra::vect(grid_xy, crs = "EPSG:4326")
  extracted <- terra::extract(rast_stack, pts)
  env_raw <- as.matrix(extracted[, -1, drop = FALSE])

  # Filter to land cells (non-NA)
  land_mask <- complete.cases(env_raw)
  grid_xy <- grid_xy[land_mask, , drop = FALSE]
  env_raw <- env_raw[land_mask, , drop = FALSE]

  # Fill remaining NAs and standardize
  for (j in seq_len(ncol(env_raw))) {
    na_mask <- is.na(env_raw[, j])
    if (any(na_mask)) env_raw[na_mask, j] <- env_mean_sd$mean[j]
  }
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
#' @param bioclim_pattern Bioclim regex
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
                               chelsa_bio_dir, bioclim_pattern,
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

  results <- vector("list", length(species_batch))

  for (i in seq_along(species_batch)) {
    sp <- species_batch[i]
    sp_id <- species_map[sp]
    if (is.na(sp_id)) next

    sp_test <- jade_test_data |> filter(species == sp)
    sp_train <- jade_train_data |> filter(species == sp)
    if (nrow(sp_test) == 0) next

    truth_xy <- cbind(sp_test$X, sp_test$Y)

    # Create prediction grid
    grid <- tryCatch(
      create_prediction_grid(truth_xy, chelsa_var_meta, env_mean_sd,
                             chelsa_bio_dir, bioclim_pattern),
      error = \(e) { message("  Grid error: ", sp, ": ", e$message); NULL }
    )
    if (is.null(grid)) next

    sp_results <- list()

    # -- NicheFlow KDE --
    tryCatch({
      gen_xy <- generate_geo_samples(
        sp_id, 10000L, flow_model, vae_model, geode_model,
        active_dims, xy_mean_sd, vae_latent_dim, device
      )
      kde_scores <- score_geographic_kde(gen_xy, grid$grid_xy)
      emd_kde <- compute_geographic_emd(kde_scores, grid$grid_xy, truth_xy)
      sp_results <- c(sp_results, list(
        tibble(species = sp, method = "nicheflow_kde", emd = emd_kde)))
    }, error = \(e) message("  EMD KDE error: ", sp, ": ", e$message))

    # -- NicheFlow LL --
    tryCatch({
      ll_scores <- compute_log_density(
        grid$grid_env_std, as.integer(sp_id),
        vae_model, flow_model, active_dims,
        K = 5L, ode_steps = 50L, batch_size = 500L, device = device
      )
      # Convert log-density to positive weights via exp
      ll_weights <- exp(ll_scores - max(ll_scores))
      emd_ll <- compute_geographic_emd(ll_weights, grid$grid_xy, truth_xy)
      sp_results <- c(sp_results, list(
        tibble(species = sp, method = "nicheflow_ll", emd = emd_ll)))
    }, error = \(e) message("  EMD LL error: ", sp, ": ", e$message))

    # -- MaxEnt --
    tryCatch({
      train_env <- standardize_env(sp_train[, env_cols], env_mean_sd)
      all_pres_xy <- rbind(cbind(sp_train$X, sp_train$Y), truth_xy)
      bg_xy <- generate_background_points(all_pres_xy, 5000L)

      rast_files <- list.files(chelsa_bio_dir, full.names = TRUE) |>
        str_subset(bioclim_pattern)
      rast_stack <- terra::rast(rast_files)
      pts <- terra::vect(bg_xy, crs = "EPSG:4326")
      bg_raw <- as.matrix(terra::extract(rast_stack, pts)[, -1])
      for (j in seq_len(ncol(bg_raw))) {
        na_m <- is.na(bg_raw[, j])
        if (any(na_m)) bg_raw[na_m, j] <- env_mean_sd$mean[j]
      }
      bg_env <- standardize_env(bg_raw, env_mean_sd)

      mx_train_env <- rbind(train_env, bg_env)
      mx_labels <- c(rep(1, nrow(train_env)), rep(0, nrow(bg_env)))
      mx_grid_scores <- run_maxnet_predict(mx_train_env, mx_labels,
                                           grid$grid_env_std)
      emd_mx <- compute_geographic_emd(mx_grid_scores, grid$grid_xy, truth_xy)
      sp_results <- c(sp_results, list(
        tibble(species = sp, method = "maxent", emd = emd_mx)))
    }, error = \(e) message("  EMD MaxEnt error: ", sp, ": ", e$message))

    # -- RF --
    tryCatch({
      rf_labels <- factor(c(rep("yes", nrow(train_env)),
                            rep("no", nrow(bg_env))),
                          levels = c("no", "yes"))
      rf_grid_scores <- run_rf_predict(mx_train_env, rf_labels,
                                       grid$grid_env_std)
      emd_rf <- compute_geographic_emd(rf_grid_scores, grid$grid_xy, truth_xy)
      sp_results <- c(sp_results, list(
        tibble(species = sp, method = "rf", emd = emd_rf)))
    }, error = \(e) message("  EMD RF error: ", sp, ": ", e$message))

    results[[i]] <- list_rbind(sp_results)

    if (i %% 5 == 0) {
      message("  EMD: ", i, "/", length(species_batch))
    }

    gc(verbose = FALSE)
    if (grepl("cuda", device)) cuda_empty_cache()
  }

  list_rbind(compact(results))
}
