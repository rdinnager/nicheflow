#' Disdat Benchmark Data Preparation
#'
#' Functions for loading species distribution benchmark data from the
#' disdat R package and extracting CHELSA-BIOCLIM+ variables at the
#' occurrence and survey coordinates.
#'
#' @author rdinnage


#' Load disdat region data and extract CHELSA-BIOCLIM+ variables
#'
#' For one of 6 regions, loads presence-only training data and
#' presence-absence test data from disdat, then extracts CHELSA-BIOCLIM+
#' variables at all coordinates.
#'
#' @param region Character, one of: AWT, NSW, CAN, NZ, SA, SWI
#' @param chelsa_var_meta Variable metadata tibble
#' @param env_mean_sd Standardization stats (mean, sd vectors)
#' @param chelsa_bio_dir Path to CHELSA bio raster directory
#' @param bioclim_pattern Regex for filtering bioclim files
#' @return List with train_data, bg_data, test_data, region
prepare_disdat_region <- function(region, chelsa_var_meta, env_mean_sd,
                                  chelsa_bio_dir, bioclim_pattern) {
  message("Preparing disdat region: ", region)

  # Load disdat data
  po <- disdat::disPo(region)  # presence-only
  bg <- disdat::disBg(region)  # background points
  pa <- disdat::disPa(region)  # presence-absence test data

  # Load CHELSA raster stack
  rast_files <- list.files(chelsa_bio_dir, full.names = TRUE) |>
    stringr::str_subset(bioclim_pattern)
  rast_stack <- terra::rast(rast_files)

  # -- Training presences --
  po_xy <- cbind(po$x, po$y)
  po_extracted <- terra::extract(rast_stack, terra::vect(po_xy, crs = "EPSG:4326"))
  po_env_raw <- as.matrix(po_extracted[, -1, drop = FALSE])

  # Fill NAs
  for (j in seq_len(ncol(po_env_raw))) {
    na_mask <- is.na(po_env_raw[, j])
    if (any(na_mask)) po_env_raw[na_mask, j] <- env_mean_sd$mean[j]
  }

  po_env_std <- sweep(po_env_raw, 2, env_mean_sd$mean, "-")
  po_env_std <- sweep(po_env_std, 2, env_mean_sd$sd, "/")

  train_data <- dplyr::tibble(
    species = po$spid,
    x = po$x,
    y = po$y
  ) |>
    dplyr::bind_cols(dplyr::as_tibble(po_env_std))

  # -- Background --
  bg_xy <- cbind(bg$x, bg$y)
  bg_extracted <- terra::extract(rast_stack, terra::vect(bg_xy, crs = "EPSG:4326"))
  bg_env_raw <- as.matrix(bg_extracted[, -1, drop = FALSE])

  for (j in seq_len(ncol(bg_env_raw))) {
    na_mask <- is.na(bg_env_raw[, j])
    if (any(na_mask)) bg_env_raw[na_mask, j] <- env_mean_sd$mean[j]
  }

  bg_env_std <- sweep(bg_env_raw, 2, env_mean_sd$mean, "-")
  bg_env_std <- sweep(bg_env_std, 2, env_mean_sd$sd, "/")

  bg_data <- dplyr::tibble(
    x = bg$x,
    y = bg$y
  ) |>
    dplyr::bind_cols(dplyr::as_tibble(bg_env_std))

  # -- Test data (presence-absence) --
  pa_xy <- cbind(pa$x, pa$y)
  pa_extracted <- terra::extract(rast_stack, terra::vect(pa_xy, crs = "EPSG:4326"))
  pa_env_raw <- as.matrix(pa_extracted[, -1, drop = FALSE])

  for (j in seq_len(ncol(pa_env_raw))) {
    na_mask <- is.na(pa_env_raw[, j])
    if (any(na_mask)) pa_env_raw[na_mask, j] <- env_mean_sd$mean[j]
  }

  pa_env_std <- sweep(pa_env_raw, 2, env_mean_sd$mean, "-")
  pa_env_std <- sweep(pa_env_std, 2, env_mean_sd$sd, "/")

  test_data <- dplyr::tibble(
    species = pa$spid,
    x = pa$x,
    y = pa$y,
    pa = pa$pa
  ) |>
    dplyr::bind_cols(dplyr::as_tibble(pa_env_std))

  message("  ", region, ": ", nrow(train_data), " train presences, ",
          nrow(bg_data), " background, ", nrow(test_data), " test PA")

  list(
    train_data = train_data,
    bg_data = bg_data,
    test_data = test_data,
    region = region
  )
}


#' JADE resample disdat training data using global Jacobian raster
#'
#' Extracts Jacobian values at training presence coordinates from the
#' precomputed global CHELSA Jacobian raster, then applies accept-reject
#' thinning to correct for geographic sampling bias.
#'
#' @param train_data Tibble from prepare_disdat_region()$train_data
#' @param jacobian_raster_path Path to global Jacobian raster
#' @return Tibble with JADE-resampled rows (subset of train_data)
jade_resample_disdat <- function(train_data, jacobian_raster_path) {
  message("JADE resampling disdat training data (", nrow(train_data), " points)")

  jac_rast <- terra::rast(jacobian_raster_path)

  # Extract Jacobian values at training coordinates
  xy <- cbind(train_data$x, train_data$y)
  jac_vals <- terra::extract(jac_rast, terra::vect(xy, crs = "EPSG:4326"))
  jac_vals <- jac_vals[[2]]  # first column is ID

  # Replace NAs with 0 (ocean/missing → reject)
  jac_vals[is.na(jac_vals)] <- 0

  # Normalize to probabilities
  if (max(jac_vals) > 0) {
    accept_prob <- jac_vals / max(jac_vals)
  } else {
    message("  Warning: all Jacobian values are 0 or NA")
    return(train_data)
  }

  # Accept-reject sampling
  keep <- runif(length(accept_prob)) < accept_prob
  result <- train_data[keep, ]

  message("  Kept ", nrow(result), " of ", nrow(train_data),
          " points (", round(100 * nrow(result) / nrow(train_data), 1), "%)")

  result
}


#' Export disdat benchmark data as parquet files
#'
#' @param region_data List from prepare_disdat_region()
#' @param jade_resampled Tibble from jade_resample_disdat()
#' @param output_dir Base output directory
#' @return Character vector of output file paths
export_disdat_parquet <- function(region_data, jade_resampled,
                                  output_dir = "data/processed/disdat") {
  region <- region_data$region
  dir.create(file.path(output_dir, region), recursive = TRUE,
             showWarnings = FALSE)

  paths <- c(
    train = file.path(output_dir, region, "train.parquet"),
    bg = file.path(output_dir, region, "background.parquet"),
    test = file.path(output_dir, region, "test_pa.parquet"),
    jade = file.path(output_dir, region, "train_jade.parquet")
  )

  arrow::write_parquet(region_data$train_data, paths["train"])
  arrow::write_parquet(region_data$bg_data, paths["bg"])
  arrow::write_parquet(region_data$test_data, paths["test"])
  arrow::write_parquet(jade_resampled, paths["jade"])

  message("Exported disdat ", region, ": ", length(paths), " files")

  as.character(paths)
}
