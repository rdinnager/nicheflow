#' CHELSA Global Environmental Tensor Pipeline Functions
#'
#' Functions for loading CHELSA-BIOCLIM+ environmental data into torch
#' float32 tensors, computing standardization statistics, and preparing
#' datasets for VAE and GeODE training.
#'
#' Pipeline: rasters -> land mask -> raw tensor -> standardization -> standardized tensor
#'
#' IMPORTANT: Two R torch limitations for tensors with >2^31 elements:
#'   1. safetensors::torch_save() segfaults (raw vector size limit)
#'   2. Whole-tensor ops (isnan, mean, std) may segfault
#' We work around both by operating column-by-column throughout.
#' Each column has ~308M elements (< 2^31), safely under the limit.
#'
#' @author rdinnage


# ===========================================================================
# Tensor save/load helpers (bypass safetensors 2 GB limit)
# ===========================================================================

#' Save a large 2D tensor column-by-column
#'
#' Works around the safetensors 2 GB crash by saving each column
#' as a separate .pt file. Each column of a 308M-row float32 tensor
#' is ~1.2 GB, safely under the limit.
#'
#' @param tensor 2D torch tensor to save
#' @param dir_path Directory to save column files + metadata
save_large_tensor <- function(tensor, dir_path) {
  dir.create(dir_path, recursive = TRUE, showWarnings = FALSE)
  n_cols <- tensor$shape[2]

  meta <- list(
    shape = as.integer(tensor$shape),
    dtype = tensor$dtype$.type(),
    n_cols = n_cols
  )
  saveRDS(meta, file.path(dir_path, "meta.rds"))

  for (i in seq_len(n_cols)) {
    col_path <- file.path(dir_path, sprintf("col_%03d.pt", i))
    torch::torch_save(tensor[, i], col_path)
  }

  total_size <- sum(file.size(
    list.files(dir_path, pattern = "\\.pt$", full.names = TRUE)
  ))
  message("Saved ", n_cols, " columns to ", dir_path,
          " (", round(total_size / 1e9, 2), " GB total)")
}


#' Load a large 2D tensor from column files
#'
#' Reconstructs a tensor from per-column .pt files saved by
#' \code{save_large_tensor()}.
#'
#' @param dir_path Directory containing column files + meta.rds
#' @return Reconstructed 2D torch tensor
load_large_tensor <- function(dir_path) {
  meta <- readRDS(file.path(dir_path, "meta.rds"))
  n_rows <- meta$shape[1]
  n_cols <- meta$shape[2]

  tensor <- torch::torch_zeros(n_rows, n_cols, dtype = torch::torch_float32())

  for (i in seq_len(n_cols)) {
    col_path <- file.path(dir_path, sprintf("col_%03d.pt", i))
    tensor[, i] <- torch::torch_load(col_path)
  }

  message("Loaded tensor: ", paste(tensor$shape, collapse = " x "),
          " from ", dir_path)
  tensor
}


# ===========================================================================
# Pipeline functions
# ===========================================================================

#' Build land mask index from Natural Earth polygons and CHELSA grid
#'
#' Rasterizes Natural Earth 10m land polygons to the CHELSA grid
#' resolution and returns the integer indices of all land pixels.
#'
#' @param land_sf sf object of Natural Earth land polygons (from ne_download())
#' @param chelsa_dir Path to CHELSA bio raster directory
#' @return List with land_idx, n_land, n_total
build_land_mask <- function(land_sf, chelsa_dir) {
  ref_rast <- terra::rast(file.path(chelsa_dir,
                                     "CHELSA_bio1_1981-2010_V.2.1.tif"))
  message("Raster dims: ", terra::nrow(ref_rast), " x ", terra::ncol(ref_rast),
          " = ", format(terra::ncell(ref_rast), big.mark = ","), " cells")
  message("Resolution: ", terra::xres(ref_rast), " degrees")

  message("Rasterizing land polygons...")
  land_mask <- terra::rasterize(terra::vect(land_sf), ref_rast,
                                 field = 1, background = NA)
  land_idx <- which(!is.na(terra::values(land_mask)))
  n_land <- length(land_idx)
  n_total <- terra::ncell(ref_rast)

  message("Land cells: ", format(n_land, big.mark = ","),
          " / ", format(n_total, big.mark = ","),
          " (", round(100 * n_land / n_total, 1), "%)")

  rm(land_mask, ref_rast)
  gc(verbose = FALSE)

  list(land_idx = land_idx, n_land = n_land, n_total = n_total)
}


#' Load all CHELSA variables and save each column individually
#'
#' Loads each variable from GeoTIFF, fills NAs, verifies per-column,
#' and saves directly as an individual .pt file. Never constructs the
#' full 38 GB tensor in memory — peak usage is ~2.5 GB (one column
#' as R vector + one as torch tensor).
#'
#' @param chelsa_dir Path to CHELSA bio raster directory
#' @param var_meta Data frame with columns: variable, chelsa_name, na_fill
#' @param land_idx Integer vector of land cell indices (from build_land_mask)
#' @param output_dir Directory to save the tensor column files
#' @return The output_dir string
load_chelsa_tensor <- function(chelsa_dir, var_meta, land_idx, output_dir) {
  n_land <- length(land_idx)
  n_vars <- nrow(var_meta)

  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

  expected_gb <- round(as.double(n_land) * n_vars * 4 / 1e9, 1)
  message("Loading ", n_vars, " variables, ", format(n_land, big.mark = ","),
          " land cells (", expected_gb, " GB total as float32)")
  message("Saving columns directly to: ", output_dir)

  # Save metadata for load_large_tensor()
  meta <- list(
    shape = c(n_land, n_vars),
    dtype = "Float",
    n_cols = n_vars
  )
  saveRDS(meta, file.path(output_dir, "meta.rds"))

  t_start <- Sys.time()
  total_nan <- 0L
  total_inf <- 0L

  for (i in seq_len(n_vars)) {
    var_name <- var_meta$variable[i]
    chelsa_name <- var_meta$chelsa_name[i]
    na_fill_val <- as.numeric(var_meta$na_fill[i])

    tif_file <- list.files(chelsa_dir,
                           pattern = paste0("CHELSA_", chelsa_name, "_"),
                           full.names = TRUE)[1]

    if (is.na(tif_file) || !file.exists(tif_file)) {
      stop("File not found for variable '", var_name,
           "' (chelsa_name: ", chelsa_name, ")")
    }

    t_var <- Sys.time()
    r <- terra::rast(tif_file)
    vals <- r[land_idx][[1]]

    # NA fill per variable metadata
    n_na <- sum(is.na(vals))
    if (n_na > 0 && !is.na(na_fill_val)) {
      vals[is.na(vals)] <- na_fill_val
      n_remaining <- sum(is.na(vals))
    } else {
      n_remaining <- n_na
    }

    # Create column tensor and verify
    col_tensor <- torch::torch_tensor(vals, dtype = torch::torch_float32())
    n_nan <- col_tensor$isnan()$sum()$item()
    n_inf <- col_tensor$isinf()$sum()$item()
    total_nan <- total_nan + n_nan
    total_inf <- total_inf + n_inf

    # Save column directly
    col_path <- file.path(output_dir, sprintf("col_%03d.pt", i))
    torch::torch_save(col_tensor, col_path)

    elapsed <- round(as.numeric(Sys.time() - t_var, units = "secs"), 1)

    na_msg <- ""
    if (n_na > 0) {
      na_msg <- paste0(" | filled ", format(n_na, big.mark = ","), " NAs")
      if (n_remaining > 0) {
        na_msg <- paste0(na_msg, " (", n_remaining, " unfillable)")
      }
    }
    verify_msg <- ""
    if (n_nan > 0 || n_inf > 0) {
      verify_msg <- paste0(" | WARN: ", n_nan, " NaN, ", n_inf, " Inf")
    }
    message("  [", i, "/", n_vars, "] ", var_name,
            " (", elapsed, "s)", na_msg, verify_msg)

    rm(r, vals, col_tensor)
    if (i %% 5 == 0) gc(verbose = FALSE)
  }

  total_time <- round(as.numeric(Sys.time() - t_start, units = "mins"), 1)
  total_size <- sum(file.size(
    list.files(output_dir, pattern = "\\.pt$", full.names = TRUE)
  ))

  message("Total load time: ", total_time, " minutes")
  message("Total size on disk: ", round(total_size / 1e9, 2), " GB")
  message("Verification: ", total_nan, " NaN, ", total_inf, " Inf across all columns")
  if (total_nan > 0) warning("Tensor has ", total_nan, " NaN values!")
  if (total_inf > 0) warning("Tensor has ", total_inf, " Inf values!")

  gc(verbose = FALSE)
  output_dir
}


#' Compute column-wise mean and SD from saved tensor columns
#'
#' Loads each column file individually and computes its mean and SD,
#' avoiding loading the full tensor into memory.
#'
#' @param tensor_dir Directory containing column .pt files
#' @return List with mean and sd numeric vectors
compute_standardization <- function(tensor_dir) {
  meta <- readRDS(file.path(tensor_dir, "meta.rds"))
  n_cols <- meta$n_cols
  n_rows <- meta$shape[1]

  message("Computing column-wise mean and SD for ",
          n_cols, " columns across ",
          format(n_rows, big.mark = ","), " rows")

  col_mean <- numeric(n_cols)
  col_sd <- numeric(n_cols)

  for (i in seq_len(n_cols)) {
    col_path <- file.path(tensor_dir, sprintf("col_%03d.pt", i))
    col <- torch::torch_load(col_path)
    col_mean[i] <- col$mean()$item()
    col_sd[i] <- col$std()$item()
    rm(col)
  }
  gc(verbose = FALSE)

  # Guard against zero SD
  zero_sd <- col_sd < 1e-10
  if (any(zero_sd)) {
    warning("Zero SD detected for columns: ",
            paste(which(zero_sd), collapse = ", "),
            ". Setting SD to 1 for these columns.")
    col_sd[zero_sd] <- 1
  }

  message("Per-column statistics:")
  for (i in seq_along(col_mean)) {
    message("  Col ", i, ": mean=", round(col_mean[i], 4),
            "  sd=", round(col_sd[i], 4))
  }

  list(mean = col_mean, sd = col_sd)
}


#' Standardize tensor columns and save to new directory
#'
#' Loads each column, applies z-score: (x - mean) / sd, saves to
#' new directory. Operates one column at a time for minimal memory.
#'
#' @param tensor_dir Directory containing input tensor column files
#' @param stats List with mean and sd numeric vectors
#' @param output_dir Directory to save standardized column files
#' @return The output_dir string
standardize_tensor <- function(tensor_dir, stats, output_dir) {
  meta <- readRDS(file.path(tensor_dir, "meta.rds"))
  n_cols <- meta$n_cols

  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  saveRDS(meta, file.path(output_dir, "meta.rds"))

  message("Standardizing ", n_cols, " columns from ", tensor_dir)

  check_means <- numeric(n_cols)
  check_sds <- numeric(n_cols)

  for (i in seq_len(n_cols)) {
    col <- torch::torch_load(file.path(tensor_dir, sprintf("col_%03d.pt", i)))
    col$sub_(stats$mean[i])$div_(stats$sd[i])

    check_means[i] <- col$mean()$item()
    check_sds[i] <- col$std()$item()

    torch::torch_save(col, file.path(output_dir, sprintf("col_%03d.pt", i)))
    rm(col)
  }
  gc(verbose = FALSE)

  message("Post-standardization check:")
  message("  Mean range: [", round(min(check_means), 6),
          ", ", round(max(check_means), 6), "] (expect ~0)")
  message("  SD range:   [", round(min(check_sds), 6),
          ", ", round(max(check_sds), 6), "] (expect ~1)")

  total_size <- sum(file.size(
    list.files(output_dir, pattern = "\\.pt$", full.names = TRUE)
  ))
  message("Saved to ", output_dir, " (", round(total_size / 1e9, 2), " GB)")

  output_dir
}


#' Extract lon/lat coordinates for land cells and save as tensor columns
#'
#' @param chelsa_dir Path to CHELSA bio raster directory
#' @param land_idx Integer vector of land cell indices
#' @param output_dir Directory to save the xy tensor column files
#' @return The output_dir string
extract_xy_coords <- function(chelsa_dir, land_idx, output_dir) {
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

  ref_rast <- terra::rast(file.path(chelsa_dir,
                                     "CHELSA_bio1_1981-2010_V.2.1.tif"))

  message("Extracting XY coordinates for ",
          format(length(land_idx), big.mark = ","), " land cells...")
  xy <- terra::xyFromCell(ref_rast, land_idx)

  message("Lon range: [", round(min(xy[, 1]), 2),
          ", ", round(max(xy[, 1]), 2), "]")
  message("Lat range: [", round(min(xy[, 2]), 2),
          ", ", round(max(xy[, 2]), 2), "]")

  # Save metadata
  meta <- list(
    shape = as.integer(dim(xy)),
    dtype = "Float",
    n_cols = 2L
  )
  saveRDS(meta, file.path(output_dir, "meta.rds"))

  # Save each column (lon, lat) separately
  for (i in 1:2) {
    col_tensor <- torch::torch_tensor(xy[, i], dtype = torch::torch_float32())
    torch::torch_save(col_tensor, file.path(output_dir, sprintf("col_%03d.pt", i)))
    rm(col_tensor)
  }

  size_gb <- round(length(land_idx) * 2 * 4 / 1e9, 2)
  message("Saved 2 columns to ", output_dir, " (", size_gb, " GB)")

  rm(ref_rast, xy)
  gc(verbose = FALSE)

  output_dir
}
