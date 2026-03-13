#' JADE Jacobian Raster Computation Functions
#'
#' Functions for computing the generalized Jacobian raster used in
#' JADE (Jacobian-Adjusted Density Estimation) sampling to correct
#' for geographic-to-environmental distortion.
#'
#' @author rdinnage


#' Compute per-variable standard deviations via random sampling
#'
#' Samples a subset of pixels rather than reading entire rasters.
#' With 1M samples from ~900M pixels, the relative standard error
#' of the SD estimate is ~0.07% -- negligible for standardization.
#'
#' @param bioclim_files Character vector of paths to bioclim raster files
#' @param sample_size Number of random pixels to sample (default 1e6)
#' @return Named numeric vector of SDs for each variable
#' @export
compute_bioclim_sds <- function(bioclim_files, sample_size = 1e6) {
  sds <- numeric(length(bioclim_files))
  var_names <- basename(bioclim_files) |>
    tools::file_path_sans_ext()
  names(sds) <- var_names

  for (j in seq_along(bioclim_files)) {
    fname <- bioclim_files[j]
    r <- terra::rast(fname)
    # Use global() which processes in chunks -- spatSample() tries to load
    # the entire raster into memory for na.rm = TRUE
    sds[j] <- terra::global(r, "sd", na.rm = TRUE)[1, 1]
    rm(r)
    gc(verbose = FALSE)
    message("  Computed SD for ", var_names[j], ": ", round(sds[j], 4))
  }

  # Guard against zero SD
  sds[sds < 1e-10] <- 1
  sds
}


#' Compute A, B, C Gram matrix component rasters
#'
#' Computes the components of the 2x2 Gram matrix J^T J by accumulating:
#' - A = sum((de_i/dlon)^2)
#' - B = sum((de_i/dlon)(de_i/dlat))
#' - C = sum((de_i/dlat)^2)
#'
#' Each variable's derivatives are standardized by its global SD.
#' Uses terra::focal() with central difference kernels.
#' Writes intermediate rasters to disk to prevent memory buildup.
#'
#' @param bioclim_files Character vector of paths to bioclim raster files
#' @param bioclim_sds Named numeric vector of SDs from compute_bioclim_sds()
#' @param output_dir Directory to write output rasters
#' @return List with paths: list(A = "...", B = "...", C = "...")
#' @export
compute_abc_rasters <- function(bioclim_files, bioclim_sds, output_dir) {

  t_start <- proc.time()
  terra::terraOptions(memfrac = 0.3, progress = 10)

  # Create output directory
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

  # Template and resolution
  template <- terra::rast(bioclim_files[1])
  h <- terra::xres(template)  # Grid spacing in degrees

  message("Resolution: ", h, " degrees")
  message("Dimensions: ", terra::nrow(template), " x ", terra::ncol(template))

  # Focal weight matrices (central differences)
  # Longitude: [west, center, east] = [-1, 0, 1] / (2h)
  w_dlon <- matrix(c(-1, 0, 1) / (2 * h), nrow = 1, ncol = 3)
  # Latitude: [north, center, south] = [1, 0, -1] / (2h)
  w_dlat <- matrix(c(1, 0, -1) / (2 * h), nrow = 3, ncol = 1)

  # Initialize accumulator rasters on disk
  A_path <- file.path(output_dir, "gram_A_temp.tif")
  B_path <- file.path(output_dir, "gram_B_temp.tif")
  C_path <- file.path(output_dir, "gram_C_temp.tif")

  terra::init(template, 0, filename = A_path, overwrite = TRUE)
  terra::init(template, 0, filename = B_path, overwrite = TRUE)
  terra::init(template, 0, filename = C_path, overwrite = TRUE)

  A <- terra::rast(A_path)
  B <- terra::rast(B_path)
  C <- terra::rast(C_path)

  rm(template)
  gc(verbose = FALSE)

  # Temp paths for focal results
  dlon_path <- file.path(output_dir, "temp_dlon.tif")
  dlat_path <- file.path(output_dir, "temp_dlat.tif")

  # Process each variable
  n_vars <- length(bioclim_files)
  var_times <- numeric(n_vars)
  var_names <- basename(bioclim_files) |> tools::file_path_sans_ext()

  for (j in seq_along(bioclim_files)) {
    t_var <- proc.time()
    var_label <- var_names[j]
    message("\n[", j, "/", n_vars, "] Processing ", var_label, "...")

    r <- terra::rast(bioclim_files[j])

    # Spatial derivatives via focal convolution - write directly to disk
    # to avoid holding ~3.6GB per result in memory
    de_dlon <- terra::focal(r, w = w_dlon, na.rm = FALSE,
                             filename = dlon_path, overwrite = TRUE)
    de_dlat <- terra::focal(r, w = w_dlat, na.rm = FALSE,
                             filename = dlat_path, overwrite = TRUE)
    rm(r)
    gc(verbose = FALSE)

    # Standardize by global SD
    sd_val <- bioclim_sds[j]
    de_dlon <- de_dlon / sd_val
    de_dlat <- de_dlat / sd_val

    # Accumulate Gram matrix components - write each to disk to break
    # lazy chains and keep memory bounded
    terra::writeRaster(A + de_dlon^2, A_path, overwrite = TRUE)
    A <- terra::rast(A_path)
    terra::writeRaster(B + de_dlon * de_dlat, B_path, overwrite = TRUE)
    B <- terra::rast(B_path)
    rm(de_dlon)
    gc(verbose = FALSE)
    terra::writeRaster(C + de_dlat^2, C_path, overwrite = TRUE)
    C <- terra::rast(C_path)
    rm(de_dlat)
    gc(verbose = FALSE)

    # Remove temp derivative files
    unlink(dlon_path)
    unlink(dlat_path)

    # Progress
    var_elapsed <- (proc.time() - t_var)[3]
    var_times[j] <- var_elapsed
    total_elapsed <- (proc.time() - t_start)[3]
    est_remaining <- mean(var_times[1:j]) * (n_vars - j)
    message("  ", var_label, " done in ", round(var_elapsed, 1), "s",
            " | Elapsed: ", format_time(total_elapsed),
            " | Est. remaining: ~", format_time(est_remaining))
  }

  message("\nAll ", n_vars, " variables processed in ",
          format_time((proc.time() - t_start)[3]))

  # Write final A, B, C rasters with proper names and compression
  A_path <- file.path(output_dir, "gram_A.tif")
  B_path <- file.path(output_dir, "gram_B.tif")
  C_path <- file.path(output_dir, "gram_C.tif")

  terra::writeRaster(A, A_path, overwrite = TRUE,
                     filetype = "GTiff",
                     gdal = c("COMPRESS=LZW", "TILED=YES"))
  terra::writeRaster(B, B_path, overwrite = TRUE,
                     filetype = "GTiff",
                     gdal = c("COMPRESS=LZW", "TILED=YES"))
  terra::writeRaster(C, C_path, overwrite = TRUE,
                     filetype = "GTiff",
                     gdal = c("COMPRESS=LZW", "TILED=YES"))

  # Clean up temp files
  unlink(file.path(output_dir, "gram_A_temp.tif"))
  unlink(file.path(output_dir, "gram_B_temp.tif"))
  unlink(file.path(output_dir, "gram_C_temp.tif"))

  c(A = A_path, B = B_path, C = C_path)
}


#' Fill coastal NA values via iterative focal mean
#'
#' focal(..., na.rm = FALSE) produces NA at land cells with ocean neighbors.
#' Since SDM occurrences frequently fall near coastlines, we fill these NAs
#' using the mean of non-NA neighbors.
#'
#' @param jacobian SpatRaster with NA values at coastal cells
#' @param land_mask SpatRaster with 1 for land, NA for ocean
#' @param max_passes Maximum number of fill iterations (default 3)
#' @return SpatRaster with filled coastal NAs
#' @export
fill_coastal_na <- function(jacobian, land_mask, max_passes = 3) {
  filled <- jacobian

  for (pass in seq_len(max_passes)) {
    focal_means <- terra::focal(filled, w = 3, fun = "mean", na.rm = TRUE,
                                na.policy = "only")
    filled <- terra::cover(filled, focal_means)
    rm(focal_means); gc(verbose = FALSE)
    filled <- terra::mask(filled, land_mask)

    n_remaining <- terra::global(is.na(filled) & !is.na(land_mask), "sum",
                                 na.rm = TRUE)[1, 1]
    message("  Coastal NA fill pass ", pass, "/", max_passes,
            ": ", n_remaining, " land NAs remaining")
    if (n_remaining == 0) break
  }

  # Fallback: remaining isolated land NAs get global mean
  if (n_remaining > 0) {
    mean_val <- terra::global(filled, "mean", na.rm = TRUE)[1, 1]
    message("  Fallback: filling ", n_remaining,
            " remaining land NAs with global mean (", round(mean_val, 4), ")")
    filled <- terra::ifel(is.na(filled) & !is.na(land_mask), mean_val, filled)
  }

  filled
}


#' Compute final Jacobian raster from A, B, C components
#'
#' Computes J = sqrt(max(AC - B^2, 0)) and applies latitude correction.
#' Fills coastal NAs via iterative focal mean.
#'
#' @param abc_rasters Named character vector with paths A, B, C from compute_abc_rasters()
#' @param output_path Path to write final Jacobian raster
#' @param lat_correct Apply latitude correction (default TRUE)
#' @return Path to final Jacobian raster
#' @export
compute_jacobian_from_abc <- function(abc_rasters, output_path, lat_correct = TRUE) {

  message("Computing Jacobian from A, B, C rasters...")

  # Load A, B, C (abc_rasters is a character vector of 3 file paths;
  # format = "file" strips names, so use positional indexing)
  A <- terra::rast(abc_rasters[1])
  B <- terra::rast(abc_rasters[2])
  C <- terra::rast(abc_rasters[3])

  gc(verbose = FALSE)
  # Create land mask for coastal NA fill and materialize to disk
  land_mask <- terra::ifel(!is.na(A), 1, NA)
  gc(verbose = FALSE)
  dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
  lm_tmp <- file.path(dirname(output_path), "land_mask_temp.tif")
  terra::writeRaster(land_mask, lm_tmp, overwrite = TRUE,
                     datatype = "INT1U", filetype = "GTiff",
                     gdal = c("COMPRESS=LZW", "TILED=YES"))
  rm(land_mask); gc(verbose = FALSE)
  land_mask <- terra::rast(lm_tmp)
  gc(verbose = FALSE)

  # Compute Jacobian determinant
  det_JtJ <- A * C - B^2
  gc(verbose = FALSE)
  jacobian <- sqrt(terra::clamp(det_JtJ, lower = 0))
  gc(verbose = FALSE)
  rm(A, B, C, det_JtJ)
  gc(verbose = FALSE)

  # Latitude correction
  if (lat_correct) {
    message("Applying latitude correction...")
    lat_rast <- terra::init(jacobian, "y")
    cos_lat <- terra::clamp(cos(lat_rast * pi / 180), lower = 0.01)
    gc(verbose = FALSE)
    jacobian <- jacobian / cos_lat
    rm(lat_rast, cos_lat)
    gc(verbose = FALSE)
  }

  # Materialize to a temp raster on disk to break lazy evaluation chain
  # before the memory-intensive focal() fill step
  tmp_path <- file.path(dirname(output_path), "jacobian_prefill_temp.tif")
  message("Writing pre-fill temp raster...")
  gc(verbose = FALSE)
  terra::writeRaster(jacobian, tmp_path, overwrite = TRUE,
                     filetype = "GTiff", gdal = c("COMPRESS=LZW", "TILED=YES"))
  rm(jacobian)
  gc(verbose = FALSE)
  jacobian <- terra::rast(tmp_path)

  # Fill coastal NAs
  message("Filling coastal NAs...")
  gc(verbose = FALSE)
  jacobian <- fill_coastal_na(jacobian, land_mask)
  rm(land_mask)
  gc(verbose = FALSE)

  # Write final output
  message("Writing: ", output_path)
  gc(verbose = FALSE)
  terra::writeRaster(jacobian, output_path, overwrite = TRUE,
                     names = "jacobian_J2", filetype = "GTiff",
                     gdal = c("COMPRESS=LZW", "TILED=YES"))

  # Clean up temp files
  unlink(tmp_path)
  unlink(lm_tmp)

  output_path
}


#' Compute A, B, C Gram matrix rasters with NA pre-filling
#'
#' Wrapper around compute_abc_rasters that fills NA values in input rasters
#' before computing focal derivatives. Without this, variables like
#' gsl/gsp/gdd*/ngd*/fcf/scd/swe that have NAs in warm regions propagate
#' NAs through focal() to the entire Gram matrix and Jacobian.
#'
#' @param bioclim_files Character vector of paths to bioclim raster files
#' @param bioclim_sds Named numeric vector of SDs from compute_bioclim_sds()
#' @param output_dir Directory to write output rasters
#' @param na_fill_values Named numeric vector of fill values. Names must match
#'   raster layer names (basename without extension).
#' @return List with paths: list(A = "...", B = "...", C = "...")
#' @export
compute_abc_rasters_filled <- function(bioclim_files, bioclim_sds, output_dir,
                                       na_fill_values) {

  t_start <- proc.time()
  terra::terraOptions(memfrac = 0.3, progress = 10)

  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

  template <- terra::rast(bioclim_files[1])
  h <- terra::xres(template)

  message("Resolution: ", h, " degrees")
  message("Dimensions: ", terra::nrow(template), " x ", terra::ncol(template))

  w_dlon <- matrix(c(-1, 0, 1) / (2 * h), nrow = 1, ncol = 3)
  w_dlat <- matrix(c(1, 0, -1) / (2 * h), nrow = 3, ncol = 1)

  A_path <- file.path(output_dir, "gram_A_temp.tif")
  B_path <- file.path(output_dir, "gram_B_temp.tif")
  C_path <- file.path(output_dir, "gram_C_temp.tif")

  terra::init(template, 0, filename = A_path, overwrite = TRUE)
  terra::init(template, 0, filename = B_path, overwrite = TRUE)
  terra::init(template, 0, filename = C_path, overwrite = TRUE)

  A <- terra::rast(A_path)
  B <- terra::rast(B_path)
  C <- terra::rast(C_path)

  rm(template)
  gc(verbose = FALSE)

  dlon_path <- file.path(output_dir, "temp_dlon.tif")
  dlat_path <- file.path(output_dir, "temp_dlat.tif")

  n_vars <- length(bioclim_files)
  var_times <- numeric(n_vars)
  var_names <- basename(bioclim_files) |> tools::file_path_sans_ext()

  for (j in seq_along(bioclim_files)) {
    t_var <- proc.time()
    var_label <- var_names[j]
    message("\n[", j, "/", n_vars, "] Processing ", var_label, "...")

    r <- terra::rast(bioclim_files[j])

    # Fill NAs with documented values before derivative computation
    if (var_label %in% names(na_fill_values)) {
      fill_val <- na_fill_values[var_label]
      message("  Filling NAs with ", fill_val, " for ", var_label)
      r <- terra::classify(r, cbind(NA, fill_val))
    }

    de_dlon <- terra::focal(r, w = w_dlon, na.rm = FALSE,
                             filename = dlon_path, overwrite = TRUE)
    de_dlat <- terra::focal(r, w = w_dlat, na.rm = FALSE,
                             filename = dlat_path, overwrite = TRUE)
    rm(r)
    gc(verbose = FALSE)

    sd_val <- bioclim_sds[j]
    de_dlon <- de_dlon / sd_val
    de_dlat <- de_dlat / sd_val

    terra::writeRaster(A + de_dlon^2, A_path, overwrite = TRUE)
    A <- terra::rast(A_path)
    terra::writeRaster(B + de_dlon * de_dlat, B_path, overwrite = TRUE)
    B <- terra::rast(B_path)
    rm(de_dlon)
    gc(verbose = FALSE)
    terra::writeRaster(C + de_dlat^2, C_path, overwrite = TRUE)
    C <- terra::rast(C_path)
    rm(de_dlat)
    gc(verbose = FALSE)

    unlink(dlon_path)
    unlink(dlat_path)

    var_elapsed <- (proc.time() - t_var)[3]
    var_times[j] <- var_elapsed
    total_elapsed <- (proc.time() - t_start)[3]
    est_remaining <- mean(var_times[1:j]) * (n_vars - j)
    message("  ", var_label, " done in ", round(var_elapsed, 1), "s",
            " | Elapsed: ", format_time(total_elapsed),
            " | Est. remaining: ~", format_time(est_remaining))
  }

  # Rename temp files to final names
  A_final <- file.path(output_dir, "gram_A.tif")
  B_final <- file.path(output_dir, "gram_B.tif")
  C_final <- file.path(output_dir, "gram_C.tif")

  file.rename(A_path, A_final)
  file.rename(B_path, B_final)
  file.rename(C_path, C_final)

  total_time <- (proc.time() - t_start)[3]
  message("\nGram matrix computation complete in ", format_time(total_time))

  c(A_final, B_final, C_final)
}


#' Time formatting helper
#' @keywords internal
format_time <- function(seconds) {
  if (seconds < 60) paste0(round(seconds, 1), "s")
  else if (seconds < 3600) paste0(round(seconds / 60, 1), " min")
  else paste0(round(seconds / 3600, 1), " hr")
}
