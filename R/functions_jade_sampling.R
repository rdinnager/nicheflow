#' JADE Accept-Reject Sampling Functions
#'
#' Functions for JADE (Jacobian-Adjusted Density Estimation) accept-reject
#' sampling to generate distortion-free environmental samples from species
#' range polygons.
#'
#' @author rdinnage


#' JADE accept-reject sampling from species range polygon
#'
#' Samples uniform random points within a species range polygon, then applies
#' accept-reject thinning based on Jacobian values to correct for
#' geographic-to-environmental distortion.
#'
#' Points in high-gradient regions (high Jacobian) are kept preferentially;
#' points in flat regions (low Jacobian) are discarded more often.
#'
#' @param spec_polys sf data frame with single species polygon (expects 'species' column)
#' @param jacobian_path Path to Jacobian raster file
#' @param bioclim_files Character vector of paths to bioclim raster files
#' @param n_target Target number of JADE-corrected samples (default 5000)
#' @param batch_size Initial batch size for sampling (default NULL = 5x n_target)
#' @param max_iter Maximum sampling iterations (default 100)
#' @param verbose Print progress messages (default TRUE)
#' @return Data frame with X, Y coordinates, environmental values, and species name
#' @export
jade_sample <- function(spec_polys, jacobian_path, bioclim_files,
                        n_target = 5000, batch_size = NULL,
                        max_iter = 100, verbose = TRUE) {

  # Load rasters
  jac_rast <- terra::rast(jacobian_path)
  bio_stack <- terra::rast(bioclim_files)

  # Get species name
  species_name <- spec_polys$species[1]

  if (verbose) {
    message("JADE sampling for: ", species_name)
  }

  # Default batch size: 5x target
  if (is.null(batch_size)) batch_size <- n_target * 5

  # Get taxon if available
  taxon_name <- if ("taxon" %in% names(spec_polys)) spec_polys$taxon[1] else NA_character_

  # Storage for accepted points
  accepted_xy <- matrix(nrow = 0, ncol = 2,
                        dimnames = list(NULL, c("X", "Y")))
  accepted_env <- data.frame()
  accepted_jac <- numeric(0)
  total_sampled <- 0
  iter <- 0

  while (nrow(accepted_xy) < n_target && iter < max_iter) {
    iter <- iter + 1

    # Sample uniform geographic points within range
    pts_sf <- sf::st_sample(spec_polys, size = batch_size, type = "random")

    # Handle case where st_sample returns empty
    if (length(pts_sf) == 0) {
      if (verbose) message("  Warning: st_sample returned 0 points")
      next
    }

    pts_xy <- sf::st_coordinates(pts_sf)
    colnames(pts_xy) <- c("X", "Y")
    total_sampled <- total_sampled + nrow(pts_xy)

    # Extract Jacobian at each point
    pts_vect <- terra::vect(pts_xy, crs = "EPSG:4326")
    jac_extract <- terra::extract(jac_rast, pts_vect)
    jac_vals <- jac_extract[[2]]

    # Remove invalid points (ocean, outside extent, zero Jacobian)
    valid <- !is.na(jac_vals) & jac_vals > 0
    if (sum(valid) == 0) {
      if (verbose) message("  Iter ", iter, ": no valid points (all NA or zero Jacobian)")
      next
    }

    pts_xy <- pts_xy[valid, , drop = FALSE]
    jac_vals <- jac_vals[valid]

    # Acceptance-rejection thinning
    p_accept <- jac_vals / max(jac_vals)
    keep <- runif(length(p_accept)) <= p_accept

    if (sum(keep) == 0) {
      if (verbose) message("  Iter ", iter, ": no points accepted")
      next
    }

    # Accumulate accepted points and their Jacobian values
    accepted_xy <- rbind(accepted_xy, pts_xy[keep, , drop = FALSE])
    accepted_jac <- c(accepted_jac, jac_vals[keep])

    # Extract environmental values at accepted points
    keep_vect <- terra::vect(pts_xy[keep, , drop = FALSE], crs = "EPSG:4326")
    env_batch <- terra::extract(bio_stack, keep_vect)
    # Remove the ID column that terra::extract adds
    env_batch <- env_batch[, -1, drop = FALSE]
    accepted_env <- rbind(accepted_env, env_batch)

    if (verbose) {
      message("  Iter ", iter, ": sampled ", nrow(pts_xy),
              ", accepted ", sum(keep),
              " (total: ", nrow(accepted_xy), "/", n_target, ")")
    }

    # Adaptive batch sizing: estimate remaining need from empirical rate
    if (iter == 1 && sum(keep) > 0) {
      empirical_rate <- sum(keep) / nrow(pts_xy)
      remaining <- n_target - nrow(accepted_xy)
      batch_size <- max(batch_size,
                        ceiling(remaining / empirical_rate * 1.3))
      if (verbose) {
        message("  Acceptance rate: ", round(100 * empirical_rate, 1), "%",
                " | Adjusted batch size: ", batch_size)
      }
    }
  }

  # Trim to exact target if we overshot
  if (nrow(accepted_xy) > n_target) {
    accepted_xy <- accepted_xy[1:n_target, , drop = FALSE]
    accepted_env <- accepted_env[1:n_target, , drop = FALSE]
    accepted_jac <- accepted_jac[1:n_target]
  }

  # Check if we reached target
  if (nrow(accepted_xy) < n_target) {
    warning("JADE sampling for ", species_name, " only achieved ",
            nrow(accepted_xy), "/", n_target, " samples after ", iter, " iterations")
  }

  # Combine results
  result <- dplyr::bind_cols(
    as.data.frame(accepted_xy),
    accepted_env
  ) |>
    dplyr::mutate(jacobian = accepted_jac,
                  species = species_name,
                  taxon = taxon_name)

  if (verbose) {
    acceptance_rate <- nrow(accepted_xy) / total_sampled
    message("  Final: ", nrow(result), " samples, ",
            round(100 * acceptance_rate, 1), "% acceptance rate")
  }

  result
}


#' Batch JADE sampling for multiple species
#'
#' Convenience wrapper for sampling multiple species sequentially.
#' For parallel execution, use targets dynamic branching instead.
#'
#' @param species_list List of sf data frames, each with a single species polygon
#' @param jacobian_path Path to Jacobian raster file
#' @param bioclim_files Character vector of paths to bioclim raster files
#' @param n_target Target samples per species (default 5000)
#' @param verbose Print progress messages (default TRUE)
#' @return Data frame with all species combined
#' @export
jade_sample_batch <- function(species_list, jacobian_path, bioclim_files,
                              n_target = 5000, verbose = TRUE) {

  results <- purrr::map(
    species_list,
    \(sp) jade_sample(sp, jacobian_path, bioclim_files,
                      n_target = n_target, verbose = verbose),
    .progress = verbose
  )

  dplyr::bind_rows(results)
}


#' Crash-protected JADE sampling via callr subprocess
#'
#' Wraps \code{jade_sample()} in \code{callr::r()} so that OOM kills,
#' segfaults, or timeouts in one species do not crash the crew worker.
#' Each call runs in an isolated subprocess with its own address space.
#'
#' @param spec_polys sf data frame with single species polygon
#' @param jacobian_path Path to Jacobian raster file
#' @param bioclim_files Character vector of paths to bioclim raster files
#' @param n_target Target number of JADE-corrected samples (default 5000)
#' @param timeout Subprocess timeout in seconds (default 300)
#' @return Data frame with samples, or NULL on failure
#' @export
jade_sample_safe <- function(spec_polys, jacobian_path, bioclim_files,
                             n_target = 5000, timeout = 600,
                             log_file = "logs/jade_sampling.log") {
  sp_name <- tryCatch(spec_polys$species[1], error = function(e2) "unknown")
  dir.create(dirname(log_file), showWarnings = FALSE, recursive = TRUE)
  cat(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), " DISPATCH ", sp_name, "\n",
      file = log_file, append = TRUE)
  tryCatch(
    {
      result <- callr::r(
        function(spec_polys, jacobian_path, bioclim_files, n_target) {
          source("R/functions_jade_sampling.R")
          jade_sample(spec_polys, jacobian_path, bioclim_files,
                      n_target = n_target, verbose = FALSE)
        },
        args = list(
          spec_polys = spec_polys,
          jacobian_path = jacobian_path,
          bioclim_files = bioclim_files,
          n_target = n_target
        ),
        timeout = timeout
      )
      n_rows <- if (is.null(result)) 0L else nrow(result)
      cat(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), " OK       ", sp_name,
          " (", n_rows, " rows)\n", sep = "",
          file = log_file, append = TRUE)
      result
    },
    error = function(e) {
      cat(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), " FAIL     ", sp_name,
          ": ", conditionMessage(e), "\n", sep = "",
          file = log_file, append = TRUE)
      warning("jade_sample failed for ", sp_name, ": ", conditionMessage(e))
      NULL
    }
  )
}


#' Load polygon-level metadata for JADE post-hoc cleaning
#'
#' Reloads raw shapefiles with the same filtering as the pipeline, but
#' retains IUCN metadata columns (presence, origin, seasonal) and computes
#' polygon areas. Rows are in the same order as \code{all_taxa_polygons}.
#'
#' @param taxa_poly_folders Named list of paths (reptiles, amphibians, mammals)
#' @return Tibble with species, taxon, presence, origin, seasonal, area_km2
#' @export
load_jade_polygon_metadata <- function(taxa_poly_folders) {

  results <- list()

  # --- Reptiles (GARD format: no IUCN metadata columns) ---
  message("Loading reptile metadata...")
  rep_polys <- sf::st_read(taxa_poly_folders$reptiles, quiet = TRUE) |>
    dplyr::mutate(species = binomial, taxon = "reptiles")
  rep_polys <- filter_taxa_polygons(rep_polys)
  old_s2 <- sf::sf_use_s2(); sf::sf_use_s2(FALSE)
  results$reptiles <- rep_polys |>
    dplyr::mutate(presence = 1L, origin = 1L, seasonal = 1L,
                  area_km2 = as.numeric(sf::st_area(geometry)) / 1e6) |>
    sf::st_drop_geometry() |>
    dplyr::select(species, taxon, presence, origin, seasonal, area_km2)
  sf::sf_use_s2(old_s2)
  rm(rep_polys); gc(verbose = FALSE)

  # --- Amphibians (IUCN format: PART1 + PART2) ---
  message("Loading amphibian metadata...")
  amp_parts <- list()
  p1 <- file.path(taxa_poly_folders$amphibians, "AMPHIBIANS_PART1.shp")
  p2 <- file.path(taxa_poly_folders$amphibians, "AMPHIBIANS_PART2.shp")
  if (file.exists(p1)) amp_parts$p1 <- sf::st_read(p1, quiet = TRUE)
  if (file.exists(p2)) amp_parts$p2 <- sf::st_read(p2, quiet = TRUE)
  amp_polys <- dplyr::bind_rows(amp_parts)
  sp_col <- if ("binomial" %in% names(amp_polys)) "binomial" else "sci_name"
  amp_polys <- amp_polys |>
    dplyr::mutate(species = .data[[sp_col]], taxon = "amphibians")
  amp_polys <- filter_taxa_polygons(amp_polys)
  old_s2 <- sf::sf_use_s2(); sf::sf_use_s2(FALSE)
  results$amphibians <- amp_polys |>
    dplyr::mutate(
      presence = if ("presence" %in% names(amp_polys)) amp_polys$presence else 1L,
      origin = if ("origin" %in% names(amp_polys)) amp_polys$origin else 1L,
      seasonal = if ("seasonal" %in% names(amp_polys)) amp_polys$seasonal else 1L,
      area_km2 = as.numeric(sf::st_area(geometry)) / 1e6
    ) |>
    sf::st_drop_geometry() |>
    dplyr::select(species, taxon, presence, origin, seasonal, area_km2)
  sf::sf_use_s2(old_s2)
  rm(amp_polys, amp_parts); gc(verbose = FALSE)

  # --- Mammals (IUCN format) ---
  message("Loading mammal metadata...")
  mam_polys <- sf::st_read(taxa_poly_folders$mammals, quiet = TRUE)
  sp_col <- if ("binomial" %in% names(mam_polys)) "binomial" else "sci_name"
  mam_polys <- mam_polys |>
    dplyr::mutate(species = .data[[sp_col]], taxon = "mammals")
  mam_polys <- filter_taxa_polygons(mam_polys)
  old_s2 <- sf::sf_use_s2(); sf::sf_use_s2(FALSE)
  results$mammals <- mam_polys |>
    dplyr::mutate(
      presence = if ("presence" %in% names(mam_polys)) mam_polys$presence else 1L,
      origin = if ("origin" %in% names(mam_polys)) mam_polys$origin else 1L,
      seasonal = if ("seasonal" %in% names(mam_polys)) mam_polys$seasonal else 1L,
      area_km2 = as.numeric(sf::st_area(geometry)) / 1e6
    ) |>
    sf::st_drop_geometry() |>
    dplyr::select(species, taxon, presence, origin, seasonal, area_km2)
  sf::sf_use_s2(old_s2)
  rm(mam_polys); gc(verbose = FALSE)

  meta <- dplyr::bind_rows(results)
  message("Polygon metadata: ", nrow(meta), " rows, ",
          dplyr::n_distinct(meta$species), " unique species")
  meta
}


#' Clean and merge JADE samples with environmental-volume-weighted resampling
#'
#' Post-hoc cleanup of JADE samples:
#' \enumerate{
#'   \item Removes samples from extinct/introduced polygons
#'   \item For species with multiple valid polygons, resamples to
#'         \code{n_target} points weighted by each polygon's environmental
#'         volume (geographic area x harmonic mean of Jacobian)
#' }
#'
#' The environmental volume weighting approximates what would have happened
#' if the polygons had been merged before JADE sampling. The key identity is:
#' \deqn{Z_i = \int J(x) dx \approx A_i / \text{mean}(1/J_k)}
#' where \eqn{Z_i} is the environmental volume of polygon i, \eqn{A_i} is its
#' geographic area, and \eqn{J_k} are the Jacobian values from its JADE samples.
#' Since the per-polygon samples are already JADE-corrected, no additional
#' per-point Jacobian weighting is needed after resampling.
#'
#' Processing is done species-by-species to avoid loading all ~125M rows
#' into memory at once.
#'
#' @param jade_samples List of data frames from dynamic branching (same order
#'   as polygon_metadata rows)
#' @param polygon_metadata Tibble from \code{load_jade_polygon_metadata()}
#' @param n_target Target samples per species after merging (default 5000)
#' @return Single data frame with cleaned, merged samples
#' @export
clean_and_merge_jade_samples <- function(jade_samples, polygon_metadata,
                                         n_target = 5000) {

  stopifnot(length(jade_samples) == nrow(polygon_metadata))

  # Tag each polygon with its index
  polygon_metadata <- polygon_metadata |>
    dplyr::mutate(.poly_idx = dplyr::row_number())

  # Keep only extant/probably extant + native polygons
  valid_meta <- polygon_metadata |>
    dplyr::filter(presence %in% c(1L, 2L), origin == 1L)

  n_removed <- nrow(polygon_metadata) - nrow(valid_meta)
  message("Filtering: ", nrow(polygon_metadata), " polygons -> ",
          nrow(valid_meta), " valid (removed ", n_removed,
          " extinct/introduced/uncertain)")

  # Group valid polygon indices by species
  species_groups <- valid_meta |>
    dplyr::group_by(species) |>
    dplyr::summarise(
      poly_indices = list(.poly_idx),
      poly_areas = list(area_km2),
      n_polys = dplyr::n(),
      .groups = "drop"
    )

  message("Processing ", nrow(species_groups), " species (",
          sum(species_groups$n_polys > 1), " multi-polygon)...")

  # Process each species independently - never bind all samples at once
  n_done <- 0L
  result_list <- purrr::pmap(species_groups, \(species, poly_indices, poly_areas, n_polys) {
    # Load samples for this species only
    sp_parts <- purrr::map2(poly_indices, poly_areas, \(idx, area) {
      s <- jade_samples[[idx]]
      if (is.null(s) || nrow(s) == 0) return(NULL)
      s |> dplyr::mutate(.poly_idx = idx, .area_km2 = area)
    }) |>
      purrr::compact()

    if (length(sp_parts) == 0) return(NULL)

    # Single polygon: pass through unchanged
    if (length(sp_parts) == 1) {
      n_done <<- n_done + 1L
      if (n_done %% 5000 == 0) message("  ", n_done, " species processed...")
      return(sp_parts[[1]] |> dplyr::select(-c(.poly_idx, .area_km2)))
    }

    # Multi-polygon: combine this species' samples only
    sp_data <- dplyr::bind_rows(sp_parts)
    n_available <- nrow(sp_data)
    n_take <- min(n_target, n_available)

    # Compute environmental volume per polygon:
    #   Z_i = area_i / mean(1/J) = area_i * harmonic_mean(J)
    poly_vols <- sp_data |>
      dplyr::summarise(
        .env_vol = dplyr::first(.area_km2) / mean(1 / jacobian, na.rm = TRUE),
        .by = .poly_idx
      )

    # Allocate samples proportional to environmental volume
    total_vol <- sum(poly_vols$.env_vol)
    poly_vols <- poly_vols |>
      dplyr::mutate(.alloc = pmax(1L, round(n_take * .env_vol / total_vol)))

    # Adjust allocations to hit exact target
    alloc_diff <- sum(poly_vols$.alloc) - n_take
    if (alloc_diff != 0) {
      idx_largest <- which.max(poly_vols$.env_vol)
      poly_vols$.alloc[idx_largest] <- poly_vols$.alloc[idx_largest] - alloc_diff
    }

    # Sample from each polygon according to allocation
    resampled <- purrr::pmap(poly_vols, \(.poly_idx, .env_vol, .alloc) {
      poly_rows <- sp_data |> dplyr::filter(.poly_idx == !!.poly_idx)
      dplyr::slice_sample(poly_rows, n = min(.alloc, nrow(poly_rows)))
    }) |>
      dplyr::bind_rows() |>
      dplyr::select(-c(.poly_idx, .area_km2))

    n_done <<- n_done + 1L
    if (n_done %% 5000 == 0) message("  ", n_done, " species processed...")

    resampled
  })

  result <- dplyr::bind_rows(purrr::compact(result_list))

  message("Clean samples: ", nrow(result), " rows, ",
          dplyr::n_distinct(result$species), " species")
  result
}


#' Extract JADE samples from introduced (non-native) ranges
#'
#' Pulls samples from polygons with `origin == 2L` (introduced) and
#' `presence %in% c(1L, 2L)` (extant/probably extant). Each polygon's
#' samples are kept separate (not aggregated per species).
#'
#' @param jade_samples List of data frames, one per polygon (from dynamic branching)
#' @param polygon_metadata Tibble from `load_jade_polygon_metadata()`
#' @return Tibble with all columns from jade_samples plus poly_idx, origin,
#'   presence, seasonal, area_km2
extract_introduced_jade_samples <- function(jade_samples, polygon_metadata) {
  stopifnot(length(jade_samples) == nrow(polygon_metadata))

  polygon_metadata <- polygon_metadata |>
    dplyr::mutate(.poly_idx = dplyr::row_number())

  introduced_meta <- polygon_metadata |>
    dplyr::filter(origin == 2L, presence %in% c(1L, 2L))

  message("Introduced + extant polygons: ", nrow(introduced_meta),
          " out of ", nrow(polygon_metadata), " total")

  if (nrow(introduced_meta) == 0) {
    message("No introduced polygons found.")
    return(tibble::tibble())
  }

  result_list <- purrr::map(seq_len(nrow(introduced_meta)), \(i) {
    row <- introduced_meta[i, ]
    idx <- row$.poly_idx
    s <- jade_samples[[idx]]
    if (is.null(s) || nrow(s) == 0) return(NULL)
    s |>
      dplyr::mutate(
        poly_idx = idx,
        origin = row$origin,
        presence = row$presence,
        seasonal = row$seasonal,
        area_km2 = row$area_km2
      )
  })

  result <- dplyr::bind_rows(purrr::compact(result_list))

  message("Introduced samples: ", nrow(result), " rows, ",
          dplyr::n_distinct(result$species), " species, ",
          dplyr::n_distinct(result$poly_idx), " polygons")
  result
}


#' Apply NA fill values from variable metadata to JADE samples
#'
#' Some BIOCLIM+ variables (fcf, scd, swe, gdd*, ngd*, gsl, gsp) have NAs
#' in warm regions where the phenomenon doesn't occur. These should be filled
#' with documented values (typically 0) from the variable metadata CSV.
#'
#' @param jade_samples Data frame of JADE samples with CHELSA column names
#' @param var_meta Tibble from chelsa_bioclim_numeric_projected_variables.csv
#' @param bioclim_files Character vector of raster file paths (for column name mapping)
#' @return jade_samples with NAs replaced according to var_meta$na_fill
#' @export
apply_na_fill_to_jade_samples <- function(jade_samples, var_meta, bioclim_files) {
  # Build mapping: raster layer name -> chelsa_name -> na_fill
  layer_names <- basename(bioclim_files) |> tools::file_path_sans_ext()

  fill_map <- tibble::tibble(layer_name = layer_names) |>
    dplyr::mutate(
      chelsa_name = purrr::map_chr(layer_name, \(ln) {
        idx <- purrr::detect_index(var_meta$chelsa_name, \(cn) {
          grepl(paste0("CHELSA_", cn, "_"), ln)
        })
        if (idx == 0) return(NA_character_)
        var_meta$chelsa_name[idx]
      })
    ) |>
    dplyr::left_join(
      var_meta |> dplyr::select(chelsa_name, na_fill),
      by = "chelsa_name"
    )

  # Apply fills to matching columns
  env_cols <- intersect(layer_names, names(jade_samples))
  n_filled <- 0L

  for (col in env_cols) {
    fill_val <- fill_map$na_fill[fill_map$layer_name == col]
    if (length(fill_val) == 1 && !is.na(fill_val)) {
      n_na <- sum(is.na(jade_samples[[col]]))
      if (n_na > 0) {
        jade_samples[[col]][is.na(jade_samples[[col]])] <- as.numeric(fill_val)
        n_filled <- n_filled + n_na
        message("  Filled ", n_na, " NAs in ", col, " with ", fill_val)
      }
    }
  }

  # Check for remaining NAs in env columns
  remaining_na <- sum(is.na(jade_samples[env_cols]))
  message("NA fill: replaced ", format(n_filled, big.mark = ","),
          " NAs across ", length(env_cols), " env columns. ",
          "Remaining NAs: ", remaining_na)

  jade_samples
}
