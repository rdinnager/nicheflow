#' GeODE Validation Functions
#'
#' Ecoregion zoom plot generation for GeODE rectified flow model validation.
#' Generates per-ecoregion comparison plots of truth vs predicted coordinates.
#' Used by train_geode.R.


#' Load ecoregions from RDS file
#'
#' @param path Path to ecoregions .rds file
#' @return sf object of WWF ecoregions
load_ecoregions <- function(path) {
  ecoregions <- readRDS(path)
  # Ensure WGS84

  sf::st_transform(ecoregions, 4326)
}


#' Select validation ecoregions with biome diversity
#'
#' Filters to terrestrial biomes (1-14), enforces area bounds using SHAPE_AREA,
#' then samples for biome diversity.
#'
#' @param ecoregions sf object of ecoregions (must have SHAPE_AREA column)
#' @param n Number of ecoregions to select
#' @param min_area Minimum SHAPE_AREA (sq degrees) to include
#' @param max_area Maximum SHAPE_AREA (sq degrees) to include
#' @return sf subset of selected ecoregions
select_validation_ecoregions <- function(ecoregions, n = 2, min_area = 1,
                                          max_area = 20) {
  # Filter to terrestrial biomes
  eco_filtered <- ecoregions[ecoregions$BIOME_NUM >= 1 &
                               ecoregions$BIOME_NUM <= 14, ]

  # Area filter using SHAPE_AREA column
  eco_filtered <- eco_filtered[eco_filtered$SHAPE_AREA >= min_area &
                                 eco_filtered$SHAPE_AREA <= max_area, ]

  if (nrow(eco_filtered) < n) {
    warning("Only ", nrow(eco_filtered), " ecoregions available, requested ", n)
    return(eco_filtered)
  }

  # Sample with biome diversity: one per biome first, then fill
  biomes <- unique(eco_filtered$BIOME_NUM)
  selected_idx <- integer(0)

  if (length(biomes) >= n) {
    sampled_biomes <- sample(biomes, n)
    for (bm in sampled_biomes) {
      candidates <- which(eco_filtered$BIOME_NUM == bm)
      selected_idx <- c(selected_idx, sample(candidates, 1))
    }
  } else {
    # Fewer biomes than n: one from each, then random fill
    for (bm in biomes) {
      candidates <- which(eco_filtered$BIOME_NUM == bm)
      selected_idx <- c(selected_idx, sample(candidates, 1))
    }
    remaining <- setdiff(seq_len(nrow(eco_filtered)), selected_idx)
    extra <- min(n - length(selected_idx), length(remaining))
    if (extra > 0) {
      selected_idx <- c(selected_idx, sample(remaining, extra))
    }
  }

  eco_filtered[selected_idx, ]
}


#' Find tensor row indices for cells within an ecoregion
#'
#' Uses bbox prefilter on raw lon/lat vectors for speed, then precise
#' sf::st_contains for accuracy.
#'
#' @param ecoregion Single-row sf object (one ecoregion polygon)
#' @param xy_raw_lon Numeric vector of raw longitudes (all land cells)
#' @param xy_raw_lat Numeric vector of raw latitudes (all land cells)
#' @param max_points Maximum number of indices to return (random subsample)
#' @return Integer vector of tensor row indices
find_ecoregion_indices <- function(ecoregion, xy_raw_lon, xy_raw_lat,
                                   max_points = 50000) {
  # Bbox prefilter (fast)
  bbox <- sf::st_bbox(ecoregion)
  bbox_mask <- xy_raw_lon >= bbox["xmin"] & xy_raw_lon <= bbox["xmax"] &
    xy_raw_lat >= bbox["ymin"] & xy_raw_lat <= bbox["ymax"]
  bbox_idx <- which(bbox_mask)

  if (length(bbox_idx) == 0) return(integer(0))

  # Simplify complex polygons to speed up containment check
  eco_simple <- sf::st_simplify(ecoregion, dTolerance = 0.01)

  # Precise containment check with sf
  pts <- sf::st_as_sf(
    data.frame(lon = xy_raw_lon[bbox_idx], lat = xy_raw_lat[bbox_idx]),
    coords = c("lon", "lat"), crs = 4326
  )

  contained <- sf::st_contains(eco_simple, pts)[[1]]
  result_idx <- bbox_idx[contained]

  if (length(result_idx) > max_points) {
    result_idx <- sort(sample(result_idx, max_points))
  }

  result_idx
}


#' Generate ecoregion zoom plots for GeODE validation
#'
#' Combines fixed and random ecoregions, generates predicted coordinates
#' via ODE integration, and saves comparison plots.
#'
#' @param model TrajNet model (on device, non-JIT for ODE integration)
#' @param fixed_ecoregions sf object of fixed ecoregions (tracked across epochs)
#' @param all_ecoregions sf object of all ecoregions (for random selection)
#' @param env_tensor Full environment tensor (CPU, standardized)
#' @param xy_raw_lon Raw longitude vector (numeric, all land cells)
#' @param xy_raw_lat Raw latitude vector (numeric, all land cells)
#' @param checkpoint_dir Directory to save plots
#' @param epoch Current epoch number
#' @param xy_mean_sd List with mean and sd vectors for un-standardizing XY
#' @param device Device string for model
#' @param x_sd SD of standardized x coordinate (for noise generation)
#' @param y_sd SD of standardized y coordinate (for noise generation)
#' @param n_random Number of additional random ecoregions per round
#' @param n_sample Number of points to sample per ecoregion
#' @param ode_steps Number of ODE integration steps
generate_ecoregion_zoom_plots <- function(model, fixed_ecoregions, all_ecoregions,
                                           env_tensor, xy_raw_lon, xy_raw_lat,
                                           checkpoint_dir, epoch,
                                           xy_mean_sd, device, x_sd, y_sd,
                                           n_random = 2, n_sample = 5000,
                                           ode_steps = 200) {
  library(deSolve)

  # Select random ecoregions (excluding fixed ones)
  fixed_names <- fixed_ecoregions$ECO_NAME
  available <- all_ecoregions[!all_ecoregions$ECO_NAME %in% fixed_names, ]
  random_ecoregions <- select_validation_ecoregions(available, n = n_random)

  # Combine: fixed (numbered 1-2) + random (numbered 3-4)
  eco_list <- list()
  for (i in seq_len(nrow(fixed_ecoregions))) {
    eco_list[[i]] <- list(
      eco = fixed_ecoregions[i, ],
      number = i,
      name = fixed_ecoregions$ECO_NAME[i]
    )
  }
  for (i in seq_len(nrow(random_ecoregions))) {
    eco_list[[nrow(fixed_ecoregions) + i]] <- list(
      eco = random_ecoregions[i, ],
      number = nrow(fixed_ecoregions) + i,
      name = random_ecoregions$ECO_NAME[i]
    )
  }

  # Generate predictions and collect plot rows for each ecoregion
  plot_rows <- list()

  for (eco_info in eco_list) {
    tryCatch({
      message("  Validating ecoregion ", eco_info$number, ": ", eco_info$name)

      # Find indices within this ecoregion
      eco_idx <- find_ecoregion_indices(
        eco_info$eco, xy_raw_lon, xy_raw_lat,
        max_points = n_sample * 2
      )

      if (length(eco_idx) < 100) {
        message("    Skipping: only ", length(eco_idx), " cells found")
        next
      }

      # Subsample to n_sample
      if (length(eco_idx) > n_sample) {
        eco_idx <- sort(sample(eco_idx, n_sample))
      }

      # Extract env data for these cells from the full tensor
      env_sub <- env_tensor[eco_idx, ]$to(device = device)

      # Truth coordinates (raw lon/lat)
      truth_lon <- xy_raw_lon[eco_idx]
      truth_lat <- xy_raw_lat[eco_idx]

      # Generate predictions via ODE integration
      n_pts <- length(eco_idx)
      x_init <- rnorm(n_pts, sd = x_sd * 1.1)
      y_init <- rnorm(n_pts, sd = y_sd * 1.1)
      y_init_tensor <- torch::torch_tensor(cbind(x_init, y_init))

      final_coords <- as.matrix(model$sample_trajectory(
        y_init_tensor$to(device = device),
        env_sub,
        steps = ode_steps
      ))

      # Un-standardize predicted coordinates
      pred_lon <- final_coords[, 1] * xy_mean_sd$sd[1] + xy_mean_sd$mean[1]
      pred_lat <- final_coords[, 2] * xy_mean_sd$sd[2] + xy_mean_sd$mean[2]

      # Create plot row (zoom + world side by side)
      row <- create_ecoregion_zoom_plot(
        truth_lon, truth_lat, pred_lon, pred_lat,
        eco_info$eco, eco_info$name
      )
      plot_rows[[length(plot_rows) + 1]] <- row
      message("    Done: ", eco_info$name)

    }, error = \(e) {
      message("    Error for ecoregion ", eco_info$number, ": ", e$message)
    })
  }

  torch::cuda_empty_cache()

  # Stack all ecoregion rows into one combined plot and save
  if (length(plot_rows) > 0) {
    library(patchwork)
    combined <- patchwork::wrap_plots(plot_rows, ncol = 1)
    output_path <- file.path(
      checkpoint_dir,
      sprintf("epoch_%04d_ecoregion_validation.png", epoch)
    )
    dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
    n_rows <- length(plot_rows)
    ragg::agg_png(output_path, width = 2000, height = 600 * n_rows, res = 150)
    print(combined)
    dev.off()
    message("  Saved combined plot: ", basename(output_path))
  }
}


#' Create a single ecoregion zoom + world row
#'
#' Returns a patchwork object with two panels side by side:
#'   Left: zoomed map with coastline, ecoregion boundary, truth + predicted points
#'   Right: world map with all points (showing where predictions land globally)
#'     and a red rectangle highlighting the zoom region
#'
#' @param truth_lon Numeric vector of truth longitudes
#' @param truth_lat Numeric vector of truth latitudes
#' @param pred_lon Numeric vector of predicted longitudes
#' @param pred_lat Numeric vector of predicted latitudes
#' @param ecoregion_sf Single-row sf ecoregion polygon
#' @param eco_name Character name of the ecoregion
#' @return A patchwork object (zoom | world) for one ecoregion row
create_ecoregion_zoom_plot <- function(truth_lon, truth_lat, pred_lon, pred_lat,
                                        ecoregion_sf, eco_name) {
  library(ggplot2)
  library(patchwork)

  coastline <- rnaturalearth::ne_coastline(scale = 50, returnclass = "sf")
  world <- rnaturalearth::ne_countries(scale = 110, returnclass = "sf")

  # Compute bounding box with padding
  bbox <- sf::st_bbox(ecoregion_sf)
  pad_lon <- (bbox["xmax"] - bbox["xmin"]) * 0.2
  pad_lat <- (bbox["ymax"] - bbox["ymin"]) * 0.2
  xlim <- c(bbox["xmin"] - pad_lon, bbox["xmax"] + pad_lon)
  ylim <- c(bbox["ymin"] - pad_lat, bbox["ymax"] + pad_lat)

  # Build data frame (predicted points only)
  pred_df <- data.frame(lon = pred_lon, lat = pred_lat)

  # Left panel: zoomed map
  # Ecoregion border drawn after points so it stays visible
  zoom_map <- ggplot() +
    geom_sf(data = coastline, colour = "grey50", linewidth = 0.3) +
    geom_point(data = pred_df,
               aes(x = lon, y = lat),
               colour = "red", size = 0.3, alpha = 0.3) +
    geom_sf(data = ecoregion_sf, fill = NA, colour = "blue",
            linewidth = 0.8) +
    coord_sf(xlim = xlim, ylim = ylim, expand = FALSE) +
    labs(title = eco_name) +
    theme_minimal() +
    theme(
      plot.title = element_text(size = 10, face = "bold"),
      axis.text = element_text(size = 7)
    )

  # Right panel: world map with predicted points + zoom rectangle
  bbox_rect <- data.frame(
    xmin = xlim[1], xmax = xlim[2],
    ymin = ylim[1], ymax = ylim[2]
  )

  world_map <- ggplot() +
    geom_sf(data = world, fill = "grey80", colour = "grey50", linewidth = 0.1) +
    geom_point(data = pred_df,
               aes(x = lon, y = lat),
               colour = "red", size = 0.1, alpha = 0.2) +
    geom_rect(data = bbox_rect,
              aes(xmin = xmin, xmax = xmax, ymin = ymin, ymax = ymax),
              fill = NA, colour = "red", linewidth = 0.6) +
    coord_sf(expand = FALSE) +
    theme_minimal() +
    theme(
      axis.text = element_blank(),
      axis.ticks = element_blank(),
      axis.title = element_blank(),
      panel.background = element_rect(fill = "white", colour = "grey30")
    )

  # Combine side by side: zoom (wider) | world
  zoom_map + world_map + plot_layout(widths = c(3, 4))
}
