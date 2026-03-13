#' Multi-Taxa Polygon Loading Functions
#'
#' Functions for loading and combining species range polygons from
#' multiple taxa datasets (reptiles, amphibians, mammals).
#'
#' @author rdinnage


#' Load and combine all taxa polygons
#'
#' Loads reptiles from GARD1.7, amphibians from AMPHIBIANS (combining PART1 + PART2),
#' and mammals from MAMMALS_TERRESTRIAL_ONLY. Applies filtering to remove invalid
#' and problematic species ranges.
#'
#' @param poly_folders Named list with paths to polygon folders:
#'   - reptiles: path to GARD1.7 folder
#'   - amphibians: path to AMPHIBIANS folder (contains PART1 + PART2)
#'   - mammals: path to MAMMALS_TERRESTRIAL_ONLY folder
#' @return sf object with columns: species, taxon, geometry
#' @export
load_all_taxa_polygons <- function(poly_folders) {

  all_polys <- list()

  # Load reptiles (GARD1.7 - single shapefile)
  if (!is.null(poly_folders$reptiles) && dir.exists(poly_folders$reptiles)) {
    message("Loading reptiles from GARD1.7...")
    reptiles <- load_reptile_polygons(poly_folders$reptiles)
    all_polys$reptiles <- reptiles
    message("  Loaded ", nrow(reptiles), " reptile species")
  }

  # Load amphibians (PART1 + PART2 combined)
  if (!is.null(poly_folders$amphibians) && dir.exists(poly_folders$amphibians)) {
    message("Loading amphibians from AMPHIBIANS...")
    amphibians <- load_amphibian_polygons(poly_folders$amphibians)
    all_polys$amphibians <- amphibians
    message("  Loaded ", nrow(amphibians), " amphibian species")
  }

  # Load mammals
  if (!is.null(poly_folders$mammals) && dir.exists(poly_folders$mammals)) {
    message("Loading mammals from MAMMALS_TERRESTRIAL_ONLY...")
    mammals <- load_mammal_polygons(poly_folders$mammals)
    all_polys$mammals <- mammals
    message("  Loaded ", nrow(mammals), " mammal species")
  }

  # Combine all taxa
  combined <- dplyr::bind_rows(all_polys)
  message("Total: ", nrow(combined), " species across ", length(all_polys), " taxa")

  # Check for duplicates
  dup_species <- combined$species[duplicated(combined$species)]
  if (length(dup_species) > 0) {
    warning("Found ", length(dup_species), " duplicate species names across taxa. ",
            "Keeping first occurrence.")
    combined <- combined |>
      dplyr::filter(!duplicated(species))
  }

  combined
}


#' Load reptile polygons from GARD1.7
#'
#' @param folder Path to GARD1.7 folder
#' @return sf object with species, taxon, geometry columns
#' @keywords internal
load_reptile_polygons <- function(folder) {
  polys <- sf::st_read(folder, quiet = TRUE)

  # GARD uses 'binomial' column for species names
  polys <- polys |>
    dplyr::mutate(
      species = binomial,
      taxon = "reptiles"
    )

  # Apply standard filtering
  polys <- filter_taxa_polygons(polys)

  polys |>
    dplyr::select(species, taxon, geometry)
}


#' Load amphibian polygons from AMPHIBIANS folder
#'
#' Combines AMPHIBIANS_PART1 and AMPHIBIANS_PART2 shapefiles.
#'
#' @param folder Path to AMPHIBIANS folder
#' @return sf object with species, taxon, geometry columns
#' @keywords internal
load_amphibian_polygons <- function(folder) {
  part1_shp <- file.path(folder, "AMPHIBIANS_PART1.shp")
  part2_shp <- file.path(folder, "AMPHIBIANS_PART2.shp")

  polys_list <- list()

  if (file.exists(part1_shp)) {
    message("  Loading AMPHIBIANS_PART1...")
    polys_list$part1 <- sf::st_read(part1_shp, quiet = TRUE)
  }

  if (file.exists(part2_shp)) {
    message("  Loading AMPHIBIANS_PART2...")
    polys_list$part2 <- sf::st_read(part2_shp, quiet = TRUE)
  }

  # Combine parts
  polys <- dplyr::bind_rows(polys_list)

  # IUCN uses 'sci_name' or 'binomial' for species names
  species_col <- if ("binomial" %in% names(polys)) "binomial" else if ("sci_name" %in% names(polys)) "sci_name" else NULL

  if (is.null(species_col)) {
    stop("Could not find species name column in amphibian data. ",
         "Available columns: ", paste(names(polys), collapse = ", "))
  }

  polys <- polys |>
    dplyr::mutate(
      species = .data[[species_col]],
      taxon = "amphibians"
    )

  # Apply standard filtering
  polys <- filter_taxa_polygons(polys)

  polys |>
    dplyr::select(species, taxon, geometry)
}


#' Load mammal polygons from MAMMALS_TERRESTRIAL_ONLY
#'
#' @param folder Path to MAMMALS_TERRESTRIAL_ONLY folder
#' @return sf object with species, taxon, geometry columns
#' @keywords internal
load_mammal_polygons <- function(folder) {
  polys <- sf::st_read(folder, quiet = TRUE)

  # IUCN uses 'sci_name' or 'binomial' for species names
  species_col <- if ("binomial" %in% names(polys)) "binomial" else if ("sci_name" %in% names(polys)) "sci_name" else NULL

  if (is.null(species_col)) {
    stop("Could not find species name column in mammal data. ",
         "Available columns: ", paste(names(polys), collapse = ", "))
  }

  polys <- polys |>
    dplyr::mutate(
      species = .data[[species_col]],
      taxon = "mammals"
    )

  # Apply standard filtering
  polys <- filter_taxa_polygons(polys)

  polys |>
    dplyr::select(species, taxon, geometry)
}


#' Filter taxa polygons
#'
#' Validates geometries, calculates areas, and removes problematic species:
#' - Invalid geometries that can't be fixed
#' - Very small ranges (< 100 km^2)
#' - Very weird shapes (area/bbox ratio < 0.01)
#' - Perfect circles (artifacts)
#'
#' @param polys sf object with species polygons
#' @return Filtered sf object
#' @keywords internal
filter_taxa_polygons <- function(polys) {
  # Fix invalid geometries
  invalid <- !sf::st_is_valid(polys)
  if (any(invalid)) {
    message("  Fixing ", sum(invalid), " invalid geometries...")
    polys$geometry[invalid] <- sf::st_make_valid(polys$geometry[invalid])

    # Check if fix worked
    still_invalid <- !sf::st_is_valid(polys)
    if (any(still_invalid)) {
      message("  Removing ", sum(still_invalid), " geometries that couldn't be fixed")
      polys <- polys[!still_invalid, ]
    }
  }

  # Calculate areas vectorized (in km^2) -- much faster than row-by-row
  message("  Computing polygon areas (vectorized)...")
  old_s2 <- sf::sf_use_s2()
  sf::sf_use_s2(FALSE)
  areas <- tryCatch(
    as.numeric(sf::st_area(polys)) / 1e6,
    error = \(e) { message("  st_area failed: ", e$message); rep(0, nrow(polys)) }
  )

  # Bounding box areas from coordinate ranges (no sf overhead)
  message("  Computing bounding box areas...")
  coords_range <- do.call(rbind, lapply(sf::st_geometry(polys), \(g) {
    bb <- sf::st_bbox(g)
    unname(c(bb[1], bb[3], bb[2], bb[4]))  # xmin, xmax, ymin, ymax
  }))
  colnames(coords_range) <- c("xmin", "xmax", "ymin", "ymax")
  # Approximate bbox area in km^2 using latitude correction
  deg_to_km <- 111.32
  mean_lat <- (coords_range[, "ymin"] + coords_range[, "ymax"]) / 2
  bb_areas <- (coords_range[, "xmax"] - coords_range[, "xmin"]) *
              (coords_range[, "ymax"] - coords_range[, "ymin"]) *
              cos(mean_lat * pi / 180) * deg_to_km^2
  rm(coords_range, mean_lat)
  sf::sf_use_s2(old_s2)
  gc(verbose = FALSE)

  area_ratios <- areas / pmax(bb_areas, 1e-10)

  # Filter criteria
  circles <- abs(area_ratios - (pi/4)) < 0.00015
  very_small <- areas < 100
  weird_shapes <- area_ratios < 0.01

  remove <- circles | very_small | weird_shapes

  message("  Filtering: ", sum(circles), " circles, ",
          sum(very_small), " very small, ",
          sum(weird_shapes), " weird shapes")

  polys |>
    dplyr::filter(!remove)
}


#' Load and filter polygons from a single folder (backwards compatible)
#'
#' This is a wrapper around the existing load_and_filter_polygons logic
#' that returns polygons with a taxon column.
#'
#' @param folder Path to polygon folder
#' @param taxon_name Name of the taxon (e.g., "reptiles")
#' @return sf object with species, taxon, geometry columns
#' @export
load_single_taxa_polygons <- function(folder, taxon_name = "unknown") {

  # Detect folder type and load appropriately
  if (grepl("GARD", folder, ignore.case = TRUE)) {
    polys <- load_reptile_polygons(folder)
  } else if (grepl("AMPHIBIAN", folder, ignore.case = TRUE)) {
    polys <- load_amphibian_polygons(folder)
  } else if (grepl("MAMMAL", folder, ignore.case = TRUE)) {
    polys <- load_mammal_polygons(folder)
  } else {
    # Generic loading
    polys <- sf::st_read(folder, quiet = TRUE)

    # Try to find species column
    species_col <- dplyr::case_when(
      "binomial" %in% names(polys) ~ "binomial",
      "sci_name" %in% names(polys) ~ "sci_name",
      "species" %in% names(polys) ~ "species",
      TRUE ~ names(polys)[1]
    )

    polys <- polys |>
      dplyr::mutate(
        species = .data[[species_col]],
        taxon = taxon_name
      )

    polys <- filter_taxa_polygons(polys)
    polys <- polys |>
      dplyr::select(species, taxon, geometry)
  }

  polys
}
