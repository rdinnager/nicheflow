#' Reptile SDM Benchmark Dataset Functions
#'
#' Functions for creating a benchmark dataset that pairs GBIF occurrence
#' records with GARD expert range polygons for evaluating species
#' distribution models.
#'
#' @author rdinnage


#' Load and filter GARD species polygons for benchmarking
#'
#' Loads the GARD 1.7 shapefile and applies `filter_taxa_polygons()` to
#' remove circular buffers, tiny ranges, and weird shapes.
#'
#' @param gard_path Path to GARD1.7 folder
#' @return sf object with columns: species, family, group, geometry
load_gard_species <- function(gard_path) {
  message("Loading GARD 1.7 polygons from: ", gard_path)
  polys <- sf::st_read(gard_path, quiet = TRUE)
  message("  Raw polygons: ", nrow(polys))

  polys <- polys |>
    dplyr::mutate(species = binomial)

  polys <- filter_taxa_polygons(polys)
  message("  After filtering: ", nrow(polys))

  polys |>
    dplyr::select(species, family, group, geometry)
}


#' Match GBIF records to GARD species names
#'
#' Efficiently reads the GBIF CSV using arrow, keeping only needed columns
#' and filtering to SPECIES rank records that match GARD species names.
#'
#' @param gbif_path Path to GBIF occurrence TSV
#' @param gard_species sf object from `load_gard_species()` with a `species` column
#' @return tibble of matched GBIF records
match_gbif_to_gard <- function(gbif_path, gard_species) {
  gard_names <- unique(gard_species$species)
  message("Matching GBIF records to ", length(gard_names), " GARD species...")

  gbif <- arrow::read_tsv_arrow(
    gbif_path,
    col_select = c("species", "taxonRank", "decimalLatitude", "decimalLongitude",
                    "coordinateUncertaintyInMeters", "countryCode",
                    "basisOfRecord", "year", "occurrenceStatus", "issue"),
    as_data_frame = FALSE
  )

  matched <- gbif |>
    dplyr::filter(taxonRank == "SPECIES", species %in% gard_names) |>
    dplyr::collect()

  message("  Matched ", nrow(matched), " records across ",
          dplyr::n_distinct(matched$species), " species")
  matched
}


#' Known artifact coordinate locations
#'
#' Returns a tibble of coordinates known to be data entry artifacts:
#' GBIF headquarters, major natural history museums, and country centroids
#' for commonly mis-geocoded countries.
#'
#' @return tibble with columns: name, lat, lon
#' @keywords internal
get_artifact_coords <- function() {
  tibble::tibble(
    name = c(
      "GBIF HQ Copenhagen", "NHM London", "Smithsonian DC",
      "MNHN Paris", "Berlin Museum", "Leiden Naturalis",
      "centroid Brazil", "centroid Australia", "centroid USA",
      "centroid India", "centroid China", "centroid Russia",
      "centroid South Africa", "centroid Mexico", "centroid Indonesia",
      "centroid Argentina"
    ),
    lat = c(
      55.6761, 51.4967, 38.8913,
      48.8441, 52.5300, 52.1643,
      -14.24, -25.27, 39.83,
      20.59, 35.86, 61.52,
      -30.56, 23.63, -0.79,
      -38.42
    ),
    lon = c(
      12.5683, -0.1764, -77.0260,
      2.3596, 13.3946, 4.4850,
      -51.93, 133.78, -98.58,
      78.96, 104.20, 105.32,
      22.94, -102.55, 113.92,
      -63.62
    )
  )
}


#' Clean GBIF occurrence records for SDM benchmarking
#'
#' Applies a standard SDM cleaning pipeline without external dependencies
#' like CoordinateCleaner. Steps:
#' 1. Remove absences
#' 2. Remove missing coordinates
#' 3. Remove zero/near-zero coordinates
#' 4. Remove equal lat/lon
#' 5. Remove sea points (not on land)
#' 6. Remove centroid/institution artifacts
#' 7. Remove high coordinate uncertainty
#' 8. Deduplicate
#'
#' @param gbif_data tibble from `match_gbif_to_gard()`
#' @return list with `data` (cleaned tibble) and `log` (cleaning summary tibble)
clean_gbif_occurrences <- function(gbif_data) {
  log_entries <- list()
  n_start <- nrow(gbif_data)
  log_entries[[1]] <- tibble::tibble(step = "start", records = n_start, removed = 0L)

  # 1. Remove absences
  dat <- gbif_data |>
    dplyr::filter(is.na(occurrenceStatus) | occurrenceStatus == "PRESENT")
  log_entries[[2]] <- tibble::tibble(step = "remove_absences",
                                     records = nrow(dat),
                                     removed = n_start - nrow(dat))

  # 2. Remove missing coordinates
  n_before <- nrow(dat)
  dat <- dat |>
    dplyr::filter(!is.na(decimalLatitude), !is.na(decimalLongitude))
  log_entries[[3]] <- tibble::tibble(step = "remove_missing_coords",
                                     records = nrow(dat),
                                     removed = n_before - nrow(dat))

  # 3. Remove zero/near-zero coordinates
  n_before <- nrow(dat)
  dat <- dat |>
    dplyr::filter(abs(decimalLatitude) + abs(decimalLongitude) >= 0.1)
  log_entries[[4]] <- tibble::tibble(step = "remove_zero_coords",
                                     records = nrow(dat),
                                     removed = n_before - nrow(dat))

  # 4. Remove equal lat/lon (data entry error pattern)
  n_before <- nrow(dat)
  dat <- dat |>
    dplyr::filter(decimalLatitude != decimalLongitude)
  log_entries[[5]] <- tibble::tibble(step = "remove_equal_latlon",
                                     records = nrow(dat),
                                     removed = n_before - nrow(dat))

  # 5. Remove sea points
  message("  Filtering sea points (spatial intersection with land)...")
  n_before <- nrow(dat)
  land <- rnaturalearth::ne_countries(scale = 50, returnclass = "sf")
  land <- sf::st_union(land)
  pts_sf <- sf::st_as_sf(dat,
                          coords = c("decimalLongitude", "decimalLatitude"),
                          crs = 4326, remove = FALSE)
  on_land <- lengths(sf::st_intersects(pts_sf, land)) > 0
  dat <- dat[on_land, ]
  rm(pts_sf, land, on_land)
  gc(verbose = FALSE)
  log_entries[[6]] <- tibble::tibble(step = "remove_sea_points",
                                     records = nrow(dat),
                                     removed = n_before - nrow(dat))

  # 6. Remove centroid/institution artifacts
  n_before <- nrow(dat)
  artifacts <- get_artifact_coords()
  artifact_flag <- rep(FALSE, nrow(dat))
  for (i in seq_len(nrow(artifacts))) {
    near <- abs(dat$decimalLatitude - artifacts$lat[i]) < 0.01 &
            abs(dat$decimalLongitude - artifacts$lon[i]) < 0.01
    artifact_flag <- artifact_flag | near
  }
  dat <- dat[!artifact_flag, ]
  log_entries[[7]] <- tibble::tibble(step = "remove_artifacts",
                                     records = nrow(dat),
                                     removed = n_before - nrow(dat))

  # 7. Remove high coordinate uncertainty
  n_before <- nrow(dat)
  dat <- dat |>
    dplyr::filter(is.na(coordinateUncertaintyInMeters) |
                  coordinateUncertaintyInMeters <= 10000)
  log_entries[[8]] <- tibble::tibble(step = "remove_high_uncertainty",
                                     records = nrow(dat),
                                     removed = n_before - nrow(dat))

  # 8. Deduplicate (round to ~111m, unique per species)
  n_before <- nrow(dat)
  dat <- dat |>
    dplyr::mutate(
      lat_round = round(decimalLatitude, 3),
      lon_round = round(decimalLongitude, 3)
    ) |>
    dplyr::distinct(species, lat_round, lon_round, .keep_all = TRUE) |>
    dplyr::select(-lat_round, -lon_round)
  log_entries[[9]] <- tibble::tibble(step = "deduplicate",
                                     records = nrow(dat),
                                     removed = n_before - nrow(dat))

  cleaning_log <- dplyr::bind_rows(log_entries)
  message("  Cleaning complete: ", n_start, " -> ", nrow(dat), " records (",
          round(100 * nrow(dat) / n_start, 1), "% retained)")

  list(data = dat, log = cleaning_log)
}


#' Build the benchmark dataset from cleaned occurrences and GARD polygons
#'
#' Filters to species meeting a minimum occurrence threshold and subsets
#' the GARD polygons to matching species.
#'
#' @param cleaned_occurrences list from `clean_gbif_occurrences()` with `data` and `log`
#' @param gard_polys sf object from `load_gard_species()`
#' @param min_occurrences minimum number of cleaned occurrences per species (default 10)
#' @return list with `occurrences`, `polygons`, `species_summary`, `cleaning_log`
build_benchmark_dataset <- function(cleaned_occurrences, gard_polys,
                                    min_occurrences = 10) {
  occ <- cleaned_occurrences$data

  species_counts <- occ |>
    dplyr::count(species, name = "n_occurrences") |>
    dplyr::filter(n_occurrences >= min_occurrences)

  message("Species with >= ", min_occurrences, " occurrences: ", nrow(species_counts))

  keep_species <- species_counts$species

  occ_filtered <- occ |>
    dplyr::filter(species %in% keep_species)

  polys_filtered <- gard_polys |>
    dplyr::filter(species %in% keep_species)

  species_summary <- species_counts |>
    dplyr::left_join(
      sf::st_drop_geometry(polys_filtered) |>
        dplyr::select(species, family) |>
        dplyr::distinct(),
      by = "species"
    ) |>
    dplyr::arrange(family, species)

  message("Benchmark dataset: ", nrow(species_summary), " species, ",
          nrow(occ_filtered), " occurrences, ",
          nrow(polys_filtered), " range polygons")

  list(
    occurrences = occ_filtered,
    polygons = polys_filtered,
    species_summary = species_summary,
    cleaning_log = cleaned_occurrences$log
  )
}


#' Export benchmark dataset to disk
#'
#' Saves the benchmark dataset as parquet, CSV, GeoPackage, shapefile, and metadata JSON.
#'
#' @param benchmark list from `build_benchmark_dataset()`
#' @param output_dir directory to write files to (created if needed)
#' @return character vector of output file paths (for targets `format = "file"`)
export_benchmark_dataset <- function(benchmark, output_dir) {
  dir.create(output_dir, showWarnings = FALSE, recursive = TRUE)

  paths <- character()

  # Occurrences as parquet
  parquet_path <- file.path(output_dir, "reptile_benchmark_occurrences.parquet")
  arrow::write_parquet(benchmark$occurrences, parquet_path)
  paths <- c(paths, parquet_path)
  message("  Wrote: ", parquet_path)

  # Occurrences as CSV
  csv_path <- file.path(output_dir, "reptile_benchmark_occurrences.csv")
  readr::write_csv(benchmark$occurrences, csv_path)
  paths <- c(paths, csv_path)
  message("  Wrote: ", csv_path)

  # Species summary
  summary_path <- file.path(output_dir, "reptile_benchmark_species.csv")
  readr::write_csv(benchmark$species_summary, summary_path)
  paths <- c(paths, summary_path)
  message("  Wrote: ", summary_path)

  # GARD polygons as GeoPackage
  gpkg_path <- file.path(output_dir, "reptile_benchmark_ranges.gpkg")
  sf::st_write(benchmark$polygons, gpkg_path, delete_dsn = TRUE, quiet = TRUE)
  paths <- c(paths, gpkg_path)
  message("  Wrote: ", gpkg_path)

  # GARD polygons as shapefile
  shp_path <- file.path(output_dir, "reptile_benchmark_ranges.shp")
  sf::st_write(benchmark$polygons, shp_path, delete_dsn = TRUE, quiet = TRUE)
  paths <- c(paths, shp_path)
  message("  Wrote: ", shp_path)

  # Metadata JSON
  meta <- list(
    dataset = "Reptile SDM Benchmark",
    created = as.character(Sys.time()),
    n_species = nrow(benchmark$species_summary),
    n_occurrences = nrow(benchmark$occurrences),
    n_polygons = nrow(benchmark$polygons),
    min_occurrences_threshold = min(benchmark$species_summary$n_occurrences),
    cleaning_log = as.list(benchmark$cleaning_log),
    files = basename(paths)
  )
  meta_path <- file.path(output_dir, "reptile_benchmark_metadata.json")
  jsonlite::write_json(meta, meta_path, pretty = TRUE, auto_unbox = TRUE)
  paths <- c(paths, meta_path)
  message("  Wrote: ", meta_path)

  message("Export complete: ", length(paths), " files to ", output_dir)
  paths
}
