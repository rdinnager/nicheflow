#!/usr/bin/env Rscript
#' Build Reptile SDM Benchmark Dataset
#'
#' Standalone script that pairs GBIF occurrence data with GARD expert range
#' polygons. Sources the benchmark functions and runs the full pipeline.
#'
#' Usage: Rscript R/build_benchmark_dataset.R

library(tidyverse)
library(sf)
library(arrow)
library(rnaturalearth)
library(conflicted)
conflicts_prefer(dplyr::filter)
conflicts_prefer(dplyr::select)

# Source the benchmark functions + polygon filter helper
source("R/functions_benchmark_dataset.R")
source("R/functions_load_taxa_polygons.R")

cat("============================================================\n")
cat("  Reptile SDM Benchmark Dataset Builder\n")
cat("============================================================\n\n")

# --- Step 1: Load GARD polygons ---
cat(">> Step 1/5: Loading GARD polygons...\n")
t0 <- Sys.time()
gard_polys <- load_gard_species("data/SDM/maps/GARD1.7")
cat("   Done in", round(difftime(Sys.time(), t0, units = "secs"), 1), "s\n")
cat("   ", nrow(gard_polys), "polygons,",
    n_distinct(gard_polys$species), "species\n\n")

# --- Step 2: Match GBIF to GARD ---
cat(">> Step 2/5: Matching GBIF records to GARD species...\n")
t0 <- Sys.time()
gbif_matched <- match_gbif_to_gard(
  "data/SDM/points/Reptiles_0000117-250212154643175.csv",
  gard_polys
)
cat("   Done in", round(difftime(Sys.time(), t0, units = "secs"), 1), "s\n")
cat("   ", nrow(gbif_matched), "records,",
    n_distinct(gbif_matched$species), "species\n\n")

# --- Step 3: Clean occurrences ---
cat(">> Step 3/5: Cleaning GBIF occurrences...\n")
t0 <- Sys.time()
gbif_cleaned <- clean_gbif_occurrences(gbif_matched)
cat("   Done in", round(difftime(Sys.time(), t0, units = "secs"), 1), "s\n")
cat("\n   Cleaning log:\n")
print(gbif_cleaned$log, n = Inf)
cat("\n")

# --- Step 4: Build benchmark dataset ---
cat(">> Step 4/5: Building benchmark dataset (min 10 occurrences)...\n")
t0 <- Sys.time()
benchmark <- build_benchmark_dataset(gbif_cleaned, gard_polys,
                                     min_occurrences = 10)
cat("   Done in", round(difftime(Sys.time(), t0, units = "secs"), 1), "s\n")
cat("   ", nrow(benchmark$species_summary), "species,",
    nrow(benchmark$occurrences), "occurrences,",
    nrow(benchmark$polygons), "polygons\n\n")

# --- Step 5: Export ---
cat(">> Step 5/5: Exporting to data/benchmark/...\n")
t0 <- Sys.time()
output_files <- export_benchmark_dataset(benchmark, "data/benchmark")
cat("   Done in", round(difftime(Sys.time(), t0, units = "secs"), 1), "s\n\n")

cat("============================================================\n")
cat("  COMPLETE\n")
cat("  Species:", nrow(benchmark$species_summary), "\n")
cat("  Occurrences:", nrow(benchmark$occurrences), "\n")
cat("  Polygons:", nrow(benchmark$polygons), "\n")
cat("  Output files:\n")
for (f in output_files) cat("    ", f, "\n")
cat("============================================================\n")
