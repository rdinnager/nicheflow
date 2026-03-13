#!/usr/bin/env Rscript
library(sf)

polys <- st_read("data/benchmark/reptile_benchmark_ranges.gpkg", quiet = TRUE)
dir.create("data/benchmark/shapefile", showWarnings = FALSE)
st_write(polys, "data/benchmark/shapefile/reptile_benchmark_ranges.shp",
         delete_dsn = TRUE, quiet = TRUE)
cat("Done:", nrow(polys), "features written to data/benchmark/shapefile/\n")
