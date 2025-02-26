library(tidyverse)
library(sf)
library(terra)
library(furrr)
library(rnaturalearth)
library(arrow)

source("R/utils.R")

options(future.globals.maxSize = 5000*1024^2)

land <- ne_download(scale = 10, type = 'land', category = 'physical')

# chelsa_files <- list.files("data/SDM/env/CHELSA-BIOMCLIM+/1981-2010/bio", 
#                            full.names = TRUE)
# chelsa <- rast(chelsa_files)

squamates2 <- read_rds("data/final_squamate_sf.rds")
## remove invalid and tiny polygons
invalid <- map_lgl(squamates2$geometry, st_is_valid, .progress = TRUE)
squamates2$geometry[!invalid] <- st_make_valid(squamates2$geometry[!invalid])
invalid2 <- map_lgl(squamates2$geometry[!invalid], st_is_valid, .progress = TRUE)
exclude <- which(!invalid)[!invalid2]
squamates2 <- squamates2 |>
  slice(-exclude)
invalid <- map_lgl(squamates2$geometry, st_is_valid, .progress = TRUE)
areas <- map(squamates2$geometry, st_area, .progress = TRUE)
areas <- list_c(areas)
squamates2 <- squamates2 |>
  filter(areas > 0.01)

write_rds(squamates2, "data/final_squamates_filtered.rds")

headers <- read_tsv("data/SDM/points/Reptiles_0000117-250212154643175.csv", n_max = 2)
reptile_pnts <- read_tsv_arrow("data/SDM/points/Reptiles_0000117-250212154643175.csv",
                               col_select = c("decimalLongitude", "decimalLatitude"))
reptile_pnts <- st_as_sf(reptile_pnts, coords = c("decimalLongitude", "decimalLatitude"),
                         crs = 4326)

#polyg <- squamates2$geometry[1]
sample_points <- function(polyg_num, polyg, spec_name, n = 10000, min_n = 4, max_n = 500,
                          prop_biased = seq(0.7, 1.0, length.out = 5),
                          folder = "output/squamate_samples_w_bias") {
  
  fp <- file.path(folder, paste0(str_pad(polyg_num, 5, pad = "0"), ".rds"))
  
  if(file.exists(fp)) {
    return(fp)
  }
  
  ## placemarker file
  write_rds(st_sfc(), fp)
  
  if(st_is_valid(polyg)) {
    
    projection <- find_equal_area_projection(polyg)
    polyg_local <- st_transform(polyg, crs = projection)
    
    bias_samp <- reptile_pnts |>
      st_join(st_sf(polyg) |> mutate(poly = TRUE)) |>
      filter(poly)
    
    bias_samp <- st_transform(bias_samp, crs = projection)
    
    coords <- bias_samp |> st_coordinates()
    
    xrange <- range(coords[ , 1])
    yrange <- range(coords[ , 2])
    xexpand <- (xrange[2] - xrange[1]) * 0.01
    yexpand <- (yrange[2] - yrange[1]) * 0.01
    
    xrange <- xrange + c(-xexpand, xexpand)
    yrange <- yrange + c(-yexpand, yexpand)
    
    dens <- MASS::kde2d(coords[ , 1], coords[ , 2], n = 100,
                        lims = c(xrange, yrange))
    dens_rast <- rast(dens)
    
    get_samps <- function(prop) {
      n_biased <- ceiling(n * prop)
      n_unbiased <- n - n_biased
      if(n_biased > 0) {
        samp <- st_sample(polyg_local, n_biased)  
        probs <- extract(dens_rast, vect(samp)) |>
          drop_na()
        ID_samp <- sample(probs$ID, n_biased, replace = TRUE, prob = probs$lyr.1) 
        bias_samp <- st_as_sf(samp[ID_samp])
      } else {
        bias_samp <- NULL
      }
      if(n_unbiased > 0) {
        unbias_samp <- st_as_sf(st_sample(polyg_local, n_unbiased))
      } else {
        unbias_samp <- NULL
      }
      samps <- rbind(bias_samp, unbias_samp) |>
        slice_sample(prop = 1) |>
        mutate(spec = spec_name)
      samps <- st_transform(samps, crs = 4326)
      datasets <- list()
      i <- 0
      while(nrow(samps) > 0) {
        i <- i + 1
        n_grab <- ceiling(runif(1, min_n - 1, max_n))
        datasets[[i]] <- samps |>
          slice(1:n_grab)
        samps <- samps |>
          slice(-1:-n_grab)
      }
      datasets
    }
    all_datasets <- map(prop_biased, get_samps) |>
      list_flatten()
    
    write_rds(all_datasets, fp)
    
    
  } else {
    fp <- ""
  }
  
  fp
  
}

future::plan(future::multisession())
future::plan(future::sequential())

samples <- future_map(seq_along(squamates2$geometry),
                      possibly(~ sample_points(.x, squamates2$geometry[.x], 
                                               squamates2$Binomial[.x]),
                               ""),
                      .progress = TRUE)

write_rds(squamates2 |>
            as_tibble() |>
            select(-geometry) |>
            mutate(output = samples), 
          "output/squamates_samples_w_bias_output.rds")

