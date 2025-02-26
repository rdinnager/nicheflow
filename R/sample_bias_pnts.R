#' .. content for \description{} (no empty lines) ..
#'
#' .. content for \details{} ..
#'
#' @title
#' @param spec_names
#' @param spec_geoms
#' @return
#' @author rdinnage
#' @export
# spec_polys <- spec_polys[[1]]
sample_bias_pnts <- function(spec_polys, reptile_bias_pnts, n = 10000, min_n = 2, max_n = 500,
                             prop_biased = c(rep(0.05, 3), rep(0.95, 3))) {

  projection <- find_equal_area_projection(spec_polys)
  polyg_local <- st_transform(spec_polys, crs = projection)
  
  bias_samp <- reptile_bias_pnts |>
    st_join(spec_polys |> mutate(poly = TRUE)) |>
    filter(poly)
  
  if(nrow(bias_samp) > 10) {
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
  } else {
    prop_biased <- rep(0, 3)
  }
  
  spec_name <- spec_polys$species[1]
  
  get_samps <- function(prop) {
    n_biased <- ceiling(n * prop)
    n_unbiased <- n - n_biased
    if(n_biased > 0) {
      samp <- st_sample(polyg_local, n_biased)  
      probs <- terra::extract(dens_rast, vect(samp)) |>
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
      slice_sample(prop = 1) |> mutate(biased = prop)
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
  all_datasets <- map(prop_biased, possibly(get_samps)) |>
    list_flatten() 
  all_df <- all_datasets |>
    map(~ .x |>
          st_coordinates()) 
  all_df <- tibble(species = spec_name, coords = all_df,
                   biased = map_dbl(all_datasets, ~ .x$biased[1]))
  
  all_df
}
