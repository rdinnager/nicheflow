#' .. content for \description{} (no empty lines) ..
#'
#' .. content for \details{} ..
#'
#' @title
#' @param species_polygons
#' @return
#' @author rdinnage
#' @export
# spec_polys <- spec_polys[[1]]
sample_ground_truth_points <- function(spec_polys, chelsa_bioclim_rast_files, n = 1000) {

  chelsa <- rast(chelsa_bioclim_rast_files)
  samp <- st_sf(st_sample(spec_polys, n))
  
  env_df <- terra::extract(chelsa, samp, ID = FALSE) 
  coord_df <- st_coordinates(samp)
  env_df <- bind_cols(coord_df, env_df) |>
    mutate(species = spec_polys$species[1])
  
  env_df
  

}
