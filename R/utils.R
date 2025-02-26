find_equal_area_projection <- function(sf_object) {
  # Ensure the dataset is in WGS84 geographic coordinates
  sf_object <- st_transform(sf_object, 4326)
  
  # Calculate the bounding box and centroid
  bbox <- st_bbox(sf_object)
  centroid <- st_coordinates(st_centroid(st_union(sf_object)))
  lon0 <- centroid[1]
  lat0 <- centroid[2]
  
  # Calculate extent in degrees
  delta_lon <- bbox$xmax - bbox$xmin
  delta_lat <- bbox$ymax - bbox$ymin
  
  # Determine if the dataset has a global extent
  is_global_extent <- delta_lon >= 180 || delta_lat >= 90
  
  if (is_global_extent) {
    # Use Equal Earth projection for global datasets
    proj_str <- "+proj=eqearth +units=m +ellps=WGS84"
  } else if (delta_lat > delta_lon) {
    # Predominantly north-south extent
    # Use Lambert Azimuthal Equal-Area projection
    proj_str <- sprintf(
      "+proj=laea +lat_0=%f +lon_0=%f +units=m +ellps=WGS84",
      lat0, lon0
    )
  } else {
    # Predominantly east-west extent
    # Use Albers Equal-Area Conic projection
    # Set standard parallels based on dataset's latitude
    std_parallel_1 <- lat0 - delta_lat / 6
    std_parallel_2 <- lat0 + delta_lat / 6
    proj_str <- sprintf(
      "+proj=aea +lat_1=%f +lat_2=%f +lat_0=%f +lon_0=%f +units=m +ellps=WGS84",
      std_parallel_1, std_parallel_2, lat0, lon0
    )
  }
  
  # Reproject the data
  #sf_projected <- st_transform(sf_object, crs = proj_str)
  
  return(proj_str)
}

#bias_file <- "output/squamate_samples_w_bias/00001.rds"
load_bias_pnts <- function(bias_file) {
  dat <- read_rds(bias_file)
  dat_df <- tibble(spec = map_chr(dat, ~ .x$spec[1]),
                   coords = map(dat, ~ st_coordinates(.x) |>
                                  as.data.frame() |>
                                  mutate(X = X / 90, Y = Y / 180) |>
                                  as.matrix()))
}
