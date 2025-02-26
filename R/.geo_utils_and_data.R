library(sf)
library(h3)
library(rnaturalearth)
library(cli)
library(fasterize)

# world <- ne_countries(scale = 10)
# ecoregions <- read_rds("data/maps/ecoregions_valid.rds")
# 
# # ecoreg_raster <- fasterize(ecoregions, raster(ecoregions, res = 0.1),
# #                            field = "OBJECTID")
# # raster::writeRaster(ecoreg_raster, "data/maps/ecoreg_raster.grd", "raster", overwrite = TRUE)
# 
# ecoreg_raster <- raster("data/maps/ecoreg_raster.grd")

localize <- function(geo_truth, geo_preds, n_hex = 1000, use_preds = FALSE,
                     train_points = NULL, test_points = NULL) {
  
  pred_sf <- geo_preds |>
    as.data.frame() |>
    st_as_sf(coords = c("V1", "V2"), crs = 4326)
  truth_sf <- geo_truth |>
    st_transform(crs = 4326) |>
    st_sample(size = 1000) |>
    st_as_sf()
  if(use_preds) {
    samp_coords <- rbind(pred_sf,
                         truth_sf) |>
      mutate(point = 1)  
  } else {
    samp_coords <- truth_sf |>
      mutate(point = 1)
  }
  
  cli_progress_message("Finding appropriate ecoregions background...")
  ecoreg_nums <- unique(na.omit(raster::extract(ecoreg_raster, truth_sf)))
  ecoreg <- ecoregions |>
    filter(OBJECTID %in% ecoreg_nums)
    # st_join(samp_coords) |>
    # filter(!is.na(point)) |>
    # distinct(ECO_NAME, .keep_all = TRUE)
  
  cli_progress_message("Finding appropriate hex grid...")
  all_ecoreg <- st_union(ecoreg)
  for(i in 1:10) {
    hexes <- unique(polyfill(all_ecoreg, i))
    if(length(hexes) > (2 * n_hex)) {
      i <- i - 1
      break
    }
    if(length(hexes) > n_hex) {
      break
    }
  }
  #hex_res <- map(1:8, ~ unique(polyfill(all_ecoreg, .x)), .progress = TRUE)
  #n_hexes <- map_int(hex_res, length)

  hexes <- unique(polyfill(all_ecoreg, i))
  hex_sf <- h3_to_geo_boundary_sf(hexes)
  
  cli_progress_message("Counting predicted points in each hexagon...")
  hex_counts_pred <- pred_sf |>
    mutate(point = 1) |>
    st_join(hex_sf) |>
    group_by(h3_index) |>
    summarise(count_pred = sum(point)) |>
    ungroup() |>
    mutate(prop_pred = count_pred / max(count_pred))
  
  hex_polys <- hex_sf |>
    left_join(hex_counts_pred |> as_tibble() |> select(-geometry)) |>
    mutate(prop_pred = count_pred / sum(count_pred, na.rm = TRUE))
  
  if(!is.null(train_points)) {
    cli_progress_message("Counting training points in each hexagon...")
    hex_counts_train <- train_points |>
      mutate(point = 1) |>
      st_join(hex_sf) |>
      group_by(h3_index) |>
      summarise(count_train = sum(point)) |>
      ungroup() |>
      mutate(prop_train = count_train / max(count_train))
    
    hex_polys <- hex_polys |>
      left_join(hex_counts_train |> as_tibble() |> select(-geometry)) |>
      mutate(prop_train = count_train / sum(count_train, na.rm = TRUE))
  }
  
  if(!is.null(test_points)) {
    cli_progress_message("Counting test points in each hexagon...")
    hex_counts_test <- test_points |>
      mutate(point = 1) |>
      st_join(hex_sf) |>
      group_by(h3_index) |>
      summarise(count_test = sum(point)) |>
      ungroup() |>
      mutate(prop_test = count_test / max(count_test))
    
    hex_polys <- hex_polys |>
      left_join(hex_counts_test |> as_tibble() |> select(-geometry)) |>
      mutate(prop_test = count_test / sum(count_test, na.rm = TRUE))
  }
  
  cli_progress_message("Intersecting hex grid with ecoregions...")
  hex_polys <- hex_polys |>
    st_intersection(ecoreg)
  
  cli_progress_message("Finding best geographic projection...")
  proj <- find_equal_area_projection(ecoreg)
  
  cli_progress_message("Reprojecting data...")
  ecoreg <- ecoreg |>
    st_transform(proj)
  countries <- world |>
    st_transform(proj)
  hex_polys <- hex_polys |>
    st_transform(proj)
  
  if(!is.null(train_points)) {
    train_points <- train_points |>
      st_transform(proj)
  }
  
  if(!is.null(test_points)) {
    test_points <- test_points |>
      st_transform(proj)
  }
  
  cli_progress_message("Putting it all together...")
  
  extent <- st_bbox(ecoreg)
  
  list(pred_sf = pred_sf |> st_transform(proj), 
       truth_sf = truth_sf |> st_transform(proj), 
       hex_polys = hex_polys,
       ecoregions = ecoreg, 
       countries = countries,
       extent = extent,
       train_points = train_points,
       test_points = test_points)
  
}

fix_problem_variables <- function(env_orig, problem_vars, env_to_samp, na_replace) {
  sampl <- env_to_samp[sample.int(nrow(env_to_samp), nrow(env_orig), replace = TRUE), problem_vars] 
  for(i in seq_along(problem_vars)) {
    env_orig[ , problem_vars[i]] <- sampl[ , i]
    env_orig[ , problem_vars[i]][is.na(env_orig[ , problem_vars[i]])] <- na_replace[problem_vars[i]]
  }
  env_orig
}
