## Load your packages, e.g. library(targets).
source("./packages.R")
conflict_prefer("select", "dplyr")
conflict_prefer("filter", "dplyr")

## Load your R files
lapply(list.files("./R", full.names = TRUE), source)

tar_option_set(
  controller = crew_controller_local(workers = 8,
                                    seconds_idle = 10),
  iteration = "list",
  memory = "transient",
  error = "null",
  retrieval = "worker",
  storage = "worker"
)

## tar_plan supports drake-style targets and also tar_target()
tar_plan(

# target = function_to_make(arg), ## drake style

# tar_target(target2, function_to_make2(arg)) ## targets style
  
  tar_target(nicheflow_vers, 0.1),
  
  tar_target(nichencoder_species_latent_dim, c(64L)),
  
  tar_target(nichencoder_env_latent_dim, c(16L)),
  
  tar_target(land, ne_download(scale = 10, type = 'land', category = 'physical')),
  
  tar_target(reptile_headers, read_tsv("data/SDM/points/Reptiles_0000117-250212154643175.csv", n_max = 2)),
  
  tar_target(reptile_bias_pnts, read_tsv_arrow("data/SDM/points/Reptiles_0000117-250212154643175.csv",
                                               col_select = c("decimalLongitude", "decimalLatitude")) |>
               st_as_sf(coords = c("decimalLongitude", "decimalLatitude"), crs = 4326),
             memory = "persistent"),
  
  tar_target(poly_folders, list(reptiles = "data/SDM/maps/GARD1.7")),
  
  tar_target(chelsa_bioclim_rast_files, list.files("data/SDM/env/CHELSA-BIOMCLIM+/1981-2010/bio", full.names = TRUE) |>
               str_subset(fixed("kg"), negate = TRUE) |>
               str_subset(fixed("lgd"), negate = TRUE) |>
               str_subset(fixed("fgd"), negate = TRUE) |>
               str_subset(fixed("fcf"), negate = TRUE) |>
               str_subset(fixed("swe"), negate = TRUE)),
  
  tar_target(species_polygons, load_and_filter_polygons(poly_folders),
             pattern = map(poly_folders)),
  
  tar_target(spec_polys, species_polygons |>
               list_rbind() |>
               select(species = binomial, geometry) |>
               rowwise() |>
               group_split()),
  
  tar_target(ground_truth_samples, sample_ground_truth_points(spec_polys,
                                                              chelsa_bioclim_rast_files),
             pattern = map(spec_polys)),
  
  tar_target(bias_samples, sample_bias_pnts(spec_polys, reptile_bias_pnts),
             pattern = map(spec_polys)),
  
  tar_target(nichencoder_script, "train_nichencoder.R")

)
