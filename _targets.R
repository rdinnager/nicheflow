## Load your packages, e.g. library(targets).
source("./packages.R")
conflict_prefer("select", "dplyr")
conflict_prefer("filter", "dplyr")

## A potentially very important trick here
## This prevents any R process from using more than 16GB RAM,
## which keeps the entire pipeline from being OOM killed.
## Instead an error will throw (e.g. cannot allocate vector of size XX), 
## and the pipeline continues!
unix::rlimit_as(1.8e10, 1.8e10)

## Load your R files
lapply(list.files("./R", full.names = TRUE), source)

tar_option_set(
  controller = crew_controller_local(workers = 8,
                                    seconds_idle = 10,
                                    garbage_collection = TRUE,
                                    retry_tasks = FALSE,
                                    options_metrics = crew_options_metrics(path = "logs/autometrics.log", seconds_interval = 10)),
  iteration = "list",
  memory = "transient",
  error = "null",
  retrieval = "worker",
  storage = "worker",
  #debug = "bias_samples_74e6869d32b8e7a8",
  garbage_collection = TRUE
)

## tar_plan supports drake-style targets and also tar_target()
tar_plan(

# target = function_to_make(arg), ## drake style

# tar_target(target2, function_to_make2(arg)) ## targets style
  
  tar_target(nicheflow_vers, 0.1,
             deployment = "main"),
  
  tar_target(nichencoder_species_latent_dim, c(64L),
             deployment = "main"),
  
  tar_target(nichencoder_env_latent_dim, c(16L),
             deployment = "main"),
  
  tar_target(land, ne_download(scale = 10, type = 'land', category = 'physical'),
             deployment = "main"),
  
  tar_target(reptile_headers, read_tsv("data/SDM/points/Reptiles_0000117-250212154643175.csv", n_max = 2),
             deployment = "main"),
  
  tar_target(reptile_bias_pnts, read_tsv_arrow("data/SDM/points/Reptiles_0000117-250212154643175.csv",
                                               col_select = c("decimalLongitude", "decimalLatitude")) |>
               st_as_sf(coords = c("decimalLongitude", "decimalLatitude"), crs = 4326),
             memory = "persistent",
             deployment = "main"),
  
  tar_target(poly_folders, list(reptiles = "data/SDM/maps/GARD1.7"),
             deployment = "main"),
  
  tar_target(chelsa_bioclim_rast_files, list.files("data/SDM/env/CHELSA-BIOMCLIM+/1981-2010/bio", full.names = TRUE) |>
               str_subset(fixed("kg"), negate = TRUE) |>
               str_subset(fixed("lgd"), negate = TRUE) |>
               str_subset(fixed("fgd"), negate = TRUE) |>
               str_subset(fixed("fcf"), negate = TRUE) |>
               str_subset(fixed("swe"), negate = TRUE),
             deployment = "main"),
  
  tar_target(species_polygons, load_and_filter_polygons(poly_folders),
             pattern = map(poly_folders),
             deployment = "main"),
  
  tar_target(spec_polys, species_polygons |>
               list_rbind() |>
               select(species = binomial, geometry) |>
               rowwise() |>
               group_split(),
             deployment = "main"),
  
  tar_target(ground_truth_samples, sample_ground_truth_points(spec_polys,
                                                              chelsa_bioclim_rast_files),
             pattern = map(spec_polys)),
  
  tar_target(bias_samples, sample_bias_pnts(spec_polys, reptile_bias_pnts),
             pattern = map(spec_polys)),
  
  tar_target(all_ground_truth_samples, list_rbind(ground_truth_samples),
             deployment = "main"),
  
  tar_target(ground_truth_sample_split, make_ground_truth_splits(all_ground_truth_samples,
                                                                 split_ratio = c(0.8, 0.1)),
             deployment = "main"),
  
  ## use recipes to do all data preprocessing
  tar_target(nichencoder_input_preprocess_recipe, recipe(head(ground_truth_sample_split$train),
                                                   vars = colnames(ground_truth_sample_split$train),
                                                   roles = c(rep("predictor", ncol(ground_truth_sample_split$train) -1), "ID")) |>
               step_normalize(all_predictors()) |>
               #step_YeoJohnson(all_predictors()) |>
               step_indicate_na(all_predictors()) |>
               prep(ground_truth_sample_split$train),
             deployment = "main"),
  
  tar_target(nichencoder_train, bake(nichencoder_input_preprocess_recipe,
                                                  new_data = NULL),
             deployment = "main"),
  
  tar_target(nichencoder_val, bake(nichencoder_input_preprocess_recipe,
                                     new_data = ground_truth_sample_split$val),
             deployment = "main"),
  
  tar_target(nichencoder_test, bake(nichencoder_input_preprocess_recipe,
                                     new_data = ground_truth_sample_split$test),
             deployment = "main"),
  
  tar_target(nichencoder_train_csv, write_csv(nichencoder_train, "data/nichencoder_train.csv"),
             deployment = "main"),
  
  tar_target(nichencoder_val_csv, write_csv(nichencoder_val, "data/nichencoder_val.csv"),
             deployment = "main"),
  
  tar_target(nichencoder_test_csv, write_csv(nichencoder_test, "data/nichencoder_test.csv"),
             deployment = "main"),
  
  tar_target(nichencoder_script, "train_nichencoder.R")

)