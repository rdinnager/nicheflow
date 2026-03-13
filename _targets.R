## Load your packages, e.g. library(targets).
source("./packages.R")
conflict_prefer("select", "dplyr")
conflict_prefer("filter", "dplyr")
conflict_prefer("fixed", "stringr")

## Per-process memory limit. Commented out for now -- heavy raster targets
## run on main (deployment = "main") and need the full SLURM allocation.
## Re-enable if workers need to be constrained.
# unix::rlimit_as(5e10, 5e10)

## Load your R files
lapply(list.files("./R", pattern = "^functions_.*\\.R$", full.names = TRUE), source)
source("R/utils.R")

tar_option_set(
  packages = c("torch", "dagnn", "zeallot", "arrow", "purrr", "dplyr",
               "tibble", "stringr", "sf", "terra"),
  controller = crew_controller_group(
    crew_controller_local(name = "default", workers = 16,
                          seconds_idle = 10,
                          garbage_collection = TRUE,
                          options_metrics = crew_options_metrics(
                            path = "logs/autometrics.log",
                            seconds_interval = 10)),
    crew_controller_local(name = "gpu0", workers = 1, seconds_idle = 120,
                          options_local = crew_options_local(
                            log_directory = "logs/crew_env_vae")),
    crew_controller_local(name = "gpu1", workers = 1, seconds_idle = 120,
                          options_local = crew_options_local(
                            log_directory = "logs/crew_geode"))
  ),
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
  
  # Shared inclusion regex: bio1-19 and npp only
  tar_target(
    bioclim_include_pattern,
    "CHELSA_(bio[0-9]+|npp)_",
    deployment = "main"
  ),

  tar_target(chelsa_bioclim_rast_files,
    list.files("data/SDM/env/CHELSA-BIOMCLIM+/1981-2010/bio", full.names = TRUE) |>
      str_subset(bioclim_include_pattern),
    deployment = "main"),

  # ===========================================================================
  # JADE Sampling Targets
  # ===========================================================================
  # JADE (Jacobian-Adjusted Density Estimation) corrects for geographic-to-

  # environmental distortion when sampling species occurrence points.
  # See notes/jade_sampling_procedure.md for algorithm details.

  # Paths to polygon folders for all three taxa
  tar_target(
    taxa_poly_folders,
    list(
      reptiles = "data/SDM/maps/GARD1.7",
      amphibians = "data/SDM/maps/AMPHIBIANS",
      mammals = "data/SDM/maps/MAMMALS_TERRESTRIAL_ONLY"
    ),
    deployment = "main"
  ),

  # Bioclim files for JADE (bio1-19 + npp only)
  tar_target(
    jade_bioclim_files,
    list.files("data/SDM/env/CHELSA-BIOMCLIM+/1981-2010/bio",
               full.names = TRUE) |>
      str_subset(bioclim_include_pattern),
    deployment = "main"
  ),

  # Compute per-variable SDs for standardization (~10 min)
  tar_target(
    bioclim_sds,
    compute_bioclim_sds(jade_bioclim_files, sample_size = 1e6),
    deployment = "main"
  ),

  # Compute A, B, C Gram matrix rasters (EXPENSIVE: ~2-3 hours for global)
  tar_target(
    abc_rasters,
    compute_abc_rasters(
      jade_bioclim_files,
      bioclim_sds,
      output_dir = "output/jacobian"
    ),
    format = "file",
    deployment = "main"
  ),

  # Compute final Jacobian raster from A, B, C
  tar_target(
    jacobian_raster_path,
    compute_jacobian_from_abc(
      abc_rasters,
      output_path = "output/jacobian/chelsa_jacobian.tif",
      lat_correct = TRUE
    ),
    format = "file",
    deployment = "main"
  ),

  # Load each taxa separately to avoid OOM (each shapefile is 1-2 GB)
  tar_target(
    reptile_polygons,
    load_single_taxa_polygons(taxa_poly_folders$reptiles, "reptiles"),
    deployment = "main"
  ),

  tar_target(
    amphibian_polygons,
    load_single_taxa_polygons(taxa_poly_folders$amphibians, "amphibians"),
    deployment = "main"
  ),

  tar_target(
    mammal_polygons,
    load_single_taxa_polygons(taxa_poly_folders$mammals, "mammals"),
    deployment = "main"
  ),

  # Combine all taxa polygons
  tar_target(
    all_taxa_polygons,
    dplyr::bind_rows(reptile_polygons, amphibian_polygons, mammal_polygons),
    deployment = "main"
  ),

  # Split into individual species for dynamic branching
  tar_target(
    jade_spec_polys,
    all_taxa_polygons |>
      rowwise() |>
      group_split(),
    iteration = "list",
    deployment = "main"
  ),

  # JADE sampling per species (dynamically branched)
  # Each branch runs in a callr subprocess for crash isolation
  tar_target(
    jade_samples,
    jade_sample_safe(
      jade_spec_polys,
      jacobian_raster_path,
      jade_bioclim_files,
      n_target = 5000
    ),
    pattern = map(jade_spec_polys),
    iteration = "list"
  ),

  # Polygon-level metadata (presence/origin/seasonal/area)
  # Reloads raw shapefiles to recover IUCN columns dropped during loading
  tar_target(
    jade_polygon_metadata,
    load_jade_polygon_metadata(taxa_poly_folders),
    deployment = "main"
  ),

  # Clean and merge: remove extinct/introduced, env-volume-weighted resampling
  tar_target(
    jade_samples_clean,
    clean_and_merge_jade_samples(jade_samples, jade_polygon_metadata,
                                 n_target = 5000),
    deployment = "main"
  ),

  # Export cleaned JADE samples to parquet
  tar_target(
    jade_samples_parquet,
    {
      path <- "data/processed/jade_samples_clean.parquet"
      dir.create(dirname(path), showWarnings = FALSE, recursive = TRUE)
      arrow::write_parquet(jade_samples_clean, path)
      path
    },
    format = "file",
    deployment = "main"
  ),

  # Extract samples from introduced (non-native) ranges
  tar_target(
    jade_samples_introduced,
    extract_introduced_jade_samples(jade_samples, jade_polygon_metadata),
    deployment = "main"
  ),

  tar_target(
    jade_samples_introduced_parquet,
    {
      path <- "data/processed/jade_samples_introduced.parquet"
      dir.create(dirname(path), showWarnings = FALSE, recursive = TRUE)
      arrow::write_parquet(jade_samples_introduced, path)
      path
    },
    format = "file",
    deployment = "main"
  ),

  # ===========================================================================
  # End JADE Sampling Targets (20-variable)
  # ===========================================================================

  # ===========================================================================
  # 31-Variable JADE Sampling + Train/Val/Test Splitting
  # ===========================================================================
  # Recomputes Jacobian with all 31 CHELSA-BIOCLIM+ variables, then generates

  # JADE-corrected samples for species-level model training/evaluation.
  # Produces GenAISDM-style splits: zero-shot, few-shot, and within-species.
  # ===========================================================================

  # Resolve 31 raster file paths from the canonical variable metadata CSV
  tar_target(
    jade_bioclim_files_31,
    {
      bio_dir <- "data/SDM/env/CHELSA-BIOMCLIM+/1981-2010/bio"
      purrr::map_chr(chelsa_var_meta$chelsa_name, \(cn) {
        f <- list.files(bio_dir, pattern = paste0("CHELSA_", cn, "_"),
                        full.names = TRUE)
        if (length(f) == 0) stop("No raster file found for: ", cn)
        f[1]
      })
    },
    deployment = "main"
  ),

  # Per-variable SDs for Jacobian standardization (~15 min)
  tar_target(
    bioclim_sds_31,
    compute_bioclim_sds(jade_bioclim_files_31, sample_size = 1e6),
    deployment = "main"
  ),

  # NA fill map: raster layer name -> fill value (from chelsa_var_meta)
  # BIOCLIM+ variables (gsl, gsp, gdd*, ngd*, fcf, scd, swe) have NAs in
  # warm regions. These must be filled BEFORE Jacobian computation or else
  # NAs propagate through focal() derivatives to the entire Gram matrix.
  tar_target(
    jade_na_fill_map_31,
    {
      layer_names <- basename(jade_bioclim_files_31) |>
        tools::file_path_sans_ext()
      fills <- purrr::map_dbl(layer_names, \(ln) {
        idx <- purrr::detect_index(chelsa_var_meta$chelsa_name, \(cn) {
          grepl(paste0("CHELSA_", cn, "_"), ln)
        })
        if (idx == 0 || is.na(chelsa_var_meta$na_fill[idx])) return(NA_real_)
        as.numeric(chelsa_var_meta$na_fill[idx])
      })
      names(fills) <- layer_names
      fills[!is.na(fills)]
    },
    deployment = "main"
  ),

  # A, B, C Gram matrix rasters for 31 variables (EXPENSIVE: ~4 hours)
  # Uses compute_abc_rasters_filled() to fill BIOCLIM+ NAs before focal()
  tar_target(
    abc_rasters_31,
    compute_abc_rasters_filled(
      jade_bioclim_files_31,
      bioclim_sds_31,
      output_dir = "output/jacobian_31",
      na_fill_values = jade_na_fill_map_31
    ),
    format = "file",
    deployment = "main"
  ),

  # Final Jacobian raster from 31-variable Gram matrices
  tar_target(
    jacobian_raster_path_31,
    compute_jacobian_from_abc(
      abc_rasters_31,
      output_path = "output/jacobian_31/chelsa_jacobian_31.tif",
      lat_correct = TRUE
    ),
    format = "file",
    deployment = "main"
  ),

  # JADE sampling per species using 31-var Jacobian (n_target=1100)
  # Reuses jade_spec_polys from 20-var pipeline
  tar_target(
    jade_samples_31,
    jade_sample_safe(
      jade_spec_polys,
      jacobian_raster_path_31,
      jade_bioclim_files_31,
      n_target = 1100,
      log_file = "logs/jade_sampling_31.log"
    ),
    pattern = map(jade_spec_polys),
    iteration = "list"
  ),

  # Clean and merge: remove extinct/introduced, env-volume resample to 1000
  tar_target(
    jade_samples_clean_31,
    clean_and_merge_jade_samples(jade_samples_31, jade_polygon_metadata,
                                 n_target = 1000),
    deployment = "main"
  ),

  # Apply NA fill values from variable metadata (0 for snow/frost/growing-season)
  tar_target(
    jade_samples_clean_31_filled,
    apply_na_fill_to_jade_samples(jade_samples_clean_31, chelsa_var_meta,
                                   jade_bioclim_files_31),
    deployment = "main"
  ),

  # Count samples per species, filter to those with >= 100
  tar_target(
    jade_species_counts_31,
    compute_species_counts(jade_samples_clean_31_filled, min_samples = 100),
    deployment = "main"
  ),

  # Assign species roles: 80% train, 10% zeroshot, 10% fewshot
  tar_target(
    jade_split_assignments_31,
    assign_species_splits(
      jade_species_counts_31,
      zeroshot_frac = 0.10,
      fewshot_frac = 0.10,
      seed = 42
    ),
    deployment = "main"
  ),

  # Multi-stage splitting: 1000 cap, 80/20 within-species train/val, 8 fewshot samples
  tar_target(
    jade_train_val_test_31,
    create_jade_splits(
      jade_samples_clean_31_filled,
      jade_split_assignments_31,
      max_samples_per_species = 1000,
      within_train_frac = 0.75,
      within_val_frac = 0.15,
      fewshot_n = 8,
      seed = 42
    ),
    deployment = "main"
  ),

  # Export splits to parquet
  tar_target(
    jade_train_parquet_31,
    {
      path <- "data/processed/jade_31_train.parquet"
      dir.create(dirname(path), showWarnings = FALSE, recursive = TRUE)
      arrow::write_parquet(jade_train_val_test_31$train, path)
      path
    },
    format = "file",
    deployment = "main"
  ),

  tar_target(
    jade_val_parquet_31,
    {
      path <- "data/processed/jade_31_val.parquet"
      arrow::write_parquet(jade_train_val_test_31$val, path)
      path
    },
    format = "file",
    deployment = "main"
  ),

  tar_target(
    jade_test_parquet_31,
    {
      path <- "data/processed/jade_31_test.parquet"
      arrow::write_parquet(jade_train_val_test_31$test, path)
      path
    },
    format = "file",
    deployment = "main"
  ),

  # Summary statistics for verification
  tar_target(
    jade_split_summary_31,
    summarize_jade_splits(jade_train_val_test_31, jade_split_assignments_31),
    deployment = "main"
  ),

  # ===========================================================================
  # End 31-Variable JADE Sampling
  # ===========================================================================

  # ===========================================================================
  # CHELSA Global Environmental Tensor Pipeline
  # ===========================================================================
  # Loads all 31 CHELSA-BIOCLIM+ variables into torch float32 tensors for:
  #   - VAE training:   (308M, 31) standardized env variables
  #   - GeODE training: (308M, 31) standardized env + (308M, 2) standardized XY
  #
  # Memory: ~38 GB for env tensor, ~2.5 GB for xy tensor (float32).
  # All targets run on main process (deployment = "main") due to memory.
  # Disk: ~81 GB total for all tensor files in output/chelsa_tensors/.
  #
  # Future: VAE and GeODE training will use separate crew controllers
  # for parallel GPU execution. Add crew_controller_group with per-GPU
  # controllers when training scripts are ready.
  # ===========================================================================

  tar_target(
    chelsa_bio_dir,
    "data/SDM/env/CHELSA-BIOMCLIM+/1981-2010/bio",
    deployment = "main"
  ),

  tar_target(
    chelsa_var_meta,
    read_csv(
      "data/SDM/env/CHELSA-BIOMCLIM+/chelsa_bioclim_numeric_projected_variables.csv",
      show_col_types = FALSE
    ),
    deployment = "main"
  ),

  # Land mask: rasterize NE 10m land polygons to CHELSA grid
  # Returns list(land_idx, n_land, n_total) — ~1.2 GB for land_idx
  tar_target(
    land_mask_idx,
    build_land_mask(land, chelsa_bio_dir),
    deployment = "main"
  ),

  # Load all 31 env variables into float32 tensor (~38 GB, ~18 min)
  # Saved as per-column .pt files (safetensors crashes on tensors > 2 GB)
  tar_target(
    env_raw_tensor_dir,
    load_chelsa_tensor(
      chelsa_bio_dir, chelsa_var_meta,
      land_mask_idx$land_idx,
      "output/chelsa_tensors/env_raw"
    ),
    deployment = "main"
  ),

  # Per-variable mean and SD for z-score standardization
  tar_target(
    env_mean_sd,
    compute_standardization(env_raw_tensor_dir),
    deployment = "main"
  ),

  # Standardized env tensor (z-scored in-place to save memory)
  # This is the VAE training input
  tar_target(
    env_standardized_dir,
    standardize_tensor(
      env_raw_tensor_dir, env_mean_sd,
      "output/chelsa_tensors/env_standardized"
    ),
    deployment = "main"
  ),

  # XY coordinates (lon/lat) for all land cells (~2.5 GB)
  # Saved as per-column .pt files
  tar_target(
    xy_coords_dir,
    extract_xy_coords(
      chelsa_bio_dir, land_mask_idx$land_idx,
      "output/chelsa_tensors/xy_coords"
    ),
    deployment = "main"
  ),

  # Per-coordinate mean and SD for XY standardization
  tar_target(
    xy_mean_sd,
    compute_standardization(xy_coords_dir),
    deployment = "main"
  ),

  # Standardized XY tensor (lon/lat z-scored)
  # Combined with env_standardized for GeODE training
  tar_target(
    xy_standardized_dir,
    standardize_tensor(
      xy_coords_dir, xy_mean_sd,
      "output/chelsa_tensors/xy_standardized"
    ),
    deployment = "main"
  ),

  # ===========================================================================
  # End CHELSA Tensor Pipeline
  # ===========================================================================

  # ===========================================================================
  # Model Training Targets
  # ===========================================================================

  # Save xy_mean_sd to disk for GeODE script (needs un-standardization stats)
  tar_target(
    xy_mean_sd_file,
    {
      path <- "output/chelsa_tensors/xy_mean_sd.rds"
      saveRDS(xy_mean_sd, path)
      path
    },
    format = "file",
    deployment = "main"
  ),

  # Loggamma initial values to sweep for VAE
  tar_target(loggamma_values, c(-2, -3, -4), deployment = "main"),

  # Environmental VAE training — 3 branches over loggamma_init (cuda:0, sequential)
  # ~107s/epoch, 500 epochs each ≈ 14.9h per branch, ~45h total
  tar_target(
    env_vae_training,
    run_script("train_env_vae.R", params = list(
      env_dir = env_standardized_dir,
      device = "cuda:0",
      latent_dim = nichencoder_env_latent_dim,
      num_epochs = 500L,
      batch_size = 1000000L,
      lr = 0.0025,
      loggamma_init = loggamma_values,
      val_every = 25L,
      checkpoint_every = 25L,
      checkpoint_dir = paste0(
        "output/checkpoints/env_vae/gamma_", loggamma_values
      )
    )),
    pattern = map(loggamma_values),
    resources = tar_resources(
      crew = tar_resources_crew(controller = "gpu0")
    ),
    deployment = "worker"
  ),

  # GeODE rectified flow training (cuda:1)
  # Resumes from epoch 600 checkpoint, runs ~400 more epochs to epoch 1000
  tar_target(
    geode_training,
    run_script("train_geode.R", params = list(
      env_dir = env_standardized_dir,
      xy_dir = xy_standardized_dir,
      xy_raw_dir = xy_coords_dir,
      xy_mean_sd_file = xy_mean_sd_file,
      device = "cuda:1",
      num_epochs = 1000L,
      batch_size = 300000L,
      val_every = 10L,
      checkpoint_every = 25L,
      n_random_ecoregions = 3L,
      checkpoint_dir = "output/checkpoints/geode"
    )),
    resources = tar_resources(
      crew = tar_resources_crew(controller = "gpu1")
    ),
    deployment = "worker"
  ),

  # ===========================================================================
  # End Model Training Targets
  # ===========================================================================

  # ===========================================================================
  # NichEncoder Pipeline: VAE Encoding + Rectified Flow
  # ===========================================================================
  # Encodes JADE samples through the trained EnvVAE into latent space,
  # then trains a conditional rectified flow (NichEncoder) that models
  # species-specific environmental distributions.

  # Chosen VAE checkpoint (gamma_-2, best ELBO: -59.87)
  tar_target(
    env_vae_checkpoint,
    "output/checkpoints/env_vae/gamma_-2/epoch_0500_model.pt",
    format = "file",
    deployment = "main"
  ),

  # Determine active latent dimensions from the original VAE training data.
  # Encodes a random 1M-row subset of the global env tensor, computes
  # mean(exp(logvar)) per dim; active = those < 0.5
  tar_target(
    vae_active_dims,
    detect_active_dims(
      env_vae_checkpoint, env_standardized_dir,
      latent_dim = nichencoder_env_latent_dim,
      device = "cuda:0", batch_size = 500000L,
      n_samples = 1000000L, threshold = 0.5
    ),
    resources = tar_resources(
      crew = tar_resources_crew(controller = "gpu0")
    ),
    deployment = "worker"
  ),

  # Encode JADE train split through VAE -> latent codes (active dims only)
  tar_target(
    jade_encoded_train,
    encode_jade_through_vae(
      jade_train_parquet_31, env_vae_checkpoint, env_mean_sd,
      chelsa_var_meta, active_dims = vae_active_dims,
      latent_dim = nichencoder_env_latent_dim,
      device = "cuda:0", batch_size = 500000L
    ),
    resources = tar_resources(
      crew = tar_resources_crew(controller = "gpu0")
    ),
    deployment = "worker"
  ),

  # Encode val split
  tar_target(
    jade_encoded_val,
    encode_jade_through_vae(
      jade_val_parquet_31, env_vae_checkpoint, env_mean_sd,
      chelsa_var_meta, active_dims = vae_active_dims,
      latent_dim = nichencoder_env_latent_dim,
      device = "cuda:0", batch_size = 500000L
    ),
    resources = tar_resources(
      crew = tar_resources_crew(controller = "gpu0")
    ),
    deployment = "worker"
  ),

  # Encode test split
  tar_target(
    jade_encoded_test,
    encode_jade_through_vae(
      jade_test_parquet_31, env_vae_checkpoint, env_mean_sd,
      chelsa_var_meta, active_dims = vae_active_dims,
      latent_dim = nichencoder_env_latent_dim,
      device = "cuda:0", batch_size = 500000L
    ),
    resources = tar_resources(
      crew = tar_resources_crew(controller = "gpu0")
    ),
    deployment = "worker"
  ),

  # Species ID mapping (train species only, for nn_embedding)
  tar_target(
    nichencoder_species_map,
    build_species_id_map(jade_encoded_train),
    deployment = "main"
  ),

  # Export encoded data as parquet files for training script
  tar_target(
    jade_encoded_train_parquet,
    {
      path <- "data/processed/jade_encoded_train.parquet"
      arrow::write_parquet(jade_encoded_train, path)
      path
    },
    format = "file",
    deployment = "main"
  ),

  tar_target(
    jade_encoded_val_parquet,
    {
      path <- "data/processed/jade_encoded_val.parquet"
      arrow::write_parquet(jade_encoded_val, path)
      path
    },
    format = "file",
    deployment = "main"
  ),

  # Save config files for training script (species map + active dims)
  tar_target(
    nichencoder_config_files,
    {
      dir.create("output/nichencoder_config", recursive = TRUE,
                 showWarnings = FALSE)
      saveRDS(vae_active_dims,
              "output/nichencoder_config/active_dims.rds")
      saveRDS(nichencoder_species_map,
              "output/nichencoder_config/species_map.rds")
      c("output/nichencoder_config/active_dims.rds",
        "output/nichencoder_config/species_map.rds")
    },
    format = "file",
    deployment = "main"
  ),

  # NichEncoder rectified flow training (cuda:0)
  # Two-cycle LR: 1500 epochs at lr=0.001, then optimizer reset + 1500 at lr=0.0001
  tar_target(
    nichencoder_training,
    run_script("train_nichencoder.R", params = list(
      encoded_train_parquet = jade_encoded_train_parquet,
      encoded_val_parquet = jade_encoded_val_parquet,
      species_map_file = "output/nichencoder_config/species_map.rds",
      device = "cuda:0",
      spec_embed_dim = 64L,
      breadths = c(512L, 256L, 128L),
      num_epochs = 3000L,
      batch_size = 500000L,
      lr = 0.001,
      loss_type = "pseudo_huber",
      n_cycles = 2L,
      cycle_2_lr_factor = 0.1,
      cycle_1_fraction = 0.5,
      ode_steps = 500L,
      n_metric_species = 100L,
      checkpoint_every = 25L,
      val_every = 25L,
      clear_checkpoints = TRUE,
      checkpoint_dir = "output/checkpoints/nichencoder"
    )),
    resources = tar_resources(
      crew = tar_resources_crew(controller = "gpu0")
    ),
    deployment = "worker"
  ),

  # ===========================================================================
  # End NichEncoder Pipeline
  # ===========================================================================

  # ===========================================================================
  # Geo-Encoder Pipeline: Occurrence Points -> Niche Embeddings
  # ===========================================================================
  # Trains a transformer to predict NichEncoder species embeddings from
  # variable-length sets of (lon,lat) coordinates. Training data is generated
  # by corrupting known species ranges with GBIF-style bias + spatial blocks.
  #
  # End-to-end inference:
  #   (lon,lat) -> Geo-Encoder -> embedding -> NichEncoder -> env distribution
  # ===========================================================================

  # GBIF bias points for amphibians and mammals (reptiles already loaded above)
  tar_target(
    amphibian_bias_pnts,
    read_tsv_arrow(
      "data/SDM/points/Amphibians_0000114-250212154643175.csv",
      col_select = c("decimalLongitude", "decimalLatitude")
    ) |>
      filter(!is.na(decimalLongitude), !is.na(decimalLatitude)) |>
      st_as_sf(coords = c("decimalLongitude", "decimalLatitude"), crs = 4326),
    memory = "persistent",
    deployment = "main"
  ),

  tar_target(
    mammal_bias_pnts,
    read_tsv_arrow(
      "data/SDM/points/Mammals_0000097-250212154643175.csv",
      col_select = c("decimalLongitude", "decimalLatitude")
    ) |>
      filter(!is.na(decimalLongitude), !is.na(decimalLatitude)) |>
      st_as_sf(coords = c("decimalLongitude", "decimalLatitude"), crs = 4326),
    memory = "persistent",
    deployment = "main"
  ),

  # Combine bias points by taxon for lookup in corruption pipeline
  tar_target(
    all_bias_pnts,
    list(
      reptiles = reptile_bias_pnts,
      amphibians = amphibian_bias_pnts,
      mammals = mammal_bias_pnts
    ),
    deployment = "main"
  ),

  # Pre-extract bias point coordinates as plain matrices (avoids each
  # worker copying the full sf objects — prevents OOM with many workers)
  tar_target(
    all_bias_coords,
    lapply(all_bias_pnts, sf::st_coordinates),
    deployment = "main"
  ),

  # Per-species coordinate list from merged JADE samples for corruption
  tar_target(
    jade_species_coord_list,
    jade_samples_clean_31 |>
      select(species, X, Y, taxon) |>
      group_by(species, taxon) |>
      group_split(),
    deployment = "main"
  ),

  # Generate corrupted coordinate sets per species (dynamic branching)
  # Resamples from existing JADE coordinates with GBIF bias weighting,
  # spatial block removal, and Beta(1.3,20)-distributed chunk sizes
  # matching empirical GBIF per-species record counts
  tar_target(
    geoencoder_corrupted_coords,
    {
      sp_coords <- jade_species_coord_list
      taxon <- sp_coords$taxon[1]
      bias_xy <- all_bias_coords[[taxon]]
      sample_corrupted_coords_safe(
        sp_coords, bias_xy,
        n_versions = 20L,
        max_n = 1000L, no_block_frac = 0.3,
        max_blocks = 3L, min_remaining_points = 10L
      )
    },
    pattern = map(jade_species_coord_list),
    iteration = "list"
  ),

  # Extract trained NichEncoder species embeddings
  tar_target(
    nichencoder_species_embeddings,
    extract_nichencoder_embeddings(
      checkpoint_dir = "output/checkpoints/nichencoder",
      species_map_file = "output/nichencoder_config/species_map.rds",
      coord_dim = 6L, n_species = 18121L,
      spec_embed_dim = 64L,
      breadths = c(512L, 256L, 128L)
    ),
    deployment = "main"
  ),

  # Assemble geo-encoder dataset: normalize, species-level split, export parquet
  tar_target(
    geoencoder_dataset,
    build_geoencoder_dataset(
      geoencoder_corrupted_coords,
      nichencoder_species_embeddings,
      xy_mean_sd
    ),
    deployment = "main"
  ),

  # Prepare downstream validation data for zero-shot species
  tar_target(
    geoencoder_val_downstream,
    build_geoencoder_val_downstream(
      geoencoder_corrupted_coords,
      jade_test_parquet_31,
      jade_split_assignments_31,
      xy_mean_sd,
      chelsa_var_meta,
      env_mean_sd
    ),
    deployment = "main"
  ),

  # Geo-encoder transformer training (GPU)
  tar_target(
    geoencoder_training,
    run_script("train_geoencoder.R", params = list(
      train_parquet = geoencoder_dataset$train_parquet,
      val_parquet = geoencoder_dataset$val_parquet,
      embeddings_file = geoencoder_dataset$embeddings_file,
      species_map_file = "output/nichencoder_config/species_map.rds",
      device = "cuda:0",
      val_device = "cuda:1",
      embed_dim = 256L, n_blocks = 8L, num_heads = 8L,
      output_dim = 64L, max_points = 500L,
      batch_size = 256L, num_epochs = 500L, lr = 0.0005,
      loss_type = "mse_cosine", cosine_weight = 0.5,
      checkpoint_every = 25L, val_every = 10L,
      clear_checkpoints = TRUE,
      checkpoint_dir = "output/checkpoints/geoencoder",
      downstream_coords_parquet = geoencoder_val_downstream$downstream_coords_parquet,
      downstream_env_parquet = geoencoder_val_downstream$downstream_env_parquet,
      nichencoder_checkpoint_dir = "output/checkpoints/nichencoder",
      nichencoder_coord_dim = 6L,
      nichencoder_n_species = 18121L,
      nichencoder_spec_embed_dim = 64L,
      nichencoder_breadths = c(512L, 256L, 128L),
      vae_checkpoint = "output/checkpoints/env_vae/gamma_-2/epoch_0500_model.pt",
      vae_input_dim = 31L,
      vae_latent_dim = 16L,
      vae_active_dims = c(7L, 9L, 11L, 13L, 15L, 16L)
    )),
    resources = tar_resources(
      crew = tar_resources_crew(controller = "gpu0")
    ),
    deployment = "worker"
  ),

  # ===========================================================================
  # End Geo-Encoder Pipeline
  # ===========================================================================

  # ===========================================================================
  # Evaluation Pipeline
  # ===========================================================================
  # Evaluates NicheFlow against MaxEnt and balanced RF baselines using
  # AUC, TSS, SWD, and geographic EMD metrics on a stratified subset of
  # training species.

  # -- Configuration --
  tar_target(eval_n_background, 5000L, deployment = "main"),
  tar_target(eval_species_batch_size, 50L, deployment = "main"),
  tar_target(eval_n_total_species, 1200L, deployment = "main"),

  # GeODE checkpoint (VAE and NichEncoder checkpoints already exist above)
  tar_target(
    geode_checkpoint,
    "output/checkpoints/geode/epoch_1000_model.pt",
    format = "file",
    deployment = "main"
  ),

  # -- Load evaluation data --
  tar_target(
    eval_test_data,
    arrow::read_parquet(jade_test_parquet_31),
    deployment = "main"
  ),

  tar_target(
    eval_train_data,
    arrow::read_parquet(jade_train_parquet_31),
    deployment = "main"
  ),

  # Species metadata (taxon, range_size_km2, median_lat, etc.)
  tar_target(
    eval_species_metadata,
    build_species_metadata(
      bind_rows(eval_train_data, eval_test_data),
      nichencoder_species_map
    ),
    deployment = "main"
  ),

  # Stratified species sampling: ~600 species
  # Stratify by taxon × latitude_band × range_size, undersample large ranges
  tar_target(
    eval_selected_species,
    {
      meta <- eval_species_metadata |>
        filter(n_points >= 70,        # rough min for 50 train + 20 test
               shot_type != "zeroshot") # no embeddings for zeroshot species

      # Create strata: taxon × shot_type × latitude_band × range_size
      meta <- meta |>
        mutate(
          lat_band = cut(abs(median_lat),
                         breaks = c(0, 15, 35, 90),
                         labels = c("low", "mid", "high")),
          size_band = cut(range_size_km2,
                          breaks = quantile(range_size_km2,
                                            c(0, 1/3, 2/3, 1)),
                          labels = c("small", "mid", "large"),
                          include.lowest = TRUE),
          stratum = paste(taxon, shot_type, lat_band, size_band, sep = "_")
        )

      set.seed(42)
      target_n <- eval_n_total_species
      n_strata <- n_distinct(meta$stratum)
      per_stratum <- ceiling(target_n / n_strata)

      # Fewer large-range species (slower to evaluate)
      sampled <- meta |>
        mutate(max_n = ifelse(size_band == "large",
                              ceiling(per_stratum * 0.5),
                              per_stratum)) |>
        group_by(stratum) |>
        slice_sample(n = min(n(), first(max_n))) |>
        ungroup()

      # Trim to target
      if (nrow(sampled) > target_n) {
        sampled <- slice_sample(sampled, n = target_n)
      }

      message("Selected ", nrow(sampled), " species for evaluation across ",
              n_strata, " strata")

      sampled$species
    },
    deployment = "main"
  ),

  # Split into batches for dynamic branching
  tar_target(
    eval_species_batches,
    split(eval_selected_species,
          ceiling(seq_along(eval_selected_species) / eval_species_batch_size)),
    deployment = "main"
  ),

  # ---------------------------------------------------------------------------
  # SWD Evaluation (GPU1 — runs in parallel with AUC NicheFlow scoring on GPU0)
  # Only needs VAE + NichEncoder, no GeODE
  # ---------------------------------------------------------------------------

  tar_target(
    eval_swd_results,
    evaluate_swd_batch(
      eval_species_batches, eval_test_data,
      nichencoder_species_map, env_vae_checkpoint,
      "output/checkpoints/nichencoder", vae_active_dims,
      env_mean_sd, device = "cuda:1"
    ),
    pattern = map(eval_species_batches),
    iteration = "list",
    resources = tar_resources(
      crew = tar_resources_crew(controller = "gpu1")
    ),
    deployment = "worker"
  ),

  tar_target(
    eval_swd_combined,
    {
      swd <- eval_swd_results |> list_rbind()
      left_join(swd, eval_species_metadata, by = "species")
    },
    deployment = "main"
  ),

  tar_target(
    eval_swd_parquet,
    {
      path <- "output/evaluation/swd_results.parquet"
      dir.create(dirname(path), showWarnings = FALSE, recursive = TRUE)
      arrow::write_parquet(eval_swd_combined, path)
      path
    },
    format = "file",
    deployment = "main"
  ),

  # ---------------------------------------------------------------------------
  # AUC/TSS Evaluation: Phase A — Data Preparation (CPU, 16 workers)
  # ---------------------------------------------------------------------------

  tar_target(
    eval_batch_data,
    prepare_eval_batch(
      eval_species_batches, eval_train_data, eval_test_data,
      chelsa_var_meta, env_mean_sd, chelsa_bio_dir,
      bioclim_include_pattern, eval_n_background
    ),
    pattern = map(eval_species_batches),
    iteration = "list"
  ),

  # Phase B: NicheFlow GPU scoring per batch
  tar_target(
    eval_nicheflow_scores,
    score_nicheflow_batch(
      eval_batch_data, nichencoder_species_map,
      env_vae_checkpoint, "output/checkpoints/nichencoder",
      geode_checkpoint, xy_mean_sd, vae_active_dims,
      device = "cuda:0"
    ),
    pattern = map(eval_batch_data),
    iteration = "list",
    resources = tar_resources(
      crew = tar_resources_crew(controller = "gpu0")
    ),
    deployment = "worker"
  ),

  # Phase C: MaxEnt + RF per species (CPU, 16 workers)
  tar_target(
    eval_species_data,
    flatten_batch_to_species(eval_batch_data),
    deployment = "main"
  ),

  tar_target(
    eval_maxent_scores,
    run_maxnet_species(eval_species_data),
    pattern = map(eval_species_data),
    iteration = "list"
  ),

  tar_target(
    eval_rf_scores,
    run_rf_species(eval_species_data),
    pattern = map(eval_species_data),
    iteration = "list"
  ),

  # Phase D: Combine + compute metrics
  tar_target(
    eval_auc_results,
    combine_and_compute_metrics(
      eval_nicheflow_scores, eval_maxent_scores, eval_rf_scores,
      eval_species_metadata
    ),
    deployment = "main"
  ),

  tar_target(
    eval_auc_parquet,
    {
      path <- "output/evaluation/auc_results.parquet"
      dir.create(dirname(path), showWarnings = FALSE, recursive = TRUE)
      arrow::write_parquet(eval_auc_results, path)
      path
    },
    format = "file",
    deployment = "main"
  ),

  # ---------------------------------------------------------------------------
  # Geographic EMD Evaluation (GPU + CPU, batched)
  # ---------------------------------------------------------------------------

  tar_target(
    eval_emd_results,
    evaluate_emd_batch(
      eval_species_batches, eval_test_data, eval_train_data,
      nichencoder_species_map, chelsa_var_meta, env_mean_sd,
      chelsa_bio_dir, bioclim_include_pattern,
      env_vae_checkpoint, "output/checkpoints/nichencoder",
      geode_checkpoint, xy_mean_sd, vae_active_dims,
      device = "cuda:0"
    ),
    pattern = map(eval_species_batches),
    iteration = "list",
    resources = tar_resources(
      crew = tar_resources_crew(controller = "gpu0")
    ),
    deployment = "worker"
  ),

  tar_target(
    eval_emd_combined,
    {
      emd <- eval_emd_results |> list_rbind()
      left_join(emd, eval_species_metadata, by = "species")
    },
    deployment = "main"
  ),

  tar_target(
    eval_emd_parquet,
    {
      path <- "output/evaluation/emd_results.parquet"
      dir.create(dirname(path), showWarnings = FALSE, recursive = TRUE)
      arrow::write_parquet(eval_emd_combined, path)
      path
    },
    format = "file",
    deployment = "main"
  ),

  # ===========================================================================
  # Disdat Benchmark Data Preparation
  # ===========================================================================

  tar_target(
    disdat_regions,
    c("AWT", "NSW", "CAN", "NZ", "SA", "SWI"),
    deployment = "main"
  ),

  tar_target(
    disdat_region_data,
    prepare_disdat_region(
      disdat_regions, chelsa_var_meta, env_mean_sd,
      chelsa_bio_dir, bioclim_include_pattern
    ),
    pattern = map(disdat_regions),
    iteration = "list"
  ),

  tar_target(
    disdat_jade_resampled,
    jade_resample_disdat(
      disdat_region_data$train_data,
      jacobian_raster_path_31
    ),
    pattern = map(disdat_region_data),
    iteration = "list"
  ),

  tar_target(
    disdat_parquets,
    export_disdat_parquet(
      disdat_region_data, disdat_jade_resampled
    ),
    pattern = map(disdat_region_data, disdat_jade_resampled),
    format = "file",
    iteration = "list"
  )

  # ===========================================================================
  # End Evaluation Pipeline
  # ===========================================================================

)