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
    crew_controller_local(name = "default", workers = 12,
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

  # cue("never"): completed, outdated from depend-hash drift
  tar_target(chelsa_bioclim_rast_files,
    list.files("data/SDM/env/CHELSA-BIOMCLIM+/1981-2010/bio", full.names = TRUE) |>
      str_subset(bioclim_include_pattern),
    cue = tar_cue("never"),
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

  # cue("never"): completed, outdated from depend-hash drift
  tar_target(
    jade_split_summary_31,
    summarize_jade_splits(jade_train_val_test_31, jade_split_assignments_31),
    cue = tar_cue("never"),
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
  # cue("never"): completed, outdated only from depend-hash drift
  tar_target(
    vae_active_dims,
    detect_active_dims(
      env_vae_checkpoint, env_standardized_dir,
      latent_dim = nichencoder_env_latent_dim,
      device = "cuda:0", batch_size = 500000L,
      n_samples = 1000000L, threshold = 0.5
    ),
    cue = tar_cue("never"),
    resources = tar_resources(
      crew = tar_resources_crew(controller = "gpu0")
    ),
    deployment = "worker"
  ),

  # cue("never"): completed, outdated only from cascade
  tar_target(
    jade_encoded_train,
    encode_jade_through_vae(
      jade_train_parquet_31, env_vae_checkpoint, env_mean_sd,
      chelsa_var_meta, active_dims = vae_active_dims,
      latent_dim = nichencoder_env_latent_dim,
      device = "cuda:0", batch_size = 500000L
    ),
    cue = tar_cue("never"),
    resources = tar_resources(
      crew = tar_resources_crew(controller = "gpu0")
    ),
    deployment = "worker"
  ),

  # cue("never"): completed, outdated only from cascade
  tar_target(
    jade_encoded_val,
    encode_jade_through_vae(
      jade_val_parquet_31, env_vae_checkpoint, env_mean_sd,
      chelsa_var_meta, active_dims = vae_active_dims,
      latent_dim = nichencoder_env_latent_dim,
      device = "cuda:0", batch_size = 500000L
    ),
    cue = tar_cue("never"),
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

  # cue("never"): completed, outdated only from cascade
  tar_target(
    jade_encoded_train_parquet,
    {
      path <- "data/processed/jade_encoded_train.parquet"
      arrow::write_parquet(jade_encoded_train, path)
      path
    },
    cue = tar_cue("never"),
    format = "file",
    deployment = "main"
  ),

  # cue("never"): completed, outdated only from cascade
  tar_target(
    jade_encoded_val_parquet,
    {
      path <- "data/processed/jade_encoded_val.parquet"
      arrow::write_parquet(jade_encoded_val, path)
      path
    },
    cue = tar_cue("never"),
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
  # cue = tar_cue("never") prevents re-running: training is complete (epoch 3000)
  # and the "outdated" status is a false positive from depend-hash drift
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
    cue = tar_cue("never"),
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

  # cue("never"): completed, outdated from depend-hash drift
  tar_target(
    nichencoder_species_embeddings,
    extract_nichencoder_embeddings(
      checkpoint_dir = "output/checkpoints/nichencoder",
      species_map_file = "output/nichencoder_config/species_map.rds",
      coord_dim = 6L, n_species = 18121L,
      spec_embed_dim = 64L,
      breadths = c(512L, 256L, 128L)
    ),
    cue = tar_cue("never"),
    deployment = "main"
  ),

  # cue("never"): completed, outdated from cascade
  tar_target(
    geoencoder_dataset,
    build_geoencoder_dataset(
      geoencoder_corrupted_coords,
      nichencoder_species_embeddings,
      xy_mean_sd
    ),
    cue = tar_cue("never"),
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
  tar_target(eval_n_test_bg, 500L, deployment = "main"),
  tar_target(eval_n_test_per_species, 100L, deployment = "main"),
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
        filter(n_points >= 2,          # must have some data
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
      n_strata <- length(unique(meta$stratum))
      per_stratum <- ceiling(target_n / n_strata)

      # Fewer large-range species (slower to evaluate)
      meta <- meta |>
        mutate(max_n = ifelse(size_band == "large",
                              ceiling(per_stratum * 0.5),
                              per_stratum))

      # Sample per stratum, capping at max_n
      sampled <- meta |>
        group_by(stratum) |>
        mutate(.rand = runif(n())) |>
        arrange(.rand, .by_group = TRUE) |>
        filter(row_number() <= max_n[1]) |>
        ungroup() |>
        select(-.rand)

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
      eval_n_background, eval_n_test_bg, eval_n_test_per_species
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
      vae_active_dims,
      device = "cuda:0"
    ),
    pattern = map(eval_batch_data),
    iteration = "list",
    deployment = "main"
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

  # ---------------------------------------------------------------------------
  # 20-variable MaxEnt with within-species held-out splits
  #
  # Per-species long-form datasets carry both split strategies as columns
  # (split_random, split_spatial). Backgrounds are reused across both
  # strategies; the 500-pt test-bg subset is flagged via is_test_bg.
  # 4 sample sizes x 2 split strategies = 8 MaxEnt result targets.
  # ---------------------------------------------------------------------------

  # Column-subset cached 31-var batches to 20 vars (bio1-19 + npp)
  tar_target(
    eval_batch_data_20,
    subset_batch_to_vars(eval_batch_data, bioclim_include_pattern),
    pattern = map(eval_batch_data),
    iteration = "list"
  ),

  # Flatten to one entry per species
  tar_target(
    eval_species_data_20,
    flatten_batch_to_species(eval_batch_data_20),
    deployment = "main"
  ),

  # Uniform area-weighted polygon sampling for MaxEnt evaluation.
  # JADE-sampled presence points are biased by the 31-var Jacobian, which is
  # the wrong sampling design for an unbiased MaxEnt benchmark. Here we draw
  # n_per_species uniform random points per species in an equal-area CRS
  # (Mollweide), then extract CHELSA at those points.
  tar_target(
    eval_uniform_presences_20_batches,
    sample_uniform_polygon_presences(
      eval_species_batches, all_taxa_polygons, jade_polygon_metadata,
      chelsa_var_meta, env_mean_sd, chelsa_bio_dir, bioclim_include_pattern,
      n_per_species = 1100L
    ),
    pattern = map(eval_species_batches),
    iteration = "list"
  ),

  # Flatten batch list -> single named list keyed by species. Plain do.call(c)
  # preserves the species-name keys; purrr::list_flatten would prefix them
  # with the batch hash and break the eval_uniform_presences_20[[sp]] lookup.
  tar_target(
    eval_uniform_presences_20,
    do.call(c, unname(eval_uniform_presences_20_batches)),
    deployment = "main"
  ),

  # Build long-form per-species dataset with both split-assignment columns,
  # then append a jacobian_20 column extracted from the 20-var Jacobian
  # raster at every point (presence + background).
  tar_target(
    eval_species_dataset_20,
    {
      jac <- terra::rast(jacobian_raster_path)
      purrr::map(eval_species_data_20, \(sp) {
        build_species_dataset(sp, eval_uniform_presences_20[[sp$species]]) |>
          add_jacobian_column(jac)
      }) |> purrr::compact()
    },
    deployment = "main",
    iteration = "list"
  ),

  # Combined parquet output for inspection / external analysis
  tar_target(
    eval_species_dataset_20_parquet,
    {
      path <- "output/evaluation/maxent_20_species_datasets.parquet"
      dir.create(dirname(path), showWarnings = FALSE, recursive = TRUE)
      arrow::write_parquet(
        purrr::list_rbind(eval_species_dataset_20),
        path
      )
      path
    },
    format = "file",
    deployment = "main"
  ),

  # 4 sample sizes x 2 split strategies = 8 MaxEnt targets via tar_map
  tarchetypes::tar_map(
    values = tibble::tibble(n_subsample = c(16L, 32L, 64L, 128L)),
    names  = "n_subsample",
    tar_target(
      eval_maxent_scores_20_random,
      run_maxnet_on_dataset(
        eval_species_dataset_20, "split_random", n_subsample
      ),
      pattern = map(eval_species_dataset_20),
      iteration = "list"
    ),
    tar_target(
      eval_maxent_scores_20_spatial,
      run_maxnet_on_class_symmetric_dataset(
        eval_species_dataset_20, "split_spatial", n_subsample
      ),
      pattern = map(eval_species_dataset_20),
      iteration = "list"
    )
  ),

  # ===========================================================================
  # Uniform-polygon training/test datasets (20-var) + 2500-species MaxEnt eval
  #
  # Replaces JADE-corrected sampling with uniform-area polygon sampling
  # (equal-area Mollweide). Train parquet: presence + 20 env cols, no
  # Jacobian. Test parquet: same plus a jacobian_20 column for inference-time
  # density correction. MaxEnt eval is run on a stratified 2500-species
  # subset of the test parquet, in parallel to the existing 1003-species path.
  # ===========================================================================

  tar_target(
    eligible_uniform_species,
    build_eligible_uniform_species(jade_polygon_metadata),
    deployment = "main"
  ),

  tar_target(
    uniform_species_metadata,
    build_uniform_species_metadata(jade_polygon_metadata, all_taxa_polygons),
    deployment = "main"
  ),

  tar_target(uniform_test_n_total, 2500L, deployment = "main"),

  tar_target(
    uniform_test_species,
    select_stratified_species(
      uniform_species_metadata, eligible_uniform_species,
      target_n = uniform_test_n_total, seed = 42L
    ),
    deployment = "main"
  ),

  # Batches over ALL eligible species for sampling
  tar_target(
    uniform_species_batches,
    split_into_batches(eligible_uniform_species,
                       batch_size = eval_species_batch_size),
    deployment = "main"
  ),

  # Per-batch uniform polygon sampling (~30-60 min total on 12 workers)
  tar_target(
    uniform_samples_20_batches,
    sample_uniform_polygon_presences(
      uniform_species_batches, all_taxa_polygons, jade_polygon_metadata,
      chelsa_var_meta, env_mean_sd, chelsa_bio_dir, bioclim_include_pattern,
      n_per_species = 1100L
    ),
    pattern = map(uniform_species_batches),
    iteration = "list"
  ),

  # Flatten -> single named list keyed by species
  tar_target(
    uniform_samples_20,
    do.call(c, unname(uniform_samples_20_batches)),
    deployment = "main"
  ),

  # Train parquet: non-test species, no Jacobian
  tar_target(
    uniform20_train_parquet,
    write_uniform_samples_parquet(
      uniform_samples_20,
      species_keep = setdiff(eligible_uniform_species, uniform_test_species),
      path = "data/processed/uniform20_train.parquet",
      species_metadata = uniform_species_metadata,
      add_jacobian = FALSE
    ),
    format = "file",
    deployment = "main"
  ),

  # Test parquet: 2500 stratified species + jacobian_20 column
  tar_target(
    uniform20_test_parquet,
    write_uniform_samples_parquet(
      uniform_samples_20,
      species_keep = uniform_test_species,
      path = "data/processed/uniform20_test.parquet",
      species_metadata = uniform_species_metadata,
      add_jacobian = TRUE,
      jacobian_raster_path = jacobian_raster_path
    ),
    format = "file",
    deployment = "main"
  ),

  # --- MaxEnt eval on the 2500 test species ---

  tar_target(
    uniform_eval_species_batches,
    split_into_batches(uniform_test_species,
                       batch_size = eval_species_batch_size),
    deployment = "main"
  ),

  # Generates 5000 backgrounds + 500 test_bg per species, pairs with uniform
  # presences. Same shape as eval_batch_data so downstream targets work.
  tar_target(
    uniform_eval_batch_data,
    prepare_uniform_eval_batch(
      uniform_eval_species_batches, uniform_samples_20,
      chelsa_var_meta, env_mean_sd, chelsa_bio_dir, bioclim_include_pattern,
      eval_n_background, eval_n_test_bg
    ),
    pattern = map(uniform_eval_species_batches),
    iteration = "list"
  ),

  tar_target(
    uniform_eval_species_data,
    flatten_batch_to_species(uniform_eval_batch_data),
    deployment = "main"
  ),

  # Long-form per-species datasets with split_random + split_spatial columns,
  # plus jacobian_20 at every point. Per-batch dynamic branching: each branch
  # = one batch of species (one small RDS file), so downstream MaxEnt patterns
  # don't need to load the whole 1 GB list to dispatch each species.
  tar_target(
    uniform_eval_dataset_20,
    {
      jac <- terra::rast(jacobian_raster_path)
      purrr::map(uniform_eval_batch_data, \(sp) {
        build_species_dataset(sp) |> add_jacobian_column(jac)
      }) |> purrr::compact()
    },
    pattern = map(uniform_eval_batch_data),
    iteration = "list"
  ),

  # Consolidated long-form parquet for downstream analysis. Sourced from
  # uniform_eval_dataset_20_envblock so it has split_spatial + split_envblock
  # (with bg-test already subsampled to 500/sp and unsampled test-block bg
  # marked as NA). split_random is dropped; the random pipeline lives only
  # in the targets cache.
  tar_target(
    uniform_eval_dataset_20_parquet,
    {
      path <- "output/evaluation/uniform2500_species_datasets_spatial_envblock.parquet"
      dir.create(dirname(path), showWarnings = FALSE, recursive = TRUE)
      df <- purrr::list_rbind(purrr::list_flatten(uniform_eval_dataset_20_envblock))
      df <- add_random_split_classes(df)
      df <- add_in_range_polygon(df, all_taxa_polygons, jade_polygon_metadata)
      arrow::write_parquet(df, path)
      path
    },
    format = "file",
    deployment = "main"
  ),

  # 10 MaxEnt targets (random + class-symmetric spatial). Per-branch = one
  # batch of ~50 species (each MaxEnt fit is fast; the dominant cost is data
  # shipping, so batching reduces total dispatches).
  tarchetypes::tar_map(
    values = tibble::tibble(n_subsample = c(8L, 16L, 32L, 64L, 128L)),
    names  = "n_subsample",
    tar_target(
      uniform2500_maxent_20_random,
      run_maxnet_batch(
        uniform_eval_dataset_20, "split_random", n_subsample
      ),
      pattern = map(uniform_eval_dataset_20),
      iteration = "list"
    ),
    tar_target(
      uniform2500_maxent_20_spatial,
      run_maxnet_class_symmetric_batch(
        uniform_eval_dataset_20, "split_spatial", n_subsample
      ),
      pattern = map(uniform_eval_dataset_20),
      iteration = "list"
    )
  ),

  # Env-block in-filling test split (Mahalanobis ellipsoidal holes).
  # Derived from uniform_eval_dataset_20 (per-batch); appends split_envblock
  # + diagnostics. Per-branch = one batch of species datasets.
  tar_target(
    uniform_eval_dataset_20_envblock,
    purrr::compact(purrr::map(uniform_eval_dataset_20,
                              \(sp) add_envblock_split(sp, n_holes = 30L))),
    pattern = map(uniform_eval_dataset_20),
    iteration = "list"
  ),

  # 5 MaxEnt targets on the env-block split. Per-branch = one batch.
  tarchetypes::tar_map(
    values = tibble::tibble(n_subsample = c(8L, 16L, 32L, 64L, 128L)),
    names  = "n_subsample",
    tar_target(
      uniform2500_maxent_20_envblock,
      run_maxnet_class_symmetric_batch(
        uniform_eval_dataset_20_envblock, "split_envblock", n_subsample
      ),
      pattern = map(uniform_eval_dataset_20_envblock),
      iteration = "list"
    )
  ),

  # Combined per-species AUC across all 15 patterns
  tar_target(
    uniform2500_maxent_auc,
    combine_maxent_auc(
      uniform2500_maxent_20_random_8,   uniform2500_maxent_20_random_16,
      uniform2500_maxent_20_random_32,  uniform2500_maxent_20_random_64,
      uniform2500_maxent_20_random_128,
      uniform2500_maxent_20_spatial_8,  uniform2500_maxent_20_spatial_16,
      uniform2500_maxent_20_spatial_32, uniform2500_maxent_20_spatial_64,
      uniform2500_maxent_20_spatial_128,
      uniform2500_maxent_20_envblock_8,   uniform2500_maxent_20_envblock_16,
      uniform2500_maxent_20_envblock_32,  uniform2500_maxent_20_envblock_64,
      uniform2500_maxent_20_envblock_128
    ),
    deployment = "main"
  ),

  tar_target(
    uniform2500_maxent_auc_parquet,
    {
      path <- "output/evaluation/uniform2500_maxent_auc.parquet"
      dir.create(dirname(path), showWarnings = FALSE, recursive = TRUE)
      arrow::write_parquet(uniform2500_maxent_auc, path)
      path
    },
    format = "file",
    deployment = "main"
  ),

  # ===========================================================================
  # Alternative bg-extent test datasets:
  #   tightbg: half the noise (radius_multiplier = 0.125)
  #   widebg:  bg sampled uniformly within the union of all RESOLVE-2017
  #            ecoregions touched by any presence
  # Each produces a parquet identical in schema to the existing
  # uniform2500_species_datasets_spatial_envblock.parquet.
  # ===========================================================================

  tar_target(
    ecoregions_sf,
    readRDS("/blue/rdinnage.fiu/rdinnage.fiu/Data/SDM/maps/ecoregions_valid.rds"),
    deployment = "main"
  ),

  # ---- TIGHTBG variant ------------------------------------------------------
  tar_target(
    uniform_eval_batch_data_tightbg,
    prepare_uniform_eval_batch(
      uniform_eval_species_batches, uniform_samples_20,
      chelsa_var_meta, env_mean_sd, chelsa_bio_dir, bioclim_include_pattern,
      eval_n_background, eval_n_test_bg,
      radius_multiplier = 0.05
    ),
    pattern = map(uniform_eval_species_batches),
    iteration = "list"
  ),

  tar_target(
    uniform_eval_dataset_20_tightbg,
    {
      jac <- terra::rast(jacobian_raster_path)
      purrr::map(uniform_eval_batch_data_tightbg, \(sp) {
        build_species_dataset(sp) |> add_jacobian_column(jac)
      }) |> purrr::compact()
    },
    pattern = map(uniform_eval_batch_data_tightbg),
    iteration = "list"
  ),

  tar_target(
    uniform_eval_dataset_20_envblock_tightbg,
    purrr::compact(purrr::map(uniform_eval_dataset_20_tightbg,
                              \(sp) add_envblock_split(sp, n_holes = 30L))),
    pattern = map(uniform_eval_dataset_20_tightbg),
    iteration = "list"
  ),

  tar_target(
    uniform_eval_dataset_20_tightbg_parquet,
    {
      path <- "output/evaluation/uniform2500_species_datasets_spatial_envblock_tightbg.parquet"
      dir.create(dirname(path), showWarnings = FALSE, recursive = TRUE)
      df <- purrr::list_rbind(purrr::list_flatten(uniform_eval_dataset_20_envblock_tightbg))
      df <- add_random_split_classes(df)
      df <- add_in_range_polygon(df, all_taxa_polygons, jade_polygon_metadata)
      arrow::write_parquet(df, path)
      path
    },
    format = "file",
    deployment = "main"
  ),

  # ---- WIDEBG variant (radius_multiplier = 0.5, 2x default expansion) -------
  tar_target(
    uniform_eval_batch_data_widebg,
    prepare_uniform_eval_batch(
      uniform_eval_species_batches, uniform_samples_20,
      chelsa_var_meta, env_mean_sd, chelsa_bio_dir, bioclim_include_pattern,
      eval_n_background, eval_n_test_bg,
      radius_multiplier = 0.5
    ),
    pattern = map(uniform_eval_species_batches),
    iteration = "list"
  ),

  tar_target(
    uniform_eval_dataset_20_widebg,
    {
      jac <- terra::rast(jacobian_raster_path)
      purrr::map(uniform_eval_batch_data_widebg, \(sp) {
        build_species_dataset(sp) |> add_jacobian_column(jac)
      }) |> purrr::compact()
    },
    pattern = map(uniform_eval_batch_data_widebg),
    iteration = "list"
  ),

  tar_target(
    uniform_eval_dataset_20_envblock_widebg,
    purrr::compact(purrr::map(uniform_eval_dataset_20_widebg,
                              \(sp) add_envblock_split(sp, n_holes = 30L))),
    pattern = map(uniform_eval_dataset_20_widebg),
    iteration = "list"
  ),

  tar_target(
    uniform_eval_dataset_20_widebg_parquet,
    {
      path <- "output/evaluation/uniform2500_species_datasets_spatial_envblock_widebg.parquet"
      dir.create(dirname(path), showWarnings = FALSE, recursive = TRUE)
      df <- purrr::list_rbind(purrr::list_flatten(uniform_eval_dataset_20_envblock_widebg))
      df <- add_random_split_classes(df)
      df <- add_in_range_polygon(df, all_taxa_polygons, jade_polygon_metadata)
      arrow::write_parquet(df, path)
      path
    },
    format = "file",
    deployment = "main"
  ),

  # ===========================================================================
  # JADE-resampled eval dataset.
  # Mirrors uniform2500 chain but feeds prepare_uniform_eval_batch with
  # presences pulled from data/processed/jade_samples_clean.parquet (JADE
  # accept-reject resampling) instead of uniform-within-polygon. Backgrounds,
  # splits, in_range_polygon, and z-score basis are all retained as-is.
  # Output: output/evaluation/uniform2500_species_datasets_spatial_envblock_jade.parquet
  # ===========================================================================

  tar_target(
    jade_samples_clean_parquet_path,
    "data/processed/jade_samples_clean.parquet",
    format = "file",
    deployment = "main"
  ),

  tar_target(
    jade_samples_20,
    build_jade_samples_for_eval(
      jade_samples_clean_parquet_path,
      eval_species = uniform_test_species,
      env_mean_sd = env_mean_sd,
      chelsa_var_meta = chelsa_var_meta,
      bioclim_include_pattern = bioclim_include_pattern,
      n_per_species = 1100L
    ),
    deployment = "main"
  ),

  tar_target(
    jade_eval_batch_data,
    prepare_uniform_eval_batch(
      uniform_eval_species_batches, jade_samples_20,
      chelsa_var_meta, env_mean_sd, chelsa_bio_dir, bioclim_include_pattern,
      eval_n_background, eval_n_test_bg
    ),
    pattern = map(uniform_eval_species_batches),
    iteration = "list"
  ),

  tar_target(
    jade_eval_dataset_20,
    {
      jac <- terra::rast(jacobian_raster_path)
      purrr::map(jade_eval_batch_data, \(sp) {
        build_species_dataset(sp) |> add_jacobian_column(jac)
      }) |> purrr::compact()
    },
    pattern = map(jade_eval_batch_data),
    iteration = "list"
  ),

  tar_target(
    jade_eval_dataset_20_envblock,
    purrr::compact(purrr::map(jade_eval_dataset_20,
                              \(sp) add_envblock_split(sp, n_holes = 30L))),
    pattern = map(jade_eval_dataset_20),
    iteration = "list"
  ),

  tar_target(
    jade_eval_dataset_20_parquet,
    {
      path <- "output/evaluation/uniform2500_species_datasets_spatial_envblock_jade.parquet"
      dir.create(dirname(path), showWarnings = FALSE, recursive = TRUE)
      df <- purrr::list_rbind(purrr::list_flatten(jade_eval_dataset_20_envblock))
      df <- add_random_split_classes(df)
      df <- add_in_range_polygon(df, all_taxa_polygons, jade_polygon_metadata)
      arrow::write_parquet(df, path)
      path
    },
    format = "file",
    deployment = "main"
  ),

  # ===========================================================================
  # Uniform-in-polygon training dataset @ 5000 pts/species, sliced into 4
  # bins matching the existing jade_samples_clean_*.parquet bin assignment.
  # Output goes to the bets-analysis project's data dir for dual-source
  # BETS training (uniform embed + JADE predict, or vice-versa).
  # ===========================================================================

  tar_target(
    bets_analysis_species_data_dir,
    "/blue/rdinnage.fiu/rdinnage.fiu/Projects/bets-analysis/bets/src/ml_domains/species/data",
    deployment = "main"
  ),

  tar_target(
    jade_bin_species_map,
    read_jade_bin_species_map(bets_analysis_species_data_dir),
    deployment = "main"
  ),

  tar_target(
    uniform_train_5000_species_set,
    unique(unlist(jade_bin_species_map)),
    deployment = "main"
  ),

  tar_target(
    uniform_train_5000_species_batches,
    split_into_batches(uniform_train_5000_species_set, batch_size = 50L),
    deployment = "main"
  ),

  tar_target(
    uniform_train_5000_batch_data,
    sample_uniform_train_batch(
      uniform_train_5000_species_batches, jade_bin_species_map,
      all_taxa_polygons, jade_polygon_metadata,
      chelsa_var_meta, chelsa_bio_dir, jacobian_raster_path,
      bioclim_include_pattern,
      n_per_species = 5000L
    ),
    pattern = map(uniform_train_5000_species_batches),
    iteration = "list"
  ),

  tar_target(
    uniform_clean_bin_parquets,
    write_uniform_clean_4bins(
      uniform_train_5000_batch_data,
      out_dir = bets_analysis_species_data_dir
    ),
    format = "file",
    deployment = "main"
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
      chelsa_bio_dir,
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
      chelsa_bio_dir
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