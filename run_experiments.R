#!/usr/bin/env Rscript
# run_experiments.R - Script to run multiple experiments with different configurations
# 
# Command line arguments:
#   --config_yaml: Path to YAML configuration file defining experiments
#   --results_base_dir: Base directory for experiment results
#   --device: Device to use (cuda or cpu)

library(torch)
library(tidyverse)
library(zeallot)
library(lubridate)
library(yaml)
library(argparser) # For command line arguments

# Load required modules
source("R/model.R")
source("R/training.R")
source("R/evaluation.R")
source("R/visualization.R")
source("R/data.R")
source("R/utils.R")
source("config/default_config.R")

# Parse command line arguments
parse_args <- function() {
  p <- arg_parser("Run multiple IWAE/SUMO experiments with different configurations")
  
  p <- add_argument(p, "--config_yaml", default="experiments.yaml",
                   help="Path to YAML configuration file defining experiments")
  p <- add_argument(p, "--results_base_dir", default="results/experiments",
                   help="Base directory for experiment results")
  p <- add_argument(p, "--device", default="cuda",
                   help="Device to use (cuda or cpu)")
  
  return(parse_args(p))
}

# Load experiment configurations from YAML
load_experiment_configs <- function(config_yaml) {
  # Read YAML configuration file
  configs <- yaml::read_yaml(config_yaml)
  
  # Process each experiment configuration
  experiment_configs <- list()
  
  for (exp_name in names(configs$experiments)) {
    # Start with default config
    config <- default_config
    
    # Apply shared overrides if any
    if (!is.null(configs$shared_params)) {
      config <- modifyList(config, configs$shared_params)
    }
    
    # Apply experiment-specific overrides
    exp_config <- configs$experiments[[exp_name]]
    config <- modifyList(config, exp_config)
    
    # Add experiment name
    config$experiment_name <- exp_name
    
    # Add to list
    experiment_configs[[exp_name]] <- config
  }
  
  return(experiment_configs)
}

# Function to run a single experiment
run_experiment <- function(config, experiment_dir, device) {
  # Create experiment directory
  if (!dir.exists(experiment_dir)) {
    dir.create(experiment_dir, recursive = TRUE)
  }
  
  # Log file
  log_file <- file.path(experiment_dir, "experiment.log")
  
  # Log start
  log_message(paste("Starting experiment:", config$experiment_name), log_file)
  log_message(paste("Configuration:", jsonlite::toJSON(config, auto_unbox = TRUE)), log_file)
  
  # Set up save directories
  config$save_dir <- file.path(experiment_dir, "models")
  config$progress_image_path <- file.path(experiment_dir, "training_progress.png")
  
  # Force device setting
  config$device <- device
  
  # Run training
  start_time <- Sys.time()
  
  tryCatch({
    results <- run_progressive_training(
      train_file = config$train_file,
      val_file = config$val_file,
      input_dim = config$input_dim,
      n_species = config$n_species,
      spec_embed_dim = config$spec_embed_dim,
      latent_dim = config$latent_dim,
      breadth = config$breadth,
      loggamma_init = config$loggamma_init,
      batch_size = config$batch_size,
      target_batch_size = config$target_batch_size,
      max_epochs = config$max_epochs,
      use_plateau_detection = config$use_plateau_detection,
      min_epochs_per_phase = config$min_epochs_per_phase,
      phase_learning_rates = config$phase_learning_rates,
      phase_K_values = config$phase_K_values,
      phase_truncation_values = config$phase_truncation_values,
      plateau_patience = config$plateau_patience,
      plateau_min_delta = config$plateau_min_delta,
      plateau_window = config$plateau_window,
      plateau_rel_improvement = config$plateau_rel_improvement,
      early_stopping = config$early_stopping,
      early_stopping_patience = config$early_stopping_patience,
      save_dir = config$save_dir,
      checkpoint_freq = config$checkpoint_freq,
      save_progress_image_freq = config$save_progress_image_freq,
      progress_image_path = config$progress_image_path,
      device = config$device,
      use_gradient_accumulation = config$use_gradient_accumulation
    )
    
    # Log success
    end_time <- Sys.time()
    total_time <- difftime(end_time, start_time, units = "hours")
    
    log_message(paste("Experiment completed successfully in", format(total_time)), log_file)
    log_message(paste("Best model saved to:", results$best_model_path), log_file)
    
    # Save results
    saveRDS(results, file.path(experiment_dir, "results.rds"))
    
    # Save plots
    plots_dir <- file.path(experiment_dir, "plots")
    if (!dir.exists(plots_dir)) {
      dir.create(plots_dir, recursive = TRUE)
    }
    
    ggsave(file.path(plots_dir, "loss_plot.png"), results$plots$loss_plot, width = 10, height = 6, dpi = 150)
    ggsave(file.path(plots_dir, "mse_plot.png"), results$plots$mse_plot, width = 10, height = 4, dpi = 150)
    
    return(list(success = TRUE, results = results))
    
  }, error = function(e) {
    # Log error
    log_message(paste("ERROR in experiment:", e$message), log_file)
    return(list(success = FALSE, error = e$message))
  })
}

# Main function to run all experiments
main <- function() {
  # Parse command line arguments
  args <- parse_args()
  
  # Check for CUDA availability if requested
  if (args$device == "cuda" && !torch_cuda_is_available()) {
    cat("CUDA requested but not available. Falling back to CPU.\n")
    args$device <- "cpu"
  }
  
  # Load experiment configurations
  cat("Loading experiment configurations from", args$config_yaml, "\n")
  experiments <- load_experiment_configs(args$config_yaml)
  
  # Create results directory with timestamp
  timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
  results_dir <- file.path(args$results_base_dir, timestamp)
  
  # Create directory if it doesn't exist
  if (!dir.exists(results_dir)) {
    dir.create(results_dir, recursive = TRUE)
  }
  
  # Save experiment configurations
  saveRDS(experiments, file.path(results_dir, "experiment_configs.rds"))
  
  # Set up log file
  log_file <- file.path(results_dir, "experiments.log")
  
  # Log start
  total_experiments <- length(experiments)
  log_message(paste("Starting", total_experiments, "experiments"), log_file)
  
  # Run each experiment
  results <- list()
  
  for (i in seq_along(experiments)) {
    exp_name <- names(experiments)[i]
    config <- experiments[[exp_name]]
    
    # Create experiment directory
    exp_dir <- file.path(results_dir, exp_name)
    
    log_message(paste("Starting experiment", i, "of", total_experiments, ":", exp_name), log_file)
    cat("==========================================\n")
    cat(sprintf("Running experiment %d/%d: %s\n", i, total_experiments, exp_name))
    cat("==========================================\n")
    
    # Run experiment
    results[[exp_name]] <- run_experiment(config, exp_dir, args$device)
    
    # Brief delay between experiments
    Sys.sleep(5)
  }
  
  # Save overall results summary
  summary <- list()
  for (exp_name in names(results)) {
    if (results[[exp_name]]$success) {
      r <- results[[exp_name]]$results
      summary[[exp_name]] <- list(
        success = TRUE,
        final_loss = tail(r$history$val_loss, 1),
        best_loss = min(r$history$val_loss),
        final_mse = tail(r$history$mse, 1),
        best_mse = min(r$history$mse),
        epochs = length(r$history$val_loss),
        training_time = r$training_time,
        transitions = r$transitions
      )
    } else {
      summary[[exp_name]] <- list(
        success = FALSE,
        error = results[[exp_name]]$error
      )
    }
  }
  
  # Save summary
  saveRDS(summary, file.path(results_dir, "experiments_summary.rds"))
  
  # Create summary table
  summary_df <- data.frame(
    experiment = character(),
    success = logical(),
    best_loss = numeric(),
    final_loss = numeric(),
    best_mse = numeric(),
    final_mse = numeric(),
    epochs = integer(),
    training_hours = numeric(),
    num_transitions = integer(),
    stringsAsFactors = FALSE
  )
  
  for (exp_name in names(summary)) {
    s <- summary[[exp_name]]
    if (s$success) {
      summary_df <- rbind(summary_df, data.frame(
        experiment = exp_name,
        success = TRUE,
        best_loss = s$best_loss,
        final_loss = s$final_loss,
        best_mse = s$best_mse,
        final_mse = s$final_mse,
        epochs = s$epochs,
        training_hours = as.numeric(s$training_time, units = "hours"),
        num_transitions = length(s$transitions),
        stringsAsFactors = FALSE
      ))
    } else {
      summary_df <- rbind(summary_df, data.frame(
        experiment = exp_name,
        success = FALSE,
        best_loss = NA,
        final_loss = NA,
        best_mse = NA,
        final_mse = NA,
        epochs = NA,
        training_hours = NA,
        num_transitions = NA,
        stringsAsFactors = FALSE
      ))
    }
  }
  
  # Save summary table
  write.csv(summary_df, file.path(results_dir, "experiments_summary.csv"), row.names = FALSE)
  
  # Log completion
  log_message("All experiments completed!", log_file)
  log_message(paste("Results saved to:", results_dir), log_file)
  
  # Return results invisibly
  invisible(list(
    results_dir = results_dir,
    summary = summary,
    summary_df = summary_df
  ))
}

# Run main function if script is run directly
if (!interactive()) {
  main()
}
