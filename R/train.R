#!/usr/bin/env Rscript
# train.R - Main training script for progressive VAE-IWAE-SUMO model
# 
# Command line arguments:
#   --train_file: Path to training data CSV
#   --val_file: Path to validation data CSV
#   --input_dim: Dimension of input features
#   --n_species: Number of species categories
#   --spec_embed_dim: Dimension of species embeddings
#   --latent_dim: Dimension of latent space
#   --batch_size: Batch size for training
#   --target_batch_size: Target batch size for gradient accumulation
#   --max_epochs: Maximum number of training epochs
#   --phase_config: Path to optional phase configuration YAML
#   --save_dir: Directory to save models
#   --results_name: Name prefix for saved results
#   --train_device: Device to use for training (cuda:0)
#   --val_device: Device to use for validation (cuda:1)
#   --use_async_validation: Whether to use asynchronous validation

library(torch)
library(tidyverse)
library(zeallot)
library(lubridate)
library(argparser) # For command line arguments
library(yaml)      # For YAML configuration

# Load required modules
source("R/model.R")
source("R/training.R")
source("R/evaluation.R")
source("R/visualization.R")
source("R/data.R")
source("R/utils.R")
source("config/default_config.R")

# Load phase configuration module if available
if (file.exists("config/phase_configs.R")) {
  source("config/phase_configs.R")
}

# Set random seed for reproducibility
set.seed(123456)

# Parse command line arguments
parse_args <- function() {
  p <- arg_parser("Train a progressive VAE-IWAE-SUMO model")
  
  # Data parameters
  p <- add_argument(p, "--train_file", default="data/processed/nichencoder_train.csv", 
                   help="Path to training data CSV")
  p <- add_argument(p, "--val_file", default="data/processed/nichencoder_val.csv", 
                   help="Path to validation data CSV")
  p <- add_argument(p, "--input_dim", default=19, type="integer",
                   help="Dimension of input features")
  p <- add_argument(p, "--n_species", default=100, type="integer",
                   help="Number of species categories")
  
  # Model architecture
  p <- add_argument(p, "--spec_embed_dim", default=64, type="integer",
                   help="Dimension of species embeddings")
  p <- add_argument(p, "--latent_dim", default=16, type="integer",
                   help="Dimension of latent space")
  p <- add_argument(p, "--breadth", default=512, type="integer",
                   help="Width of hidden layers")
  p <- add_argument(p, "--loggamma_init", default=-3, type="numeric",
                   help="Initial log observation noise")
  
  # Training parameters
  p <- add_argument(p, "--batch_size", default=128, type="integer",
                   help="Batch size for training")
  p <- add_argument(p, "--target_batch_size", default=64, type="integer",
                   help="Target batch size for gradient accumulation")
  p <- add_argument(p, "--max_epochs", default=500, type="integer",
                   help="Maximum number of training epochs")
  
  # Phase configuration
  p <- add_argument(p, "--phase_config", default=NULL,
                   help="Path to phase configuration YAML file")
  
  # Progressive training parameters
  p <- add_argument(p, "--use_plateau_detection", default=TRUE, type="logical",
                   help="Use plateau detection for phase transitions")
  p <- add_argument(p, "--early_stopping", default=TRUE, type="logical",
                   help="Use early stopping")
  
  # Checkpointing and saving
  p <- add_argument(p, "--save_dir", default="results/models",
                   help="Directory to save models")
  p <- add_argument(p, "--checkpoint_freq", default=25, type="integer",
                   help="Frequency of checkpoint saving")
  p <- add_argument(p, "--val_freq", default=5, type="integer",
                   help="Frequency of validation")
  p <- add_argument(p, "--results_name", default=NULL,
                   help="Name prefix for saved results")
  
  # Hardware
  p <- add_argument(p, "--train_device", default="cuda:0",
                   help="Device to use for training (cuda:0)")
  p <- add_argument(p, "--val_device", default="cuda:1",
                   help="Device to use for validation (cuda:1)")
  p <- add_argument(p, "--use_async_validation", default=TRUE, type="logical",
                   help="Use asynchronous validation")
  p <- add_argument(p, "--use_gradient_accumulation", default=TRUE, type="logical",
                   help="Use gradient accumulation")
  
  return(parse_args(p))
}

# Load phase configuration from YAML file
load_phase_config <- function(config_file) {
  if (is.null(config_file) || !file.exists(config_file)) {
    return(NULL)
  }
  
  # Load YAML configuration
  yaml_config <- yaml::read_yaml(config_file)
  
  # If we have the yaml_to_phases function from phase_configs.R, use it
  if (exists("yaml_to_phases")) {
    phases <- yaml_to_phases(yaml_config)
    if (!is.null(phases)) {
      return(phases)
    }
  }
  
  # Fallback direct parsing
  if (!is.null(yaml_config$phases)) {
    phases <- list()
    
    for (i in seq_along(yaml_config$phases)) {
      p <- yaml_config$phases[[i]]
      
      phase <- list(
        mode = p$mode,
        name = p$name,
        K = p$K,
        truncation = ifelse(is.null(p$truncation), 0, p$truncation),
        base_lr = p$base_lr,
        min_epochs = ifelse(is.null(p$min_epochs), 20, p$min_epochs)
      )
      
      phases[[i]] <- phase
    }
    
    return(phases)
  }
  
  # If no valid phase configuration found
  return(NULL)
}

# Main function
main <- function() {
  # Parse command line arguments
  args <- parse_args()
  
  # Create results directory and log file
  timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
  results_name <- if (!is.null(args$results_name)) args$results_name else timestamp
  
  results_dir <- file.path(args$save_dir, results_name)
  if (!dir.exists(results_dir)) {
    dir.create(results_dir, recursive = TRUE)
  }
  
  log_file <- file.path(results_dir, "training.log")
  
  # Log start message
  log_message(paste("Starting progressive VAE-IWAE-SUMO training with configuration:"), log_file)
  log_message(paste(jsonlite::toJSON(args, auto_unbox = TRUE, pretty = TRUE)), log_file)
  
  # Check CUDA device availability
  if (startsWith(args$train_device, "cuda") && !torch_cuda_is_available()) {
    log_message("CUDA requested but not available. Falling back to CPU.", log_file)
    args$train_device <- "cpu"
    args$val_device <- "cpu"
    args$use_async_validation <- FALSE  # Disable async validation when using CPU
  }
  
  # Load validation module if using async validation
  if (args$use_async_validation) {
    if (file.exists("R/async_validation.R")) {
      source("R/async_validation.R")
      log_message(paste("Using asynchronous validation on device:", args$val_device), log_file)
    } else {
      args$use_async_validation <- FALSE
      log_message("Async validation module not found. Using synchronous validation.", log_file)
    }
  }
  
  # Load phase configuration from YAML file if provided
  phase_configs <- NULL
  if (!is.null(args$phase_config)) {
    log_message(paste("Loading phase configuration from:", args$phase_config), log_file)
    phase_configs <- load_phase_config(args$phase_config)
    
    if (is.null(phase_configs)) {
      log_message("Failed to load phase configuration. Using default phases.", log_file)
    } else {
      log_message(paste("Loaded", length(phase_configs), "training phases"), log_file)
    }
  } else if (exists("get_training_phases")) {
    # Use phases from phase_configs.R
    log_message("Using default phases from phase_configs.R", log_file)
    phase_configs <- get_training_phases()
  }
  
  # Copy the phase config file to results directory if it exists
  if (!is.null(args$phase_config) && file.exists(args$phase_config)) {
    file.copy(args$phase_config, file.path(results_dir, "phase_config.yaml"), overwrite = TRUE)
  }
  
  # Load and prepare data
  log_message("Loading data...", log_file)
  data <- prepare_data(
    read_csv(args$train_file), 
    read_csv(args$val_file), 
    batch_size = args$batch_size
  )
  
  # Create model
  log_message("Creating model...", log_file)
  model <- env_vae_mod(
    input_dim = args$input_dim,
    n_spec = args$n_species,
    spec_embed_dim = args$spec_embed_dim,
    latent_dim = args$latent_dim,
    breadth = args$breadth,
    loggamma_init = args$loggamma_init
  )
  
  # Move model to device
  model <- model$to(device = args$train_device)
  
  # Progress image path
  progress_image_path <- file.path(results_dir, "training_progress.png")
  
  # Run training
  log_message("Starting training...", log_file)
  start_time <- Sys.time()
  
  results <- train_with_progressive_phases(
    model = model,
    train_dl = data$train_dl,
    val_dl = data$val_dl,
    max_epochs = args$max_epochs,
    phase_configs = phase_configs,
    use_plateau_detection = args$use_plateau_detection,
    use_gradient_accumulation = args$use_gradient_accumulation,
    target_batch_size = args$target_batch_size,
    checkpoint_freq = args$checkpoint_freq,
    val_freq = args$val_freq,
    save_dir = results_dir,
    progress_image_path = progress_image_path,
    early_stopping = args$early_stopping,
    train_device = args$train_device,
    val_device = args$val_device,
    use_async_validation = args$use_async_validation
  )
  
  end_time <- Sys.time()
  training_time <- difftime(end_time, start_time, units = "hours")
  
  # Log completion
  log_message(paste("Training completed in", format(training_time)), log_file)
  log_message(paste("Best model saved to:", results$best_model_path), log_file)
  log_message(paste("Final model saved to:", results$final_model_path), log_file)
  
  # Save final plots and results
  phase_names <- sapply(results$phases, function(p) p$name)
  plots <- plot_training_progress(results$history, results$transition_epochs, phase_names)
  
  plots_dir <- file.path(results_dir, "plots")
  if (!dir.exists(plots_dir)) {
    dir.create(plots_dir, recursive = TRUE)
  }
  
  loss_plot_path <- file.path(plots_dir, "loss_plot.png")
  mse_plot_path <- file.path(plots_dir, "mse_plot.png")
  lr_plot_path <- file.path(plots_dir, "lr_plot.png")
  
  ggsave(loss_plot_path, plots$loss_plot, width = 10, height = 6, dpi = 150)
  ggsave(mse_plot_path, plots$mse_plot, width = 10, height = 4, dpi = 150)
  if (!is.null(plots$lr_plot)) {
    ggsave(lr_plot_path, plots$lr_plot, width = 10, height = 3, dpi = 150)
  }
  
  # Save configuration and results
  results_data <- list(
    args = args,
    phases = results$phases,
    history = results$history,
    transitions = results$transition_epochs,
    best_model_path = results$best_model_path,
    final_model_path = results$final_model_path,
    training_time = training_time
  )
  
  saveRDS(results_data, file.path(results_dir, "training_results.rds"))
  
  log_message("Results saved. Training complete!", log_file)
  
  # Return results invisibly
  invisible(results)
}

# Run main function if script is run directly
if (!interactive()) {
  main()
}
