#!/usr/bin/env Rscript
# evaluate.R - Script to evaluate trained models on test data
#
# Command line arguments:
#   --model_path: Path to the trained model file (.pt)
#   --config_path: Path to the model configuration file (.rds)
#   --test_file: Path to test data CSV
#   --output_dir: Directory to save evaluation results
#   --K: Number of importance samples for evaluation (default: 100)
#   --device: Device to use (cuda or cpu)

library(torch)
library(tidyverse)
library(zeallot)
library(ggplot2)
library(argparser) # For command line arguments

# Load required modules
source("R/model.R")
source("R/evaluation.R")
source("R/visualization.R")
source("R/data.R")
source("R/utils.R")

# Parse command line arguments
parse_args <- function() {
  p <- arg_parser("Evaluate a trained progressive VAE-IWAE-SUMO model")
  
  # Required parameters
  p <- add_argument(p, "--model_path", help="Path to the trained model file (.pt)")
  p <- add_argument(p, "--config_path", help="Path to the model configuration file (.rds)")
  p <- add_argument(p, "--test_file", help="Path to test data CSV")
  
  # Optional parameters
  p <- add_argument(p, "--output_dir", default="evaluation_results",
                   help="Directory to save evaluation results")
  p <- add_argument(p, "--K", default=100, type="integer",
                   help="Number of importance samples for evaluation")
  p <- add_argument(p, "--device", default="cuda",
                   help="Device to use (cuda or cpu)")
  
  return(parse_args(p))
}

# Function to load a trained model
load_trained_model <- function(model_path, model_config, device) {
  # Create model with the same configuration
  model <- env_vae_mod(
    input_dim = model_config$input_dim,
    n_spec = model_config$n_species,
    spec_embed_dim = model_config$spec_embed_dim,
    latent_dim = model_config$latent_dim,
    breadth = model_config$breadth,
    loggamma_init = model_config$loggamma_init
  )
  
  # Load trained weights
  model$load_state_dict(torch_load(model_path, device = device))
  
  # Move to appropriate device
  model <- model$to(device = device)
  
  # Set to evaluation mode
  model$eval()
  
  return(model)
}

# Analyze and visualize model performance
analyze_model_performance <- function(evaluation_results, model, output_dir) {
  # Create output directory if it doesn't exist
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }
  
  # Extract results
  test_results <- evaluation_results$test_results
  predictions <- evaluation_results$predictions
  
  # 1. Calculate reconstruction MSE by species
  species_ids <- as.integer(predictions$species)
  unique_species <- unique(species_ids)
  
  # Convert to R matrices for easier manipulation
  inputs <- as.matrix(predictions$inputs)
  recons <- as.matrix(predictions$reconstructions)
  masks <- as.matrix(predictions$masks)
  
  # Calculate MSE for each species
  species_mse <- data.frame(
    species_id = integer(),
    mse = numeric(),
    num_samples = integer(),
    stringsAsFactors = FALSE
  )
  
  for (species in unique_species) {
    idx <- which(species_ids == species)
    if (length(idx) > 0) {
      # Get species data
      species_inputs <- inputs[idx, , drop = FALSE]
      species_recons <- recons[idx, , drop = FALSE]
      species_masks <- masks[idx, , drop = FALSE]
      
      # Calculate MSE (accounting for masks)
      if (nrow(species_inputs) > 1) {
        mse <- mean(((species_inputs - species_recons) * species_masks)^2 / rowSums(species_masks))
      } else {
        mse <- sum(((species_inputs - species_recons) * species_masks)^2) / sum(species_masks)
      }
      
      # Add to dataframe
      species_mse <- rbind(species_mse, data.frame(
        species_id = species,
        mse = mse,
        num_samples = length(idx),
        stringsAsFactors = FALSE
      ))
    }
  }
  
  # Sort by MSE
  species_mse <- species_mse[order(species_mse$mse), ]
  
  # Save species MSE data
  write.csv(species_mse, file.path(output_dir, "species_mse.csv"), row.names = FALSE)
  
  # 2. Plot latent space visualization using PHATE
  latent_means <- as.matrix(predictions$means)
  latent_plot <- plot_latent_space(latent_means, species_ids, method = "phate")
  
  # Save latent plot
  ggsave(file.path(output_dir, "latent_space_phate.png"), latent_plot, width = 10, height = 8, dpi = 150)
  
  # Also save a PCA plot for comparison
  latent_plot_pca <- plot_latent_space(latent_means, species_ids, method = "pca")
  ggsave(file.path(output_dir, "latent_space_pca.png"), latent_plot_pca, width = 10, height = 8, dpi = 150)
  
  # 3. Analyze latent dimensions
  # Calculate variance of each latent dimension
  latent_vars <- colMeans(torch_exp(predictions$logvars)$numpy())
  latent_mean_vars <- apply(latent_means, 2, var)
  
  # Create data frame for latent dimension analysis
  latent_dim_df <- data.frame(
    dimension = 1:length(latent_vars),
    encoder_variance = latent_vars,
    mean_variance = latent_mean_vars,
    activity = latent_mean_vars / latent_vars
  )
  
  # Save latent dimension analysis
  write.csv(latent_dim_df, file.path(output_dir, "latent_dimensions.csv"), row.names = FALSE)
  
  # Plot latent dimension activity
  latent_activity_plot <- ggplot(latent_dim_df, aes(x = dimension, y = activity)) +
    geom_bar(stat = "identity", fill = "steelblue") +
    theme_minimal() +
    labs(title = "Latent Dimension Activity",
         x = "Latent Dimension", y = "Activity (Mean Variance / Encoder Variance)")
  
  # Save latent activity plot
  ggsave(file.path(output_dir, "latent_activity.png"), latent_activity_plot, width = 10, height = 6, dpi = 150)
  
  # 4. Save summary statistics
  summary_stats <- list(
    test_loss = test_results$loss,
    test_mse = test_results$mse,
    mean_species_mse = mean(species_mse$mse),
    min_species_mse = min(species_mse$mse),
    max_species_mse = max(species_mse$mse),
    num_active_dims = sum(latent_dim_df$activity > 0.5),
    observation_noise = torch_exp(model$loggamma$cpu())$item()
  )
  
  # Save summary stats
  saveRDS(summary_stats, file.path(output_dir, "summary_stats.rds"))
  write.csv(as.data.frame(t(unlist(summary_stats))), file.path(output_dir, "summary_stats.csv"))
  
  # Return analysis results
  return(list(
    species_mse = species_mse,
    latent_dim_df = latent_dim_df,
    summary_stats = summary_stats
  ))
}

# Main function
main <- function() {
  # Parse command line arguments
  args <- parse_args()
  
  # Check for CUDA availability if requested
  if (args$device == "cuda" && !torch_cuda_is_available()) {
    cat("CUDA requested but not available. Falling back to CPU.\n")
    args$device <- "cpu"
  }
  
  # Create output directory if it doesn't exist
  if (!dir.exists(args$output_dir)) {
    dir.create(args$output_dir, recursive = TRUE)
  }
  
  # Set up log file
  log_file <- file.path(args$output_dir, "evaluation.log")
  
  # Log start
  log_message("Starting model evaluation with configuration:", log_file)
  log_message(paste(jsonlite::toJSON(args, auto_unbox = TRUE, pretty = TRUE)), log_file)
  
  # Load model configuration
  log_message("Loading model configuration...", log_file)
  config_data <- readRDS(args$config_path)
  
  # Extract model parameters from config
  if ("params" %in% names(config_data)) {
    model_config <- config_data$params
  } else if ("args" %in% names(config_data)) {
    model_config <- config_data$args
  } else {
    model_config <- config_data
  }
  
  # Load the trained model
  log_message("Loading trained model...", log_file)
  model <- tryCatch({
    load_trained_model(args$model_path, model_config, args$device)
  }, error = function(e) {
    log_message(paste("ERROR loading model:", e$message), log_file)
    stop(e)
  })
  
  # Load test data
  log_message("Loading test data...", log_file)
  test_df <- read_csv(args$test_file)
  
  # Create empty results structure
  test_data <- list()
  
  # Prepare test data using information from model_config
  log_message("Preparing test data...", log_file)
  
  # Get training data for species mapping
  train_df <- read_csv(model_config$train_file)
  species_names <- train_df$species
  
  # Filter test to species in training
  test_df <- test_df[test_df$species %in% species_names, ]
  
  # Extract features and masks
  test_features <- test_df %>%
    select(-species, -X, -Y, -starts_with("na_ind_")) %>%
    as.matrix()
    
  # Extract NA masks and invert (1 = present, 0 = missing)
  test_mask <- test_df %>%
    select(starts_with("na_ind_"), -na_ind_X, -na_ind_Y) %>%
    as.matrix()
    
  # Invert masks (1 = valid, 0 = NA)
  test_mask <- 1 - test_mask
  
  # Apply masks to features (zero out missing values)
  test_features[test_mask == 0] <- 0
  
  # Get species for testing
  test_species <- as.integer(as.factor(test_df$species))
  
  # Create dataset
  env_dataset <- dataset(name = "env_ds",
    initialize = function(env, mask, spec) {
      self$env <- torch_tensor(env)
      self$mask <- torch_tensor(mask)
      self$spec <- torch_tensor(spec)
    },
    .getbatch = function(i) {
      list(env = self$env[i, ], mask = self$mask[i,], spec = self$spec[i])
    },
    .length = function() {
      self$env$size()[[1]]
    }
  )
  
  # Create test dataset and dataloader
  test_ds <- env_dataset(test_features, test_mask, test_species)
  test_dl <- dataloader(test_ds, batch_size = 128, shuffle = FALSE)
  
  test_data$test_dl <- test_dl
  
  # Set evaluation phase config (use SUMO with high K for best results)
  phase_config <- list(
    mode = "sumo",
    K = args$K,
    truncation = 4
  )
  
  # Evaluate model
  log_message(paste("Evaluating model with K =", args$K, "samples..."), log_file)
  test_results <- evaluate(model, test_dl, phase_config, device = args$device)
  
  # Generate predictions
  log_message("Generating predictions...", log_file)
  predictions <- predict_with_model(model, test_dl, K = args$K, device = args$device)
  
  # Analyze results
  log_message("Analyzing model performance...", log_file)
  evaluation_results <- list(
    test_results = test_results,
    predictions = predictions
  )
  
  analysis_results <- analyze_model_performance(evaluation_results, model, args$output_dir)
  
  # Log completion
  log_message("Evaluation complete!", log_file)
  log_message(paste("Test loss:", analysis_results$summary_stats$test_loss), log_file)
  log_message(paste("Test MSE:", analysis_results$summary_stats$test_mse), log_file)
  log_message(paste("Mean species MSE:", analysis_results$summary_stats$mean_species_mse), log_file)
  log_message(paste("Active dimensions:", analysis_results$summary_stats$num_active_dims, "out of", model_config$latent_dim), log_file)
  log_message(paste("Observation noise:", analysis_results$summary_stats$observation_noise), log_file)
  
  # Return results invisibly
  invisible(list(
    evaluation_results = evaluation_results,
    analysis_results = analysis_results
  ))
}

# Run main function if script is run directly
if (!interactive()) {
  main()
}
