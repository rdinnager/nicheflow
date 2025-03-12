library(tidyverse)
library(torch)

# Prepare datasets from dataframes
prepare_data <- function(train_df, val_df, batch_size = 128) {
  # Extract species information
  species_names <- train_df$species
  species <- as.integer(as.numeric(as.factor(species_names)))
  
  # Filter validation to species in training
  val_df <- val_df[val_df$species %in% species_names, ]
  
  # Extract features and masks
  train_features <- train_df %>%
    select(-species, -X, -Y, -starts_with("na_ind_")) %>%
    as.matrix()
    
  val_features <- val_df %>%
    select(-species, -X, -Y, -starts_with("na_ind_")) %>%
    as.matrix()
    
  # Extract NA masks and invert (1 = present, 0 = missing)
  train_mask <- train_df %>%
    select(starts_with("na_ind_"), -na_ind_X, -na_ind_Y) %>%
    as.matrix()
    
  val_mask <- val_df %>%
    select(starts_with("na_ind_"), -na_ind_X, -na_ind_Y) %>%
    as.matrix()
    
  # Invert masks (1 = valid, 0 = NA)
  train_mask <- 1 - train_mask
  val_mask <- 1 - val_mask
  
  # Apply masks to features (zero out missing values)
  train_features[train_mask == 0] <- 0
  val_features[val_mask == 0] <- 0
  
  # Get species for validation
  val_species <- as.integer(as.numeric(as.factor(val_df$species)))
  
  # Create dataset class
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
  
  # Create datasets
  train_ds <- env_dataset(train_features, train_mask, species)
  val_ds <- env_dataset(val_features, val_mask, val_species)
  
  # Create dataloaders
  train_dl <- dataloader(train_ds, batch_size = batch_size, shuffle = TRUE)
  val_dl <- dataloader(val_ds, batch_size = batch_size, shuffle = FALSE)
  
  # Return datasets and loaders
  return(list(
    train_ds = train_ds,
    val_ds = val_ds,
    train_dl = train_dl,
    val_dl = val_dl,
    n_species = length(unique(species))
  ))
}

# Function to load a dataset from CSV files
load_data_from_csv <- function(train_file, val_file, test_file = NULL, batch_size = 128) {
  # Load training data
  train_df <- read_csv(train_file)
  
  # Load validation data
  val_df <- read_csv(val_file)
  
  # Prepare train and validation data
  data <- prepare_data(train_df, val_df, batch_size)
  
  # If test file provided, load and prepare test data
  if (!is.null(test_file)) {
    test_df <- read_csv(test_file)
    
    # Filter test to species in training
    species_names <- train_df$species
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
    test_species <- as.integer(as.numeric(as.factor(test_df$species)))
    
    # Create test dataset
    test_ds <- data$train_ds$clone()$initialize(test_features, test_mask, test_species)
    
    # Create test dataloader
    test_dl <- dataloader(test_ds, batch_size = batch_size, shuffle = FALSE)
    
    # Add test to returned data
    data$test_ds <- test_ds
    data$test_dl <- test_dl
  }
  
  return(data)
}
