library(torch)

# Function to adjust batch size based on K parameter
adjust_batch_size <- function(original_batch_size, K, min_batch = 4) {
  # Estimate memory requirements based on K
  estimated_ratio <- 1 + min(1, log10(K) / 2)
  new_batch_size <- floor(original_batch_size / estimated_ratio)
  return(max(min_batch, new_batch_size))
}

# Gradient accumulation helper
gradient_accumulation_step <- function(model, optimizer, loss_fn, batch, 
                                     accumulation_steps, current_step, device = "cuda") {
  # Process a subset of the batch
  batch_size <- batch$env$size(1)
  start_idx <- floor((current_step - 1) * batch_size / accumulation_steps) + 1
  end_idx <- min(floor(current_step * batch_size / accumulation_steps), batch_size)
  
  # Create mini-batch
  mini_batch <- list(
    env = batch$env[start_idx:end_idx, ]$to(device = device),
    mask = batch$mask[start_idx:end_idx, ]$to(device = device),
    spec = batch$spec[start_idx:end_idx]$to(device = device)
  )
  
  # Process mini-batch
  loss <- loss_fn(model, mini_batch) / accumulation_steps
  loss$backward()
  
  return(loss$item() * accumulation_steps)
}

# Create variance tracker for monitoring estimator variance
create_variance_tracker <- function(window_size = 50) {
  list(
    window_size = window_size,
    values = numeric(0),
    moving_avg = 0,
    moving_var = 0,
    update = function(new_value) {
      # Add new value
      self$values <- c(self$values, new_value)
      
      # Keep only the most recent window_size values
      if (length(self$values) > self$window_size) {
        self$values <- self$values[(length(self$values) - self$window_size + 1):length(self$values)]
      }
      
      # Update statistics
      self$moving_avg <- mean(self$values)
      self$moving_var <- ifelse(length(self$values) > 1, var(self$values), 0)
      
      return(self)
    }
  )
}

# Plateau detection for adaptive phase transitions
create_plateau_detector <- function(patience = 10, min_delta = 0.001, window_size = 5, 
                                   rel_improvement_threshold = 0.01) {
  # Initialize state
  state <- list(
    best_value = Inf,
    patience_counter = 0,
    history = numeric(0),
    window_history = numeric(0),
    last_transition_epoch = 0
  )
  
  # Return detection function
  function(val_loss, current_epoch, min_epochs_in_phase = 20) {
    # Add to history
    state$history <- c(state$history, val_loss)
    
    # Update best observed value
    if (val_loss < state$best_value - min_delta) {
      state$best_value <- val_loss
      state$patience_counter <- 0
    } else {
      state$patience_counter <- state$patience_counter + 1
    }
    
    # Update window for slope calculation
    state$window_history <- c(state$window_history, val_loss)
    if (length(state$window_history) > window_size) {
      state$window_history <- state$window_history[(length(state$window_history) - window_size + 1):length(state$window_history)]
    }
    
    # Don't transition if we haven't spent enough epochs in current phase
    if (current_epoch - state$last_transition_epoch < min_epochs_in_phase) {
      return(FALSE)
    }
    
    # Method 1: Patience-based plateau detection
    patience_plateau <- state$patience_counter >= patience
    
    # Method 2: Slope-based plateau detection (if we have enough history)
    slope_plateau <- FALSE
    if (length(state$window_history) == window_size) {
      # Fit a line to recent losses
      x <- 1:window_size
      y <- state$window_history
      if (sd(y) > 1e-6) {  # Avoid fitting to constant values
        slope <- stats::coef(stats::lm(y ~ x))[2]
        
        # If slope is very small (close to flat or slightly increasing)
        slope_plateau <- abs(slope) < min_delta || slope > 0
      } else {
        slope_plateau <- TRUE  # Flat line is a plateau
      }
    }
    
    # Method 3: Relative improvement plateau
    rel_plateau <- FALSE
    if (length(state$history) >= window_size * 2) {
      # Compare average of recent window to average of previous window
      recent_avg <- mean(state$history[(length(state$history) - window_size + 1):length(state$history)])
      previous_avg <- mean(state$history[(length(state$history) - 2*window_size + 1):(length(state$history) - window_size)])
      
      # Calculate relative improvement
      rel_improvement <- (previous_avg - recent_avg) / previous_avg
      rel_plateau <- rel_improvement < rel_improvement_threshold
    }
    
    # Combine detection methods - require at least two methods to agree
    is_plateau <- sum(c(patience_plateau, slope_plateau, rel_plateau)) >= 2
    
    if (is_plateau) {
      # Reset state for next phase
      state$patience_counter <- 0
      state$last_transition_epoch <- current_epoch
    }
    
    return(is_plateau)
  }
}

# Function to create results directory with timestamp
create_results_dir <- function(base_dir = "results") {
  timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
  results_dir <- file.path(base_dir, timestamp)
  
  # Create directories
  for (dir in c(results_dir, 
                file.path(results_dir, "models"),
                file.path(results_dir, "plots"),
                file.path(results_dir, "checkpoints"),
                file.path(results_dir, "logs"))) {
    if (!dir.exists(dir)) {
      dir.create(dir, recursive = TRUE)
    }
  }
  
  return(results_dir)
}

# Log message to file and console
log_message <- function(message, log_file = NULL) {
  # Format message with timestamp
  timestamp <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
  formatted_message <- paste0("[", timestamp, "] ", message)
  
  # Print to console
  cat(formatted_message, "\n")
  
  # Log to file if provided
  if (!is.null(log_file)) {
    write(formatted_message, file = log_file, append = TRUE)
  }
}

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

