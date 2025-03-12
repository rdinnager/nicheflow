library(torch)
library(zeallot)

#' Setup asynchronous validation management
#' @param device GPU device to use for validation (e.g., "cuda:1")
#' @return Initialization status
setup_async_validation <- function(device = "cuda:1") {
  # Check if mirai package is available
  if (!requireNamespace("mirai", quietly = TRUE)) {
    stop("Package 'mirai' is needed for asynchronous validation. Please install it with:\n",
         "install.packages('mirai')")
  }
  
  # Initialize global validation state if not already initialized
  if (!exists("validation_state", envir = .GlobalEnv)) {
    validation_state <- list(
      device = device,
      pending = list(),
      completed = list(),
      next_id = 1
    )
    assign("validation_state", validation_state, envir = .GlobalEnv)
    
    # Log initialization
    message("Asynchronous validation initialized on device: ", device)
    
    TRUE
  } else {
    # Already initialized
    FALSE
  }
}

#' Get model metadata for rebuilding in another process
#' @param model The VAE model
#' @return List of metadata needed to recreate model
get_model_metadata <- function(model) {
  list(
    input_dim = model$input_dim,
    n_spec = model$n_spec,
    spec_embed_dim = model$spec_embed_dim,
    latent_dim = model$latent_dim,
    breadth = model$breadth,
    loggamma_init = as.numeric(model$loggamma$cpu())
  )
}

#' Submit a model for asynchronous validation
#' @param model The model to validate
#' @param val_dl Validation dataloader
#' @param phase_config Current phase configuration
#' @param epoch Current epoch number
#' @param total_epoch Overall epoch count
#' @return Validation ID
validate_async <- function(model, val_dl, phase_config, epoch, total_epoch) {
  # Ensure validation setup is initialized
  if (!exists("validation_state", envir = .GlobalEnv)) {
    setup_async_validation()
  }
  
  # Get validation state
  validation_state <- get("validation_state", envir = .GlobalEnv)
  device <- validation_state$device
  
  # Generate a unique validation ID
  val_id <- validation_state$next_id
  validation_state$next_id <- validation_state$next_id + 1
  
  # Get model metadata
  model_metadata <- get_model_metadata(model)
  
  # Clone model state_dict
  model_state <- model$state_dict()
  
  # Create validation task
  val_task <- mirai::mirai({
    # Source required modules
    source("R/model.R")
    source("R/evaluation.R")
    
    # Get local copies of inputs
    local_state <- .env$model_state
    local_metadata <- .env$model_metadata
    local_phase_config <- .env$phase_config
    local_device <- .env$device
    local_val_dl <- .env$val_dl
    
    # Recreate model
    model <- env_vae_mod(
      input_dim = local_metadata$input_dim,
      n_spec = local_metadata$n_spec,
      spec_embed_dim = local_metadata$spec_embed_dim,
      latent_dim = local_metadata$latent_dim,
      breadth = local_metadata$breadth,
      loggamma_init = local_metadata$loggamma_init
    )
    
    # Load state dict
    model$load_state_dict(local_state)
    model <- model$to(local_device)
    
    # Run evaluation
    val_results <- evaluate(model, local_val_dl, local_phase_config, device = local_device)
    
    # Return results
    val_results
  }, 
  .args = list(
    model_state = model_state,
    model_metadata = model_metadata,
    phase_config = phase_config,
    device = device,
    val_dl = val_dl
  ))
  
  # Store task info
  validation_state$pending[[as.character(val_id)]] <- list(
    task = val_task,
    epoch = epoch,
    total_epoch = total_epoch,
    phase = phase_config$mode,
    K = phase_config$K,
    submitted_at = Sys.time()
  )
  
  assign("validation_state", validation_state, envir = .GlobalEnv)
  
  message("Submitted validation ", val_id, " for epoch ", epoch, " (", phase_config$mode, "-K=", phase_config$K, ") to device ", device)
  
  # Return validation ID
  val_id
}

#' Check for completed validation tasks
#' @return List of completed validation IDs and results
check_validation_results <- function() {
  # If validation state doesn't exist, nothing to check
  if (!exists("validation_state", envir = .GlobalEnv)) {
    return(list())
  }
  
  # Get validation state
  validation_state <- get("validation_state", envir = .GlobalEnv)
  
  # Check which tasks are complete
  completed_ids <- character(0)
  new_results <- list()
  
  for (id in names(validation_state$pending)) {
    task_info <- validation_state$pending[[id]]
    task <- task_info$task
    
    if (mirai::resolved(task)) {
      # Get results
      result <- tryCatch({
        mirai::value(task)
      }, error = function(e) {
        # If there was an error, return the error
        list(
          error = TRUE,
          message = e$message,
          loss = NA,
          mse = NA
        )
      })
      
      # Store completed result
      validation_state$completed[[id]] <- list(
        result = result,
        epoch = task_info$epoch,
        total_epoch = task_info$total_epoch,
        phase = task_info$phase,
        K = task_info$K, 
        submitted_at = task_info$submitted_at,
        completed_at = Sys.time()
      )
      
      # Add to list of completed IDs
      completed_ids <- c(completed_ids, id)
      
      # Add to new results
      new_results[[id]] <- list(
        result = result,
        epoch = task_info$epoch,
        total_epoch = task_info$total_epoch,
        phase = task_info$phase
      )
      
      message("Validation ", id, " for epoch ", task_info$epoch, " completed: loss=", 
             round(result$loss, 4), ", mse=", round(result$mse, 4))
    }
  }
  
  # Remove completed tasks from pending
  if (length(completed_ids) > 0) {
    for (id in completed_ids) {
      validation_state$pending[[id]] <- NULL
    }
    assign("validation_state", validation_state, envir = .GlobalEnv)
  }
  
  # Return new results
  new_results
}

#' Get most recent validation result
#' @return The most recent validation result or NULL if none available
get_latest_validation_result <- function() {
  if (!exists("validation_state", envir = .GlobalEnv)) {
    return(NULL)
  }
  
  validation_state <- get("validation_state", envir = .GlobalEnv)
  
  if (length(validation_state$completed) == 0) {
    return(NULL)
  }
  
  # Find the most recent result by completion timestamp
  latest_id <- NULL
  latest_time <- as.POSIXct("1970-01-01")
  
  for (id in names(validation_state$completed)) {
    completed_at <- validation_state$completed[[id]]$completed_at
    if (completed_at > latest_time) {
      latest_time <- completed_at
      latest_id <- id
    }
  }
  
  if (is.null(latest_id)) {
    return(NULL)
  }
  
  validation_state$completed[[latest_id]]
}

#' Get validation result for a specific epoch
#' @param epoch The epoch number to find results for
#' @return Validation result or NULL if not found
get_validation_for_epoch <- function(epoch) {
  if (!exists("validation_state", envir = .GlobalEnv)) {
    return(NULL)
  }
  
  validation_state <- get("validation_state", envir = .GlobalEnv)
  
  # Search in completed validations
  for (id in names(validation_state$completed)) {
    if (validation_state$completed[[id]]$epoch == epoch) {
      return(validation_state$completed[[id]])
    }
  }
  
  # Search in pending validations
  for (id in names(validation_state$pending)) {
    if (validation_state$pending[[id]]$epoch == epoch) {
      return(list(
        pending = TRUE,
        epoch = validation_state$pending[[id]]$epoch,
        submitted_at = validation_state$pending[[id]]$submitted_at
      ))
    }
  }
  
  # Not found
  NULL
}

#' Clean up validation resources
#' @return Invisible NULL
cleanup_validation <- function() {
  if (exists("validation_state", envir = .GlobalEnv)) {
    # Clean up mirai tasks
    validation_state <- get("validation_state", envir = .GlobalEnv)
    
    # Cancel any pending tasks
    for (id in names(validation_state$pending)) {
      mirai::cancel(validation_state$pending[[id]]$task)
    }
    
    # Remove validation state
    rm("validation_state", envir = .GlobalEnv)
    
    message("Cleaned up validation resources")
  }
  
  invisible(NULL)
}
