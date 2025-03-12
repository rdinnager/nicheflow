library(torch)
library(zeallot)  # For %<-% unpacking operator
library(lubridate)  # For time calculations

# Training for one epoch
train_one_epoch <- function(model, dataloader, optimizer, phase_config,
                          gradient_accumulation = FALSE, target_batch_size = 64, 
                          device = "cuda") {
  model$train()
  total_loss <- 0
  total_data_loss <- 0
  total_reg_loss <- 0
  batch_count <- 0
  
  mode <- phase_config$mode
  K <- phase_config$K
  truncation <- phase_config$truncation
  weight_var_sum <- 0
  
  # Setup gradient accumulation if needed
  batch_size <- dataloader$batch_size
  effective_batch_size <- batch_size
  accumulation_steps <- 1
  
  if (gradient_accumulation) {
    # Adjust accumulation steps based on K
    actual_batch_size <- adjust_batch_size(target_batch_size, K)
    accumulation_steps <- max(1, ceiling(target_batch_size / actual_batch_size))
  }
  
  coro::loop(for (batch in dataloader) {
    batch_count <- batch_count + 1
    
    if (gradient_accumulation && accumulation_steps > 1) {
      optimizer$zero_grad()
      batch_loss <- 0
      
      for (step in 1:accumulation_steps) {
        # Process subset of batch with accumulation
        step_loss <- gradient_accumulation_step(
          model, optimizer, 
          function(model, mini_batch) {
            c(x, z, means, log_vars, spec_lat) %<-% model(mini_batch$env, mini_batch$spec, K = K)
            
            if (mode == "vae") {
              c(loss, ., ., spec_loss, _) %<-% model$loss_function(
                z, x, mini_batch$mask, means, log_vars, spec_lat, 
                mode = mode, K = K
              )
            } else if (mode == "iwae") {
              c(loss, main_loss, spec_loss, ., weight_var) %<-% model$loss_function(
                z, x, mini_batch$mask, means, log_vars, spec_lat, 
                mode = mode, K = K
              )
              weight_var_sum <- weight_var_sum + weight_var$item()
            } else if (mode == "sumo") {
              c(loss, main_loss, spec_loss, ., weight_var) %<-% model$loss_function(
                z, x, mini_batch$mask, means, log_vars, spec_lat, 
                mode = mode, K = K, truncation = truncation
              )
              weight_var_sum <- weight_var_sum + weight_var$item()
            }
            
            return(loss)
          },
          batch, accumulation_steps, step, device
        )
        
        batch_loss <- batch_loss + step_loss
      }
      
      # Update weights
      optimizer$step()
      
      # Update scheduler if passed
      if (!is.null(phase_config$scheduler)) {
        phase_config$scheduler$step()
      }
      
      # Update tracking
      total_loss <- total_loss + batch_loss
      
    } else {
      # Standard single-step training
      optimizer$zero_grad()
      
      # Move data to device
      input <- batch$env$to(device = device)
      mask <- batch$mask$to(device = device)
      spec <- batch$spec$to(device = device)
      
      # Forward pass
      c(x, z, means, log_vars, spec_lat) %<-% model(input, spec, K = K)
      
      # Loss calculation based on mode
      if (mode == "vae") {
        c(loss, recon_loss, kl_loss, spec_loss, .) %<-% model$loss_function(
          z, x, mask, means, log_vars, spec_lat, 
          mode = mode, K = K
        )
        data_loss <- recon_loss$mean()$item() + kl_loss$mean()$item()
        reg_loss <- spec_loss$item()
      } else if (mode == "iwae") {
        c(loss, main_loss, spec_loss, ., weight_var) %<-% model$loss_function(
          z, x, mask, means, log_vars, spec_lat, 
          mode = mode, K = K
        )
        data_loss <- main_loss$item()
        reg_loss <- spec_loss$item()
        weight_var_sum <- weight_var_sum + weight_var$item()
      } else if (mode == "sumo") {
        c(loss, main_loss, spec_loss, ., weight_var) %<-% model$loss_function(
          z, x, mask, means, log_vars, spec_lat, 
          mode = mode, K = K, truncation = truncation
        )
        data_loss <- main_loss$item()
        reg_loss <- spec_loss$item()
        weight_var_sum <- weight_var_sum + weight_var$item()
      }
      
      # Backward pass and optimization
      loss$backward()
      optimizer$step()
      
      # Update scheduler if passed
      if (!is.null(phase_config$scheduler)) {
        phase_config$scheduler$step()
      }
      
      # Update tracking
      total_loss <- total_loss + loss$item()
      total_data_loss <- total_data_loss + data_loss
      total_reg_loss <- total_reg_loss + reg_loss
    }
  })
  
  # Calculate averages
  avg_loss <- total_loss / batch_count
  avg_data_loss <- total_data_loss / batch_count
  avg_reg_loss <- total_reg_loss / batch_count
  avg_weight_var <- weight_var_sum / batch_count
  
  return(list(
    loss = avg_loss,
    data_loss = avg_data_loss,
    reg_loss = avg_reg_loss,
    weight_var = avg_weight_var
  ))
}

# Progressive training main function with async validation
train_with_progressive_phases <- function(model, train_dl, val_dl, 
                                         max_epochs = 1000,
                                         phase_configs = NULL,
                                         use_plateau_detection = TRUE,
                                         plateau_patience = 15,
                                         plateau_min_delta = 0.001,
                                         plateau_window_size = 7,
                                         plateau_rel_improvement = 0.005,
                                         use_gradient_accumulation = TRUE,
                                         target_batch_size = 64,
                                         save_dir = "models",
                                         checkpoint_freq = 25,
                                         val_freq = 5,  # Validation frequency
                                         save_progress_image_freq = 5,
                                         progress_image_path = "training_progress.png",
                                         early_stopping = TRUE,
                                         early_stopping_patience = 30,
                                         train_device = "cuda:0",
                                         val_device = "cuda:1",  # Separate validation device
                                         use_async_validation = TRUE) { # Enable/disable async validation
  
  # Load phase configuration function
  if (file.exists("config/phase_configs.R")) {
    source("config/phase_configs.R")
  } else {
    warning("Phase config file not found. Using default phases.")
  }
  
  # Get training phases
  if (is.null(phase_configs) && exists("get_training_phases")) {
    phases <- get_training_phases()
  } else if (!is.null(phase_configs)) {
    phases <- phase_configs
  } else {
    # Fallback to hardcoded default phases
    phases <- list(
      # Phase 1: Standard VAE with analytical KL
      list(
        mode = "vae",
        name = "VAE",
        K = 1,
        truncation = 0,
        base_lr = 0.001,
        min_epochs = 20
      ),
      
      # Phase 2: IWAE with low K
      list(
        mode = "iwae",
        name = "IWAE-10",
        K = 10,
        truncation = 0,
        base_lr = 0.0007,
        min_epochs = 20
      ),
      
      # Phase 3: IWAE with medium K
      list(
        mode = "iwae",
        name = "IWAE-30",
        K = 30,
        truncation = 0,
        base_lr = 0.0005,
        min_epochs = 20
      ),
      
      # Phase 4: SUMO with low truncation
      list(
        mode = "sumo",
        name = "SUMO-2",
        K = 30,
        truncation = 2,
        base_lr = 0.0004,
        min_epochs = 20
      ),
      
      # Phase 5: SUMO with higher truncation
      list(
        mode = "sumo",
        name = "SUMO-4",
        K = 30,
        truncation = 4,
        base_lr = 0.0003,
        min_epochs = 20
      )
    )
  }
  
  # Log the phase configurations
  cat("Training with the following phases:\n")
  for (i in seq_along(phases)) {
    p <- phases[[i]]
    cat(sprintf("Phase %d: %s (K=%d, truncation=%d, lr=%.5f, min_epochs=%d)\n", 
                i, p$name, p$K, p$truncation, p$base_lr, p$min_epochs))
  }
  
  # Make sure save directory exists
  if (!dir.exists(save_dir)) {
    dir.create(save_dir, recursive = TRUE)
  }
  
  # Generate timestamp for this training run
  timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
  
  # Initialize asynchronous validation if enabled
  if (use_async_validation) {
    # Check if the module is available
    if (!file.exists("R/async_validation.R")) {
      warning("Async validation module not found. Falling back to synchronous validation.")
      use_async_validation <- FALSE
    } else {
      # Source module and set up async validation
      source("R/async_validation.R")
      setup_async_validation(device = val_device)
    }
  }
  
  # History tracking
  training_history <- list(
    train_loss = numeric(0),
    val_loss = numeric(0),
    mse = numeric(0),
    weight_var = numeric(0),
    learning_rate = numeric(0)
  )
  
  # Tracking variables
  current_phase <- 1
  phase_epoch <- 0
  total_epoch <- 0
  early_stop_counter <- 0
  best_val_loss <- Inf
  transition_epochs <- integer(0)
  last_validated_epoch <- 0
  
  # Create plateau detector if needed
  if (use_plateau_detection) {
    plateau_detector <- create_plateau_detector(
      patience = plateau_patience,
      min_delta = plateau_min_delta,
      window_size = plateau_window_size,
      rel_improvement_threshold = plateau_rel_improvement
    )
  }
  
  # Create variance tracker for monitoring
  weight_var_tracker <- create_variance_tracker(window_size = 20)
  
  # Save initial model
  best_model_path <- file.path(save_dir, paste0("best_model_", timestamp, ".pt"))
  torch_save(model$state_dict(), best_model_path)
  
  # Save phase configuration
  phase_config_path <- file.path(save_dir, paste0("phase_config_", timestamp, ".rds"))
  saveRDS(phases, phase_config_path)
  
  # Main training loop
  while (total_epoch < max_epochs && current_phase <= length(phases)) {
    total_epoch <- total_epoch + 1
    phase_epoch <- phase_epoch + 1
    
    # Get current phase configuration
    phase <- phases[[current_phase]]
    
    # Setup optimizer and scheduler if starting a new phase
    if (phase_epoch == 1) {
      cat("\n======== Starting phase:", phase$name, 
          "(K=", phase$K, "/truncation=", phase$truncation, ") ========\n")
      
      # Create optimizer
      optimizer <- optim_adamw(model$parameters, lr = phase$base_lr, weight_decay = 1e-4)
      
      # Estimate phase length for scheduler
      estimated_phase_length <- min(50, max_epochs - total_epoch + 1)
      
      # Create scheduler
      scheduler <- lr_one_cycle(
        optimizer,
        max_lr = phase$base_lr * 2,  # Peak at 2x base rate
        epochs = estimated_phase_length,
        steps_per_epoch = length(train_dl),
        pct_start = 0.3,  # Warm up for first 30% of phase
        div_factor = 10,  # Start at base_lr/10
        final_div_factor = 5,  # End at base_lr/50
        cycle_momentum = FALSE
      )
      
      # Add scheduler to phase config
      phase$scheduler <- scheduler
      phase$optimizer <- optimizer
    }
    
    # Get current learning rate
    current_lr <- optimizer$param_groups[[1]]$lr
    training_history$learning_rate <- c(training_history$learning_rate, current_lr)
    
    # Training epoch
    epoch_start_time <- Sys.time()
    train_results <- train_one_epoch(
      model, train_dl, optimizer, phase,
      gradient_accumulation = use_gradient_accumulation,
      target_batch_size = target_batch_size,
      device = train_device
    )
    
    # Track training results
    training_history$train_loss <- c(training_history$train_loss, train_results$loss)
    training_history$weight_var <- c(training_history$weight_var, train_results$weight_var)
    
    # Update weight variance tracker
    weight_var_tracker$update(train_results$weight_var)
    
    # Print training status
    epoch_time <- difftime(Sys.time(), epoch_start_time, units = "secs")
    cat(sprintf("Epoch %d (Phase: %s) - Train Loss: %.4f | LR: %.6f | Weight Var: %.6f | Time: %.2fs\n",
              total_epoch, phase$name, train_results$loss,
              current_lr, train_results$weight_var, 
              as.numeric(epoch_time)))
    
    # Submit for async validation or perform sync validation
    do_validation <- (total_epoch - last_validated_epoch >= val_freq) || 
                    (phase_epoch == 1) || 
                    (total_epoch == max_epochs)
    
    val_results <- NULL
    
    if (do_validation) {
      last_validated_epoch <- total_epoch
      
      if (use_async_validation) {
        # Submit for asynchronous validation
        validate_async(model, val_dl, phase, phase_epoch, total_epoch)
        
        # Check for any completed validations
        new_results <- check_validation_results()
        
        # Process new validation results
        for (id in names(new_results)) {
          result <- new_results[[id]]
          
          # Add validation result to history
          if (length(training_history$val_loss) < result$total_epoch) {
            # Fill in missing values with NA
            while (length(training_history$val_loss) < result$total_epoch - 1) {
              training_history$val_loss <- c(training_history$val_loss, NA)
              training_history$mse <- c(training_history$mse, NA)
            }
            
            # Add this result
            training_history$val_loss <- c(training_history$val_loss, result$result$loss)
            training_history$mse <- c(training_history$mse, result$result$mse)
            
            # Print validation results
            cat(sprintf("Validation for epoch %d - Val Loss: %.4f | MSE: %.4f\n",
                      result$total_epoch, result$result$loss, result$result$mse))
            
            # Update val_results if this is for the current epoch
            if (result$total_epoch == total_epoch) {
              val_results <- result$result
            }
          }
        }
      } else {
        # Synchronous validation
        val_results <- evaluate(
          model, val_dl, phase,
          device = train_device  # Use training device for sync validation
        )
        
        # Track validation results
        training_history$val_loss <- c(training_history$val_loss, val_results$loss)
        training_history$mse <- c(training_history$mse, val_results$mse)
        
        # Print validation results
        cat(sprintf("Validation for epoch %d - Val Loss: %.4f | MSE: %.4f\n",
                  total_epoch, val_results$loss, val_results$mse))
      }
    }
    
    # Save progress image if interval reached
    if (total_epoch %% save_progress_image_freq == 0) {
      # Try to save image and catch any errors
      tryCatch({
        # Get phase names from configuration
        phase_names <- sapply(phases, function(p) p$name)
        save_progress_image(training_history, transition_epochs, progress_image_path, phase_names)
        cat("Updated progress image at", progress_image_path, "\n")
      }, error = function(e) {
        warning("Error saving progress image: ", e$message)
      })
    }
    
    # Save checkpoint if interval reached
    if (total_epoch %% checkpoint_freq == 0) {
      checkpoint_path <- file.path(save_dir, paste0("checkpoint_epoch_", total_epoch, "_", timestamp, ".pt"))
      torch_save(model$state_dict(), checkpoint_path)
      cat("Saved checkpoint at epoch", total_epoch, "\n")
    }
    
    # Early stopping and plateau detection (only if we have validation results)
    if (!is.null(val_results)) {
      # Early stopping check
      if (val_results$loss < best_val_loss) {
        best_val_loss <- val_results$loss
        early_stop_counter <- 0
        
        # Save best model
        torch_save(model$state_dict(), best_model_path)
        cat("New best model saved (val_loss:", val_results$loss, ")\n")
        
      } else {
        early_stop_counter <- early_stop_counter + 1
        
        if (early_stopping && early_stop_counter >= early_stopping_patience && current_phase == length(phases)) {
          cat("Early stopping: No improvement in validation loss for", early_stopping_patience, "epochs in final phase.\n")
          break
        }
      }
      
      # Check for plateau and transition if needed
      if (use_plateau_detection && 
          phase_epoch >= phase$min_epochs && 
          current_phase < length(phases)) {
        
        is_plateau <- plateau_detector(val_results$loss, total_epoch, min_epochs_in_phase = phase$min_epochs)
        
        if (is_plateau) {
          # Move to next phase
          current_phase <- current_phase + 1
          phase_epoch <- 0
          early_stop_counter <- 0
          transition_epochs <- c(transition_epochs, total_epoch)
          
          # Get old and new phase names
          old_phase_name <- phases[[current_phase-1]]$name
          new_phase_name <- phases[[current_phase]]$name
          
          # Save phase transition checkpoint
          transition_path <- file.path(save_dir, paste0("transition_", old_phase_name, 
                                                      "_to_", new_phase_name, "_", timestamp, ".pt"))
          torch_save(model$state_dict(), transition_path)
          
          cat("\n======== PLATEAU DETECTED! ========\n")
          cat("Transitioning to phase:", new_phase_name, "\n")
        }
      }
    }
  }
  
  # Clean up async validation resources if used
  if (use_async_validation) {
    cleanup_validation()
  }
  
  # Save final model
  final_model_path <- file.path(save_dir, paste0("final_model_", timestamp, ".pt"))
  torch_save(model$state_dict(), final_model_path)
  cat("Saved final model\n")
  
  # Generate final training progress image
  phase_names <- sapply(phases, function(p) p$name)
  save_progress_image(training_history, transition_epochs, progress_image_path, phase_names)
  final_image_path <- file.path(save_dir, paste0("training_progress_", timestamp, ".png"))
  file.copy(progress_image_path, final_image_path, overwrite = TRUE)
  
  # Return history and transitions
  return(list(
    model = model,
    history = training_history,
    transition_epochs = transition_epochs,
    phases = phases,
    best_model_path = best_model_path,
    final_model_path = final_model_path
  ))
}
