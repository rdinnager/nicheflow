library(torch)
library(zeallot)  # For %<-% unpacking operator

# Evaluation function
evaluate <- function(model, dataloader, phase_config, device = "cuda") {
  model$eval()
  total_loss <- 0
  total_data_loss <- 0
  total_mse <- 0
  batch_count <- 0
  
  mode <- phase_config$mode
  K <- phase_config$K  # For evaluation, use a larger K
  truncation <- phase_config$truncation
  
  with_no_grad({
    coro::loop(for (batch in dataloader) {
      batch_count <- batch_count + 1
      
      # Move data to device
      input <- batch$env$to(device = device)
      mask <- batch$mask$to(device = device)
      spec <- batch$spec$to(device = device)
      
      # Forward pass
      c(x, z, means, log_vars, spec_lat) %<-% model(input, spec, K = K)
      
      # Loss calculation based on mode
      if (mode == "vae") {
        c(loss, ., ., ., mse) %<-% model$loss_function(
          z, x, mask, means, log_vars, spec_lat, 
          mode = mode, K = K
        )
        data_loss <- loss$item()
        approx_mse <- mse$item()
      } else if (mode == "iwae" || mode == "sumo") {
        c(loss, ., ., approx_mse, _) %<-% model$loss_function(
          z, x, mask, means, log_vars, spec_lat, 
          mode = mode, K = K, truncation = truncation
        )
        data_loss <- loss$item()
        approx_mse <- approx_mse$item()
      }
      
      # Update tracking
      total_loss <- total_loss + loss$item()
      total_data_loss <- total_data_loss + data_loss
      total_mse <- total_mse + approx_mse
    })
  })
  
  # Calculate averages
  avg_loss <- total_loss / batch_count
  avg_data_loss <- total_data_loss / batch_count
  avg_mse <- total_mse / batch_count
  
  return(list(
    loss = avg_loss, 
    data_loss = avg_data_loss,
    mse = avg_mse
  ))
}

# Function to evaluate model on test data and generate predictions
predict_with_model <- function(model, dataloader, K = 100, device = "cuda") {
  model$eval()
  all_reconstructions <- list()
  all_means <- list()
  all_logvars <- list()
  all_inputs <- list()
  all_masks <- list()
  all_species <- list()
  
  with_no_grad({
    coro::loop(for (batch in dataloader) {
      # Move data to device
      input <- batch$env$to(device = device)
      mask <- batch$mask$to(device = device)
      spec <- batch$spec$to(device = device)
      
      # Forward pass
      c(x, z, means, log_vars, spec_lat) %<-% model(input, spec, K = K)
      
      # For large K, we want to average predictions over all samples
      if (K > 1) {
        # Reshape z for decoder and expand species embedding
        batch_size <- means$size(1)
        latent_dim <- means$size(2)
        z_reshape <- z$reshape(c(-1, latent_dim))
        s_reshape <- spec_lat$unsqueeze(2)$expand(c(-1, K, -1))$reshape(c(-1, model$spec_embed_dim))
        
        # Get reconstructions for all K samples
        recons <- model$decode_latent(z_reshape, s_reshape)
        recons <- recons$reshape(c(batch_size, K, -1))
        
        # Average reconstructions across samples
        recon <- torch_mean(recons, dim = 2)
      } else {
        # For K=1, just use the single reconstruction
        z_0 <- z[, 1, ]
        recon <- model$decode_latent(z_0, spec_lat)
      }
      
      # Store results
      all_reconstructions[[length(all_reconstructions) + 1]] <- recon$cpu()
      all_means[[length(all_means) + 1]] <- means$cpu()
      all_logvars[[length(all_logvars) + 1]] <- log_vars$cpu()
      all_inputs[[length(all_inputs) + 1]] <- input$cpu()
      all_masks[[length(all_masks) + 1]] <- mask$cpu()
      all_species[[length(all_species) + 1]] <- spec$cpu()
    })
  })
  
  # Combine batches
  reconstructions <- torch_cat(all_reconstructions, dim = 1)
  means <- torch_cat(all_means, dim = 1)
  logvars <- torch_cat(all_logvars, dim = 1)
  inputs <- torch_cat(all_inputs, dim = 1)
  masks <- torch_cat(all_masks, dim = 1)
  species <- torch_cat(all_species, dim = 1)
  
  # Return results as list
  return(list(
    reconstructions = reconstructions,
    means = means,
    logvars = logvars,
    inputs = inputs,
    masks = masks,
    species = species
  ))
}
