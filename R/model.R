library(torch)

# VAE model definition
env_vae_mod <- nn_module("ProgressiveVAE",
  initialize = function(input_dim, n_spec, spec_embed_dim, latent_dim, 
                        breadth = 1024L, loggamma_init = -3) {
    self$latent_dim <- latent_dim
    self$input_dim <- input_dim
    self$n_spec <- n_spec
    self$spec_embed_dim <- spec_embed_dim
    
    # Encoder network
    self$encoder <- nn_sequential(
      nn_linear(input_dim + spec_embed_dim, breadth),
      nn_relu(),
      nn_linear(breadth + spec_embed_dim, breadth),
      nn_relu(),
      nn_linear(breadth + spec_embed_dim, breadth),
      nn_relu()
    )
    
    # Mean and log variance projections
    self$fc_mean <- nn_linear(breadth, latent_dim)
    self$fc_logvar <- nn_linear(breadth, latent_dim)
    
    # Decoder network
    self$decoder <- nn_sequential(
      nn_linear(latent_dim + spec_embed_dim, breadth),
      nn_relu(),
      nn_linear(breadth + spec_embed_dim, breadth),
      nn_relu(),
      nn_linear(breadth + spec_embed_dim, breadth),
      nn_relu(),
      nn_linear(breadth, input_dim)
    )
    
    # Species embedding layer
    self$species_embedder_mean <- nn_embedding(n_spec, spec_embed_dim)
    
    # Observation noise parameter (learned)
    self$loggamma <- nn_parameter(torch_tensor(loggamma_init))
    
    # Prevent the variance from becoming too small (optional bounds)
    self$min_loggamma <- -10
    self$max_loggamma <- 5
  },
  
  # Sample latent vectors with reparameterization trick
  sample_latent = function(mean, logvar, K = 1) {
    # Reparameterization trick
    std = torch_exp(0.5 * logvar)
    
    # For K samples, expand dimensions
    batch_size = mean$size(1)
    
    # Expand mu and std to [batch_size, K, latent_dim]
    mean_expanded = mean$unsqueeze(2)$expand(c(-1, K, -1))  # [B, K, D]
    std_expanded = std$unsqueeze(2)$expand(c(-1, K, -1))  # [B, K, D]
    
    # Sample epsilon from normal distribution
    eps = torch_randn_like(std_expanded)  # [B, K, D]
    
    # Get K samples for each input
    z = mean_expanded + eps * std_expanded  # [B, K, D]
    
    return(z)
  },
  
  # Log normal PDF calculation helper
  .log_normal_pdf = function(sample, mean, logvar) {
    # Compute log probability of sample under a normal distribution
    # sample: [B, K, D], mean: [B, K, D], logvar: [B, K, D]
    const <- -0.5 * torch_log(2 * torch_tensor(pi, device = mean$device))
    log_probs <- const - 0.5 * logvar - 0.5 * ((sample - mean) ^ 2 / torch_exp(logvar))
    # Sum over dimensions
    return(torch_sum(log_probs, dim = -1))  # [B, K]
  },
  
  # Compute all log probabilities needed for IWAE/SUMO
  compute_log_probs = function(x, z, mu, logvar, mask, spec_lat) {
    c(batch_size, K, latent_dim) %<-% z$shape
    
    # 1. Log prior: log p(z) - standard Gaussian prior
    log_prior <- self$.log_normal_pdf(z, 
                                      torch_zeros_like(z, device = z$device), 
                                      torch_zeros_like(z, device = z$device))  # [B, K]
    
    # 2. Log encoder prob: log q(z|x)
    log_q_z <- self$.log_normal_pdf(z,
                                    mu$unsqueeze(2), 
                                    logvar$unsqueeze(2))  # [B, K]
    
    # 3. Log decoder prob: log p(x|z)
    # Reshape z to process through decoder
    z_reshape <- z$reshape(c(-1, latent_dim))  # [B*K, D]
    
    # Expand species embeddings for each latent sample
    s_reshape <- spec_lat$unsqueeze(2)$expand(c(-1, K, -1))$reshape(c(-1, self$spec_embed_dim))
    
    # Get reconstructed x for each z
    x_recon <- self$decode_latent(z_reshape, s_reshape)  # [B*K, input_dim]
    
    # Reshape back to [B, K, input_dim]
    x_recon <- x_recon$reshape(c(batch_size, K, -1))
    
    # Compute log p(x|z) using Gaussian likelihood
    # Apply mask to handle missing values
    masked_x <- (x * mask)$unsqueeze(2)
    masked_x_recon <- x_recon * mask$unsqueeze(2)
    
    # Compute log probability with fixed observation variance
    log_p_x_given_z <- self$.log_normal_pdf(
      masked_x_recon, 
      masked_x, 
      torch_ones_like(masked_x) * self$loggamma
    )
    
    # Normalize by number of observed features to handle missing data
    observed_count <- torch_sum(mask, dim = -1, keepdim = TRUE)
    log_p_x_given_z <- (log_p_x_given_z / observed_count) * self$input_dim
    
    return(list(log_p_x_given_z, log_q_z, log_prior))
  },
  
  # Standard VAE loss with analytical KL
  vae_loss = function(x, z, mask, mean, logvar, spec_lat, lambda = 1, alpha = 0) {
    # Extract first sample for VAE training (K=1)
    z_0 <- z[, 1, ]  # [B, D]
    
    # Expand species embedding
    s_reshape <- spec_lat
    
    # Get reconstruction
    x_recon <- self$decode_latent(z_0, s_reshape)
    
    # Reconstruction loss (Gaussian negative log-likelihood)
    # Apply mask to handle missing values
    masked_diff <- (x - x_recon) * mask
    recon_loss_per_sample <- torch_sum(
      masked_diff^2 / torch_exp(self$loggamma), dim = -1
    )
    
    # Normalize by number of observed features
    observed_count <- torch_sum(mask, dim = -1)
    recon_loss_per_sample <- (recon_loss_per_sample / observed_count) * self$input_dim
    
    # Add log determinant term
    recon_loss_per_sample <- recon_loss_per_sample + 
                           self$input_dim * self$loggamma + 
                           self$input_dim * torch_log(2 * torch_tensor(pi, device = x$device))
    
    # KL divergence loss (analytical form for Gaussian)
    # KL = 0.5 * sum(mu^2 + sigma^2 - log(sigma^2) - 1)
    kl_loss <- 0.5 * torch_sum(
      torch_exp(logvar) + torch_square(mean) - logvar - 1, 
      dim = -1
    )
    
    # Species regularization (optional)
    if (lambda > 0) {
      # L2 and L1 regularization mixture
      spec_loss <- lambda * ((1 - alpha) * torch_sum(torch_square(spec_lat), dim = -1) + 
                             alpha * torch_sum(torch_abs(spec_lat), dim = -1))
    } else {
      spec_loss <- torch_zeros_like(kl_loss)
    }
    
    # Total loss
    loss <- torch_mean(recon_loss_per_sample + kl_loss + spec_loss)
    
    # Compute approximate MSE for monitoring
    with_no_grad({
      mse_approx <- torch_mean(torch_sum(masked_diff^2, dim = -1) / observed_count)
    })
    
    return(list(loss, recon_loss_per_sample, kl_loss, spec_loss, mse_approx))
  },
  
  # IWAE loss function
  iwae_loss = function(log_p_x_given_z, log_q_z, log_prior, K) {
    # Compute unnormalized weights - IMPORTANT: get the order right!
    log_weights <- log_p_x_given_z + log_prior - log_q_z  # [B, K]
    
    # Use the logsumexp trick for numerical stability
    # log(1/K * sum(exp(log_w_i))) = log(sum(exp(log_w_i))) - log(K)
    iwae_obj <- torch_logsumexp(log_weights, dim = 2) - torch_log(torch_tensor(K, device = log_weights$device))  # [B]
    
    # Negative because we want to maximize this objective
    return(-torch_mean(iwae_obj))
  },
  
  # SUMO loss function
  sumo_loss = function(log_weights, K, truncation = 3) {
    # Convert log weights to weights (for SUMO correction terms)
    weights <- torch_exp(log_weights - torch_logsumexp(log_weights, dim = 2, keepdim = TRUE))
    
    # IWAE term (first term in SUMO)
    iwae_term <- torch_logsumexp(log_weights, dim = 2) - torch_log(torch_tensor(K, device = log_weights$device))
    
    # Calculate mean weight for each batch element
    mean_weight <- torch_mean(weights, dim = 2, keepdim = TRUE)
    
    # Prepare to accumulate correction terms
    correction <- torch_zeros_like(iwae_term)
    
    # Normalized weights for power series
    normalized_weights <- weights / mean_weight - 1.0
    
    # Add correction terms from truncated series
    for (i in 1:truncation) {
      # Compute power term
      if (i == 1) {
        power_term <- torch_mean(normalized_weights, dim = 2)
      } else {
        power_term <- torch_mean(torch_pow(normalized_weights, i), dim = 2)
      }
      
      # Sign alternates based on term
      sign <- ifelse(i %% 2 == 1, 1, -1)
      
      # Add to correction
      correction <- correction + sign * power_term / i
    }
    
    # SUMO objective is IWAE term plus correction
    sumo_obj <- iwae_term + correction
    
    # Return negative for minimization
    return(-torch_mean(sumo_obj))
  },
  
  # Combined loss function that can handle all three modes
  loss_function = function(z, input, mask, mean, logvar, spec_lat, 
                         mode = "vae", K = 1, truncation = 3,
                         lambda = 1, alpha = 0) {
    # Different handling based on mode
    if (mode == "vae") {
      # Standard VAE loss with analytical KL
      c(loss, recon_loss, kl_loss, spec_loss, mse_approx) %<-% 
        self$vae_loss(input, z, mask, mean, logvar, spec_lat, lambda, alpha)
      
      return(list(loss, recon_loss, kl_loss, spec_loss, mse_approx))
      
    } else if (mode == "iwae" || mode == "sumo") {
      # Compute all log probabilities
      c(log_p_x_given_z, log_q_z, log_prior) %<-% 
        self$compute_log_probs(input, z, mean, logvar, mask, spec_lat)
      
      # Compute log weights (same for both IWAE and SUMO)
      log_weights <- log_p_x_given_z + log_prior - log_q_z
      
      # Species regularization
      spec_loss <- lambda * ((1 - alpha) * torch_sum(torch_square(spec_lat), dim = -1) + 
                           alpha * torch_sum(torch_abs(spec_lat), dim = -1))
      spec_loss <- torch_mean(spec_loss)
      
      # Apply appropriate estimator based on mode
      if (mode == "iwae") {
        main_loss <- self$iwae_loss(log_p_x_given_z, log_q_z, log_prior, K)
      } else {
        main_loss <- self$sumo_loss(log_weights, K, truncation)
      }
      
      # Monitor variance
      with_no_grad({
        # Estimate MSE for monitoring
        weights <- torch_exp(log_weights - torch_logsumexp(log_weights, dim = 2, keepdim = TRUE))
        
        # Estimate using importance weights
        approx_mse <- -torch_mean(torch_exp(self$loggamma) * 
                               torch_sum(weights * (log_p_x_given_z - log_prior + log_q_z), dim = 2))
        
        # Also calculate weight variance for monitoring
        weight_var <- torch_var(weights, dim = 2)$mean()
      })
      
      # Total loss
      loss <- main_loss + spec_loss
      
      return(list(loss, main_loss, spec_loss, approx_mse, weight_var))
    }
  },
  
  # Helper method to encode data
  encode = function(x, s = NULL) {
    if (is.null(s)) {
      spec_embedding <- torch_zeros(x$size()[[1]], self$spec_embed_dim, device = x$device)
    } else {
      if (s$size()[[1]] == 1 || s$size()[[1]] == self$n_spec) {
        s <- s$expand(c(x$size()[[1]], 1))
      }
      spec_embedding <- self$species_embedder_mean(s)
    }
    
    # Forward pass through encoder with residual connections for species embedding
    h <- torch_cat(list(x, spec_embedding), dim = 2)
    h <- self$encoder[[1]](h)
    h <- self$encoder[[2]](h)
    h <- torch_cat(list(h, spec_embedding), dim = 2)
    h <- self$encoder[[3]](h)
    h <- self$encoder[[4]](h)
    h <- torch_cat(list(h, spec_embedding), dim = 2)
    h <- self$encoder[[5]](h)
    h <- self$encoder[[6]](h)
    
    # Get mu and logvar
    mu <- self$fc_mean(h)
    logvar <- self$fc_logvar(h)
    
    return(list(mu, logvar))
  },
  
  # Helper method to decode latent vectors
  decode_latent = function(z, s = NULL) {
    if (is.null(s)) {
      spec_embedding <- torch_zeros(z$size()[[1]], self$spec_embed_dim, device = z$device)
    } else {
      spec_embedding <- s
    }
    
    # Forward pass through decoder with residual connections for species embedding
    h <- torch_cat(list(z, spec_embedding), dim = 2)
    h <- self$decoder[[1]](h)
    h <- self$decoder[[2]](h)
    h <- torch_cat(list(h, spec_embedding), dim = 2)
    h <- self$decoder[[3]](h)
    h <- self$decoder[[4]](h)
    h <- torch_cat(list(h, spec_embedding), dim = 2)
    h <- self$decoder[[5]](h)
    h <- self$decoder[[6]](h)
    x_recon <- self$decoder[[7]](h)
    
    return(x_recon)
  },
  
  # Main forward method
  forward = function(x, s = NULL, K = 1) {
    if (is.null(s)) {
      spec_embedding <- torch_zeros(x$size()[[1]], self$spec_embed_dim, device = x$device)
    } else {
      spec_embedding <- self$species_embedder_mean(s)
    }
    
    # Encode input to get distribution parameters
    c(means, log_vars) %<-% self$encode(x, s)
    
    # Sample K latent vectors per input
    z <- self$sample_latent(means, log_vars, K)
    
    # Enforce bounds on loggamma if needed
    self$loggamma$data <- torch_clamp(
      self$loggamma, 
      min = self$min_loggamma,
      max = self$max_loggamma
    )
    
    return(list(x, z, means, log_vars, spec_embedding))
  }
)
