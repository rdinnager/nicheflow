library(torch)
library(tidyverse)
library(deSolve)
library(cli)
library(tidymodels)
library(probably)
library(sf)

load_nichencoder_flow_rectified <- function(checkpoint_dir = "output/checkpoints/squamate_env_model_fixed_rectified_flow_stage2_distill_7d", 
                                            device = "cuda") {
  old_opt <- options(torch.serialization_version = 2)
  checkpoint_files <- list.files(checkpoint_dir, full.names = TRUE, pattern = ".pt")
  checkpoints <- file.info(checkpoint_files)
  most_recent <- which.max(checkpoints$mtime)
  checkpoint <- checkpoint_files[most_recent]
  flow_2 <- torch_load(checkpoint)
  flow_2 <- flow_2$to(device = device)
  options(old_opt)
  flow_2
}

load_geode_flow <- function(checkpoint_dir = "output/checkpoints/geo_env_model_3", device = "cuda") {
  files_geo <- list.files(checkpoint_dir, full.names = TRUE, pattern = ".pt")
  file_info_geo <- file.info(files_geo)
  latest_geo <- which.max(file_info_geo$mtime)
  mod_file_geo <- files_geo[latest_geo]
  geode <- torch_load(mod_file_geo)
  geode <- geode$to(device = device)
  attr(geode, "scaling") <- read_rds("output/chelsa_geo_scaling.rds")
  geode
}

load_geode_flow_v2 <- function(checkpoint_dir = "output/checkpoints/geo_env_model_v2_stage2_distill_2", device = "cuda") {
  files_geo <- list.files(checkpoint_dir, full.names = TRUE, pattern = ".pt")
  file_info_geo <- file.info(files_geo)
  latest_geo <- which.max(file_info_geo$mtime)
  mod_file_geo <- files_geo[latest_geo]
  geode <- torch_load(mod_file_geo)
  geode <- geode$to(device = device)
  attr(geode, "scaling") <- read_rds("output/chelsa_geo_scaling_v2.rds")
  geode
}

load_nichencoder_vae <- function(checkpoint_dir = "data/env_vae_trained_fixed2_alpha_0.5_32d.to", 
                                 device = "cuda") {
  env_vae <- torch_load(checkpoint_dir)
  env_vae <- env_vae$to(device = device)
  attr(env_vae, "active_dims") <- read_rds("output/squamate_env_mani2.rds")
  attr(env_vae, "scaling") <- read_rds("output/squamate_env_scaling2.rds")
  attr(env_vae, "species") <- read_csv("output/training_species.csv")
  env_vae
}

run_nichencoder <- function(latent, env_vae = NULL, flow_2 = NULL, n = 10000, device = "cuda", gradient = FALSE, recode = FALSE, fast = TRUE) {
  
  if(!is.matrix(latent)) {
    stop("latent must be a matrix")
  }
  if(dim(latent)[2] != env_vae$spec_embed_dim) {
    stop("latent must have ", env_vae$spec_embed_dim, " columns")
  }
  
  if(is.null(env_vae)) {
    env_vae <- load_nichencoder_vae(device = device)
  }
  
  if(is.null(flow_2)) {
    flow_2 <- load_geode_flow(device = device)
  }
  
  if(is.null(dim(latent)) || dim(latent)[1] == 1) {
    latent <- matrix(as.vector(latent), nrow = n, ncol = length(latent), byrow = TRUE)
  }
  
  scaling <- attr(env_vae, "scaling")
  active_dims <- attr(env_vae, "active_dims")
  
  if(gradient) {
    latent_tens <- torch_tensor(latent, device = device)
    samp1 <- torch_randn(n, length(active_dims), device = device)
    if(fast) { 
      samp2 <- samp1 + flow_2(samp1, t = torch_zeros(samp1$size()[[1]], 1, device = device), latent_tens)
      samp <- torch_zeros(n, env_vae$latent_dim, device = device)
      samp[ , active_dims] <- samp2
    } else {
      samp2 <- flow_2$sample_trajectory(initial_vals = samp1, spec_vals = latent_tens, steps = 2) 
      samp <- matrix(0, ncol = env_vae$latent_dim, nrow = n)
      samp[ , active_dims] <- samp2$trajectories[2, , ]
      samp <- torch_tensor(samp, device = device)
    }
    
    decoded <- env_vae$decoder(z = samp, s = latent_tens)
    if(recode) {
      reencoded <- env_vae$encoder(y = decoded, s = latent_tens)
      resamp <- torch_randn_like(reencoded$logvars) * torch_exp(reencoded$logvars) + reencoded$means
      decoded <- env_vae$decoder(z = resamp, s = latent_tens)
    }
  } else {
    with_no_grad({
      latent_tens <- torch_tensor(latent, device = device)
      samp1 <- torch_randn(n, length(active_dims), device = device)
      if(fast) {
        samp2 <- samp1 + flow_2(samp1, t = torch_zeros(samp1$size()[[1]], 1, device = "cuda"), latent_tens)
        samp <- torch_zeros(n, env_vae$latent_dim, device = device)
        samp[ , active_dims] <- samp2
      } else {
        samp2 <- flow_2$sample_trajectory(initial_vals = samp1, spec_vals = latent_tens, steps = 2) 
        samp <- matrix(0, ncol = env_vae$latent_dim, nrow = n)
        samp[ , active_dims] <- samp2$trajectories[2, , ]
        samp <- torch_tensor(samp, device = device)
      }
      
      decoded <- env_vae$decoder(z = samp, s = latent_tens)
      if(recode) {
        reencoded <- env_vae$encoder(y = decoded, s = latent_tens)
        resamp <- torch_randn_like(reencoded$logvars) * torch_exp(reencoded$logvars) + reencoded$means
        decoded <- env_vae$decoder(z = resamp, s = latent_tens)
      }
    })
  }
  env_pred <- as.matrix(decoded$cpu())
  colnames(env_pred) <- names(scaling$means)
  env_pred <- t((t(env_pred) * scaling$sd) + scaling$means)
  return(env_pred)
}
