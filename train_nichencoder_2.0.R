#| requires:
#|     - file: data
#|       target-type: link
#|     - file: output

library(torch)
library(dagnn)
library(tidyverse)
library(tidymodels)
library(zeallot)
library(abind)
library(coro)
library(conflicted)

conflict_prefer("select", "dplyr")
conflict_prefer("filter", "dplyr")

set.seed(536567678)

train_file <- "data/nichencoder_train.csv"
train <- read_csv(train_file)

val_file <- "data/nichencoder_val.csv"
val <- read_csv(val_file)

train_reshape <- train |>
  group_by(species) |>
  summarise(env = list(as.matrix(pick(c(-X, -Y, -starts_with("na_ind_"))))),
            na_mask = list(as.matrix(pick(c(starts_with("na_ind_"), -na_ind_X, -na_ind_Y)))))

na_mask_train <- map(train_reshape$na_mask, ~ 1 - .x)
env_train <- map2(train_reshape$env, na_mask_train,
                  ~ {.x[.y == 0] <- 0; .x})

max_pnts <- 800
cols <- ncol(train_reshape$env[[1]])
## append zeros so every species has max_pnts (zero padding for transformer)
env_train <- map_if(env_train, ~ nrow(.x) < max_pnts, 
                    ~ rbind(.x, matrix(0, nrow = max_pnts - nrow(.x), ncol = cols)))
padding_mask_train <- do.call(rbind, map(env_train, ~ rowSums(.x) == 0)) |>
  torch_tensor()
env_train <- env_train |>
  abind(along = 0)
species_names <- train_reshape$species
species_train <- as.integer(as.numeric(as.factor(species_names)))

rm(train, train_reshape)
gc()

## only validate on species that occur in training data for this model
val <- val |>
  filter(species %in% species_names)

val_reshape <- val |>
  group_by(species) |>
  summarise(env = list(as.matrix(pick(c(-X, -Y, -starts_with("na_ind_"))))),
            na_mask = list(as.matrix(pick(c(starts_with("na_ind_"), -na_ind_X, -na_ind_Y)))))

na_mask_val <- map(val_reshape$na_mask, ~ 1 - .x)
env_val <- map2(val_reshape$env, na_mask_val,
                  ~ {.x[.y == 0] <- 0; .x})

species_val <- as.integer(as.numeric(as.factor(val_reshape$species)))

env_dataset <- dataset(name = "env_ds",
                       initialize = function(env, na_mask, padding_mask, spec) {
                         self$env <- torch_tensor(env)
                         self$na_mask <- map(na_mask, torch_tensor)
                         self$padding_mask <- padding_mask
                         self$spec <- torch_tensor(spec)
                       },
                       .getbatch = function(i) {
                         list(env = self$env[i, , ], na_mask = self$na_mask[i], 
                              padding_mask = self$padding_mask[i, ], spec = self$spec[i])
                       },
                       .length = function() {
                         self$env$size()[[1]]
                       })

env_val_dataset <- dataset(name = "env_ds",
                       initialize = function(env, mask, spec) {
                         self$env <- map(env, torch_tensor)
                         self$mask <- map(mask, torch_tensor)
                         self$spec <- torch_tensor(spec)
                       },
                       .getbatch = function(i) {
                         list(env = self$env[i, , ], mask = self$mask[i], spec = self$spec[i])
                       },
                       .length = function() {
                         length(self$env)
                       })

batch_size <- 16

train_ds <- env_dataset(env_train, na_mask_train, padding_mask_train, species_train)
train_dl <- dataloader(train_ds, batch_size, shuffle = TRUE)
n_batch <- length(train_dl)

val_ds <- env_val_dataset(env_val, na_mask_val, species_val)
val_dl <- dataloader(val_ds, batch_size / 2, shuffle = TRUE)

#test <- train_dl$.iter()$.next()
source("R/models.R")

input_dim <- dim(env_train)[3]
spec_embed_dim <- 128L
latent_dim <- 16L
breadth <- 1024L
alpha <- 0
K <- 25
lambda <- 1
transformer_embed_dim <- 128L 
transformer_output_dim <- 512L
n_blocks <- 12L 
num_heads <- 4L 
dropout_prob <- 0.1
loggamma_init <- -1

net <- nichencoder_trainer(input_dim, 
                           spec_embed_dim = spec_embed_dim,
                           latent_dim = latent_dim,
                           breadth = breadth,
                           loggamma_init = loggamma_init, 
                           transformer_embed_dim = transformer_embed_dim,
                           transformer_output_dim = transformer_output_dim,
                           n_blocks = n_blocks,
                           num_heads = num_heads,
                           dropout_prob = dropout_prob)
net$cuda()
with_no_grad({
  iwalk(net$parameters,
      ~ {
          if(str_detect(.y, "weight") & str_detect(.y, "layernorm", negate = TRUE)) {
            nn_init_xavier_normal_(.x, 0.2)
          }
          if(str_detect(.y, "bias")) {
            nn_init_zeros_(.x)
          }
        })
})

with_no_grad({
  # Initialize bias of the variance network to start with low values
  nn_init_constant_(net$spec_embedder_var$bias, -3.0)
  
  # Optional: Initialize weights small to reduce impact of transformer outputs
  nn_init_normal_(net$spec_embedder_var$weight, mean = 0.0, std = 0.01)
})

num_epochs <- 200

epoch_times <- numeric(num_epochs * length(train_dl))
i <- 0
#b <- train_dl$.iter()$.next()
accum_steps <- 16  # Number of batches to accumulate (adjust as needed)
effective_batch_size <- batch_size * accum_steps
steps_per_epoch <- ceiling(length(train_dl) / accum_steps)
lr <- 1e-3
optimizer <- optim_adamw(net$parameters, lr = lr)
scheduler <- lr_one_cycle(optimizer, max_lr = lr,
                          epochs = num_epochs, 
                          steps_per_epoch = steps_per_epoch,
                          cycle_momentum = FALSE)


for (epoch in 1:num_epochs) {
  
  epoch_time <- Sys.time()
  batchnum <- 0
  accumulated_batches <- 0
  optimizer$zero_grad()
  coro::loop(for (b in train_dl) {
    
    batchnum <- batchnum + 1
    i <- i + 1
    
    c(input, z, means, log_vars, s, mu_s, logvar_s, na_mask) %<-% net(b$env$cuda(), b$padding_mask$cuda(), torch_cat(b$na_mask)$cuda(), K = K)
    
    c(iwae_loss, log_prior) %<-% net$nichencoder$loss_function(z, input, na_mask, means, log_vars,
                                                             s, K = K, lambda = lambda,
                                                             alpha = alpha)
    
    kl_loss_s <- 0.5 * torch_sum(
      torch_exp(logvar_s) +    # variance terms
        torch_square(mu_s) -     # squared means
        1 -                    # from standard normal variance
        logvar_s,                # log-variance terms
      dim = 2L               # sum across latent dimensions
    )$mean()                # average over batch
    
    loss <- (iwae_loss + kl_loss_s) / accum_steps
    
    # Backward pass
    loss$backward()
    
    # Update accumulated_batches counter
    accumulated_batches <- accumulated_batches + 1
   
    # c(reconstruction, input, mean_spec, means, log_vars) %<-% env_vae(b$env$cuda(), b$spec$cuda())
    # c(loss, reconstruction_loss, kl_loss, kl_loss_spec) %<-% env_vae$loss_function(reconstruction, input, 
    #                                                                                b$mask$cuda(), means, 
    #                                                                                log_vars, mean_spec, 
    #                                                                                alpha = alpha)
    if (accumulated_batches == accum_steps || batchnum == n_batch) {
      cat("Step:", i,
          "\nEpoch:", epoch,
          "\nbatch:", batchnum, "of ", n_batch,
          "\nloss:", as.numeric(loss$cpu() * accum_steps),
          "\nIWAE loss:", as.numeric(iwae_loss$cpu()),
          "\nIWAE prior loss:", as.numeric(log_prior$cpu()),
          "\nspecies KL loss:", as.numeric(kl_loss_s$cpu()),
          "\nloggamma:", as.numeric(net$nichencoder$loggamma$cpu()),
          "\ncond. active dims:", as.numeric((torch_exp(log_vars)$mean(dim = 1L) < 0.5)$sum()$cpu()),
          "\nspecies cond. active dims:", as.numeric((torch_exp(logvar_s)$mean(dim = 1L) < 0.5)$sum()$cpu()),
          "\nSpecies latent variance:", as.numeric(torch_var(mu_s, dim = 1)$mean()$cpu()),
          "\n\n")
      
      #loss$backward()
      nn_utils_clip_grad_norm_(net$parameters, max_norm = 1.0)
      optimizer$step()
      scheduler$step()
      
      optimizer$zero_grad()
      # Reset accumulation counter
      accumulated_batches <- 0
      cuda_empty_cache()
    }
  })
  
  time <- Sys.time() - epoch_time
  epoch_times[i] <- time
  cat("Estimated time remaining: ")
  print(lubridate::as.duration(mean(epoch_times[epoch_times > 0]) * (num_epochs - epoch)))
  cat("\n")
  
}

#options(torch.serialization_version = 2)
torch_save(env_vae, "output/testing/env_vae_trained_test_256d_iwae.to")

#sink()

##### testing validation
env_vae2 <- torch_load("output/testing/env_vae_trained_test_256d_iwae.to")
env_vae$load_state_dict(env_vae2$state_dict())
env_vae <- env_vae$cuda()

v <- coro::collect(val_dl, 1)
if(coro::is_exhausted(v)) {
  val_dl <- dataloader(val_ds, batch_size, shuffle = TRUE)
  v <- coro::collect(val_dl, 1)
}
c(input, z, means, log_vars, spec_lat) %<-% env_vae(v[[1]]$env$cuda(), v[[1]]$spec$cuda(), K = 100)
x_recon <- env_vae$decode(z, spec_lat)