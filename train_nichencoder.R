#| requires:
#|     - file: data
#|       target-type: link
#|     - file: output

library(torch)
library(dagnn)
library(tidyverse)
library(tidymodels)
library(zeallot)
library(patchwork)
library(conflicted)

conflict_prefer("select", "dplyr")
conflict_prefer("filter", "dplyr")

source("R/models.R")

set.seed(536567678)

train_file <- "data/nichencoder_train.csv"
train <- read_csv(train_file)

val_file <- "data/nichencoder_val.csv"
val <- read_csv(val_file)

species_names <- train$species
species_train <- as.integer(as.numeric(as.factor(species_names)))
species_inds <- tibble(species = species_names, index = species_train) |>
  distinct()

## only validate on species that occur in training data for this model
val <- val |>
  filter(species %in% species_names)

val_reshape <- val |>
  left_join(species_inds) |>
  group_by(species) |>
  summarise(env = list(as.matrix(pick(c(-X, -Y, -starts_with("na_ind_"), -index)))),
            na_mask = list(as.matrix(pick(c(starts_with("na_ind_"), -na_ind_X, -na_ind_Y)))),
            spec = list(as.matrix(pick(c(index)))))

na_mask_val <- map(val_reshape$na_mask, ~ 1 - .x)
env_val <- map2(val_reshape$env, na_mask_val,
                ~ {.x[.y == 0] <- 0; .x})
species_val <- val_reshape$spec

# species_val <- val |>
#   left_join(species_inds) |>
#   pull(index)

env_train <- train |>
  select(-species, -X, -Y, -starts_with("na_ind_")) |>
  as.matrix()
na_mask_train <- train |>
  select(starts_with("na_ind_"), -na_ind_X, -na_ind_Y) |>
  as.matrix()
  
na_mask_train <- 1 - na_mask_train
env_train[na_mask_train == 0] <- 0

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
                       })

env_val_dataset <- dataset(name = "env_ds",
                           initialize = function(env, mask, spec) {
                             self$env <- map(env, torch_tensor)
                             self$mask <- map(mask, torch_tensor)
                             self$spec <- map(spec, ~ torch_tensor(.x[ , 1]))
                           },
                           .getbatch = function(i) {
                             list(env = self$env[i], mask = self$mask[i], spec = self$spec[i])
                           },
                           .length = function() {
                             length(self$env)
                           })

batch_size <- 1000000 / 50
batch_size_val <- 50

train_ds <- env_dataset(env_train, na_mask_train, species_train)
train_dl <- dataloader(train_ds, batch_size, shuffle = TRUE)
n_batch <- length(train_dl)

val_ds <- env_val_dataset(env_val, na_mask_val, species_val)
val_dl <- dataloader(val_ds, batch_size_val, shuffle = TRUE)

input_dim <- ncol(env_train)
n_spec <- n_distinct(species_train)
spec_embed_dim <- 256L
latent_dim <- 16L
breadth <- 1024L
alpha <- 0
K <- 50
lambda <- 1

nchencdr <- nichencoder(input_dim, n_spec, spec_embed_dim, latent_dim, breadth, loggamma_init = -3)
nchencdr$cuda()

# env_vae <- env_vae_mod(input_dim, n_spec, spec_embed_dim, latent_dim, breadth, loggamma_init = -3)
# env_vae <- env_vae$cuda()

num_epochs <- 1000

lr <- 0.002
optimizer <- optim_adamw(nchencdr$parameters, lr = lr)
scheduler <- lr_one_cycle(optimizer, max_lr = lr,
                          epochs = num_epochs, steps_per_epoch = length(train_dl),
                          cycle_momentum = FALSE)

epoch_times <- numeric(num_epochs * length(train_dl))
i <- 0
#b <- train_dl$.iter()$.next()

# zz <- file("output/logs/squamate_env_model_fixed2_run_1.txt", open = "wt")
# sink(zz, type = "output", split = TRUE)
# sink(zz, type = "message")
mse <- torch_tensor(-99)

for (epoch in 1:num_epochs) {
  
  epoch_time <- Sys.time()
  batchnum <- 0
  coro::loop(for (b in train_dl) {
    
    batchnum <- batchnum + 1
    i <- i + 1
    optimizer$zero_grad()
    
    c(x, z, s, means, log_vars, iwae_loss, spec_loss) %<-% nchencdr(b$env$cuda(),
                                                                    b$spec$cuda(),
                                                                    na_mask = b$mask$cuda(),
                                                                    K = K)
    
    
    loss <- iwae_loss + spec_loss
   
    
      cat("Step:", i,
          "\nEpoch:", epoch,
          "\nbatch:", batchnum, "of ", n_batch,
          "\nloss:", as.numeric(loss$cpu()),
          "\nIWAE loss:", as.numeric(iwae_loss$cpu()),
          "\nSpecies loss:", as.numeric(spec_loss$cpu()),
          "\nValidation MSE:", as.numeric(mse$cpu()),
          "\nloggamma:", as.numeric(nchencdr$vae$loggamma$cpu()),
          "\ncond. active dims:", as.numeric((torch_exp(log_vars)$mean(dim = 1L) < 0.5)$sum()$cpu()),
          "\nSpecies latent variance:", as.numeric(torch_var(s, dim = 1)$mean()$cpu()),
          "\n\n")
      
    loss$backward()
    optimizer$step()
    scheduler$step()
  })
  
  ## do validation here
  gc()
  cuda_empty_cache()
  
  v <- coro::collect(val_dl, 1)[[1]]
  if(coro::is_exhausted(v)) {
    val_dl <- dataloader(val_ds, batch_size, shuffle = TRUE)
    v <- coro::collect(val_dl, 1)[[1]]
  }
  
  with_no_grad({
    x <- torch_cat(v$env)$cuda()
    recon <- nchencdr$reconstruct_from_species_index(x, 
                                                     torch_cat(v$spec)$cuda(),
                                                     K = 1)
    
    mse <- torch_mean(torch_square(x - recon))
  })
  
  prob_species <- function(env_spec, spec) {
    with_no_grad({
      dim_choose <- sample.int(30, 2)
      env_dat <- as.matrix(env_spec)[ , dim_choose]
      range_1 <- range(env_dat[ , 1])
      range_2 <- range(env_dat[ , 2])
      range_1 <- range_1 + c(-0.3*diff(range_1), 0.3*diff(range_1))
      range_2 <- range_2 + c(-0.3*diff(range_2), 0.3*diff(range_2))
      test_grid <- expand_grid(var_1 = seq(range_1[1], range_1[2], length.out = 100),
                               var_2 = seq(range_2[1], range_2[2], length.out = 100)) |>
        as.matrix() |>
        torch_tensor(device = "cuda")
      x_means <- torch_mean(torch_tensor(env_spec, device = "cuda"), dim = 1, keepdim = TRUE)$expand(c(nrow(test_grid), -1))$clone()
      x_means[ , dim_choose] <- test_grid
      
      spec_expanded <- spec[1]$expand(c(x_means$shape[1]))
      iwae_losses <- nchencdr$iwae_loss_from_species_index(x_means,
                                                           spec_expanded$cuda(),
                                                           torch_ones_like(x_means),
                                                           K = 100)
      
      plot_df <- as.matrix(test_grid$cpu()) |>
        as.data.frame() |>
        mutate(prob = as.numeric(iwae_losses$cpu()))
    })
    
    p <- ggplot(plot_df, aes(V1, V2)) +
      geom_raster(aes(fill = prob)) +
      geom_point(data = as.data.frame(env_dat) |> setNames(c("V1", "V2")),
                 alpha = 0.25) +
      scale_fill_viridis_c() +
      theme_minimal()
    gc()
    cuda_empty_cache()
    p
  }
  
  spec_plots <- map2(v$env[1:9], v$spec[1:9],
                     prob_species, .progress = TRUE)
  
  ps <- wrap_plots(spec_plots, ncol = 3, nrow = 3)
  ggsave("prob_progress_temp.png", ps, width = 16, height = 16)
  plot(ps)
  
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
env_vae <- torch_load("output/testing/env_vae_trained_test_256d_iwae.to")
#env_vae$load_state_dict(env_vae2$state_dict())
env_vae <- env_vae$cuda()





x_recon <- env_vae$decode(z, spec_lat)