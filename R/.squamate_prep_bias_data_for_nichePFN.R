library(tidyverse)
library(tidymodels)
library(sf)

source("R/utils.R")

set.seed(756348)

bias_files <- list.files("output/squamate_samples_w_bias", full.names = TRUE)
bias_data <- map(bias_files, load_bias_pnts, .progress = TRUE) |>
  list_rbind()

latent_codes <- read_rds("output/squamate_env_latent_codes_for_stage2_2.rds") |>
  distinct(species, .keep_all = TRUE)

bias_data <- bias_data |>
  left_join(latent_codes |>
              select(spec = species, starts_with("L")))

bias_data_split <- initial_validation_split(bias_data, c(0.9, 0.05))

write_rds(bias_data_split, "output/bias_data_split_for_nichePFN.rds")
write_rds(training(bias_data_split), "output/bias_data_for_nichePFN_train.rds")
write_rds(validation(bias_data_split), "output/bias_data_for_nichePFN_validation.rds")
write_rds(testing(bias_data_split), "output/bias_data_for_nichePFN_test.rds")
