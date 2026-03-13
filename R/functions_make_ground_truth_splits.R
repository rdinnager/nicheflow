#' .. content for \description{} (no empty lines) ..
#'
#' .. content for \details{} ..
#'
#' @title
#' @param all_ground_truth_samples
#' @param split_ratio
#' @return
#' @author rdinnage
#' @export
make_ground_truth_splits <- function(all_ground_truth_samples, split_ratio =
                                     c(0.8, 0.1)) {

  squamate_split_spec <- group_initial_validation_split(all_ground_truth_samples, species, prop = split_ratio)
  squamate_train_spec <- training(squamate_split_spec)
  squamate_val_spec <- validation(squamate_split_spec)
  squamate_test_spec <- testing(squamate_split_spec)
  
  squamate_split <- initial_validation_split(squamate_train_spec, prop = split_ratio,
                                             strata = species, pool = 0)
  
  squamate_train <- training(squamate_split)
  squamate_val <- validation(squamate_split) |>
    mutate(type = "within_species") |>
    bind_rows(squamate_val_spec |>
                mutate(type = "between_species"))
  squamate_test <- testing(squamate_split) |>
    mutate(type = "within_species") |>
    bind_rows(squamate_test_spec |>
                mutate(type = "between_species"))
  
  ### Add some rare species
  squamate_split_spec2 <- group_initial_validation_split(squamate_train, species, prop = split_ratio)
  squamate_train <- training(squamate_split_spec2)
  squamate_test_spec2 <- testing(squamate_split_spec2)
  squamate_split2 <- initial_split(squamate_test_spec2, prop = 0.01,
                                   strata = species, pool = 0)
  squamate_extra_train <- training(squamate_split2)
  
  squamate_val_spec2 <- validation(squamate_split_spec2)
  squamate_split3 <- initial_split(squamate_val_spec2, prop = 0.01,
                                   strata = species, pool = 0)
  squamate_extra_train2 <- training(squamate_split3)
  
  squamate_train <- bind_rows(squamate_train, squamate_extra_train, squamate_extra_train2)
  squamate_test <- bind_rows(squamate_test, testing(squamate_split2))
  squamate_val <- bind_rows(squamate_val, testing(squamate_split3))
  
  list(train = squamate_train, val = squamate_val, test = squamate_test)

}
