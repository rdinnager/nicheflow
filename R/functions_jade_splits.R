#' JADE Sample Splitting Functions
#'
#' Functions for creating GenAISDM-style train/val/test splits from
#' JADE-corrected environmental samples, including between-species
#' (zero-shot), within-species, and few-shot splits.
#'
#' @author rdinnage


#' Count samples per species and filter by minimum
#'
#' @param jade_samples Data frame of JADE samples with 'species' column
#' @param min_samples Minimum samples required to include a species (default 100)
#' @return Tibble with species, n_samples columns (filtered)
#' @export
compute_species_counts <- function(jade_samples, min_samples = 100) {
  counts <- jade_samples |>
    dplyr::count(species, name = "n_samples") |>
    dplyr::arrange(dplyr::desc(n_samples))

  n_total_species <- nrow(counts)
  counts <- counts |> dplyr::filter(n_samples >= min_samples)

  message("Species counts: ", n_total_species, " total, ",
          nrow(counts), " with >= ", min_samples, " samples (",
          n_total_species - nrow(counts), " removed)")
  message("  Sample range: ", min(counts$n_samples), " - ", max(counts$n_samples))
  message("  Total samples in qualifying species: ",
          format(sum(counts$n_samples), big.mark = ","))

  counts
}


#' Assign species to split roles
#'
#' Randomly assigns species to train_species, zeroshot, or fewshot roles.
#'
#' @param species_counts Tibble from compute_species_counts()
#' @param zeroshot_frac Fraction of species for zero-shot holdout (default 0.10)
#' @param fewshot_frac Fraction of species for few-shot (default 0.10)
#' @param seed Random seed for reproducibility (default 42)
#' @return Tibble with species, n_samples, split_role columns
#' @export
assign_species_splits <- function(species_counts,
                                  zeroshot_frac = 0.10,
                                  fewshot_frac = 0.10,
                                  seed = 42) {
  set.seed(seed)
  n <- nrow(species_counts)

  # Shuffle and assign
  shuffled <- species_counts[sample(n), ]
  n_zeroshot <- round(n * zeroshot_frac)
  n_fewshot <- round(n * fewshot_frac)

  shuffled$split_role <- "train_species"
  shuffled$split_role[seq_len(n_zeroshot)] <- "zeroshot"
  shuffled$split_role[(n_zeroshot + 1):(n_zeroshot + n_fewshot)] <- "fewshot"

  message("Species split assignments:")
  message("  train_species: ", sum(shuffled$split_role == "train_species"))
  message("  zeroshot:      ", sum(shuffled$split_role == "zeroshot"))
  message("  fewshot:       ", sum(shuffled$split_role == "fewshot"))

  shuffled
}


#' Create multi-stage train/val/test splits
#'
#' Implements GenAISDM-style splitting:
#' 1. Zero-shot species: ALL samples to test
#' 2. Few-shot species: fewshot_n samples to train, rest to test
#' 3. Train species: cap at max_samples, split 80/10/10 within-species
#'
#' @param jade_samples Data frame of JADE samples
#' @param split_assignments Tibble from assign_species_splits()
#' @param max_samples_per_species Cap per species before splitting (default 1000)
#' @param within_train_frac Within-species train fraction (default 0.80)
#' @param within_val_frac Within-species val fraction (default 0.10)
#' @param fewshot_n Number of samples per few-shot species in train (default 8)
#' @param seed Random seed (default 42)
#' @return List with train, val, test data frames, each with split_type column
#' @export
create_jade_splits <- function(jade_samples, split_assignments,
                               max_samples_per_species = 1000,
                               within_train_frac = 0.80,
                               within_val_frac = 0.10,
                               fewshot_n = 8,
                               seed = 42) {
  set.seed(seed)

  # Filter to qualifying species only
  qualifying <- split_assignments$species
  jade_filtered <- jade_samples |>
    dplyr::filter(species %in% qualifying) |>
    dplyr::left_join(
      split_assignments |> dplyr::select(species, split_role),
      by = "species"
    )

  # --- 1. Zero-shot species: ALL samples go to test ---
  zeroshot_data <- jade_filtered |>
    dplyr::filter(split_role == "zeroshot") |>
    dplyr::mutate(split_type = "zeroshot") |>
    dplyr::select(-split_role)

  # --- 2. Few-shot species: fewshot_n samples to train, rest to test ---
  fewshot_data <- jade_filtered |>
    dplyr::filter(split_role == "fewshot")

  fewshot_train <- fewshot_data |>
    dplyr::slice_sample(n = fewshot_n, by = species) |>
    dplyr::mutate(split_type = "fewshot") |>
    dplyr::select(-split_role)

  # Anti-join on all coordinate columns to avoid duplicates
  fewshot_remaining <- fewshot_data |>
    dplyr::anti_join(fewshot_train, by = c("species", "X", "Y")) |>
    dplyr::mutate(split_type = "fewshot") |>
    dplyr::select(-split_role)

  # --- 3. Train species: cap at max_samples, then 80/10/10 ---
  train_species_data <- jade_filtered |>
    dplyr::filter(split_role == "train_species") |>
    dplyr::select(-split_role)

  # Cap at max_samples_per_species
  train_species_capped <- train_species_data |>
    dplyr::slice_sample(n = max_samples_per_species, by = species)

  # Within-species random split
  within_splits <- train_species_capped |>
    dplyr::group_by(species) |>
    dplyr::mutate(
      .rand = runif(dplyr::n()),
      .within_split = dplyr::case_when(
        .rand <= within_train_frac ~ "train",
        .rand <= within_train_frac + within_val_frac ~ "val",
        TRUE ~ "test"
      ),
      split_type = "within_species"
    ) |>
    dplyr::ungroup()

  within_train <- within_splits |>
    dplyr::filter(.within_split == "train") |>
    dplyr::select(-c(.rand, .within_split))
  within_val <- within_splits |>
    dplyr::filter(.within_split == "val") |>
    dplyr::select(-c(.rand, .within_split))
  within_test <- within_splits |>
    dplyr::filter(.within_split == "test") |>
    dplyr::select(-c(.rand, .within_split))

  # --- 4. Assemble final splits ---
  train <- dplyr::bind_rows(within_train, fewshot_train)
  val <- within_val
  test <- dplyr::bind_rows(within_test, fewshot_remaining, zeroshot_data)

  message("Split summary:")
  message("  Train: ", format(nrow(train), big.mark = ","), " samples, ",
          dplyr::n_distinct(train$species), " species")
  message("  Val:   ", format(nrow(val), big.mark = ","), " samples, ",
          dplyr::n_distinct(val$species), " species")
  message("  Test:  ", format(nrow(test), big.mark = ","), " samples, ",
          dplyr::n_distinct(test$species), " species")
  message("    - within_species: ", sum(test$split_type == "within_species"))
  message("    - fewshot:        ", sum(test$split_type == "fewshot"))
  message("    - zeroshot:       ", sum(test$split_type == "zeroshot"))

  list(train = train, val = val, test = test)
}


#' Summarize jade splits for verification
#'
#' @param splits List from create_jade_splits()
#' @param split_assignments Tibble from assign_species_splits()
#' @return List with overall, test_by_type, and species_roles summaries
#' @export
summarize_jade_splits <- function(splits, split_assignments) {
  overall <- dplyr::bind_rows(
    splits$train |>
      dplyr::summarise(n_samples = dplyr::n(),
                       n_species = dplyr::n_distinct(species),
                       split = "train"),
    splits$val |>
      dplyr::summarise(n_samples = dplyr::n(),
                       n_species = dplyr::n_distinct(species),
                       split = "val"),
    splits$test |>
      dplyr::summarise(n_samples = dplyr::n(),
                       n_species = dplyr::n_distinct(species),
                       split = "test")
  )

  test_by_type <- splits$test |>
    dplyr::summarise(
      n_samples = dplyr::n(),
      n_species = dplyr::n_distinct(species),
      .by = split_type
    )

  species_roles <- split_assignments |>
    dplyr::count(split_role, name = "n_species")

  message("\n=== Split Summary ===")
  message("Overall:")
  purrr::pwalk(overall, \(split, n_samples, n_species) {
    message("  ", split, ": ", format(n_samples, big.mark = ","),
            " samples, ", n_species, " species")
  })
  message("\nTest set breakdown:")
  purrr::pwalk(test_by_type, \(split_type, n_samples, n_species) {
    message("  ", split_type, ": ", format(n_samples, big.mark = ","),
            " samples, ", n_species, " species")
  })

  list(overall = overall, test_by_type = test_by_type,
       species_roles = species_roles)
}
