# =============================================================================
# JADE Sample Visualization Functions
# =============================================================================
# Functions for visualizing JADE (Jacobian-Adjusted Density Estimation) samples
# in environmental space. Includes:
#   - Species selection based on Jacobian variation
#   - 2D environment-space scatterplot pairs
#   - Circular parallel coordinates across all bioclim variables

#' Select species with contrasting Jacobian variation for visualization
#'
#' Samples completed jade_samples branches and picks species at different
#' quantiles of Jacobian ratio (max/min within species).
#'
#' @param n_sample Number of branches to sample for candidate screening
#' @param quantiles Numeric vector of quantile targets (default: 5th, 33rd, 66th, 95th)
#' @param seed Random seed for reproducibility
#' @return A tibble with columns: branch, species, taxon, n, jac_ratio, jac_cv, x_range, y_range
select_jade_viz_species <- function(n_sample = 300,
                                    quantiles = c(0.05, 0.33, 0.66, 0.95),
                                    seed = 42) {
  p <- targets::tar_progress() |>
    filter(grepl("^jade_samples_", name), progress == "completed")

  set.seed(seed)
  sample_names <- sample(p$name, min(n_sample, nrow(p)))

  stats <- purrr::map(sample_names, \(bn) {
    d <- targets::tar_read_raw(bn)
    if (is.null(d) || nrow(d) < 100) return(NULL)
    tibble(
      branch = bn,
      species = d$species[1],
      taxon = d$taxon[1],
      n = nrow(d),
      jac_ratio = max(d$jacobian) / min(d$jacobian),
      jac_cv = sd(d$jacobian) / mean(d$jacobian),
      x_range = diff(range(d$X)),
      y_range = diff(range(d$Y))
    )
  }) |> purrr::list_rbind()

  picks <- purrr::map(quantiles, \(q) {
    target_val <- quantile(stats$jac_ratio, q)
    stats |>
      mutate(dist = abs(jac_ratio - target_val)) |>
      slice_min(dist, n = 1)
  }) |> purrr::list_rbind() |>
    select(-dist)

  picks
}

#' Load JADE sample data for selected species
#'
#' @param picks Tibble from select_jade_viz_species() with a `branch` column
#' @return A tibble combining all selected species' samples
load_jade_viz_data <- function(picks) {
  purrr::map(picks$branch, \(bn) targets::tar_read_raw(bn)) |>
    purrr::list_rbind()
}

#' Plot JADE samples as 2D environment-space scatterplot pairs
#'
#' Creates a multi-panel plot with one row per pair of environmental variables
#' and one column per species, coloured by log10(Jacobian).
#'
#' @param species_data Tibble of JADE samples (from load_jade_viz_data())
#' @param key_patterns Character vector of regex patterns to select bioclim
#'   variables (matched against column names). Default picks bio1, bio4,
#'   bio12, bio15, and npp.
#' @param short_names Character vector of short display names (same length
#'   as key_patterns)
#' @param output_path File path for saving the plot (NULL to return plot object)
#' @param width Plot width in inches
#' @param height Plot height in inches
#' @param dpi Plot resolution
#' @return A patchwork plot object (invisibly if saved to file)
plot_jade_env_pairs <- function(species_data,
                                key_patterns = c("bio1_", "bio4_", "bio12_",
                                                 "bio15_", "npp_"),
                                short_names = c("Mean Temp", "Temp Seasonality",
                                                "Annual Precip",
                                                "Precip Seasonality", "NPP"),
                                output_path = NULL,
                                width = 14, height = 22, dpi = 200) {
  env_cols <- grep("CHELSA", names(species_data), value = TRUE)

  key_vars <- purrr::map_chr(key_patterns, \(pat) {
    grep(pat, env_cols, value = TRUE)[1]
  })
  names(short_names) <- key_vars

  # Species labels with Jacobian ratio
  species_data <- species_data |>
    mutate(
      species_label = paste0(
        species, " (", taxon, ", JR=",
        round(jacobian / min(jacobian)), "x)"
      ),
      log_jac = log10(jacobian),
      .by = species
    )

  # Generate all pairwise combinations
  pairs <- combn(key_vars, 2, simplify = FALSE)

  plots <- purrr::map(pairs, \(pair) {
    x_var <- pair[1]
    y_var <- pair[2]
    ggplot(species_data, aes(x = .data[[x_var]], y = .data[[y_var]],
                              colour = log_jac)) +
      geom_point(alpha = 0.15, size = 0.3) +
      scale_colour_viridis_c(option = "inferno", name = "log10(J)") +
      facet_wrap(~ species, scales = "free", nrow = 1) +
      labs(x = short_names[x_var], y = short_names[y_var]) +
      theme_minimal(base_size = 7) +
      theme(
        legend.position = "none",
        strip.text = element_text(size = 5),
        axis.title = element_text(size = 6)
      )
  })

  combined <- patchwork::wrap_plots(plots, ncol = 1) +
    patchwork::plot_annotation(
      title = "JADE Samples in Environmental Space (coloured by log Jacobian)",
      subtitle = "Species ordered by increasing Jacobian variation (left to right)",
      theme = theme(
        plot.title = element_text(size = 11),
        plot.subtitle = element_text(size = 8)
      )
    )

  if (!is.null(output_path)) {
    dir.create(dirname(output_path), showWarnings = FALSE, recursive = TRUE)
    ggsave(output_path, combined, width = width, height = height, dpi = dpi)
    message("Saved ", output_path)
  }

  invisible(combined)
}

#' Plot circular parallel coordinates of JADE samples
#'
#' Creates a radar-style plot where each sample point traces a path through
#' all bioclim variables (ordered by correlation clustering), coloured by
#' log10(Jacobian). One facet per species.
#'
#' @param species_data Tibble of JADE samples (from load_jade_viz_data())
#' @param n_subsample Number of points to subsample per species (default 500)
#' @param alpha Line transparency (default 0.04)
#' @param seed Random seed for subsampling
#' @param output_path File path for saving the plot (NULL to return plot object)
#' @param width Plot width in inches
#' @param height Plot height in inches
#' @param dpi Plot resolution
#' @return A ggplot object (invisibly if saved to file)
plot_jade_circular_parcoord <- function(species_data,
                                        n_subsample = 500,
                                        alpha = 0.04,
                                        seed = 123,
                                        output_path = NULL,
                                        width = 20, height = 7, dpi = 200) {
  env_cols <- grep("CHELSA", names(species_data), value = TRUE)

  # Order variables by hierarchical clustering on correlation
  env_mat <- species_data |> select(all_of(env_cols)) |> as.matrix()
  cor_mat <- cor(env_mat, use = "pairwise.complete.obs")
  hc <- hclust(as.dist(1 - abs(cor_mat)), method = "ward.D2")
  ordered_vars <- env_cols[hc$order]

  short_names <- ordered_vars |>
    sub("CHELSA_", "", x = _) |>
    sub("_1981-2010_V\\.2\\.1", "", x = _)

  # Subsample per species
  set.seed(seed)
  plot_data <- species_data |>
    group_by(species) |>
    slice_sample(n = min(n_subsample, n())) |>
    ungroup()

  # Normalize 0-1 globally across all species
  plot_norm <- plot_data |>
    mutate(across(
      all_of(ordered_vars),
      \(x) (x - min(x, na.rm = TRUE)) /
           (max(x, na.rm = TRUE) - min(x, na.rm = TRUE) + 1e-10)
    ))

  # Species labels with Jacobian ratio
  jac_ratios <- plot_data |>
    summarise(jr = round(max(jacobian) / min(jacobian)), .by = species)
  plot_norm <- plot_norm |>
    left_join(jac_ratios, by = "species") |>
    mutate(species_label = paste0(species, "\n(JR=", jr, "x)"))

  # Pivot to long format
  plot_long <- plot_norm |>
    mutate(pt_id = row_number()) |>
    select(pt_id, species_label, jacobian, all_of(ordered_vars)) |>
    pivot_longer(all_of(ordered_vars),
                 names_to = "variable", values_to = "value") |>
    mutate(variable = factor(variable,
                             levels = ordered_vars, labels = short_names))

  # Close the loop: repeat the first variable as a closing segment
  closing <- plot_long |>
    filter(variable == short_names[1]) |>
    mutate(variable = "..close")

  all_levels <- c(short_names, "..close")
  plot_long <- bind_rows(plot_long, closing) |>
    mutate(variable = factor(variable, levels = all_levels)) |>
    arrange(pt_id, variable)

  # Order species facets by Jacobian ratio
  sp_order <- plot_norm |>
    distinct(species_label, jr) |>
    arrange(jr)
  plot_long <- plot_long |>
    mutate(species_label = factor(species_label,
                                  levels = sp_order$species_label))

  circ_plot <- ggplot(plot_long,
                       aes(x = variable, y = value,
                           group = pt_id, colour = log10(jacobian))) +
    geom_path(alpha = alpha, linewidth = 0.3) +
    coord_polar() +
    facet_wrap(~ species_label, nrow = 1) +
    scale_colour_viridis_c(option = "inferno", name = "log10(J)") +
    scale_y_continuous(limits = c(-0.1, 1), breaks = c(0, 0.5, 1)) +
    scale_x_discrete(labels = \(x) ifelse(x == "..close", "", x)) +
    labs(
      title = "Circular Parallel Coordinates: JADE Samples Across Bioclim Variables",
      subtitle = paste0(
        "Each line = one sample point, coloured by Jacobian. ",
        "Variables ordered by correlation clustering."
      )
    ) +
    theme_minimal(base_size = 9) +
    theme(
      axis.text.x = element_text(size = 5),
      axis.text.y = element_blank(),
      axis.title = element_blank(),
      strip.text = element_text(size = 8, face = "italic"),
      legend.position = "bottom",
      panel.grid.minor = element_blank()
    )

  if (!is.null(output_path)) {
    dir.create(dirname(output_path), showWarnings = FALSE, recursive = TRUE)
    ggsave(output_path, circ_plot, width = width, height = height, dpi = dpi)
    message("Saved ", output_path)
  }

  invisible(circ_plot)
}
