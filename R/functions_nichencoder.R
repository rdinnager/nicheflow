#' NichEncoder Rectified Flow Model
#'
#' Conditional rectified flow (TrajNet) that learns species-specific
#' environmental distributions in VAE latent space. Uses GPU-native Euler
#' ODE integration and species embeddings via nn_embedding.


#' NichEncoder TrajNet: Species-conditioned rectified flow
#'
#' U-Net trajectory network that predicts velocity fields v(x, t, species)
#' for ODE-based generative modeling. Conditions on learned species embeddings
#' to generate species-specific latent environmental distributions.
nichencoder_traj_net <- nn_module("NichEncoderTrajNet",
  initialize = function(coord_dim, n_species, spec_embed_dim = 32L,
                        breadths = c(256L, 128L, 64L),
                        t_encode = 32L, spec_encode = 64L,
                        model_device = "cuda:0",
                        loss_type = "pseudo_huber") {
    if (length(breadths) != 3) stop("breadths should be length 3!")

    self$coord_dim <- coord_dim
    self$n_species <- n_species
    self$model_device <- model_device
    self$loss_type <- loss_type

    # Species embedding (1-based indexing for R)
    self$species_embedding <- nn_embedding(n_species, spec_embed_dim)

    # Time and species-context encoding
    self$encode_t <- nn_linear(1L, t_encode)
    self$encode_spec <- nn_linear(spec_embed_dim, spec_encode)

    # U-Net with skip connections, conditioned on t + species
    self$unet <- nndag(
      input = ~ coord_dim,
      t_encoded = ~ t_encode,
      spec_encoded = ~ spec_encode,
      e_1 = input + t_encoded + spec_encoded ~ breadths[1],
      e_2 = e_1 + t_encoded + spec_encoded ~ breadths[2],
      e_3 = e_2 + t_encoded + spec_encoded ~ breadths[3],
      d_1 = e_3 + t_encoded + spec_encoded ~ breadths[3],
      d_2 = d_1 + e_3 + t_encoded + spec_encoded ~ breadths[2],
      d_3 = d_2 + e_2 + t_encoded + spec_encoded ~ breadths[1],
      output = d_3 + e_1 + t_encoded + spec_encoded ~ coord_dim,
      .act = list(nn_relu, output = nn_identity)
    )

    self$loss_function <- function(input, target) {
      if (self$loss_type == "pseudo_huber") {
        c <- 0.00054 * self$coord_dim
        diff_sq <- torch_sum((target - input)^2, dim = 2L)
        torch_mean(torch_sqrt(diff_sq + c^2) - c)
      } else {
        torch_mean((target - input)^2)
      }
    }

    # GPU-native Euler integration (stays on GPU, no deSolve)
    self$sample_trajectory <- function(initial_vals, spec_ids, steps = 200L) {
      with_no_grad({
        dt <- 1.0 / steps
        n <- initial_vals$shape[1]
        y <- initial_vals$detach()

        # Embed species once (outside JIT) and encode
        spec_emb <- self$species_embedding(spec_ids$detach())
        spec_enc <- self$encode_spec(spec_emb)

        for (s in seq_len(steps)) {
          t_val <- (s - 1) * dt
          t_tensor <- torch_full(c(n, 1L), t_val, device = y$device)
          t_enc <- self$encode_t(t_tensor)
          velocity <- self$unet(
            input = y, t_encoded = t_enc,
            spec_encoded = spec_enc
          )
          y <- y + velocity * dt
        }

        # Single GPU->CPU transfer at end
        y$cpu()
      })
    }
  },

  forward = function(coords, t, spec_encoded) {
    # NOTE: spec_encoded is the pre-encoded species vector (after encode_spec),
    # NOT raw species IDs. This allows JIT tracing of the forward pass since
    # all inputs are float tensors. The species_embedding + encode_spec steps
    # happen outside this call.
    t_encoded <- self$encode_t(t)
    self$unet(input = coords, t_encoded = t_encoded,
              spec_encoded = spec_encoded)
  }
)


#' Generate rectified flow training data
#'
#' Creates noise-interpolation pairs for training the NichEncoder.
#' For each sample: z ~ N(0,I), t sampled from chosen distribution,
#' target = v - z, coords = z + t * target.
#'
#' @param latent_codes Batch of latent codes (CPU tensor, N x coord_dim)
#' @param spec_ids Batch of species integer IDs (CPU tensor, N)
#' @param device Target device string
#' @param noise_scale Gaussian noise added to latent codes for regularization
#' @param time_sampling Time sampling strategy: "uniform" for t ~ U(0,1),
#'   "logit_normal" for t ~ sigmoid(N(mean, sd)) which concentrates samples
#'   near t=0.5 where multimodal structure must be resolved
#'   (from Esser et al. 2024, Stable Diffusion 3).
#' @param time_logit_normal_mean Mean of the normal in logit space (default 0)
#' @param time_logit_normal_sd SD of the normal in logit space (default 1)
#' @return List with coords, t, spec_ids, target (all on device)
generate_nichencoder_training_data <- function(latent_codes, spec_ids,
                                               device = "cuda:0",
                                               noise_scale = 0.01,
                                               time_sampling = "logit_normal",
                                               time_logit_normal_mean = 0,
                                               time_logit_normal_sd = 1) {
  with_no_grad({
    n <- latent_codes$size()[1]
    coord_dim <- latent_codes$size()[2]

    if (time_sampling == "logit_normal") {
      # t = sigmoid(N(mean, sd)) — concentrates near 0.5
      t <- torch_sigmoid(
        torch_randn(n, 1, device = device) * time_logit_normal_sd +
          time_logit_normal_mean
      )
    } else {
      t <- torch_rand(n, 1, device = device)
    }
    z <- torch_randn(n, coord_dim, device = device)

    # Add small noise to latent codes for regularization
    target <- (latent_codes$to(device = device) +
                 torch_randn_like(latent_codes, device = device) * noise_scale) - z
    coords <- z + t * target
  })

  list(
    coords = coords,
    t = t,
    spec_ids = spec_ids$to(device = device),
    target = target
  )
}


#' Compute energy distance between two sample sets
#'
#' Energy distance measures the distance between two distributions.
#' It captures both location and shape differences, unlike centroid MSE.
#' Computed on CPU to avoid GPU memory issues with large distance matrices.
#'
#' @param x Matrix (n1 x d)
#' @param y Matrix (n2 x d)
#' @param max_n Cap sample sizes to this for speed (default 500)
#' @return Scalar energy distance
compute_energy_distance <- function(x, y, max_n = 500L) {
  if (nrow(x) > max_n) x <- x[sample.int(nrow(x), max_n), , drop = FALSE]
  if (nrow(y) > max_n) y <- y[sample.int(nrow(y), max_n), , drop = FALSE]

  # E[||X-Y||] - 0.5*E[||X-X'||] - 0.5*E[||Y-Y'||]
  xy_dist <- mean(as.matrix(dist(rbind(x, y)))[
    seq_len(nrow(x)), nrow(x) + seq_len(nrow(y))
  ])
  xx_dist <- if (nrow(x) > 1) mean(dist(x)) else 0
  yy_dist <- if (nrow(y) > 1) mean(dist(y)) else 0

  2 * xy_dist - xx_dist - yy_dist
}


#' Compute sliced Wasserstein distance between two sample sets
#'
#' Projects samples onto random 1D directions and computes the exact
#' 1D Wasserstein-1 distance (sorted difference) for each slice, then
#' averages. Fast and effective for moderate dimensions.
#'
#' @param x Matrix (n1 x d)
#' @param y Matrix (n2 x d)
#' @param n_slices Number of random projection directions (default 100)
#' @return Scalar sliced Wasserstein distance
compute_swd <- function(x, y, n_slices = 100L) {
  d <- ncol(x)
  n1 <- nrow(x)
  n2 <- nrow(y)

  # Random unit directions on the d-sphere
  directions <- matrix(rnorm(n_slices * d), nrow = n_slices, ncol = d)
  directions <- directions / sqrt(rowSums(directions^2))

  # Project and compute 1D Wasserstein for each slice
  # W1 in 1D = mean(|sort(p1) - sort(p2)|) with matched quantiles
  proj_x <- x %*% t(directions)  # n1 x n_slices
  proj_y <- y %*% t(directions)  # n2 x n_slices

  # Match sample sizes via quantile interpolation to the smaller set
  n_q <- min(n1, n2, 500L)
  probs <- seq(0, 1, length.out = n_q)

  slice_dists <- vapply(seq_len(n_slices), \(s) {
    qx <- quantile(proj_x[, s], probs, names = FALSE)
    qy <- quantile(proj_y[, s], probs, names = FALSE)
    mean(abs(qx - qy))
  }, numeric(1))

  mean(slice_dists)
}


#' Validate NichEncoder generative quality
#'
#' Generates latent codes via ODE for species, computes energy distance,
#' sliced Wasserstein distance, and centroid MSE. Generates n_true samples
#' per species (matching the reference count). Computes metrics over many
#' species for stable averages, but only plots a subset.
#'
#' @param model NichEncoder model (non-JIT, on device)
#' @param ref_latent Reference latent codes tensor (CPU) — typically training data
#' @param ref_species_ids Reference species ID tensor (CPU)
#' @param species_map Named integer vector (species_name -> id)
#' @param fixed_species_ids Integer vector of species IDs to always plot
#'   (tracked across epochs for consistency). NULL = no fixed species.
#' @param n_random_plot_species Number of additional random species to plot
#' @param n_metric_species Total number of species to compute metrics over
#'   (includes plotted species + additional random ones)
#' @param device Torch device string
#' @param ode_steps Number of Euler integration steps
#' @param checkpoint_dir Directory to save plots (NULL = no plots)
#' @param epoch Current epoch (for plot filenames)
#' @return Named list with val_loss (energy distance), val_swd,
#'   val_centroid_mse averaged over n_metric_species
validate_nichencoder <- function(model, ref_latent, ref_species_ids,
                                 species_map,
                                 fixed_species_ids = NULL,
                                 n_random_plot_species = 5L,
                                 n_metric_species = 100L,
                                 device = "cuda:0", ode_steps = 200L,
                                 checkpoint_dir = NULL, epoch = NULL) {

  unique_species <- unique(as.integer(ref_species_ids$cpu()))

  # Build plot species list: fixed + random (for visualization)
  plot_species <- integer(0)
  if (!is.null(fixed_species_ids)) {
    plot_species <- intersect(fixed_species_ids, unique_species)
  }
  remaining <- setdiff(unique_species, plot_species)
  n_random <- min(n_random_plot_species, length(remaining))
  if (n_random > 0) {
    plot_species <- c(plot_species, sample(remaining, n_random))
  }

  # Build metric species list: plot species + additional random
  metric_species <- plot_species
  remaining_for_metrics <- setdiff(unique_species, metric_species)
  n_extra <- min(n_metric_species - length(metric_species),
                 length(remaining_for_metrics))
  if (n_extra > 0) {
    metric_species <- c(metric_species,
                        sample(remaining_for_metrics, n_extra))
  }
  message("  Validating ", length(metric_species), " species (",
          length(plot_species), " plotted)")

  # Per-species results
  results <- vector("list", length(metric_species))

  for (i in seq_along(metric_species)) {
    sp_id <- metric_species[i]
    sp_mask <- as.logical(ref_species_ids == sp_id)
    sp_true <- ref_latent[sp_mask, ]

    n_true <- sp_true$shape[1]
    coord_dim <- sp_true$shape[2]

    # Generate n_true samples (match truth count)
    z0 <- torch_randn(n_true, coord_dim, device = device)
    sp_ids_gen <- torch_full(c(n_true), sp_id,
                              dtype = torch_long(), device = device)
    generated <- model$sample_trajectory(z0, sp_ids_gen, steps = ode_steps)

    # Convert to R matrices for metrics
    true_mat <- as.matrix(sp_true)
    gen_mat <- as.matrix(generated)

    # Centroid MSE
    centroid_mse <- sum((colMeans(gen_mat) - colMeans(true_mat))^2)

    # Energy distance
    energy_dist <- compute_energy_distance(true_mat, gen_mat)

    # Sliced Wasserstein distance
    swd <- compute_swd(true_mat, gen_mat)

    sp_name <- names(species_map)[species_map == sp_id]
    if (length(sp_name) == 0) sp_name <- paste0("ID_", sp_id)

    is_plot <- sp_id %in% plot_species
    is_fixed <- sp_id %in% fixed_species_ids

    # Print details for plotted species only
    if (is_plot) {
      tag <- if (is_fixed) " [fixed]" else ""
      message("    ", sp_name, tag, " (n=", n_true,
              "): energy=", round(energy_dist, 4),
              " swd=", round(swd, 4),
              " centroid_mse=", round(centroid_mse, 4))
    }

    results[[i]] <- list(
      species = sp_name, sp_id = sp_id, n_true = n_true,
      energy_dist = energy_dist, swd = swd, centroid_mse = centroid_mse,
      is_fixed = is_fixed, is_plot = is_plot,
      # Only keep matrices for plotted species (memory)
      true_mat = if (is_plot) true_mat else NULL,
      gen_mat = if (is_plot) gen_mat else NULL
    )

    # Progress for metric-only species (every 25)
    if (!is_plot && i %% 25 == 0) {
      message("    ... ", i, "/", length(metric_species), " species evaluated")
    }
  }

  mean_energy <- mean(purrr::map_dbl(results, "energy_dist"))
  mean_swd <- mean(purrr::map_dbl(results, "swd"))
  mean_centroid <- mean(purrr::map_dbl(results, "centroid_mse"))
  message("  Mean energy (", length(metric_species), " spp): ",
          round(mean_energy, 4),
          " | Mean SWD: ", round(mean_swd, 4),
          " | Mean centroid MSE: ", round(mean_centroid, 4))

  # Generate plots from plotted species only
  plot_results <- purrr::keep(results, \(r) r$is_plot)
  if (!is.null(checkpoint_dir) && !is.null(epoch) && length(plot_results) > 0) {
    tryCatch({
      generate_nichencoder_validation_plots(
        plot_results, checkpoint_dir, epoch
      )
    }, error = \(e) {
      message("  Scatter plot generation error: ", e$message)
    })
    tryCatch({
      generate_nichencoder_hexbin_plots(
        plot_results, checkpoint_dir, epoch
      )
    }, error = \(e) {
      message("  Hex-bin plot generation error: ", e$message)
    })
  }

  list(
    val_loss = mean_energy,
    val_swd = mean_swd,
    val_centroid_mse = mean_centroid,
    per_species = purrr::map(results, \(r) {
      list(species = r$species, energy_dist = r$energy_dist,
           swd = r$swd, centroid_mse = r$centroid_mse, n_true = r$n_true)
    })
  )
}


#' Generate pairwise latent scatter plots for NichEncoder validation
#'
#' For each validated species, creates a grid of pairwise scatter plots
#' showing true latent codes (blue) vs generated (red). Saves as a combined
#' PNG in the checkpoint directory.
#'
#' @param results List of per-species results from validate_nichencoder
#' @param checkpoint_dir Directory to save the plot
#' @param epoch Current epoch number
generate_nichencoder_validation_plots <- function(results, checkpoint_dir,
                                                   epoch) {
  library(ggplot2)
  library(patchwork)

  coord_dim <- ncol(results[[1]]$true_mat)
  latent_names <- paste0("L", seq_len(coord_dim))

  # Select up to 3 most informative dimension pairs for scatter plots
  # Use the pairs with highest variance in the combined data
  combined <- rbind(results[[1]]$true_mat, results[[1]]$gen_mat)
  col_vars <- apply(combined, 2, var)
  top_dims <- order(col_vars, decreasing = TRUE)

  # Pick pairs: (1st, 2nd), (1st, 3rd), (2nd, 3rd) of highest-variance dims
  if (coord_dim >= 3) {
    pairs <- list(
      c(top_dims[1], top_dims[2]),
      c(top_dims[1], top_dims[3]),
      c(top_dims[2], top_dims[3])
    )
  } else {
    pairs <- list(c(1, 2))
  }

  n_species <- length(results)
  n_pairs <- length(pairs)

  # Compute global per-dimension ranges across all species (true + generated)
  # so axis limits are fixed and comparable across species rows
  dim_min <- rep(Inf, coord_dim)
  dim_max <- rep(-Inf, coord_dim)
  for (r in results) {
    for (d in seq_len(coord_dim)) {
      vals <- c(r$true_mat[, d], r$gen_mat[, d])
      dim_min[d] <- min(dim_min[d], min(vals))
      dim_max[d] <- max(dim_max[d], max(vals))
    }
  }
  # Add small padding (2% of range)
  dim_pad <- (dim_max - dim_min) * 0.02
  dim_min <- dim_min - dim_pad
  dim_max <- dim_max + dim_pad

  species_plots <- vector("list", n_species)

  for (s in seq_len(n_species)) {
    r <- results[[s]]

    fixed_tag <- if (r$is_fixed) " *" else ""
    sp_label <- paste0(r$species, fixed_tag,
                       "  (n=", r$n_true,
                       ", E=", round(r$energy_dist, 3),
                       ", SWD=", round(r$swd, 3), ")")

    pair_plots <- vector("list", n_pairs)
    for (p in seq_along(pairs)) {
      d1 <- pairs[[p]][1]
      d2 <- pairs[[p]][2]

      df <- rbind(
        data.frame(x = r$true_mat[, d1], y = r$true_mat[, d2],
                   type = "Truth"),
        data.frame(x = r$gen_mat[, d1], y = r$gen_mat[, d2],
                   type = "Generated")
      )

      pp <- ggplot(df, aes(x = x, y = y, colour = type)) +
        geom_point(size = 0.5, alpha = 0.5) +
        scale_colour_manual(values = c(Truth = "#2166AC", Generated = "#B2182B")) +
        coord_cartesian(xlim = c(dim_min[d1], dim_max[d1]),
                        ylim = c(dim_min[d2], dim_max[d2])) +
        labs(x = latent_names[d1], y = latent_names[d2]) +
        theme_minimal(base_size = 10) +
        theme(legend.position = if (p == 1) "bottom" else "none",
              legend.title = element_blank(),
              legend.text = element_text(size = 11),
              legend.key.size = unit(1.2, "lines"),
              plot.margin = margin(4, 4, 4, 4))

      # Add species label as subtitle on first panel of each row
      if (p == 1) {
        pp <- pp + ggtitle(sp_label) +
          theme(plot.title = element_text(size = 12, face = "bold.italic"))
      }

      pair_plots[[p]] <- pp
    }

    species_plots[[s]] <- wrap_plots(pair_plots, nrow = 1)
  }

  # Stack species rows
  combined_plot <- wrap_plots(species_plots, ncol = 1)

  output_path <- file.path(
    checkpoint_dir,
    sprintf("epoch_%04d_nichencoder_validation.png", epoch)
  )
  dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)

  height_per_species <- 450
  ragg::agg_png(output_path,
                width = 2400,
                height = height_per_species * n_species + 80,
                res = 150)
  print(combined_plot)
  dev.off()
  message("  Saved validation plot: ", basename(output_path))
}


#' Generate hex-bin overlap plots for NichEncoder validation
#'
#' For each validated species, creates hex-bin plots showing the proportion
#' of generated vs truth points in each cell. Perfect overlap = 0.5 (equal
#' mix). Uses ggplot2's stat_summary_hex for correct hex tessellation.
#'
#' @param results List of per-species results from validate_nichencoder
#'   (only those with is_plot = TRUE and non-NULL matrices)
#' @param checkpoint_dir Directory to save the plot
#' @param epoch Current epoch number
#' @param xbins Number of hex bins across the x-axis (default 40)
generate_nichencoder_hexbin_plots <- function(results, checkpoint_dir,
                                              epoch, xbins = 40L) {
  library(ggplot2)
  library(patchwork)

  coord_dim <- ncol(results[[1]]$true_mat)
  latent_names <- paste0("L", seq_len(coord_dim))

  # Same dimension pair selection as scatter plots
  combined <- rbind(results[[1]]$true_mat, results[[1]]$gen_mat)
  col_vars <- apply(combined, 2, var)
  top_dims <- order(col_vars, decreasing = TRUE)

  if (coord_dim >= 3) {
    pairs <- list(
      c(top_dims[1], top_dims[2]),
      c(top_dims[1], top_dims[3]),
      c(top_dims[2], top_dims[3])
    )
  } else {
    pairs <- list(c(1, 2))
  }

  n_species <- length(results)
  n_pairs <- length(pairs)

  # Compute global per-dimension ranges (same as scatter plots)
  dim_min <- rep(Inf, coord_dim)
  dim_max <- rep(-Inf, coord_dim)
  for (r in results) {
    for (d in seq_len(coord_dim)) {
      vals <- c(r$true_mat[, d], r$gen_mat[, d])
      dim_min[d] <- min(dim_min[d], min(vals))
      dim_max[d] <- max(dim_max[d], max(vals))
    }
  }
  dim_pad <- (dim_max - dim_min) * 0.02
  dim_min <- dim_min - dim_pad
  dim_max <- dim_max + dim_pad

  species_plots <- vector("list", n_species)

  for (s in seq_len(n_species)) {
    r <- results[[s]]

    fixed_tag <- if (r$is_fixed) " *" else ""
    sp_label <- paste0(r$species, fixed_tag,
                       "  (n=", r$n_true,
                       ", E=", round(r$energy_dist, 3),
                       ", SWD=", round(r$swd, 3), ")")

    pair_plots <- vector("list", n_pairs)
    for (p in seq_along(pairs)) {
      d1 <- pairs[[p]][1]
      d2 <- pairs[[p]][2]

      true_x <- r$true_mat[, d1]
      true_y <- r$true_mat[, d2]
      gen_x <- r$gen_mat[, d1]
      gen_y <- r$gen_mat[, d2]

      # Combine: z = 0 for truth, 1 for generated; mean(z) = prop_generated
      df <- data.frame(
        x = c(true_x, gen_x),
        y = c(true_y, gen_y),
        is_gen = c(rep(0, length(true_x)), rep(1, length(gen_x)))
      )

      # Use equal binwidth for both axes so hexagons stay regular
      # under coord_fixed(ratio = 1)
      x_range <- dim_max[d1] - dim_min[d1]
      y_range <- dim_max[d2] - dim_min[d2]
      bw <- max(x_range, y_range) / xbins

      pp <- ggplot(df, aes(x = x, y = y, z = is_gen)) +
        stat_summary_hex(fun = mean, binwidth = bw) +
        scico::scale_fill_scico(
          palette = "vanimo", direction = -1,
          limits = c(0, 1), midpoint = 0.5,
          name = "Prop.\nGenerated"
        ) +
        coord_fixed(
          ratio = 1,
          xlim = c(dim_min[d1], dim_max[d1]),
          ylim = c(dim_min[d2], dim_max[d2])
        ) +
        labs(x = latent_names[d1], y = latent_names[d2]) +
        theme_minimal(base_size = 10) +
        theme(legend.position = if (p == n_pairs) "right" else "none",
              plot.margin = margin(4, 4, 4, 4))

      if (p == 1) {
        pp <- pp + ggtitle(sp_label) +
          theme(plot.title = element_text(size = 12, face = "bold.italic"))
      }

      pair_plots[[p]] <- pp
    }

    species_plots[[s]] <- wrap_plots(pair_plots, nrow = 1)
  }

  combined_plot <- wrap_plots(species_plots, ncol = 1)

  output_path <- file.path(
    checkpoint_dir,
    sprintf("epoch_%04d_nichencoder_hexbin.png", epoch)
  )

  height_per_species <- 450
  ragg::agg_png(output_path,
                width = 2600,
                height = height_per_species * n_species + 80,
                res = 150)
  print(combined_plot)
  dev.off()
  message("  Saved hex-bin plot: ", basename(output_path))
}
