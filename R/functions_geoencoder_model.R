#' Geo-Encoder Model Functions
#'
#' Transformer-based model that maps variable-length sets of (lon,lat)
#' occurrence points to NichEncoder species embeddings. Uses masked mean
#' pooling for permutation invariance.
#'
#' @author rdinnage


#' Geo-Encoder Transformer: Occurrence points -> niche embedding
#'
#' Self-attention transformer with masked mean pooling that maps
#' variable-length coordinate sets to fixed-dimensional niche embeddings.
#' Reuses transformer_block from R/models.R.
#'
#' @param input_dim Input feature dimension (2 for lon/lat)
#' @param embed_dim Internal transformer dimension
#' @param output_dim Output embedding dimension (matches NichEncoder spec_embed_dim)
#' @param n_blocks Number of transformer blocks
#' @param num_heads Number of attention heads
#' @param dropout_prob Dropout probability
geoencoder_transformer <- nn_module("GeoEncoderTransformer",
  initialize = function(input_dim = 2L, embed_dim = 256L, output_dim = 64L,
                        n_blocks = 8L, num_heads = 8L, dropout_prob = 0.1) {
    self$input_dim <- input_dim
    self$embed_dim <- embed_dim
    self$output_dim <- output_dim
    self$embedding <- nn_linear(input_dim, embed_dim, bias = FALSE)
    self$embed_dropout <- nn_dropout(dropout_prob)
    self$blocks <- nn_module_list(
      map(seq_len(n_blocks),
          ~ transformer_block(embed_dim, num_heads, dropout_prob))
    )
    self$output_norm <- nn_layer_norm(embed_dim)
    self$mlp_final <- nn_sequential(
      nn_linear(embed_dim, embed_dim * 4),
      nn_gelu(),
      nn_dropout(dropout_prob),
      nn_linear(embed_dim * 4, output_dim)
    )
  },
  forward = function(input, mask) {
    # input: (batch, seq_len, 2)
    # mask: (batch, seq_len) — TRUE = padding position
    x <- self$embed_dropout(self$embedding(input))

    for (i in seq_along(self$blocks)) {
      x <- self$blocks[[i]](x, mask)
    }

    # Masked mean pooling: zero out padding, divide by real sequence length
    mask_expanded <- (!mask)$unsqueeze(-1L)$to(dtype = x$dtype)
    x <- x * mask_expanded
    x <- torch_sum(x, dim = 2L)
    seq_lengths <- mask_expanded$sum(dim = 2L)$clamp(min = 1)
    x <- x / seq_lengths

    x <- self$output_norm(x)
    self$mlp_final(x)
  }
)


#' Geo-Encoder loss function
#'
#' Composite loss combining MSE and cosine similarity for embedding prediction.
#'
#' @param predicted Predicted embeddings (batch x output_dim)
#' @param target Target embeddings (batch x output_dim)
#' @param loss_type Loss type: "mse_cosine" or "mse"
#' @param cosine_weight Weight for cosine similarity component
#' @return Scalar loss tensor
geoencoder_loss <- function(predicted, target,
                            loss_type = "mse_cosine",
                            cosine_weight = 0.5) {
  if (loss_type == "mse_cosine") {
    mse <- torch_mean((predicted - target)^2)
    cos <- 1 - torch_mean(nnf_cosine_similarity(predicted, target))
    mse + cosine_weight * cos
  } else {
    torch_mean((predicted - target)^2)
  }
}


#' ODE trajectory from raw embedding tensor (bypasses nn_embedding)
#'
#' Like nichencoder_traj_net$sample_trajectory() but accepts a pre-computed
#' embedding tensor instead of integer species IDs. Defined as a standalone
#' function to avoid modifying nichencoder_traj_net (which would invalidate
#' the nichencoder_training target).
#'
#' @param model A nichencoder_traj_net model instance (on device)
#' @param initial_vals Initial latent values (batch x coord_dim)
#' @param spec_emb Raw species embedding tensor (batch x spec_embed_dim)
#' @param steps Number of Euler integration steps
#' @return Generated latent codes on CPU (batch x coord_dim)
sample_trajectory_from_embedding <- function(model, initial_vals, spec_emb,
                                              steps = 200L) {
  with_no_grad({
    dt <- 1.0 / steps
    n <- initial_vals$shape[1]
    y <- initial_vals$detach()
    spec_enc <- model$encode_spec(spec_emb$detach())

    for (s in seq_len(steps)) {
      t_val <- (s - 1) * dt
      t_tensor <- torch_full(c(n, 1L), t_val, device = y$device)
      t_enc <- model$encode_t(t_tensor)
      velocity <- model$unet(
        input = y, t_encoded = t_enc,
        spec_encoded = spec_enc
      )
      y <- y + velocity * dt
    }

    y$cpu()
  })
}


#' Validate geo-encoder with embedding-space and downstream evaluation
#'
#' Two validation modes:
#' 1. Embedding-space: predict embeddings for val species, compare to true
#'    NichEncoder embeddings (MSE, cosine similarity, scatter plots)
#' 2. Downstream: predict embeddings for zero-shot species, run through
#'    NichEncoder ODE + VAE decoder, compare generated environmental
#'    distributions to raw CHELSA-BIOCLIM data (energy distance, SWD)
#'
#' @param model Geo-encoder model (on device)
#' @param val_groups List of validation data groups
#' @param target_embeddings Matrix of species embeddings (n_species x embed_dim)
#' @param species_map Named integer vector (species_name -> id)
#' @param device Torch device string
#' @param batch_size Batch size for validation
#' @param downstream_groups Optional list of downstream coord groups (zero-shot)
#' @param downstream_env Optional dataframe of raw env data for zero-shot species
#' @param nichencoder_model Optional NichEncoder model for ODE generation
#' @param vae_model Optional VAE model for latent-to-env decoding
#' @param n_plot_species Number of embedding-space species to plot
#' @param n_downstream_plot_species Number of downstream species to plot
#' @param n_metric_species Number of downstream species for metric computation
#' @param ode_steps Number of ODE integration steps
#' @param checkpoint_dir Directory for saving plots
#' @param epoch Current epoch number
#' @return Named list with val_mse, val_cosine, val_loss, and optionally
#'   val_energy, val_swd, val_centroid_mse
validate_geoencoder <- function(model, val_groups = NULL, target_embeddings,
                                species_map, device = "cuda:0",
                                batch_size = 256L,
                                val_flat = NULL,
                                val_subsample = 0L,
                                downstream_groups = NULL,
                                downstream_flat = NULL,
                                downstream_env = NULL,
                                nichencoder_model = NULL,
                                vae_model = NULL,
                                n_plot_species = 5L,
                                n_downstream_plot_species = 6L,
                                n_metric_species = 50L,
                                fixed_plot_species = NULL,
                                ode_steps = 200L,
                                vae_active_dims = NULL,
                                vae_latent_dim = 16L,
                                checkpoint_dir = NULL, epoch = NULL) {

  # If model is on a different device than the validation device, copy weights
  model_device <- model$parameters[[1]]$device
  if (as.character(model_device) != as.character(torch_device(device))) {
    message("  Copying geo-encoder to validation device: ", device)
    val_model <- model$clone(deep = TRUE)$to(device = device)
    val_model$eval()
  } else {
    val_model <- model
    val_model$eval()
  }

  # ===== Embedding-space validation =====
  use_flat <- !is.null(val_flat)
  n_total <- if (use_flat) val_flat$n_groups else length(val_groups)

  # Subsample for speed
  if (val_subsample > 0L && val_subsample < n_total) {
    eval_idx <- sample.int(n_total, val_subsample)
    n_eval <- val_subsample
    message("  Subsampling ", n_eval, " of ", format(n_total, big.mark = ","),
            " groups for embedding validation")
  } else {
    eval_idx <- seq_len(n_total)
    n_eval <- n_total
  }

  n_batches <- ceiling(n_eval / batch_size)

  all_mse <- numeric(0)
  all_cos <- numeric(0)
  all_species <- character(0)
  all_predicted <- list()
  all_true <- list()

  with_no_grad({
    for (b in seq_len(n_batches)) {
      b_start <- (b - 1) * batch_size + 1
      b_end <- min(b * batch_size, n_eval)
      batch_idx <- eval_idx[b_start:b_end]

      if (use_flat) {
        batch_result <- collate_geoencoder_batch_flat(
          batch_idx, val_flat, target_embeddings, species_map, device)
        batch_species <- val_flat$group_species[batch_idx]
      } else {
        batch_groups <- val_groups[batch_idx]
        batch_result <- collate_geoencoder_batch(batch_groups,
                                                  target_embeddings,
                                                  species_map, device)
        batch_species <- purrr::map_chr(batch_groups, "species")
      }

      predicted <- val_model(batch_result$coords, batch_result$mask)
      target <- batch_result$target_emb

      mse_per <- torch_sum((predicted - target)^2, dim = 2L)
      cos_per <- nnf_cosine_similarity(predicted, target)

      all_mse <- c(all_mse, as.numeric(mse_per$cpu()))
      all_cos <- c(all_cos, as.numeric(cos_per$cpu()))
      all_species <- c(all_species, batch_species)
      all_predicted <- c(all_predicted,
                         list(as.matrix(predicted$cpu())))
      all_true <- c(all_true,
                    list(as.matrix(target$cpu())))
    }
  })

  val_mse <- mean(all_mse)
  val_cosine <- mean(all_cos)
  val_loss <- val_mse + 0.5 * (1 - val_cosine)

  message("  Embedding validation (", n_eval, " samples): ",
          "MSE=", round(val_mse, 6),
          " Cosine=", round(val_cosine, 4),
          " Loss=", round(val_loss, 6))

  # Generate embedding scatter plots
  if (!is.null(checkpoint_dir) && !is.null(epoch)) {
    tryCatch({
      pred_mat <- do.call(rbind, all_predicted)
      true_mat <- do.call(rbind, all_true)
      generate_geoencoder_embedding_plots(
        pred_mat, true_mat, all_species,
        n_plot_species = n_plot_species,
        checkpoint_dir = checkpoint_dir, epoch = epoch
      )
    }, error = \(e) {
      message("  Embedding plot error: ", e$message)
    })
  }

  result <- list(
    val_mse = val_mse,
    val_cosine = val_cosine,
    val_loss = val_loss
  )

  # ===== Downstream validation (zero-shot species) =====
  has_downstream <- (!is.null(downstream_groups) || !is.null(downstream_flat)) &&
    !is.null(downstream_env) &&
    !is.null(nichencoder_model) && !is.null(vae_model)

  if (has_downstream) {
    message("  Running downstream validation...")
    ds_result <- validate_geoencoder_downstream(
      model = val_model,
      downstream_groups = downstream_groups,
      downstream_flat = downstream_flat,
      downstream_env = downstream_env,
      nichencoder_model = nichencoder_model,
      vae_model = vae_model,
      device = device,
      batch_size = batch_size,
      n_plot_species = n_downstream_plot_species,
      n_metric_species = n_metric_species,
      fixed_plot_species = fixed_plot_species,
      ode_steps = ode_steps,
      vae_active_dims = vae_active_dims,
      vae_latent_dim = vae_latent_dim,
      checkpoint_dir = checkpoint_dir,
      epoch = epoch
    )
    result$val_energy <- ds_result$mean_energy
    result$val_swd <- ds_result$mean_swd
    result$val_centroid_mse <- ds_result$mean_centroid
  }

  result
}


#' Downstream validation: geo-encoder → NichEncoder ODE → VAE decoder → env space
#'
#' For each zero-shot species: predict embedding from one corrupted coord set,
#' generate latent codes via NichEncoder ODE, decode to environmental space
#' via VAE, compare to raw environmental data.
validate_geoencoder_downstream <- function(model, downstream_groups = NULL,
                                            downstream_flat = NULL,
                                            downstream_env,
                                            nichencoder_model, vae_model,
                                            device, batch_size = 256L,
                                            n_plot_species = 6L,
                                            n_metric_species = 50L,
                                            fixed_plot_species = NULL,
                                            ode_steps = 200L,
                                            vae_active_dims = NULL,
                                            vae_latent_dim = 16L,
                                            checkpoint_dir = NULL,
                                            epoch = NULL) {

  nichencoder_model$eval()
  vae_model$eval()

  use_flat <- !is.null(downstream_flat)

  # Get unique species in downstream data
  if (use_flat) {
    ds_species <- unique(downstream_flat$group_species)
  } else {
    ds_species <- unique(purrr::map_chr(downstream_groups, "species"))
  }

  # Ensure fixed plot species are included in evaluation set
  valid_fixed <- if (!is.null(fixed_plot_species)) {
    intersect(fixed_plot_species, ds_species)
  } else {
    character(0)
  }

  n_ds_species <- min(n_metric_species, length(ds_species))
  random_pool <- setdiff(ds_species, valid_fixed)
  n_random <- n_ds_species - length(valid_fixed)
  selected_species <- c(valid_fixed, sample(random_pool, min(n_random, length(random_pool))))

  # Plot species: fixed first, then random to fill
  n_plot <- min(n_plot_species, length(selected_species))
  n_random_plot <- n_plot - length(valid_fixed)
  random_selected <- setdiff(selected_species, valid_fixed)
  plot_species <- c(valid_fixed,
                    if (n_random_plot > 0) random_selected[seq_len(min(n_random_plot, length(random_selected)))]
                    else character(0))

  message("  Downstream: ", length(selected_species), " species (",
          length(plot_species), " plotted, ", length(valid_fixed), " fixed)")

  # NichEncoder coord_dim (from model architecture)
  coord_dim <- nichencoder_model$coord_dim

  results <- vector("list", n_ds_species)

  for (i in seq_along(selected_species)) {
    sp <- selected_species[i]

    # Find coordinates for this species
    if (use_flat) {
      sp_idx <- which(downstream_flat$group_species == sp)[1]
      if (is.na(sp_idx)) next
      start <- downstream_flat$group_start[sp_idx]
      len <- downstream_flat$group_length[sp_idx]
      coords_mat <- downstream_flat$coords_all[start:(start + len - 1L), , drop = FALSE]
    } else {
      sp_groups <- purrr::keep(downstream_groups, \(g) g$species == sp)
      if (length(sp_groups) == 0) next
      coords_mat <- sp_groups[[1]]$coords
    }

    # Predict embedding via geo-encoder
    n_pts <- nrow(coords_mat)

    with_no_grad({
      # Single-sample batch: (1, n_pts, 2), mask all FALSE (real data)
      coords_t <- torch_tensor(coords_mat, device = device)$unsqueeze(1L)
      mask_t <- torch_zeros(1L, n_pts, dtype = torch_bool(), device = device)

      predicted_emb <- model(coords_t, mask_t)  # 1 x output_dim
    })

    # Get ground truth env data for this species
    sp_env <- downstream_env |>
      dplyr::filter(species == sp)
    env_cols <- grep("^env_", names(sp_env), value = TRUE)
    true_env_mat <- as.matrix(sp_env[, env_cols])
    n_true <- nrow(true_env_mat)

    if (n_true == 0) next

    # Generate latent codes via NichEncoder ODE
    n_gen <- n_true  # Match sample count
    with_no_grad({
      z0 <- torch_randn(n_gen, coord_dim, device = device)
      spec_emb_expanded <- predicted_emb$expand(c(n_gen, -1L))
      generated_latent <- sample_trajectory_from_embedding(
        nichencoder_model, z0, spec_emb_expanded, steps = ode_steps
      )

      # Map active dims back to full latent space for VAE decoder
      gen_latent_device <- generated_latent$to(device = device)
      if (!is.null(vae_active_dims)) {
        full_latent <- torch_zeros(n_gen, vae_latent_dim, device = device)
        for (j in seq_along(vae_active_dims)) {
          full_latent[, vae_active_dims[j]] <- gen_latent_device[, j]
        }
        gen_latent_device <- full_latent
      }

      # Decode latent codes to environmental space via VAE decoder
      generated_env <- vae_model$decoder(gen_latent_device)
      gen_env_mat <- as.matrix(generated_env$cpu())
    })

    # Compute metrics in environmental space
    energy_dist <- compute_energy_distance(true_env_mat, gen_env_mat)
    swd <- compute_swd(true_env_mat, gen_env_mat)
    centroid_mse <- sum((colMeans(gen_env_mat) - colMeans(true_env_mat))^2)

    is_plot <- sp %in% plot_species

    if (is_plot) {
      message("    ", sp, " (n=", n_true,
              "): energy=", round(energy_dist, 4),
              " swd=", round(swd, 4),
              " centroid_mse=", round(centroid_mse, 4))
    }

    results[[i]] <- list(
      species = sp, n_true = n_true,
      energy_dist = energy_dist, swd = swd, centroid_mse = centroid_mse,
      is_plot = is_plot,
      true_mat = if (is_plot) true_env_mat else NULL,
      gen_mat = if (is_plot) gen_env_mat else NULL
    )

    if (!is_plot && i %% 10 == 0) {
      message("    ... ", i, "/", n_ds_species, " species evaluated")
    }

    gc()
    cuda_empty_cache()
  }

  results <- purrr::compact(results)

  mean_energy <- mean(purrr::map_dbl(results, "energy_dist"))
  mean_swd <- mean(purrr::map_dbl(results, "swd"))
  mean_centroid <- mean(purrr::map_dbl(results, "centroid_mse"))

  message("  Downstream mean (", length(results), " spp): ",
          "Energy=", round(mean_energy, 4),
          " | SWD=", round(mean_swd, 4),
          " | Centroid MSE=", round(mean_centroid, 4))

  # Generate plots
  plot_results <- purrr::keep(results, \(r) r$is_plot)
  if (!is.null(checkpoint_dir) && !is.null(epoch) && length(plot_results) > 0) {
    tryCatch({
      generate_geoencoder_downstream_scatter_plots(
        plot_results, checkpoint_dir, epoch
      )
    }, error = \(e) {
      message("  Downstream scatter plot error: ", e$message)
    })
    tryCatch({
      generate_geoencoder_downstream_hexbin_plots(
        plot_results, checkpoint_dir, epoch
      )
    }, error = \(e) {
      message("  Downstream hex-bin plot error: ", e$message)
    })
  }

  list(
    mean_energy = mean_energy,
    mean_swd = mean_swd,
    mean_centroid = mean_centroid,
    per_species = purrr::map(results, \(r) {
      list(species = r$species, energy_dist = r$energy_dist,
           swd = r$swd, centroid_mse = r$centroid_mse, n_true = r$n_true)
    })
  )
}


#' Generate embedding scatter plots for geo-encoder validation
#'
#' Shows predicted vs true embeddings in top-variance dimension pairs.
#' Each species has many predicted embeddings (from corrupted versions)
#' clustering around the single true embedding point.
generate_geoencoder_embedding_plots <- function(pred_mat, true_mat,
                                                 species_vec,
                                                 n_plot_species = 5L,
                                                 checkpoint_dir = NULL,
                                                 epoch = NULL) {
  library(ggplot2)
  library(patchwork)

  embed_dim <- ncol(pred_mat)

  # Select top-3 variance dimensions
  col_vars <- apply(pred_mat, 2, var)
  top_dims <- order(col_vars, decreasing = TRUE)
  if (embed_dim >= 3) {
    pairs <- list(
      c(top_dims[1], top_dims[2]),
      c(top_dims[1], top_dims[3]),
      c(top_dims[2], top_dims[3])
    )
  } else {
    pairs <- list(c(1, 2))
  }

  # Select species to plot
  unique_sp <- unique(species_vec)
  plot_sp <- unique_sp[seq_len(min(n_plot_species, length(unique_sp)))]
  n_pairs <- length(pairs)

  # Compute global per-dim ranges
  dim_min <- rep(Inf, embed_dim)
  dim_max <- rep(-Inf, embed_dim)
  for (sp in plot_sp) {
    sp_mask <- species_vec == sp
    sp_pred <- pred_mat[sp_mask, , drop = FALSE]
    sp_true <- true_mat[sp_mask, , drop = FALSE]
    for (d in seq_len(embed_dim)) {
      vals <- c(sp_pred[, d], sp_true[1, d])
      dim_min[d] <- min(dim_min[d], min(vals))
      dim_max[d] <- max(dim_max[d], max(vals))
    }
  }
  dim_pad <- (dim_max - dim_min) * 0.02
  dim_min <- dim_min - dim_pad
  dim_max <- dim_max + dim_pad

  species_plots <- vector("list", length(plot_sp))

  for (s in seq_along(plot_sp)) {
    sp <- plot_sp[s]
    sp_mask <- species_vec == sp
    sp_pred <- pred_mat[sp_mask, , drop = FALSE]
    sp_true_val <- true_mat[sp_mask, , drop = FALSE][1, , drop = FALSE]

    n_pred <- nrow(sp_pred)
    sp_mse <- mean(rowSums((sp_pred - sp_true_val[rep(1, n_pred), ])^2))
    sp_cos <- mean(
      rowSums(sp_pred * sp_true_val[rep(1, n_pred), ]) /
        (sqrt(rowSums(sp_pred^2)) * sqrt(sum(sp_true_val^2)))
    )

    sp_label <- paste0(sp, "  (n=", n_pred,
                       ", MSE=", round(sp_mse, 3),
                       ", cos=", round(sp_cos, 3), ")")

    pair_plots <- vector("list", n_pairs)
    for (p in seq_along(pairs)) {
      d1 <- pairs[[p]][1]
      d2 <- pairs[[p]][2]

      df <- rbind(
        data.frame(x = sp_pred[, d1], y = sp_pred[, d2], type = "Predicted"),
        data.frame(x = sp_true_val[, d1], y = sp_true_val[, d2], type = "True")
      )

      pp <- ggplot(df, aes(x = x, y = y, colour = type)) +
        geom_point(
          data = df[df$type == "Predicted", ],
          size = 0.5, alpha = 0.4
        ) +
        geom_point(
          data = df[df$type == "True", ],
          size = 4, shape = 18
        ) +
        scale_colour_manual(
          values = c(True = "#2166AC", Predicted = "#B2182B")
        ) +
        coord_cartesian(xlim = c(dim_min[d1], dim_max[d1]),
                        ylim = c(dim_min[d2], dim_max[d2])) +
        labs(x = paste0("Dim ", d1), y = paste0("Dim ", d2)) +
        theme_minimal(base_size = 10) +
        theme(legend.position = if (p == 1) "bottom" else "none",
              legend.title = element_blank(),
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
    sprintf("epoch_%04d_geoencoder_embedding.png", epoch)
  )
  dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)

  ragg::agg_png(output_path, width = 2400,
                height = 450 * length(plot_sp) + 80, res = 150)
  print(combined_plot)
  dev.off()
  message("  Saved embedding plot: ", basename(output_path))
}


#' Generate downstream scatter plots in environmental space
#'
#' Pairwise scatter plots comparing true vs generated environmental
#' distributions for zero-shot species.
generate_geoencoder_downstream_scatter_plots <- function(results,
                                                          checkpoint_dir,
                                                          epoch) {
  library(ggplot2)
  library(patchwork)

  env_dim <- ncol(results[[1]]$true_mat)

  # Select top-3 variance env dimensions
  combined <- rbind(results[[1]]$true_mat, results[[1]]$gen_mat)
  col_vars <- apply(combined, 2, var)
  top_dims <- order(col_vars, decreasing = TRUE)
  if (env_dim >= 3) {
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

  dim_min <- rep(Inf, env_dim)
  dim_max <- rep(-Inf, env_dim)
  for (r in results) {
    for (d in seq_len(env_dim)) {
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
    sp_label <- paste0(r$species,
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
        scale_colour_manual(
          values = c(Truth = "#2166AC", Generated = "#B2182B")
        ) +
        coord_cartesian(xlim = c(dim_min[d1], dim_max[d1]),
                        ylim = c(dim_min[d2], dim_max[d2])) +
        labs(x = paste0("Env ", d1), y = paste0("Env ", d2)) +
        theme_minimal(base_size = 10) +
        theme(legend.position = if (p == 1) "bottom" else "none",
              legend.title = element_blank(),
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
    sprintf("epoch_%04d_geoencoder_downstream_scatter.png", epoch)
  )

  ragg::agg_png(output_path, width = 2400,
                height = 450 * n_species + 80, res = 150)
  print(combined_plot)
  dev.off()
  message("  Saved downstream scatter: ", basename(output_path))
}


#' Generate downstream hex-bin overlap plots in environmental space
#'
#' Hex-bin plots showing proportion of generated vs truth environmental
#' data in each cell. Same style as NichEncoder validation hex-bins.
generate_geoencoder_downstream_hexbin_plots <- function(results,
                                                         checkpoint_dir,
                                                         epoch,
                                                         xbins = 40L) {
  library(ggplot2)
  library(patchwork)

  env_dim <- ncol(results[[1]]$true_mat)

  combined <- rbind(results[[1]]$true_mat, results[[1]]$gen_mat)
  col_vars <- apply(combined, 2, var)
  top_dims <- order(col_vars, decreasing = TRUE)
  if (env_dim >= 3) {
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

  dim_min <- rep(Inf, env_dim)
  dim_max <- rep(-Inf, env_dim)
  for (r in results) {
    for (d in seq_len(env_dim)) {
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
    sp_label <- paste0(r$species,
                       "  (n=", r$n_true,
                       ", E=", round(r$energy_dist, 3),
                       ", SWD=", round(r$swd, 3), ")")

    pair_plots <- vector("list", n_pairs)
    for (p in seq_along(pairs)) {
      d1 <- pairs[[p]][1]
      d2 <- pairs[[p]][2]

      df <- data.frame(
        x = c(r$true_mat[, d1], r$gen_mat[, d1]),
        y = c(r$true_mat[, d2], r$gen_mat[, d2]),
        is_gen = c(rep(0, r$n_true), rep(1, nrow(r$gen_mat)))
      )

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
        labs(x = paste0("Env ", d1), y = paste0("Env ", d2)) +
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
    sprintf("epoch_%04d_geoencoder_downstream_hexbin.png", epoch)
  )

  ragg::agg_png(output_path, width = 2600,
                height = 450 * n_species + 80, res = 150)
  print(combined_plot)
  dev.off()
  message("  Saved downstream hex-bin: ", basename(output_path))
}


#' Collate a batch of geo-encoder samples with padding
#'
#' Pads variable-length coordinate sets to the maximum length in the batch,
#' creates padding mask, and looks up target embeddings.
#'
#' @param batch_groups List of lists, each with: species, coords (n x 2 matrix)
#' @param target_embeddings Matrix of species embeddings (n_species x embed_dim)
#' @param species_map Named integer vector (species_name -> id)
#' @param device Torch device string
#' @param jitter_sd SD for coordinate jitter augmentation (0 = no jitter)
#' @param dropout_frac Fraction of points to randomly drop (0 = no dropout)
#' @return List with coords (batch x max_len x 2), mask (batch x max_len),
#'   target_emb (batch x embed_dim)
collate_geoencoder_batch <- function(batch_groups, target_embeddings,
                                      species_map, device,
                                      jitter_sd = 0, dropout_frac = 0) {

  batch_size <- length(batch_groups)
  embed_dim <- ncol(target_embeddings)

  # Apply dropout (remove random fraction of points)
  if (dropout_frac > 0) {
    batch_groups <- map(batch_groups, \(g) {
      n <- nrow(g$coords)
      keep <- runif(n) >= dropout_frac
      # Ensure at least 1 point
      if (sum(keep) == 0) keep[sample.int(n, 1)] <- TRUE
      g$coords <- g$coords[keep, , drop = FALSE]
      g
    })
  }

  lengths <- map_int(batch_groups, ~ nrow(.x$coords))
  max_len <- max(lengths)

  # Build padded arrays on CPU, then transfer to GPU in one shot
  coords_array <- array(0, dim = c(batch_size, max_len, 2L))
  mask_mat <- matrix(TRUE, nrow = batch_size, ncol = max_len)
  target_mat <- matrix(0, nrow = batch_size, ncol = embed_dim)

  for (i in seq_len(batch_size)) {
    g <- batch_groups[[i]]
    n <- lengths[i]
    coord_mat <- g$coords

    # Apply jitter
    if (jitter_sd > 0) {
      coord_mat <- coord_mat + matrix(rnorm(n * 2, sd = jitter_sd),
                                       nrow = n, ncol = 2)
    }

    coords_array[i, 1:n, ] <- coord_mat
    mask_mat[i, 1:n] <- FALSE  # FALSE = real data, TRUE = padding

    # Look up target embedding
    sp_id <- species_map[g$species]
    target_mat[i, ] <- target_embeddings[sp_id, ]
  }

  # Single CPU→GPU transfer
  list(
    coords = torch_tensor(coords_array, device = device),
    mask = torch_tensor(mask_mat, dtype = torch_bool(), device = device),
    target_emb = torch_tensor(target_mat, device = device)
  )
}

#' Flatten grouped coordinate data into a compact representation
#'
#' Converts a list-of-lists structure into a flat representation with a single
#' coordinate matrix and index vectors. This dramatically reduces the number
#' of R objects, preventing GC from becoming a bottleneck during training.
#'
#' @param groups List of lists, each with $species (character) and $coords (matrix Nx2)
#' @return List with:
#'   - coords_all: single matrix (total_rows x 2) of all coordinates
#'   - group_species: character vector of species per group
#'   - group_start: integer vector of 1-based start row index per group
#'   - group_length: integer vector of number of rows per group
#'   - n_groups: total number of groups
flatten_groups <- function(groups) {
  n_groups <- length(groups)
  group_species <- character(n_groups)
  group_length <- integer(n_groups)

  for (i in seq_len(n_groups)) {
    group_species[i] <- groups[[i]]$species
    group_length[i] <- nrow(groups[[i]]$coords)
  }

  n_total <- sum(group_length)
  group_start <- c(1L, cumsum(group_length[-n_groups]) + 1L)

  coords_all <- matrix(0, nrow = n_total, ncol = 2L)
  for (i in seq_len(n_groups)) {
    rows <- group_start[i]:(group_start[i] + group_length[i] - 1L)
    coords_all[rows, ] <- groups[[i]]$coords
  }

  list(
    coords_all = coords_all,
    group_species = group_species,
    group_start = group_start,
    group_length = group_length,
    n_groups = n_groups
  )
}

#' Collate a batch from flat group representation
#'
#' Like collate_geoencoder_batch but works with the flat representation from
#' flatten_groups(). Avoids creating intermediate R list objects, reducing GC.
#'
#' @param idx Integer vector of group indices for this batch
#' @param flat Flat group representation from flatten_groups()
#' @param target_embeddings Matrix of target embeddings
#' @param species_map Named integer vector mapping species to row indices
#' @param device Torch device string
#' @param jitter_sd SD for coordinate jitter augmentation
#' @param dropout_frac Fraction of points to randomly drop
#' @return List with coords, mask, target_emb tensors
collate_geoencoder_batch_flat <- function(idx, flat, target_embeddings,
                                           species_map, device,
                                           jitter_sd = 0, dropout_frac = 0) {

  batch_size <- length(idx)
  embed_dim <- ncol(target_embeddings)

  # Get lengths, applying dropout fraction to determine effective lengths
  raw_lengths <- flat$group_length[idx]

  if (dropout_frac > 0) {
    # For dropout: compute kept count per group (at least 1)
    kept_lengths <- pmax(1L, as.integer(rbinom(batch_size, raw_lengths,
                                                1 - dropout_frac)))
  } else {
    kept_lengths <- raw_lengths
  }

  max_len <- max(kept_lengths)

  # Build padded arrays on CPU
  coords_array <- array(0, dim = c(batch_size, max_len, 2L))
  mask_mat <- matrix(TRUE, nrow = batch_size, ncol = max_len)
  target_mat <- matrix(0, nrow = batch_size, ncol = embed_dim)

  for (i in seq_len(batch_size)) {
    gi <- idx[i]
    start <- flat$group_start[gi]
    raw_n <- raw_lengths[i]
    n <- kept_lengths[i]

    # Extract coordinates from flat matrix
    rows <- start:(start + raw_n - 1L)

    if (dropout_frac > 0 && n < raw_n) {
      # Subsample rows (dropout)
      rows <- rows[sample.int(raw_n, n)]
    }

    coord_mat <- flat$coords_all[rows, , drop = FALSE]

    # Apply jitter
    if (jitter_sd > 0) {
      coord_mat <- coord_mat + matrix(rnorm(n * 2L, sd = jitter_sd),
                                       nrow = n, ncol = 2L)
    }

    coords_array[i, 1:n, ] <- coord_mat
    mask_mat[i, 1:n] <- FALSE

    # Look up target embedding
    sp_id <- species_map[flat$group_species[gi]]
    target_mat[i, ] <- target_embeddings[sp_id, ]
  }

  # Single CPU->GPU transfer
  list(
    coords = torch_tensor(coords_array, device = device),
    mask = torch_tensor(mask_mat, dtype = torch_bool(), device = device),
    target_emb = torch_tensor(target_mat, device = device)
  )
}
