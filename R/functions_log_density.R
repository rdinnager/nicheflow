#' IWAE Log-Density Estimation with NichEncoder Flow Prior
#'
#' Estimates log p(x | species) for environmental observations using the
#' unconditional VAE + NichEncoder rectified flow as a species-conditional
#' prior. Uses importance-weighted autoencoder (IWAE) bound with Hutchinson
#' trace estimation for the flow log-determinant.

library(zeallot)  # for %<-%

# ---------------------------------------------------------------------------
# Helper: Diagonal Gaussian log-density
# ---------------------------------------------------------------------------

#' Log probability under diagonal Gaussian, summed over last dim
#'
#' @param sample Tensor [*, D]
#' @param mean Tensor [*, D]
#' @param logvar Tensor [*, D]
#' @return Tensor [*] (summed over D)
.log_normal_pdf <- function(sample, mean, logvar) {
  const <- -0.5 * log(2 * pi)
  log_probs <- const - 0.5 * logvar - 0.5 * ((sample - mean)^2 / torch_exp(logvar))
  torch_sum(log_probs, dim = -1)
}


# ---------------------------------------------------------------------------
# Helper: Resolve species conditioning to embedding tensor
# ---------------------------------------------------------------------------

#' Resolve species argument to embedding tensor
#'
#' Accepts integer species IDs (looked up via flow_model$species_embedding),
#' raw embedding matrices/vectors, or single values broadcast to n rows.
#'
#' @param species Integer vector [N], matrix [N, embed_dim], single integer,
#'   or single numeric vector of length embed_dim
#' @param flow_model nichencoder_traj_net instance (for embedding lookup)
#' @param n Number of data points (for broadcasting)
#' @param device Torch device
#' @return Tensor [n, spec_embed_dim] on device
.resolve_species_embedding <- function(species, flow_model, n, device) {
  if (is.matrix(species)) {
    # Raw embedding matrix [N, embed_dim]
    emb <- torch_tensor(species, device = device)
    if (emb$shape[1] != n) {
      stop("species embedding matrix has ", emb$shape[1], " rows but data has ", n)
    }
    return(emb)
  }

  if (is.numeric(species) && length(species) > 1 && !is.integer(species)) {
    # Single numeric vector (embed_dim length) — broadcast to n rows
    emb <- torch_tensor(matrix(species, nrow = 1), device = device)
    return(emb$expand(c(n, -1L)))
  }

  if (is.integer(species) || (is.numeric(species) && all(species == floor(species)))) {
    species <- as.integer(species)
    ids <- torch_tensor(species, dtype = torch_long(), device = device)
    if (length(species) == 1L) {
      ids <- ids$expand(c(n))
    }
    with_no_grad({
      emb <- flow_model$species_embedding(ids)
    })
    return(emb)
  }

  stop("species must be integer IDs, a numeric embedding vector, or a matrix of embeddings")
}


# ---------------------------------------------------------------------------
# Core: Reverse ODE log-density via Hutchinson trace
# ---------------------------------------------------------------------------

#' Compute log p_flow(z_active | species) via reverse ODE + Hutchinson trace
#'
#' Integrates the NichEncoder velocity field backward from t=1 (data) to
#' t=0 (noise), accumulating the log-determinant via Hutchinson trace
#' estimation. Returns log p(z₀; N(0,I)) + log_det.
#'
#' @param z_active Tensor [M, active_dim] — latent codes at t=1
#' @param spec_enc Tensor [M, spec_encode_dim] — pre-encoded species vectors
#' @param flow_model nichencoder_traj_net instance
#' @param ode_steps Number of reverse Euler steps
#' @param n_hutchinson Number of Hutchinson probe vectors per step
#' @return Tensor [M] of log-densities under the flow
.reverse_ode_log_density <- function(z_active, spec_enc, flow_model,
                                      ode_steps = 50L, n_hutchinson = 1L) {
  dt <- 1.0 / ode_steps
  n <- z_active$shape[1]
  active_dim <- z_active$shape[2]

  y <- z_active$detach()$clone()
  log_det <- torch_zeros(n, device = y$device)

  for (s in seq(ode_steps, 1)) {
    # Current time (going backward: t = s/steps -> (s-1)/steps)
    t_val <- s * dt
    t_tensor <- torch_full(c(n, 1L), t_val, device = y$device)
    t_enc <- flow_model$encode_t(t_tensor)

    # Enable gradients for trace computation
    y <- y$detach()$requires_grad_(TRUE)

    # Velocity at current point
    velocity <- flow_model$unet(
      input = y, t_encoded = t_enc,
      spec_encoded = spec_enc
    )

    # Hutchinson trace estimate: tr(dv/dy) ≈ ε^T (dv/dy) ε
    trace_est <- torch_zeros(n, device = y$device)
    for (h in seq_len(n_hutchinson)) {
      eps <- torch_randn_like(y)
      # VJP: grad_outputs = eps gives ε^T J
      vjp <- torch::autograd_grad(
        outputs = velocity,
        inputs = y,
        grad_outputs = eps,
        retain_graph = (h < n_hutchinson),
        create_graph = FALSE
      )
      # ε^T J ε = sum(eps * vjp) per sample
      trace_est <- trace_est + torch_sum(eps * vjp[[1]], dim = -1L)
    }
    trace_est <- trace_est / n_hutchinson

    # Accumulate log-determinant (negative because reverse direction)
    log_det <- log_det - trace_est * dt

    # Reverse Euler step: y_{s-1} = y_s - v * dt
    y <- y$detach() - velocity$detach() * dt
  }

  # Base distribution: log N(y; 0, I)
  log_p_base <- -0.5 * active_dim * log(2 * pi) -
    0.5 * torch_sum(y^2, dim = -1L)

  log_p_base + log_det
}


# ---------------------------------------------------------------------------
# Core: Single batch IWAE log-density
# ---------------------------------------------------------------------------

#' Compute IWAE log-density for a single batch
#'
#' @param env_batch Tensor [B, input_dim] — standardized environmental data
#' @param spec_emb_batch Tensor [B, spec_embed_dim] — resolved species embeddings
#' @param vae_model env_vae_mod instance (eval mode, on device)
#' @param flow_model nichencoder_traj_net instance (eval mode, on device)
#' @param active_dims Integer vector of active latent dimensions (1-based)
#' @param K Number of importance samples
#' @param ode_steps Reverse ODE steps
#' @param n_hutchinson Hutchinson probes per ODE step
#' @return Tensor [B] of log p(x | species)
.compute_log_density_batch <- function(env_batch, spec_emb_batch,
                                        vae_model, flow_model,
                                        active_dims, K = 10L,
                                        ode_steps = 50L, n_hutchinson = 1L) {
  B <- env_batch$shape[1]
  latent_dim <- vae_model$latent_dim
  input_dim <- vae_model$input_dim
  active_dim <- length(active_dims)
  inactive_dims <- setdiff(seq_len(latent_dim), active_dims)

  # --- Step 1: Encode ---
  with_no_grad({
    c(means, logvars) %<-% vae_model$encoder(env_batch)
    # means, logvars: [B, latent_dim]
  })

  # --- Step 2: Sample K latent codes ---
  # Expand to [B, K, latent_dim]
  mu_k <- means$unsqueeze(2L)$expand(c(B, K, latent_dim))
  lv_k <- logvars$unsqueeze(2L)$expand(c(B, K, latent_dim))
  std_k <- torch_exp(0.5 * lv_k)
  eps <- torch_randn_like(mu_k)
  z_k <- mu_k + std_k * eps
  # z_k: [B, K, latent_dim]

  # --- Step 3: log q(z | x) ---
  log_q_z <- .log_normal_pdf(z_k, mu_k, lv_k)  # [B, K]

  # --- Step 4: log p(z) = log p(z_active | species) + log p(z_inactive) ---
  # Split active/inactive dims
  z_flat <- z_k$reshape(c(B * K, latent_dim))

  # Inactive dims: standard normal prior
  if (length(inactive_dims) > 0) {
    z_inactive <- z_flat[, inactive_dims]
    log_p_inactive <- -0.5 * length(inactive_dims) * log(2 * pi) -
      0.5 * torch_sum(z_inactive^2, dim = -1L)  # [B*K]
  } else {
    log_p_inactive <- torch_zeros(B * K, device = env_batch$device)
  }

  # Active dims: flow density
  z_active <- z_flat[, active_dims]  # [B*K, active_dim]

  # Expand species embeddings: [B, embed_dim] -> [B*K, embed_dim]
  spec_emb_k <- spec_emb_batch$unsqueeze(2L)$expand(
    c(B, K, spec_emb_batch$shape[2])
  )$reshape(c(B * K, spec_emb_batch$shape[2]))

  # Pre-encode species for the flow model
  with_no_grad({
    spec_enc <- flow_model$encode_spec(spec_emb_k)
  })

  # Reverse ODE for flow density (this part needs gradients for trace)
  log_p_active <- .reverse_ode_log_density(
    z_active, spec_enc, flow_model,
    ode_steps = ode_steps, n_hutchinson = n_hutchinson
  )  # [B*K]

  log_prior <- (log_p_active + log_p_inactive)$reshape(c(B, K))  # [B, K]

  # --- Step 5: log p(x | z) ---
  with_no_grad({
    x_recon <- vae_model$decoder(z_flat)  # [B*K, input_dim]
    x_recon <- x_recon$reshape(c(B, K, input_dim))

    # Gaussian reconstruction log-likelihood with learned gamma
    loggamma <- vae_model$loggamma
    x_expanded <- env_batch$unsqueeze(2L)$expand(c(B, K, input_dim))
    log_p_x_given_z <- .log_normal_pdf(
      x_recon, x_expanded,
      torch_ones_like(x_recon) * loggamma
    )  # [B, K]
  })

  # --- Step 6: IWAE ---
  log_weights <- log_p_x_given_z + log_prior - log_q_z  # [B, K]
  log_density <- torch_logsumexp(log_weights, dim = 2L) -
    log(K)  # [B]

  log_density
}


# ---------------------------------------------------------------------------
# Main API: compute_log_density
# ---------------------------------------------------------------------------

#' Estimate log p(x | species) via IWAE with NichEncoder flow prior
#'
#' Combines an unconditional VAE encoder/decoder with the NichEncoder
#' rectified flow as a species-conditional prior to estimate the marginal
#' log-likelihood of environmental observations for given species.
#'
#' @param env_data Matrix [N, input_dim], standardized environmental observations
#' @param species Species conditioning — one of:
#'   - Integer vector [N] of 1-based species IDs
#'   - Matrix [N, spec_embed_dim] of raw embedding vectors (zero-shot)
#'   - Single integer (applied to all N points)
#'   - Single numeric vector of length spec_embed_dim (applied to all N)
#' @param vae_model env_vae_mod instance (eval mode, on device)
#' @param flow_model nichencoder_traj_net instance (eval mode, on device)
#' @param active_dims Integer vector of active VAE latent dimensions (1-based),
#'   e.g. c(7L, 9L, 11L, 13L, 15L, 16L)
#' @param K Number of importance samples (higher = tighter bound)
#' @param ode_steps Number of reverse ODE steps for density estimation
#' @param n_hutchinson Number of Hutchinson probe vectors per ODE step
#' @param batch_size Processing batch size (for memory management)
#' @param device Torch device string
#' @return Numeric vector [N] of log p(x_i | species_i)
compute_log_density <- function(env_data, species,
                                 vae_model, flow_model,
                                 active_dims,
                                 K = 10L, ode_steps = 50L,
                                 n_hutchinson = 1L,
                                 batch_size = 1000L,
                                 device = "cpu") {
  N <- nrow(env_data)

  # Resolve species to embedding tensor [N, embed_dim]
  spec_emb <- .resolve_species_embedding(species, flow_model, N, device)

  # Process in batches
  results <- numeric(N)
  n_batches <- ceiling(N / batch_size)

  for (b in seq_len(n_batches)) {
    b_start <- (b - 1L) * batch_size + 1L
    b_end <- min(b * batch_size, N)
    idx <- b_start:b_end

    env_batch <- torch_tensor(env_data[idx, , drop = FALSE], device = device)
    spec_batch <- spec_emb[idx, ]

    log_d <- .compute_log_density_batch(
      env_batch, spec_batch,
      vae_model, flow_model,
      active_dims, K = K,
      ode_steps = ode_steps,
      n_hutchinson = n_hutchinson
    )

    results[idx] <- as.numeric(log_d$detach()$cpu())

    if (b %% 10 == 0 || b == n_batches) {
      message("  Batch ", b, "/", n_batches, " complete")
    }

    # Free GPU memory
    gc(verbose = FALSE)
    if (grepl("cuda", device)) cuda_empty_cache()
  }

  results
}


# ---------------------------------------------------------------------------
# Convenience: Load from checkpoints
# ---------------------------------------------------------------------------

#' Compute log-density from saved checkpoints
#'
#' Loads VAE and NichEncoder from checkpoints, computes log p(x | species).
#'
#' @param env_data Matrix [N, input_dim]
#' @param species Flexible species arg (see compute_log_density)
#' @param vae_checkpoint Path to VAE checkpoint file
#' @param flow_checkpoint_dir Directory containing NichEncoder checkpoints
#' @param active_dims Active VAE latent dimensions
#' @param vae_input_dim VAE input dimension (default 31)
#' @param vae_latent_dim VAE latent dimension (default 16)
#' @param flow_coord_dim NichEncoder coordinate dimension (default 6)
#' @param flow_n_species Number of species in NichEncoder (default 18121)
#' @param flow_spec_embed_dim Species embedding dimension (default 64)
#' @param flow_breadths NichEncoder U-Net breadths
#' @param K Importance samples
#' @param ode_steps Reverse ODE steps
#' @param n_hutchinson Hutchinson probes
#' @param batch_size Processing batch size
#' @param device Torch device
#' @return Numeric vector [N] of log p(x_i | species_i)
compute_log_density_from_checkpoints <- function(
    env_data, species,
    vae_checkpoint, flow_checkpoint_dir,
    active_dims = c(7L, 9L, 11L, 13L, 15L, 16L),
    vae_input_dim = 31L, vae_latent_dim = 16L,
    flow_coord_dim = 6L, flow_n_species = 18121L,
    flow_spec_embed_dim = 64L,
    flow_breadths = c(512L, 256L, 128L),
    K = 10L, ode_steps = 50L, n_hutchinson = 1L,
    batch_size = 1000L, device = "cpu") {

  # Load VAE
  message("Loading VAE from: ", vae_checkpoint)
  vae_model <- env_vae_mod(vae_input_dim, vae_latent_dim)
  load_model_checkpoint(vae_model, vae_checkpoint)
  vae_model <- vae_model$to(device = device)
  vae_model$eval()

  # Load NichEncoder flow
  flow_ckpt <- find_latest_checkpoint(flow_checkpoint_dir)
  message("Loading NichEncoder from: ", flow_ckpt$path,
          " (epoch ", flow_ckpt$epoch, ")")
  flow_model <- nichencoder_traj_net(
    coord_dim = flow_coord_dim,
    n_species = flow_n_species,
    spec_embed_dim = flow_spec_embed_dim,
    breadths = flow_breadths
  )
  load_model_checkpoint(flow_model, flow_ckpt$path)
  flow_model <- flow_model$to(device = device)
  flow_model$eval()

  compute_log_density(
    env_data, species,
    vae_model, flow_model,
    active_dims,
    K = K, ode_steps = ode_steps,
    n_hutchinson = n_hutchinson,
    batch_size = batch_size,
    device = device
  )
}


# ---------------------------------------------------------------------------
# Latent-space only: log p(z_active | species) without IWAE
# ---------------------------------------------------------------------------

#' Compute flow log-density in latent space only (no VAE)
#'
#' Estimates log p(z_active | species) directly from latent codes.
#' Faster than full IWAE but only gives density in the 6-dim active
#' latent subspace, not in the original 31-dim environmental space.
#'
#' @param z_active Matrix [N, active_dim] of latent codes (active dims only)
#' @param species Flexible species arg (see compute_log_density)
#' @param flow_model nichencoder_traj_net instance (eval mode, on device)
#' @param ode_steps Reverse ODE steps
#' @param n_hutchinson Hutchinson probes per step
#' @param batch_size Processing batch size
#' @param device Torch device
#' @return Numeric vector [N] of log p(z_i | species_i)
compute_latent_log_density <- function(z_active, species,
                                        flow_model,
                                        ode_steps = 50L,
                                        n_hutchinson = 1L,
                                        batch_size = 1000L,
                                        device = "cpu") {
  N <- nrow(z_active)
  spec_emb <- .resolve_species_embedding(species, flow_model, N, device)

  results <- numeric(N)
  n_batches <- ceiling(N / batch_size)

  for (b in seq_len(n_batches)) {
    b_start <- (b - 1L) * batch_size + 1L
    b_end <- min(b * batch_size, N)
    idx <- b_start:b_end

    z_batch <- torch_tensor(z_active[idx, , drop = FALSE], device = device)
    spec_batch <- spec_emb[idx, ]

    with_no_grad({
      spec_enc <- flow_model$encode_spec(spec_batch)
    })

    log_d <- .reverse_ode_log_density(
      z_batch, spec_enc, flow_model,
      ode_steps = ode_steps, n_hutchinson = n_hutchinson
    )

    results[idx] <- as.numeric(log_d$detach()$cpu())

    if (b %% 10 == 0 || b == n_batches) {
      message("  Batch ", b, "/", n_batches, " complete")
    }

    gc(verbose = FALSE)
    if (grepl("cuda", device)) cuda_empty_cache()
  }

  results
}
