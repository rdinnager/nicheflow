#' Environmental VAE Model Definition
#'
#' Unconditional VAE that learns the intrinsic dimensionality of environmental
#' space via a trainable loggamma parameter. Used by train_env_vae.R for
#' training and by functions_vae_encoding.R for inference.

env_vae_mod <- nn_module("EnvVAE",
  initialize = function(input_dim, latent_dim, loggamma_init = -3) {
    self$latent_dim <- latent_dim
    self$input_dim <- input_dim
    self$encoder <- nndag(
      i_1 = ~ input_dim,
      e_1 = i_1 ~ latent_dim * 2,
      e_2 = e_1 ~ latent_dim * 2,
      e_3 = e_2 ~ latent_dim * 2,
      means = e_3 ~ latent_dim,
      logvars = e_3 ~ latent_dim,
      .act = list(nn_relu,
                  logvars = nn_identity,
                  means = nn_identity)
    )
    self$decoder <- nndag(
      z = ~ latent_dim,
      d_1 = z ~ latent_dim * 2,
      d_2 = d_1 ~ latent_dim * 2,
      d_3 = d_2 ~ latent_dim * 2,
      out = d_3 ~ input_dim,
      .act = list(nn_relu,
                  out = nn_identity)
    )
    self$loggamma <- nn_parameter(torch_tensor(loggamma_init))
  },

  reparameterize = function(mean, logvar) {
    std <- torch_exp(torch_tensor(0.5, device = mean$device) * logvar)
    eps <- torch_randn_like(std)
    eps * std + mean
  },

  loss_function = function(reconstruction, input, mean, log_var) {
    kl <- torch_sum(
      torch_exp(log_var) + torch_square(mean) - log_var, dim = 2L
    ) - self$latent_dim
    recon1 <- torch_sum(
      torch_square(input - reconstruction), dim = 2L
    ) / torch_exp(self$loggamma)
    recon2 <- self$input_dim * self$loggamma +
      torch_log(torch_tensor(2 * pi, device = input$device)) * self$input_dim
    loss <- torch_mean(recon1 + recon2 + kl)
    list(loss, torch_mean(recon1 * torch_exp(self$loggamma)), torch_mean(kl))
  },

  forward = function(x) {
    c(means, log_vars) %<-% self$encoder(x)
    z <- self$reparameterize(means, log_vars)
    list(self$decoder(z), x, means, log_vars)
  }
)
