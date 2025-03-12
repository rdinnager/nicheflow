library(torch)
library(dagnn)
library(zeallot)

transformer_block <- nn_module("TransformerBlock",
                               initialize = function(embed_dim, num_heads = 8L, dropout_prob = 0.1) {
                                 
                                 self$embed_dim <- embed_dim
                                 self$linear_q <- nn_linear(embed_dim, embed_dim)
                                 self$linear_k <- nn_linear(embed_dim, embed_dim)
                                 self$linear_v <- nn_linear(embed_dim, embed_dim)
                                 self$layernorm_pre_att <- nn_layer_norm(embed_dim)
                                 self$layernorm_pre_mlp <- nn_layer_norm(embed_dim)
                                 self$attention <- nn_multihead_attention(embed_dim, num_heads = num_heads,
                                                                          batch_first = TRUE,
                                                                          dropout = dropout_prob)
                                 self$output_dropout <- nn_dropout(dropout_prob)
                                 self$mlp <- nn_sequential(nn_linear(embed_dim, embed_dim * 4),
                                                           nn_gelu(),
                                                           nn_dropout(dropout_prob),
                                                           nn_linear(embed_dim * 4, embed_dim))
                                 self$mlp_dropout <- nn_dropout(dropout_prob)
                               },
                               forward = function(input, mask) {
                                 #browser()
                                 x <- self$layernorm_pre_att(input)
                                 q <- self$linear_q(x)
                                 k <- self$linear_k(x)
                                 v <- self$linear_v(x)
                                 x <- self$attention(q, k, v, key_padding_mask = mask)
                                 x <- self$output_dropout(x[[1]]) + input
                                 x <- self$layernorm_pre_mlp(x)
                                 x <- self$mlp_dropout(self$mlp(x))
                                 return(x + input)
                                 
                               })

nichencoder_transformer <- nn_module("NichEncoderTransformer",
                                     initialize = function(input_dim = 2L, embed_dim = 384L, output_dim = 32L, n_blocks = 16L, num_heads = 8L, dropout_prob = 0.1) {
                                       self$input_dim <- input_dim
                                       self$embed_dim <- embed_dim
                                       self$embedding <- nn_linear(input_dim, embed_dim)
                                       self$embed_dropout <- nn_dropout(p = dropout_prob)
                                       self$blocks <- nn_module_list(map(seq_len(n_blocks),
                                                                         ~ transformer_block(embed_dim, num_heads = num_heads, dropout_prob = dropout_prob)))
                                       self$mlp_final <- nn_sequential(nn_linear(embed_dim, embed_dim * 4),
                                                                       nn_gelu(),
                                                                       nn_dropout(dropout_prob),
                                                                       nn_linear(embed_dim * 4, output_dim))
                                       },
                                     forward = function(input, mask) {
                                       x <- self$embed_dropout(self$embedding(input))
                                       for(i in 1:length(self$blocks)) {
                                         x <- self$blocks[[i]](x, mask)
                                       }
                                       x <- torch_sum(x, dim = 2L)
                                       out <- self$mlp_final(x)
                                       out
                                     })


nichencoder_vae <- nn_module("NichEncoderVAE",
                             initialize = function(input_dim, spec_embed_dim, latent_dim, breadth = 1024L, loggamma_init = 0, lambda = 1, alpha = 0) {
                               self$latent_dim <- latent_dim
                               self$input_dim <- input_dim
                               self$spec_embed_dim <- spec_embed_dim
                               self$lambda <- lambda
                               self$alpha <- alpha
                               self$encoder <- nndag(y = ~ input_dim,
                                                     s = ~ spec_embed_dim,
                                                     e_1 = y + s ~ breadth,
                                                     e_2 = e_1 + s ~ breadth,
                                                     e_3 = e_2 + s ~ breadth,
                                                     means = e_3 ~ latent_dim,
                                                     logvars = e_3 ~ latent_dim,
                                                     .act = list(nn_relu,
                                                                 logvars = nn_identity,
                                                                 means = nn_identity))
                               
                               self$decoder <- nndag(z = ~ latent_dim,
                                                     s = ~ spec_embed_dim,
                                                     d_1 = z + s ~ breadth,
                                                     d_2 = d_1 + s ~ breadth,
                                                     d_3 = d_2 + s ~ breadth,
                                                     out = d_3 ~ input_dim,
                                                     .act = list(nn_relu,
                                                                 out = nn_identity))
                               
                               self$loggamma <- nn_parameter(torch_tensor(loggamma_init))
                               
                             },
                             sample_latent = function(mean, logvar, K = 50) {
                               # Reparameterization trick
                               std = torch_exp(0.5 * logvar)
                               
                               # For K samples, expand dimensions
                               batch_size = mean$size(1)
                               
                               # Expand mu and std to [batch_size, K, latent_dim]
                               mean_expanded = mean$unsqueeze(2)$expand(c(-1, K, -1))  # [B, K, D]
                               std_expanded = std$unsqueeze(2)$expand(c(-1, K, -1))  # [B, K, D]
                               
                               # Sample epsilon from normal distribution
                               eps = torch_randn_like(std_expanded)  # [B, K, D]
                               
                               # Get K samples for each input
                               z = mean_expanded + eps * std_expanded  # [B, K, D]
                               
                               z
                             },
                             .log_normal_pdf = function(sample, mean, logvar) {
                               # Compute log probability of sample under a normal distribution
                               # sample: [B, K, D], mean: [B, K, D], logvar: [B, K, D]
                               const <- -0.5 * torch_log(2 * torch_tensor(pi, device = mean$device))
                               log_probs <- const - 0.5 * logvar - 0.5 * ((sample - mean) ^ 2 / torch_exp(logvar))
                               # Sum over dimensions
                               return(torch_sum(log_probs, dim = -1))  # [B, K]
                             }, 
                             compute_log_probs = function(x, z, mu, logvar, mask, s) {
                               #browser()
                               c(batch_size, K, latent_dim) %<-% z$shape
                               # 1. Log prior: log p(z) - standard Gaussian prior
                               log_prior <- self$.log_normal_pdf(z, 
                                                                 torch_zeros_like(z, device = z$device), 
                                                                 torch_zeros_like(z, device = z$device))  # [B, K]
                               # 2. Log encoder prob: log q(z|x)
                               log_q_z <- self$.log_normal_pdf(z,
                                                               mu$unsqueeze(2), 
                                                               logvar$unsqueeze(2))  # [B, K]
                               # 3. Log decoder prob: log p(x|z)
                               # Reshape z to process through decoder
                               z_reshape <- z$reshape(c(-1, latent_dim))  # [B*K, D]
                               #s_reshape <- s$reshape(c(-1, latent_dim))  # [B*K, D]
                               s_reshape <- s$unsqueeze(2)$expand(c(-1, K, -1))$reshape(c(-1, self$spec_embed_dim))
                               
                               # Get reconstructed x for each z
                               x_recon <- self$decoder(z = z_reshape, s = s_reshape)  # [B*K, input_dim]
                               
                               # Reshape back to [B, K, input_dim]
                               x_recon <- x_recon$reshape(c(batch_size, K, -1))
                               
                               log_p_x_given_z <- self$.log_normal_pdf(x_recon * mask$unsqueeze(2), (x * mask)$unsqueeze(2), self$loggamma)
                               log_p_x_given_z <- (log_p_x_given_z / self$input_dim) * torch_sum(mask, dim = 2, keepdim = TRUE)
                               
                               return(list(log_p_x_given_z, log_q_z, log_prior))
                             },
                             iwae_loss = function(log_p_x_given_z, log_q_z, log_prior, K) {
                               # Compute unnormalized weights
                               log_weights <- log_p_x_given_z + log_prior - log_q_z  # [B, K]
                               
                               # Use the logsumexp trick for numerical stability
                               # log(1/K * sum(exp(log_w_i))) = log(sum(exp(log_w_i))) - log(K)
                               iwae_obj <- torch_logsumexp(log_weights, dim = 2) - torch_log(K)$to(device = log_weights$device)  # [B]
                               prior_loss <- torch_logsumexp(log_weights - log_p_x_given_z, dim = 2) - torch_log(K)$to(device = log_weights$device)
                               
                               # Negative because we want to maximize this objective
                               return(list(iwae_obj, prior_loss$detach()))
                             },
                             loss_function = function(z, input, mask, mean, 
                                                      log_var, s,
                                                      K = 50, 
                                                      reduce = torch::torch_mean) {
                               #browser()
                               # Compute all log probabilities
                               c(log_p_x_given_z, log_q_z, log_prior) %<-% self$compute_log_probs(input, z, mean, log_var, mask, s)
                               
                               # Compute IWAE loss
                               c(iwae_loss, log_prior) %<-% self$iwae_loss(log_p_x_given_z, log_q_z, log_prior, K)
                               
                               #kl <- torch_sum(torch_exp(log_var) + torch_square(mean) - log_var, dim = 2L) - self$latent_dim
                               #kl_spec <- (((1 - alpha) * torch_sum(torch_square(mean_spec), dim = 2L) + alpha * torch_sum(torch_abs(mean_spec), dim = 2L)) * lambda)
                               #recon1 <- torch_sum(torch_square(input - reconstruction) * mask, dim = 2L) / torch_exp(self$loggamma)
                               #recon2 <- self$input_dim * self$loggamma + torch_log(torch_tensor(2 * pi, device = "cuda")) * self$input_dim
                               #recon_loss <- (torch_sum(recon1) / torch_sum(mask)) + torch_mean(recon2)
                               #spec_loss <- torch_mean(kl_spec)
                               #loss <- iwae_loss #+ spec_loss
                               if(!is.null(reduce)) {
                                 return(list(-reduce(iwae_loss), -reduce(log_prior)))
                               }
                               list(iwae_loss, log_prior)
                             },
                             encode = function(x, s) {
                               self$encoder(x, spec_embedding)
                             },
                             ## s is the species embedding
                             decode = function(z, s) {
                               
                               c(batch_size, K, latent_dim) %<-% z$shape
                               # Reshape z to process through decoder
                               z_reshape <- z$reshape(c(-1, latent_dim))  # [B*K, D]
                               s_reshape <- s$unsqueeze(2)$expand(c(-1, K, -1))$reshape(c(-1, self$spec_embed_dim))
                               
                               # Get reconstructed x for each z
                               x_recon <- self$decoder(z = z_reshape, s = s_reshape)  # [B*K, input_dim]
                               
                               # Reshape back to [B, K, input_dim]
                               x_recon <- x_recon$reshape(c(batch_size, K, -1))
                               x_recon
                             },
                             forward = function(x, s, K = 50) {
                               #browser()
                               
                               c(means, log_vars) %<-% self$encoder(y = x, s = s)
                               z <- self$sample_latent(means, log_vars, K)
                               
                               list(x, z, means, log_vars, s)
                               
                               #list(self$decoder(z = z, s = spec_embedding_mean), x, spec_embedding_mean, means, log_vars)
                             }
                             
)

nichencoder_trainer <- nn_module("NichEncoderTrainer",
                                 initialize = function(input_dim, #max_pnts, 
                                                       spec_embed_dim, latent_dim, 
                                                       breadth = 1024L, loggamma_init = 0,
                                                       transformer_embed_dim = 384L, 
                                                       transformer_output_dim = 32L, 
                                                       n_blocks = 16L, num_heads = 8L, 
                                                       dropout_prob = 0.1) {
                                   
                                   self$nichencoder <- nichencoder_vae(input_dim, spec_embed_dim,
                                                                       latent_dim, breadth, loggamma_init)
                                   self$transformer <- nichencoder_transformer(input_dim, transformer_embed_dim,
                                                                               transformer_output_dim,
                                                                               n_blocks, num_heads, dropout_prob)
                                   self$spec_embedder_mu <- nn_linear(transformer_output_dim, spec_embed_dim)
                                   self$spec_embedder_var <- nn_linear(transformer_output_dim, spec_embed_dim)
                                   
                                 },
                                 spec_reparameterize = function(mean, logvar) {
                                   std <- torch_exp(torch_tensor(0.5, device = "cuda") * logvar)
                                   eps <- torch_randn_like(std)
                                   eps * std + mean
                                 },
                                 spec_encode = function(env, padding_mask) {
                                   x <- self$transformer(env, padding_mask)
                                   mu_s <- self$spec_embedder_mu(x)
                                   logvar_s <- self$spec_embedder_var(x)
                                   list(mu_s, logvar_s)
                                 },
                                 forward = function(env, padding_mask, na_mask, K = 50) {
                                   #browser()
                                   c(batch_size, pnts, input_dim) %<-% env$shape
                                   #env <- env[ , torch_randperm(pnts, device = env$device) + 1L, ]
                                   c(mu_s, logvar_s) %<-% self$spec_encode(env, padding_mask)
                                   s <- self$spec_reparameterize(mu_s, logvar_s)
                                   s_reshape <- s$unsqueeze(2)$expand(c(-1, pnts, -1))
                                   env_for_vae <- env$detach()
                                   env_reshape <- env_for_vae$reshape(c(-1, input_dim)) 
                                   ind <- !padding_mask$reshape(c(-1))
                                   env_reshape <- env_reshape[ind, ]
                                   #sampsize <- env_reshape$size()[1]
                                   #samp <- (torch_randperm(sampsize, device = env$device) + 1L)[seq_len(ceiling(0.9 * sampsize))]
                                   #env_reshape <- env_reshape[samp, ]
                                   s_reshape <- s_reshape$reshape(c(-1, s_reshape$size(-1)))[ind, ]#[samp, ]
                                   #na_mask <- na_mask[samp, ]
                                   c(x, z, means, log_vars, s) %<-% self$nichencoder(env_reshape, s_reshape, K)
                                   list(x, z, means, log_vars, s, mu_s, logvar_s, na_mask)
                                 })

nichencoder <- nn_module("NichEncoder",
                         initialize = function(input_dim, n_spec, spec_embed_dim, latent_dim, breadth = 1024L, loggamma_init = 0) {
                           self$latent_dim <- latent_dim
                           self$input_dim <- input_dim
                           self$n_spec <- n_spec
                           self$spec_embed_dim <- spec_embed_dim
                           self$loggamma_init <- loggamma_init
                           self$vae <- nichencoder_vae(input_dim, spec_embed_dim, 
                                                       latent_dim, breadth, loggamma_init)
                           self$species_embedder_mean <- nn_embedding(n_spec, spec_embed_dim)
                           
                         },
                         encode = function(x, s, K = 1) {
                           c(input, z, means, log_vars, spec_lat) %<-% self$vae(x, s, K = K)
                           if(K == 1) {
                             z <- z$squeeze(2)
                           }
                           return(z)
                         },
                         encode_from_species_index = function(x, s_ind, K = 1) {
                           s <- self$species_embedder_mean(s_ind)
                           z <- self$encode(x, s, K)
                           return(z)
                         },
                         decode = function(z, s) {
                           if(length(z$shape) == 3) {
                             recon <- self$vae$decode(z = z, s = s)
                             if(K == 1) {
                               recon <- recon$squeeze(2)
                             }
                           } else {
                             recon <- self$vae$decoder(z = z, s = s)
                           }
                           return(recon)
                         },
                         decode_from_species_index = function(z, s_ind) {
                           s <- self$species_embedder_mean(s_ind)
                           recon <- self$decode(z, s)
                           return(recon)
                         },
                         reconstruct_from_species_index = function(x, s_ind, K = 1) {
                           s <- self$species_embedder_mean(s_ind)
                           z <- self$encode(x, s, K = K)
                           recon <- self$decode(z, s)
                           return(recon)
                         },
                         iwae_loss_from_species_index = function(x, s, na_mask, K = 50) {
                           s <- self$species_embedder_mean(s)
                           c(input, z, means, log_vars, spec_lat) %<-% self$vae(x, s, K = K)
                           
                           c(iwae_loss, prior_loss) %<-% self$vae$loss_function(z, input, na_mask, means, log_vars,
                                                                                s, K = K, 
                                                                                reduce = NULL)
                           iwae_loss
                         },
                         forward = function(x, s, na_mask, K = 50) {
                           #browser()
                           #c(means, log_vars) %<-% self$encoder(y = x, s = spec_embedding_mean)
                           #z <- self$sample_latent(means, log_vars, K)
                           
                           s <- self$species_embedder_mean(s)
                           c(input, z, means, log_vars, spec_lat) %<-% self$vae(x, s, K = K)
                           
                           c(iwae_loss, prior_loss) %<-% self$vae$loss_function(z, input, na_mask, means, log_vars,
                                                                                s, K = K)
                           
                           spec_loss <- ((1 - alpha) * torch_sum(torch_square(s), dim = 2L) + alpha * torch_sum(torch_abs(s), dim = 2L)) * lambda
                           
                           list(x, z, s, means, log_vars, iwae_loss, torch_mean(spec_loss))
                           
                           #list(self$decoder(z = z, s = spec_embedding_mean), x, spec_embedding_mean, means, log_vars)
                         }
                         
)
