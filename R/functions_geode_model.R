#' GeODE Model Definition
#'
#' Rectified flow trajectory network for mapping environmental variables
#' to geographic coordinates. Extracted from train_geode.R for reuse
#' in evaluation pipelines.
#'
#' @author rdinnage

library(torch)
library(dagnn)


#' GeODE Trajectory Network
#'
#' U-Net architecture trained with rectified flow to map standardized
#' environmental features → standardized (lon, lat) coordinates.
#' Conditioning on environment is encoded once and reused across ODE steps.
#'
#' @field coord_dim Coordinate dimension (2 for lon/lat)
#' @field encode_t Time encoding layer (1 → t_encode)
#' @field encode_env Environment encoding layer (env_dim → env_encode)
#' @field unet U-Net trajectory network with skip connections
#' @field sample_trajectory ODE integration method
geode_traj_net <- nn_module("GeodeTrajNet",
  initialize = function(coord_dim, env_dim, breadths = c(512, 256, 128),
                        t_encode = 32L, env_encode = 64L,
                        model_device = "cpu") {
    if (length(breadths) != 3) stop("breadths should be length 3!")

    self$coord_dim <- coord_dim
    self$model_device <- model_device

    self$encode_t <- nn_linear(1L, t_encode)
    self$encode_env <- nn_linear(env_dim, env_encode)

    self$unet <- nndag(
      input = ~ coord_dim,
      t_encoded = ~ t_encode,
      env_encoded = ~ env_encode,
      e_1 = input + t_encoded + env_encoded ~ breadths[1],
      e_2 = e_1 + t_encoded + env_encoded ~ breadths[2],
      e_3 = e_2 + t_encoded + env_encoded ~ breadths[3],
      d_1 = e_3 + t_encoded + env_encoded ~ breadths[3],
      d_2 = d_1 + e_3 + t_encoded + env_encoded ~ breadths[2],
      d_3 = d_2 + e_2 + t_encoded + env_encoded ~ breadths[1],
      output = d_3 + e_1 + t_encoded + env_encoded ~ coord_dim,
      .act = list(nn_relu, output = nn_identity)
    )

    self$loss_function <- function(input, target) {
      torch_mean((target - input)^2)
    }

    # GPU-native Euler integration (much faster than deSolve for rectified flows)
    self$sample_trajectory <- function(initial_vals, env_vals, steps = 200L) {
      with_no_grad({
        dt <- 1.0 / steps
        n <- initial_vals$shape[1]
        y <- initial_vals$detach()

        # Precompute env encoding once (same for all steps)
        env_enc <- self$encode_env(env_vals$detach())

        for (s in seq_len(steps)) {
          t_val <- (s - 1) * dt
          t_tensor <- torch_full(c(n, 1L), t_val,
                                 device = y$device)
          t_enc <- self$encode_t(t_tensor)
          velocity <- self$unet(
            input = y, t_encoded = t_enc,
            env_encoded = env_enc
          )
          y <- y + velocity * dt
        }

        # Return final coordinates (single GPU→CPU transfer)
        y$cpu()
      })
    }
  },

  forward = function(coords, t, env) {
    t_encoded <- self$encode_t(t)
    env_encoded <- self$encode_env(env)
    self$unet(input = coords, t_encoded = t_encoded, env_encoded = env_encoded)
  }
)


#' Load GeODE model from checkpoint
#'
#' Constructs a GeODE trajectory network and loads weights from a checkpoint.
#' The checkpoint must have been saved via torch_save(model$state_dict(), path).
#'
#' @param checkpoint_path Path to the .pt checkpoint file
#' @param env_dim Environment input dimension (default 31 for CHELSA-BIOCLIM+)
#' @param coord_dim Coordinate dimension (default 2 for lon/lat)
#' @param breadths U-Net breadths (default c(512, 256, 128))
#' @param device Torch device string
#' @return geode_traj_net model in eval mode on device
load_geode_model <- function(checkpoint_path,
                             env_dim = 31L, coord_dim = 2L,
                             breadths = c(512L, 256L, 128L),
                             device = "cpu") {
  model <- geode_traj_net(
    coord_dim = coord_dim,
    env_dim = env_dim,
    breadths = breadths,
    model_device = device
  )
  load_model_checkpoint(model, checkpoint_path)
  model <- model$to(device = device)
  model$eval()
  model
}
