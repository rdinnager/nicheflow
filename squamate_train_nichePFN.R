library(tidyverse)
library(sf)
library(torch)
library(abind)

source("R/utils.R")

bias_data_train <- read_rds("output/bias_data_for_nichePFN_train.rds") |>
  drop_na()

bias_data_val <- read_rds("output/bias_data_for_nichePFN_validation.rds") |>
  drop_na()

all_specs <- unique(c(bias_data_train$spec, bias_data_val$spec))
spec_fact_train <- bias_data_train |>
  pull(spec) |>
  factor(levels = all_specs)

spec_names_train <- levels(spec_fact_train) 
spec_nums_train <- matrix(as.numeric(spec_fact_train), ncol = 1)

spec_fact_val <- bias_data_val |>
  pull(spec) |>
  factor(levels = all_specs)

spec_names_val <- levels(spec_fact_val) 
spec_nums_val <- matrix(as.numeric(spec_fact_val), ncol = 1)

dataset_tensor <- function(dataset_batch, max_pnts = 500L) {
  mats <- map(dataset_batch, ~ rbind(.x, matrix(0, nrow = max_pnts - nrow(.x), ncol = 2)))
  arr <- mats |>
    abind(along = 0) |>
    torch_tensor()
  mask <- do.call(rbind, map(mats, ~.x[, 1] == 0)) |>
    torch_tensor()
  list(arr, mask)
}

bias_dataset <- dataset(name = "bias_ds",
                       initialize = function(spec_nums, bias_data, tensor_fun) {
                         self$specs <- spec_nums |>
                           torch_tensor()
                         self$coord_list <- bias_data |>
                           pull(coords)
                         self$lats <- bias_data |>
                           select(starts_with("L")) |>
                           as.matrix() |>
                           torch_tensor()
                         self$dataset_tensor <- tensor_fun
                       },
                       .getbatch = function(i) {
                         coords <- self$dataset_tensor(self$coord_list[i])
                         list(spec = self$specs[i, ], 
                              coords = coords[[1]], 
                              mask = coords[[2]],
                              lats = self$lats[i, ])
                       },
                       .length = function() {
                         self$lats$size()[[1]]
                       })

batch_size <- 1200

train_ds <- bias_dataset(spec_nums_train, bias_data_train, dataset_tensor)
train_dl <- dataloader(train_ds, batch_size, shuffle = TRUE, pin_memory = TRUE)

val_ds <- bias_dataset(spec_nums_val, bias_data_val, dataset_tensor)
val_dl <- dataloader(val_ds, ceiling(batch_size/2), shuffle = FALSE, pin_memory = TRUE)

# x -> LayerNorm -> Attention -> Skip -> LayerNorm -> MLP -> Skip
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

nichePFN <- nn_module("NichePFN",
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
                        for(i in length(self$blocks)) {
                          x <- self$blocks[[i]](x, mask)
                        }
                        x <- torch_sum(x, dim = 2L)
                        out <- self$mlp_final(x)
                        out
                      })

npfn <- nichePFN()
npfn <- npfn$cuda()

## trick to make sure enough CUDA memory gets preallocated
# test1 <- train_dl$.iter()$.next()
# test2 <- val_dl$.iter()$.next()
# out1 <- npfn(test1$coords$cuda(), test1$mask$cuda())
# out2 <- with_no_grad(npfn$eval()(test2$coords$cuda(), test2$mask$cuda()))
# l <- nnf_mse_loss(out1, test1$lats$cuda())
# l$backward()
# npfn$train()
#test <- train_dl$.iter()$.next()
#npfn(test$coords$cuda(), test$mask$cuda())
rm(test1, test2, out1, out2, l)
gc()

num_epochs <- 300

lr <- 0.0002
optimizer <- optim_adamw(npfn$parameters, lr = lr)
scheduler <- lr_one_cycle(optimizer, max_lr = lr,
                          epochs = num_epochs, steps_per_epoch = length(train_dl),
                          cycle_momentum = FALSE)
#scaler <- cuda_amp_grad_scaler()

i <- 0

n_batch <- length(train_dl)

for (epoch in 1:num_epochs) {
  
  epoch_time <- Sys.time()
  batchnum <- 0
  coro::loop(for (b in train_dl) {
    
    batchnum <- batchnum + 1
    i <- i + 1
    
    #with_autocast(device_type = "cuda", {
    out <- npfn(b$coords$cuda(), b$mask$cuda())
    l <- nnf_smooth_l1_loss(out, b$lats$cuda())
    #})
    
    if(batchnum %% 50 == 0) {
    cat("Epoch: ", epoch,
        "  batch: ", batchnum,
        "  Done: %", ceiling((batchnum / n_batch) * 100),
        "  loss: ", as.numeric(l$cpu()),
        "\n", sep = "")
    }
    
    # scaler$scale(l)$backward()
    # scaler$step(optimizer)
    # scaler$update()
    l$backward()
    optimizer$step()
    scheduler$step()
    optimizer$zero_grad()
  })
  npfn$eval()
  vls <- numeric(length(val_dl))
  i <- 0
  coro::loop(for (v in val_dl) {
    i <- i + 1
    with_no_grad({
      vout <- npfn(v$coords$cuda(), v$mask$cuda())
      vls[i] <- as.numeric(nnf_smooth_l1_loss(vout, v$lats$cuda())$cpu())
    })
  })
  npfn$train()
  cat("\nEpoch: ", epoch,
      "    mean validation loss: ", mean(vls),
      "\n\n", sep = "")
  torch_save(npfn, "output/nichePFN/checkpoint.to")
}