#!/usr/bin/env Rscript
# run.R - Generic script runner with parameter override
#
# Usage:
#   Rscript run.R <script.R> [key=value ...]
#
# Examples:
#   Rscript run.R train.R latent_dim=64 lr=0.001
#   Rscript run.R train.R train_file="data/train.csv" batch_size=256
#   Rscript run.R train.R 'hidden_dims=c(512, 256)' use_cuda=TRUE

source("R/functions_run_script.R")

args <- commandArgs(trailingOnly = TRUE)

if (length(args) == 0) {
  cat("Usage: Rscript run.R <script.R> [key=value ...]\n")
  cat("\nRuns an R script with parameter overrides.\n")
  cat("Parameters in the script must be flagged with #| param comments.\n")
  cat("Outputs flagged with #| output are collected and saved.\n")
  quit(status = 1)
}

script <- args[1]
param_args <- args[-1]

# Parse key=value pairs
params <- list()
if (length(param_args) > 0) {
  for (arg in param_args) {
    if (!grepl("=", arg)) {
      stop("Invalid argument '", arg, "'. Expected key=value format.")
    }
    key <- sub("=.*", "", arg)
    value_str <- sub("^[^=]+=", "", arg)
    params[[key]] <- coerce_cli_value(value_str)
  }
}

message("--- run.R ---")
message("Script: ", script)
if (length(params) > 0) {
  message("CLI params: ",
          paste(names(params), "=",
                vapply(params, deparse1, character(1)),
                collapse = ", "))
}
message("---")

results <- run_script(script, params = params)

if (length(results) > 0) {
  # Print output variables (skip .data)
  output_names <- setdiff(names(results), ".data")
  if (length(output_names) > 0) {
    message("\n--- Outputs ---")
    for (nm in output_names) {
      val <- results[[nm]]
      if (is.atomic(val) && length(val) == 1) {
        message(nm, ": ", val)
      } else {
        message(nm, ": ", class(val)[1], " [", length(val), "]")
      }
    }
  }

  # Print data summaries
  if (!is.null(results$.data) && length(results$.data) > 0) {
    message("\n--- Data (scalars) ---")
    for (nm in names(results$.data)) {
      vals <- results$.data[[nm]]
      n <- length(vals)
      last_val <- vals[n]
      message(nm, ": ", n, " values, last = ", round(last_val, 6))
    }
  }

  out_dir <- "output/runs"
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  out_path <- file.path(out_dir,
                        paste0(tools::file_path_sans_ext(basename(script)), "_",
                               format(Sys.time(), "%Y%m%d_%H%M%S"), ".rds"))
  saveRDS(results, out_path)
  message("Saved to: ", out_path)
}
