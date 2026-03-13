#' Write an RDS file atomically via temp file + rename
#' Prevents readers from seeing partial writes on shared filesystems.
write_atomic_rds <- function(obj, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  tmp <- paste0(path, ".tmp.", Sys.getpid())
  saveRDS(obj, tmp)
  file.rename(tmp, path)
}

#' Write a torch state dict atomically via temp file + rename
write_atomic_pt <- function(state_dict, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  tmp <- paste0(path, ".tmp.", Sys.getpid())
  torch::torch_save(state_dict, tmp)
  file.rename(tmp, path)
}

#' Wait for a file to appear, polling at intervals
#' @param path Path to wait for
#' @param timeout Maximum seconds to wait
#' @param poll_interval Seconds between checks
#' @return TRUE if file appeared, FALSE if timeout
wait_for_file <- function(path, timeout = 1800, poll_interval = 5) {
  t_start <- Sys.time()
  while (!file.exists(path)) {
    elapsed <- as.numeric(Sys.time() - t_start, units = "secs")
    if (timeout > 0 && elapsed > timeout) {
      message("Timeout waiting for: ", path, " (", round(elapsed), "s)")
      return(FALSE)
    }
    Sys.sleep(poll_interval)
  }
  message("Found: ", path, " after ",
          round(as.numeric(Sys.time() - t_start, units = "secs")), "s")
  TRUE
}
