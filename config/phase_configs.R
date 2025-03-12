# Configuration file for training phases in progressive VAE-IWAE-SUMO model
# This central configuration makes it easy to add, remove, or modify phases

#' Define training phases
#' @param custom_config Optional configuration overrides
#' @return List of phase configurations
get_training_phases <- function(custom_config = NULL) {
  # Default phase configurations
  default_phases <- list(
    # Phase 1: Standard VAE with analytical KL
    list(
      mode = "vae",
      name = "VAE",
      K = 1,
      truncation = 0,
      base_lr = 0.001,
      min_epochs = 20
    ),
    
    # Phase 2: IWAE with low K
    list(
      mode = "iwae",
      name = "IWAE-10",
      K = 10,
      truncation = 0,
      base_lr = 0.0007,
      min_epochs = 20
    ),
    
    # Phase 3: IWAE with medium K
    list(
      mode = "iwae",
      name = "IWAE-30",
      K = 30,
      truncation = 0,
      base_lr = 0.0005,
      min_epochs = 20
    ),
    
    # Phase 4: IWAE with high K
    list(
      mode = "iwae",
      name = "IWAE-50",
      K = 50,
      truncation = 0,
      base_lr = 0.0004,
      min_epochs = 20
    ),
    
    # Phase 5: SUMO with low truncation
    list(
      mode = "sumo",
      name = "SUMO-2",
      K = 50,
      truncation = 2,
      base_lr = 0.0003,
      min_epochs = 20
    ),
    
    # Phase 6: SUMO with higher truncation
    list(
      mode = "sumo",
      name = "SUMO-4",
      K = 50,
      truncation = 4,
      base_lr = 0.0002,
      min_epochs = 20
    )
  )
  
  # If custom configuration provided, use it to override defaults
  if (!is.null(custom_config)) {
    # Apply overrides if provided
    if (!is.null(custom_config$phases)) {
      return(custom_config$phases)
    }
    
    # Handle phase-specific overrides
    for (i in seq_along(default_phases)) {
      if (i <= length(custom_config) && !is.null(custom_config[[i]])) {
        # Merge configuration
        default_phases[[i]] <- modifyList(default_phases[[i]], custom_config[[i]])
      }
    }
    
    # Handle phase additions (if there are more phases in custom_config)
    if (length(custom_config) > length(default_phases)) {
      for (i in (length(default_phases) + 1):length(custom_config)) {
        if (!is.null(custom_config[[i]])) {
          default_phases[[i]] <- custom_config[[i]]
        }
      }
    }
    
    # Handle phase removals (if specific phases should be removed)
    if (!is.null(custom_config$exclude_phases)) {
      default_phases <- default_phases[-custom_config$exclude_phases]
    }
  }
  
  # Return the final phase configuration
  default_phases
}

#' Get phase names for plotting
#' @param phases List of phase configurations
#' @return Vector of phase names
get_phase_names <- function(phases) {
  sapply(phases, function(phase) phase$name)
}

#' Convert from yaml configuration to phase configuration
#' @param yaml_config YAML configuration from experiments.yaml
#' @return List of phase configurations
yaml_to_phases <- function(yaml_config) {
  if (is.null(yaml_config$phases)) {
    return(NULL)
  }
  
  # Process YAML phase configuration
  phases <- list()
  
  for (i in seq_along(yaml_config$phases)) {
    phase_config <- yaml_config$phases[[i]]
    
    # Create phase from YAML
    phase <- list(
      mode = phase_config$mode,
      name = phase_config$name,
      K = phase_config$K,
      truncation = ifelse(is.null(phase_config$truncation), 0, phase_config$truncation),
      base_lr = phase_config$base_lr,
      min_epochs = ifelse(is.null(phase_config$min_epochs), 20, phase_config$min_epochs)
    )
    
    phases[[i]] <- phase
  }
  
  phases
}
