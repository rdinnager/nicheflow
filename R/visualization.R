library(ggplot2)
library(gridExtra)

# Plot training progress with phase transitions
plot_training_progress <- function(training_history, transition_epochs, phase_names = NULL) {
  # Create data frame for plotting
  df <- data.frame(
    epoch = 1:length(training_history$val_loss),
    val_loss = training_history$val_loss,
    train_loss = training_history$train_loss,
    mse = training_history$mse
  )
  
  # Create main loss plot
  p1 <- ggplot(df, aes(x = epoch)) +
    geom_line(aes(y = val_loss, color = "Validation Loss")) +
    geom_line(aes(y = train_loss, color = "Training Loss")) +
    geom_vline(xintercept = transition_epochs, 
              linetype = "dashed", color = "red", alpha = 0.7) +
    scale_color_manual(values = c("Validation Loss" = "blue", "Training Loss" = "darkgreen")) +
    theme_minimal() +
    labs(title = "Loss with Training Phase Transitions",
         x = "Epoch", y = "Loss", color = "")
  
  # Add phase labels if transitions exist
  if (length(transition_epochs) > 0) {
    # If phase names not provided, create default names
    if (is.null(phase_names)) {
      # Default phase names based on standard progression
      phase_names <- c("VAE", "IWAE-10", "IWAE-30", "IWAE-50", "SUMO-2", "SUMO-4")
      # Ensure we have enough names
      if (length(phase_names) < length(transition_epochs) + 1) {
        # Generate generic names if needed
        additional_names <- paste0("Phase-", (length(phase_names) + 1):(length(transition_epochs) + 1))
        phase_names <- c(phase_names, additional_names)
      }
    }
    
    # Ensure we only use as many phase names as we need
    phase_names <- phase_names[1:(length(transition_epochs) + 1)]
    
    # Create data frame for phase labels
    phase_df <- data.frame(
      x = c(1, transition_epochs),
      xend = c(transition_epochs, max(df$epoch)),
      phase = phase_names
    )
    
    p1 <- p1 + 
      geom_segment(data = phase_df, 
                  aes(x = x, xend = xend, y = -Inf, yend = -Inf),
                  color = "transparent") +
      geom_text(data = phase_df, 
               aes(x = (x + xend) / 2, y = min(df$val_loss, na.rm = TRUE) * 0.95, label = phase),
               color = "red", vjust = 1)
  }
  
  # Create MSE plot
  p2 <- ggplot(df, aes(x = epoch, y = mse)) +
    geom_line(color = "purple") +
    geom_vline(xintercept = transition_epochs, 
              linetype = "dashed", color = "red", alpha = 0.7) +
    theme_minimal() +
    labs(title = "Mean Squared Error",
         x = "Epoch", y = "MSE")
  
  # Create learning rate plot if available
  if ("learning_rate" %in% names(training_history)) {
    df_lr <- data.frame(
      epoch = 1:length(training_history$learning_rate),
      learning_rate = training_history$learning_rate
    )
    
    p3 <- ggplot(df_lr, aes(x = epoch, y = learning_rate)) +
      geom_line(color = "orange") +
      geom_vline(xintercept = transition_epochs, 
                linetype = "dashed", color = "red", alpha = 0.7) +
      theme_minimal() +
      labs(title = "Learning Rate Schedule",
           x = "Epoch", y = "Learning Rate")
    
    return(list(loss_plot = p1, mse_plot = p2, lr_plot = p3))
  }
  
  return(list(loss_plot = p1, mse_plot = p2))
}

# Function to save current training progress as image
save_progress_image <- function(training_history, transition_epochs, output_file, phase_names = NULL) {
  # Create plots
  plots <- plot_training_progress(training_history, transition_epochs, phase_names)
  
  # Combine plots into a single figure
  if ("learning_rate" %in% names(training_history)) {
    combined_plot <- gridExtra::grid.arrange(
      plots$loss_plot, plots$mse_plot, plots$lr_plot,
      ncol = 1, heights = c(3, 2, 1)
    )
  } else {
    combined_plot <- gridExtra::grid.arrange(
      plots$loss_plot, plots$mse_plot,
      ncol = 1, heights = c(3, 2)
    )
  }
  
  # Save to file, overwriting previous version
  ggsave(output_file, combined_plot, width = 10, height = 8, dpi = 100)
  
  # Return invisible NULL
  invisible(NULL)
}

# Plot latent space visualizations using PHATE
plot_latent_space <- function(latent_means, species, n_dims = 2, method = "phate") {
  # Convert to matrix for dimensionality reduction
  z_matrix <- as.matrix(latent_means)
  
  # Apply dimensionality reduction if needed
  if (ncol(z_matrix) > n_dims) {
    if (method == "phate") {
      # Check if phateR is available
      if (!requireNamespace("phateR", quietly = TRUE)) {
        stop("Package 'phateR' is needed for PHATE visualization. Please install it with:\n",
             "install.packages('phateR')")
      }
      
      # Apply PHATE - try to use reasonable defaults
      phate_result <- phateR::phate(z_matrix, 
                                   k = min(30, nrow(z_matrix) - 1),
                                   n.landmarks = min(1000, nrow(z_matrix)),
                                   ndim = n_dims,
                                   verbose = FALSE)
      
      # Create dataframe for plotting
      plot_df <- data.frame(
        x = phate_result$embedding[, 1],
        y = phate_result$embedding[, 2],
        species_id = as.factor(species)
      )
      
      # Create plot
      p <- ggplot(plot_df, aes(x = x, y = y, color = species_id)) +
        geom_point(alpha = 0.7) +
        theme_minimal() +
        labs(title = "PHATE Visualization of Latent Space",
             x = "PHATE Dimension 1",
             y = "PHATE Dimension 2",
             color = "Species")
      
    } else if (method == "pca") {
      # Apply PCA
      pca_result <- prcomp(z_matrix, scale. = TRUE)
      z_reduced <- pca_result$x[, 1:n_dims]
      
      # Create dataframe for plotting
      plot_df <- data.frame(
        x = z_reduced[, 1],
        y = z_reduced[, 2],
        species_id = as.factor(species)
      )
      
      # Create plot
      p <- ggplot(plot_df, aes(x = x, y = y, color = species_id)) +
        geom_point(alpha = 0.7) +
        theme_minimal() +
        labs(title = "PCA of Latent Space",
             x = "Principal Component 1",
             y = "Principal Component 2",
             color = "Species")
    }
  } else {
    # Create dataframe for plotting
    plot_df <- data.frame(
      x = z_matrix[, 1],
      y = if (ncol(z_matrix) >= 2) z_matrix[, 2] else rep(0, nrow(z_matrix)),
      species_id = as.factor(species)
    )
    
    # Create plot
    p <- ggplot(plot_df, aes(x = x, y = y, color = species_id)) +
      geom_point(alpha = 0.7) +
      theme_minimal() +
      labs(title = "Latent Space",
           x = "Latent Dimension 1",
           y = "Latent Dimension 2",
           color = "Species")
  }
  
  return(p)
}
