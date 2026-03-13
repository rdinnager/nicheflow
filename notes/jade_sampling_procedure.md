# JADE-Corrected Sampling from Species Range Polygons

A self-contained procedure for generating distortion-free environmental samples from species range polygons using JADE (Jacobian-Adjusted Density Estimation).

## Overview

### The Problem

When you sample geographic points uniformly from a species' range and read off their environmental values, the resulting distribution in environmental space is **not** the niche. It is distorted by the geography-to-environment map: environments that occupy large geographic areas are overrepresented, while environments restricted to small areas (e.g., montane climates) are underrepresented.

Formally, if $f(\mathbf{e})$ is the true niche density, then the pushforward of a uniform geographic sample has density:

$$p(\mathbf{e}) \propto \frac{f(\mathbf{e})}{\mathcal{J}_2(\mathbf{s}(\mathbf{e}))}$$

where $\mathcal{J}_2$ is the generalized Jacobian of the geography-to-environment map — the local area distortion factor. Large $\mathcal{J}_2$ means the environment changes rapidly (steep gradients), so that environment is **underrepresented** in the pushforward.

### The Solution

JADE corrects this by thinning a uniform geographic sample according to each point's Jacobian value. Points in steep-gradient regions (high Jacobian) are kept preferentially; points in flat regions (low Jacobian) are discarded more often. Since we have range polygons (not fixed occurrence records), we can sample as many geographic points as we want and thin them down, giving us:

- As many corrected samples as desired (not limited by original sample size)
- No duplicate points (unlike resampling with replacement)
- The exact target distribution in environmental space

The tradeoff is that in regions with high Jacobian variation, more initial points must be sampled — but this is a computational cost, not a statistical limitation.

### What You Need

| Input | Description | Format |
|-------|-------------|--------|
| Environmental rasters | Bioclimatic variables covering the study area | GeoTIFF (e.g., CHELSA-BIOCLIM+) |
| Species range polygons | Geographic boundaries of species ranges | Shapefile / GeoPackage / GeoJSON |

The procedure will generate:
- A **Jacobian raster** (computed once, reused for all species)
- A **per-variable SD file** (for standardization)
- **JADE-corrected samples** for each species

### Required R Packages

```r
install.packages(c("terra", "sf", "ggplot2"))

library(terra)    # Raster I/O, focal operations, extraction
library(sf)       # Vector geometry, polygon sampling
library(ggplot2)  # Visualization
```

**`terra`** handles raster operations (loading GeoTIFFs, focal convolutions for spatial derivatives, point extraction). It uses file-backed processing to handle global rasters that exceed RAM.

**`sf`** handles vector geometry — loading shapefiles and sampling random points within polygons.

---

## Step 1: Prepare Environmental Rasters

You need a set of environmental rasters in GeoTIFF format, all sharing the same extent, resolution, and coordinate reference system. This procedure uses CHELSA-BIOCLIM+ v2.1 (19 bioclimatic variables at 30 arc-second resolution) as the running example, but any set of aligned environmental rasters will work.

```r
# ── Configuration ────────────────────────────────────────────────────────
# Adjust these paths and patterns to match your data
bioclim_dir <- "data/chelsa-bioclim/1981-2010"
bio_indices <- 1:19
file_pattern <- "CHELSA_bio%02d_1981-2010_V.2.1.tif"

# ── Load rasters ─────────────────────────────────────────────────────────
bio_files <- file.path(bioclim_dir, sprintf(file_pattern, bio_indices))
stopifnot(all(file.exists(bio_files)))

bio_stack <- rast(bio_files)
names(bio_stack) <- paste0("bio", sprintf("%02d", bio_indices))

# ── Verify ───────────────────────────────────────────────────────────────
cat("Resolution:", res(bio_stack)[1], "degrees\n")
cat("Extent:", as.vector(ext(bio_stack)), "\n")
cat("CRS:", crs(bio_stack, describe = TRUE)$name, "\n")
cat("Layers:", nlyr(bio_stack), "\n")
```

If using your own environmental rasters (not CHELSA-BIOCLIM), ensure they:
1. Share the same spatial extent, resolution, and CRS
2. Use geographic coordinates (longitude/latitude in degrees)
3. Have NA for ocean/missing cells

---

## Step 2: Compute the Jacobian Raster

The Jacobian raster encodes the local area distortion factor $\mathcal{J}_2(\mathbf{s})$ at every grid cell. **It is computed once for a given set of environmental rasters and reused for all species.** This is the most computationally expensive step (~2-3 hours for global 30 arc-second rasters).

### 2.1 Mathematical Background

The geography-to-environment map $g: (\text{lon}, \text{lat}) \to (e_1, \ldots, e_d)$ has a $d \times 2$ Jacobian matrix:

$$J_g(\mathbf{s}) = \begin{pmatrix} \partial e_1/\partial \text{lon} & \partial e_1/\partial \text{lat} \\ \vdots & \vdots \\ \partial e_d/\partial \text{lon} & \partial e_d/\partial \text{lat} \end{pmatrix}$$

The generalized Jacobian (2-volume, area distortion factor) is:

$$\mathcal{J}_2(\mathbf{s}) = \sqrt{\det(J_g^T J_g)}$$

where $J_g^T J_g$ is a $2 \times 2$ Gram matrix that can be accumulated incrementally over the $d$ environmental variables:

$$J_g^T J_g = \begin{pmatrix} A & B \\ B & C \end{pmatrix}, \quad A = \sum_i \left(\frac{\partial e_i}{\partial \text{lon}}\right)^2, \quad B = \sum_i \frac{\partial e_i}{\partial \text{lon}} \frac{\partial e_i}{\partial \text{lat}}, \quad C = \sum_i \left(\frac{\partial e_i}{\partial \text{lat}}\right)^2$$

$$\mathcal{J}_2 = \sqrt{\max(AC - B^2, \; 0)}$$

**Standardization:** Environmental variables have wildly different scales (temperature in $°C \times 10$ vs precipitation in mm). Without standardization, high-variance variables dominate the Jacobian. Dividing each variable's spatial derivatives by its global standard deviation computes the Jacobian in z-scored environmental space, making all variables contribute proportionally to their relative spatial variation.

**Latitude correction:** On a geographic (lon/lat) grid, one degree of longitude shrinks toward the poles as $\cos(\text{lat})$. The derivatives computed via `focal()` are in per-degree units, so the Jacobian must be divided by $\cos(\text{lat})$ to get the correct per-km distortion factor.

### 2.2 Helper Functions

```r
# ── Compute per-variable SDs via random sampling ────────────────────────
# Samples a subset of pixels rather than reading entire rasters.
# With 1M samples from ~900M pixels, the relative standard error
# of the SD estimate is ~0.07% — negligible for standardization.
compute_bioclim_sds <- function(bioclim_dir, bio_indices, file_pattern,
                                sample_size = 1e6) {
  sds <- numeric(length(bio_indices))
  names(sds) <- paste0("bio", sprintf("%02d", bio_indices))

  for (j in seq_along(bio_indices)) {
    i <- bio_indices[j]
    fname <- file.path(bioclim_dir, sprintf(file_pattern, i))
    r <- rast(fname)
    vals <- spatSample(r, size = sample_size, method = "random",
                       na.rm = TRUE, as.df = FALSE)
    sds[j] <- sd(vals[, 1], na.rm = TRUE)
    rm(r, vals); gc(verbose = FALSE)
  }
  sds
}


# ── Fill coastal NAs via iterative focal mean ───────────────────────────
# focal(..., na.rm = FALSE) produces NA at land cells with ocean neighbors.
# Since SDM occurrences frequently fall near coastlines, we fill these NAs
# using the mean of non-NA neighbors (justified: Jacobian varies smoothly,
# so neighbors ~1km away have very similar values).
fill_coastal_na <- function(jacobian, land_mask, max_passes = 3) {
  filled <- jacobian

  for (pass in seq_len(max_passes)) {
    focal_means <- focal(filled, w = 3, fun = "mean", na.rm = TRUE,
                         na.policy = "only")
    filled <- cover(filled, focal_means)
    filled <- mask(filled, land_mask)

    n_remaining <- global(is.na(filled) & !is.na(land_mask), "sum",
                          na.rm = TRUE)[1, 1]
    message("  Coastal NA fill pass ", pass, "/", max_passes,
            ": ", n_remaining, " land NAs remaining")
    if (n_remaining == 0) break
  }

  # Fallback: remaining isolated land NAs get global mean
  if (n_remaining > 0) {
    mean_val <- global(filled, "mean", na.rm = TRUE)[1, 1]
    message("  Fallback: filling ", n_remaining,
            " remaining land NAs with global mean (", round(mean_val, 4), ")")
    filled <- ifel(is.na(filled) & !is.na(land_mask), mean_val, filled)
  }

  filled
}


# ── Time formatting helper ──────────────────────────────────────────────
format_time <- function(seconds) {
  if (seconds < 60) paste0(round(seconds, 1), "s")
  else if (seconds < 3600) paste0(round(seconds / 60, 1), " min")
  else paste0(round(seconds / 3600, 1), " hr")
}
```

### 2.3 Main Jacobian Computation Function

This function processes environmental rasters one at a time, computes spatial derivatives via `terra::focal()`, and accumulates the Gram matrix components. This keeps peak memory usage low (~6 raster layers at any time, regardless of how many environmental variables you have).

```r
compute_jacobian_raster <- function(bioclim_dir, output_path,
                                    bio_indices = 1:19,
                                    standardize = TRUE,
                                    lat_correct = TRUE,
                                    file_pattern = "CHELSA_bio%02d_1981-2010_V.2.1.tif",
                                    overwrite = FALSE) {

  if (file.exists(output_path) && !overwrite) {
    message("Output exists: ", output_path, " (use overwrite = TRUE to recompute)")
    return(rast(output_path))
  }

  t_start <- proc.time()
  terraOptions(memfrac = 0.6, progress = 10)

  # ── Template and resolution ──────────────────────────────────────────
  template <- rast(file.path(bioclim_dir,
                             sprintf(file_pattern, bio_indices[1])))
  h <- xres(template)  # Grid spacing in degrees (0.008333 for CHELSA)

  message("Resolution: ", h, " degrees")
  message("Dimensions: ", nrow(template), " x ", ncol(template))

  # ── Focal weight matrices (central differences) ──────────────────────
  # Longitude: [west, center, east] = [-1, 0, 1] / (2h)
  w_dlon <- matrix(c(-1, 0, 1) / (2 * h), nrow = 1, ncol = 3)
  # Latitude: [north, center, south] = [1, 0, -1] / (2h)
  w_dlat <- matrix(c(1, 0, -1) / (2 * h), nrow = 3, ncol = 1)

  # ── Compute per-variable SDs ─────────────────────────────────────────
  if (standardize) {
    message("\nComputing per-variable SDs...")
    t_sd <- proc.time()
    sds <- compute_bioclim_sds(bioclim_dir, bio_indices, file_pattern)
    message("  Done in ", round((proc.time() - t_sd)[3], 1), "s")
    sds[sds < 1e-10] <- 1  # Guard against zero SD

    # Save SDs for later use (standardizing SDM inputs)
    sd_path <- sub("\\.[^.]+$", "_bioclim_sds.csv", output_path)
    dir.create(dirname(sd_path), recursive = TRUE, showWarnings = FALSE)
    sd_df <- data.frame(
      bio_index = bio_indices,
      variable = paste0("bio", sprintf("%02d", bio_indices)),
      sd = as.numeric(sds)
    )
    write.csv(sd_df, sd_path, row.names = FALSE)
    message("  SDs saved to: ", sd_path)
  }

  # ── Initialize accumulator rasters ───────────────────────────────────
  # Written to disk to prevent terra's lazy evaluation from building
  # a 19-deep chain of deferred operations
  tmp_dir <- tempdir()
  A <- init(template, 0)
  writeRaster(A, file.path(tmp_dir, "gram_A.tif"), overwrite = TRUE)
  A <- rast(file.path(tmp_dir, "gram_A.tif"))
  B <- init(template, 0)
  writeRaster(B, file.path(tmp_dir, "gram_B.tif"), overwrite = TRUE)
  B <- rast(file.path(tmp_dir, "gram_B.tif"))
  C <- init(template, 0)
  writeRaster(C, file.path(tmp_dir, "gram_C.tif"), overwrite = TRUE)
  C <- rast(file.path(tmp_dir, "gram_C.tif"))

  # ── Land mask ────────────────────────────────────────────────────────
  land_mask <- ifel(!is.na(template), 1, NA)
  rm(template); gc(verbose = FALSE)

  # ── Process each variable ────────────────────────────────────────────
  n_vars <- length(bio_indices)
  var_times <- numeric(n_vars)

  for (j in seq_along(bio_indices)) {
    i <- bio_indices[j]
    t_var <- proc.time()
    var_label <- paste0("bio", sprintf("%02d", i))
    message("\n[", j, "/", n_vars, "] Processing ", var_label, "...")

    r <- rast(file.path(bioclim_dir, sprintf(file_pattern, i)))

    # Spatial derivatives via focal convolution
    de_dlon <- focal(r, w = w_dlon, na.rm = FALSE)
    de_dlat <- focal(r, w = w_dlat, na.rm = FALSE)
    rm(r); gc(verbose = FALSE)

    # Standardize
    if (standardize) {
      de_dlon <- de_dlon / sds[j]
      de_dlat <- de_dlat / sds[j]
    }

    # Accumulate Gram matrix components
    A <- A + de_dlon^2
    B <- B + de_dlon * de_dlat
    C <- C + de_dlat^2

    # Force write to disk (prevents lazy chain buildup)
    writeRaster(A, file.path(tmp_dir, "gram_A.tif"), overwrite = TRUE)
    A <- rast(file.path(tmp_dir, "gram_A.tif"))
    writeRaster(B, file.path(tmp_dir, "gram_B.tif"), overwrite = TRUE)
    B <- rast(file.path(tmp_dir, "gram_B.tif"))
    writeRaster(C, file.path(tmp_dir, "gram_C.tif"), overwrite = TRUE)
    C <- rast(file.path(tmp_dir, "gram_C.tif"))
    rm(de_dlon, de_dlat); gc(verbose = FALSE)

    # Progress
    var_elapsed <- (proc.time() - t_var)[3]
    var_times[j] <- var_elapsed
    total_elapsed <- (proc.time() - t_start)[3]
    est_remaining <- mean(var_times[1:j]) * (n_vars - j)
    message("  ", var_label, " done in ", round(var_elapsed, 1), "s",
            " | Elapsed: ", format_time(total_elapsed),
            " | Est. remaining: ~", format_time(est_remaining))
  }

  message("\nAll ", n_vars, " variables processed in ",
          format_time((proc.time() - t_start)[3]))

  # ── Compute Jacobian ─────────────────────────────────────────────────
  message("Computing Jacobian determinant...")
  det_JtJ <- A * C - B^2
  jacobian <- sqrt(clamp(det_JtJ, lower = 0))
  rm(A, B, C, det_JtJ); gc(verbose = FALSE)

  # ── Latitude correction ──────────────────────────────────────────────
  if (lat_correct) {
    message("Applying latitude correction...")
    template <- rast(file.path(bioclim_dir,
                               sprintf(file_pattern, bio_indices[1])))
    lat_rast <- init(template, "y")
    cos_lat <- clamp(cos(lat_rast * pi / 180), lower = 0.01)
    jacobian <- jacobian / cos_lat
    rm(template, lat_rast, cos_lat); gc(verbose = FALSE)
  }

  # ── Fill coastal NAs ─────────────────────────────────────────────────
  message("Filling coastal NAs...")
  template <- rast(file.path(bioclim_dir,
                             sprintf(file_pattern, bio_indices[1])))
  land_mask <- ifel(!is.na(template), 1, NA)
  jacobian <- fill_coastal_na(jacobian, land_mask)
  rm(template, land_mask); gc(verbose = FALSE)

  # ── Write output ─────────────────────────────────────────────────────
  dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
  message("Writing: ", output_path)
  writeRaster(jacobian, output_path, overwrite = TRUE,
              names = "jacobian_J2", filetype = "GTiff",
              gdal = c("COMPRESS=LZW", "TILED=YES"))

  message("Total time: ", format_time((proc.time() - t_start)[3]))
  rast(output_path)
}
```

### 2.4 Run the Computation

```r
# Compute (or load existing) Jacobian raster
jac_rast <- compute_jacobian_raster(
  bioclim_dir = bioclim_dir,
  output_path = "output/jacobian/chelsa_jacobian.tif",
  bio_indices = bio_indices,
  standardize = TRUE,
  lat_correct = TRUE,
  file_pattern = file_pattern,
  overwrite = FALSE
)

# Load the SDs (saved automatically during computation)
sd_df <- read.csv("output/jacobian/chelsa_jacobian_bioclim_sds.csv")
```

**Runtime**: ~2-3 hours for 19 global CHELSA-BIOCLIM variables (dominated by 19 `focal()` calls on ~900 million cell rasters). For regional extents, proportionally faster.

### 2.5 What the Jacobian Looks Like

- **High values** ($\mathcal{J}_2 \gg 1$): Mountain ranges, climate transition zones (Andes, Himalayas, Sahel edge). Environment changes rapidly per unit distance.
- **Low values** ($\mathcal{J}_2 \approx 0$): Flat, climatically uniform regions (Sahara interior, deep Amazon). Environment barely changes.
- **Zero**: Completely flat in all environmental dimensions — degenerate.

```r
# Quick visualization (downsampled for plotting)
agg_fact <- max(1, round(nrow(jac_rast) / 2000))
jac_plot <- aggregate(jac_rast, fact = agg_fact, fun = "mean", na.rm = TRUE)
df <- as.data.frame(jac_plot, xy = TRUE, na.rm = TRUE)
names(df) <- c("lon", "lat", "jacobian")
df$jacobian <- log10(pmax(df$jacobian, 1e-10))

ggplot(df, aes(x = lon, y = lat, fill = jacobian)) +
  geom_raster() +
  scale_fill_viridis_c(option = "inferno") +
  coord_quickmap() +
  labs(title = "Generalized Jacobian (log10 scale)",
       fill = expression(log[10](J[2]))) +
  theme_minimal()
```

---

## Step 3: Load Species Range Polygons

```r
# Load shapefile (or GeoPackage, GeoJSON, etc.)
ranges <- st_read("path/to/species_ranges.shp")

# Must be in geographic CRS (lon/lat) to match rasters
ranges <- st_transform(ranges, crs = 4326)

# Inspect
cat("Number of features:", nrow(ranges), "\n")
cat("Geometry type:", unique(st_geometry_type(ranges)), "\n")
plot(st_geometry(ranges), main = "Species ranges")
```

### Multiple Species

If the shapefile contains ranges for multiple species, select one at a time:

```r
# Assuming a column 'species' identifies each species
species_list <- unique(ranges$species)
range_sp <- ranges[ranges$species == species_list[1], ]
```

`sf::st_sample()` handles MULTIPOLYGON geometries correctly — it samples across all disjoint parts proportionally to their area.

---

## Step 4: JADE-Corrected Sampling

This is the core procedure. We sample geographic points uniformly within the range polygon, extract their Jacobian values, and thin using acceptance-rejection sampling. This is wrapped in an iterative loop that continues until we have the desired number of corrected samples.

### 4.1 The Thinning Algorithm

For each geographic point $\mathbf{s}_i$, compute an acceptance probability:

$$p_{\text{accept}}(i) = \frac{\mathcal{J}_2(\mathbf{s}_i)}{\max_j \mathcal{J}_2(\mathbf{s}_j)}$$

Draw $u_i \sim \text{Uniform}(0, 1)$. Accept the point if $u_i \leq p_{\text{accept}}(i)$.

This is standard acceptance-rejection sampling: the point with the highest Jacobian is always accepted; all others are accepted in proportion to their Jacobian relative to the maximum.

### 4.2 Sampling Function

```r
jade_sample <- function(range_polygon, jac_rast, bio_stack,
                        n_target, batch_size = NULL,
                        max_iter = 100, verbose = TRUE) {
  # Default batch size: 5x target (adjust based on expected acceptance rate)
  if (is.null(batch_size)) batch_size <- n_target * 5

  accepted_xy  <- matrix(nrow = 0, ncol = 2,
                         dimnames = list(NULL, c("lon", "lat")))
  accepted_env <- data.frame()
  total_sampled <- 0
  iter <- 0

  while (nrow(accepted_xy) < n_target && iter < max_iter) {
    iter <- iter + 1

    # ── Sample uniform geographic points within range ──────────────────
    pts_sf <- st_sample(range_polygon, size = batch_size, type = "random")
    pts_xy <- st_coordinates(pts_sf)
    colnames(pts_xy) <- c("lon", "lat")
    total_sampled <- total_sampled + nrow(pts_xy)

    # ── Extract Jacobian at each point ─────────────────────────────────
    jac_vals <- extract(jac_rast, pts_xy, method = "simple")[, 1]

    # Remove invalid points (ocean, outside extent, zero Jacobian)
    valid <- !is.na(jac_vals) & jac_vals > 0
    pts_xy   <- pts_xy[valid, , drop = FALSE]
    jac_vals <- jac_vals[valid]

    if (length(jac_vals) == 0) next

    # ── Acceptance-rejection thinning ──────────────────────────────────
    p_accept <- jac_vals / max(jac_vals)
    keep <- runif(length(p_accept)) <= p_accept

    if (sum(keep) == 0) next

    # ── Accumulate accepted points ─────────────────────────────────────
    accepted_xy <- rbind(accepted_xy, pts_xy[keep, , drop = FALSE])

    # Extract environmental values at accepted points
    env_batch <- extract(bio_stack, pts_xy[keep, , drop = FALSE],
                         method = "simple")
    # Remove the ID column that terra::extract adds
    env_batch <- env_batch[, -1, drop = FALSE]
    accepted_env <- rbind(accepted_env, env_batch)

    if (verbose) {
      message("  Iter ", iter, ": sampled ", nrow(pts_xy),
              ", accepted ", sum(keep),
              " (total: ", nrow(accepted_xy), "/", n_target, ")")
    }

    # Adaptive batch sizing: estimate remaining need from empirical rate
    if (iter == 1 && sum(keep) > 0) {
      empirical_rate <- sum(keep) / nrow(pts_xy)
      remaining <- n_target - nrow(accepted_xy)
      batch_size <- max(batch_size,
                        ceiling(remaining / empirical_rate * 1.3))
    }
  }

  # Trim to exact target
  if (nrow(accepted_xy) > n_target) {
    accepted_xy  <- accepted_xy[1:n_target, , drop = FALSE]
    accepted_env <- accepted_env[1:n_target, , drop = FALSE]
  }

  list(
    xy             = as.data.frame(accepted_xy),
    env            = accepted_env,
    n_target       = n_target,
    n_sampled      = total_sampled,
    n_iterations   = iter,
    acceptance_rate = nrow(accepted_xy) / total_sampled
  )
}
```

### 4.3 Run It

```r
result <- jade_sample(
  range_polygon = range_sp,
  jac_rast      = jac_rast,
  bio_stack     = bio_stack,
  n_target      = 10000,
  batch_size    = 50000
)

cat("Target:          ", result$n_target, "points\n")
cat("Total sampled:   ", result$n_sampled, "\n")
cat("Iterations:      ", result$n_iterations, "\n")
cat("Acceptance rate: ", round(100 * result$acceptance_rate, 1), "%\n")
```

---

## Step 5: Standardize Environmental Values

The accepted points' environmental values should be standardized using the **same SDs** that were used during Jacobian computation. This ensures the environmental space matches the space in which distortion was corrected.

```r
sds <- sd_df$sd
names(sds) <- sd_df$variable

# Standardize: divide each variable by its global SD
env_standardized <- as.data.frame(
  scale(result$env, center = FALSE, scale = sds)
)

# Final result: coordinates + standardized environment
jade_result <- cbind(result$xy, env_standardized)

cat("JADE-corrected sample:", nrow(jade_result), "points x",
    ncol(env_standardized), "environmental dimensions\n")
```

**When to standardize:**
- For density estimation, niche modeling, or niche overlap analysis: **yes** — use standardized values
- For downstream analysis that expects raw units (e.g., plotting temperature distributions): keep raw values in `result$env`. The JADE correction is in the *sampling*, not in the values

---

## Why Thinning Instead of Weighted Resampling?

| Approach | Pros | Cons |
|----------|------|------|
| **Weighted resampling** (with replacement) | Simple; always returns n points | Duplicates; effective sample size $\ll n$ when Jacobian varies widely |
| **JADE thinning** (acceptance-rejection) | No duplicates; exact target distribution; unlimited sample size | Variable yield; need to oversample |

With range polygons we can sample as many geographic points as we want, so the variable yield of thinning is not a limitation — we just sample more. The result is a set of **unique, independent** points whose density in environmental space is exactly correct.

### Acceptance Rate Guide

The expected acceptance rate is $\text{mean}(\mathcal{J}_2) / \max(\mathcal{J}_2)$ within the range.

| Species range type | Expected acceptance | Suggested initial oversample |
|---|---|---|
| Small, flat range (e.g., lowland endemic) | 30–60% | 3–4x |
| Mixed terrain (e.g., continental range) | 10–30% | 5–10x |
| Mountains + plains (e.g., Andes to Amazon) | 1–10% | 15–100x |

---

## Mathematical Justification

### Why Thinning by Jacobian Corrects Distortion

**Setup**: Let $g: \mathbb{R}^2 \to \mathbb{R}^d$ be the geography-to-environment map with generalized Jacobian $\mathcal{J}_2(\mathbf{s})$. A species occupies a range $\mathcal{R}$.

**Step 1**: Sample $\mathbf{s}_1, \ldots, \mathbf{s}_N$ uniformly from $\mathcal{R}$. Geographic density: $p_{\text{geo}}(\mathbf{s}) = 1/|\mathcal{R}|$.

**Step 2**: The pushforward to environmental space has density:

$$p_{\text{env}}(\mathbf{e}) = \frac{1}{|\mathcal{R}| \cdot \mathcal{J}_2(\mathbf{s}(\mathbf{e}))}$$

This is distorted — environments with large $\mathcal{J}_2$ are underrepresented.

**Step 3**: Accept each point with probability $\propto \mathcal{J}_2(\mathbf{s}_i)$. The accepted points have geographic density:

$$p_{\text{accepted}}(\mathbf{s}) \propto \frac{\mathcal{J}_2(\mathbf{s})}{|\mathcal{R}|}$$

**Step 4**: Push accepted points to environmental space:

$$p_{\text{accepted, env}}(\mathbf{e}) \propto \frac{\mathcal{J}_2(\mathbf{s}(\mathbf{e}))}{|\mathcal{R}| \cdot \mathcal{J}_2(\mathbf{s}(\mathbf{e}))} = \frac{1}{|\mathcal{R}|}$$

**The Jacobian factors cancel exactly.** The result is a uniform density in environmental space over the image of $\mathcal{R}$ — the undistorted available niche.

### Non-Uniform Occurrence Within the Range

If you have information about spatially varying occurrence intensity $\lambda(\mathbf{s})$ (e.g., from abundance data or a prior SDM), modify the acceptance probability:

$$p_{\text{accept}}(\mathbf{s}_i) \propto \mathcal{J}_2(\mathbf{s}_i) \cdot \lambda(\mathbf{s}_i)$$

For the typical use case — "characterize what environments fall within the range" — uniform sampling is appropriate and $\lambda = \text{const}$.

### Connection to Background-Based SDMs

Background-based methods (MaxEnt, logistic regression with background points) correct distortion via the density ratio $r(\mathbf{e}) = p_{\text{pres}}(\mathbf{e}) / p_{\text{bg}}(\mathbf{e})$, in which the Jacobian cancels between numerator and denominator. JADE achieves the same cancellation analytically — no background points needed.

| | Background-based | JADE thinning |
|--|---|---|
| **Corrects via** | Density ratio cancellation | Jacobian weighting cancellation |
| **Needs background?** | Yes | No |
| **Output** | Density ratio function $r(\mathbf{e})$ | Sample from niche $f(\mathbf{e})$ |
| **Robustness** | Works even if injectivity fails | Requires $d_{\text{eff}} > 2$ (always true with 19 bioclim vars) |
| **Best for** | Modeling the niche as a function | Generating samples for downstream analysis |

---

## Diagnostics

### Jacobian Distribution Within the Range

```r
# Sample many points and extract Jacobians (no thinning)
diag_pts <- st_sample(range_sp, size = 50000, type = "random")
diag_xy  <- st_coordinates(diag_pts)
diag_jac <- extract(jac_rast, diag_xy, method = "simple")[, 1]
diag_jac <- diag_jac[!is.na(diag_jac) & diag_jac > 0]

# Histogram
ggplot(data.frame(jac = diag_jac), aes(x = log10(jac))) +
  geom_histogram(bins = 100, fill = "steelblue", alpha = 0.7) +
  labs(title = "Jacobian Distribution Within Species Range",
       x = expression(log[10](J[2])), y = "Count") +
  theme_minimal()

# Key statistics
cat("Jacobian CV:              ", round(sd(diag_jac) / mean(diag_jac), 2), "\n")
cat("Max/Min ratio:            ", round(max(diag_jac) / min(diag_jac)), "\n")
cat("Expected acceptance rate: ", round(100 * mean(diag_jac) / max(diag_jac), 1), "%\n")
```

### Effective Sample Size Comparison

With weighted resampling, the effective sample size is $n_{\text{eff}} = (\sum w_i)^2 / \sum w_i^2$. With thinning, **all accepted points are independent** — $n_{\text{eff}}$ equals the actual count.

```r
w <- diag_jac / sum(diag_jac)
n_eff <- 1 / sum(w^2)
cat("Weighted resampling from", length(diag_jac), "points: n_eff =", round(n_eff), "\n")
cat("JADE thinning:", nrow(result$xy), "independent points\n")
```

### Visual Check: Before vs After Thinning

```r
pts_all <- st_sample(range_sp, size = 50000, type = "random")
xy_all  <- st_coordinates(pts_all)
jac_all <- extract(jac_rast, xy_all, method = "simple")[, 1]
valid   <- !is.na(jac_all) & jac_all > 0
xy_all  <- xy_all[valid, ]
jac_all <- jac_all[valid]
keep    <- runif(length(jac_all)) <= jac_all / max(jac_all)

df_both <- rbind(
  data.frame(lon = xy_all[, 1], lat = xy_all[, 2], type = "Before (uniform)"),
  data.frame(lon = xy_all[keep, 1], lat = xy_all[keep, 2], type = "After (JADE)")
)

ggplot(df_both, aes(x = lon, y = lat)) +
  geom_point(size = 0.3, alpha = 0.3) +
  facet_wrap(~ type) +
  coord_quickmap() +
  labs(title = "Geographic Distribution: Before vs After JADE Thinning") +
  theme_minimal()
```

After thinning, expect **higher density in mountainous/high-gradient regions** and lower density in flat areas. This is correct — steep-gradient points are underrepresented in environmental space and must be retained preferentially.

---

## Notes and Edge Cases

**Jacobian = 0 regions**: Points in cells where all environmental variables are locally flat (Jacobian = 0) are excluded. This is correct — they contribute zero environmental volume and represent degenerate points.

**Coastal cells**: The Jacobian raster fills coastal NA cells via focal-mean interpolation from neighbors. For island species, verify coastal values visually.

**Scaling to many species**: The Jacobian raster is computed once and shared. Per-species cost is dominated by `st_sample()` + `terra::extract()`, both fast. Use `future`/`furrr` for parallelism across species; `terra`'s file-backed rasters are safe to share across workers.

**Variable selection**: The Jacobian depends on which environmental variables you include. If your downstream analysis uses a subset of variables, consider computing a Jacobian for that subset. More variables generally means more Jacobian variation (lower acceptance rate but better distortion correction).

**Non-CHELSA rasters**: This procedure works with any set of aligned environmental rasters in geographic (lon/lat) coordinates. Replace the `bioclim_dir`, `file_pattern`, and `bio_indices` arguments accordingly.
