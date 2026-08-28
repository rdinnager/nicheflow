# Fast background-point generation for SDMs in R

This is a self-contained recipe for generating pseudo-absence (background)
points around a set of presence locations and pairing them with
environmental data. Written for an agent / human picking this up in any R
project. Requires only the presence (X, Y) coordinates and an
environmental raster (or any per-coordinate env extractor); **no range
polygons required**.

## What this gets you

For each species (or any focal entity with a presence cloud) you produce a
fixed number of bg points scattered geographically around the presences,
density-corrected to avoid clustering bias, with environmental values
attached and ocean / no-data cells removed. Suitable as the negative class
for MaxEnt, presence-vs-background logistic regression, or any
discriminative SDM.

## The algorithm (presence-anchored circular noise)

For each presence cloud:

1. Compute the **bbox-diagonal** of the presence (X, Y) cloud in degrees.
2. Set **`noise_radius = bbox_diagonal × radius_multiplier`** — the only
   knob you tune. `0.25` is a common default; `0.05` hugs the presences,
   `0.5` gives a substantially wider buffer.
3. Generate `n_background × oversample_factor` candidates (default
   `oversample_factor = 5`):
   - Pick a random presence (with replacement).
   - Add a uniform-disk offset: angle `~ U(0, 2π)`, radius
     `~ √U × noise_radius`. The square-root sampling ensures uniform
     density inside the disk (without it points cluster near the
     centre).
4. **Density correction.** For each candidate, count the number of
   presences within `noise_radius` via a KD-tree (`RANN::nn2`,
   `searchtype = "radius"`, `k = min(50, n_pres)`). Weight each candidate
   inversely by that count, so candidates near dense presence clusters
   are down-weighted.
5. Weighted-subsample down to exactly `n_background` candidates.

Returns an XY matrix in WGS84.

The generator does NOT drop ocean or no-data points. The caller does that
*after* env extraction (next section), which is why we oversample 5×.

```r
generate_background_points <- function(presence_xy,
                                        n_background,
                                        radius_multiplier = 0.25,
                                        oversample_factor = 5L) {
  n_pres <- nrow(presence_xy)

  bbox_diag <- sqrt(diff(range(presence_xy[, 1]))^2 +
                     diff(range(presence_xy[, 2]))^2)
  noise_radius <- bbox_diag * radius_multiplier

  n_candidates <- n_background * oversample_factor
  source_idx <- sample.int(n_pres, n_candidates, replace = TRUE)

  angles <- runif(n_candidates, 0, 2 * pi)
  radii  <- sqrt(runif(n_candidates)) * noise_radius
  noise_x <- radii * cos(angles)
  noise_y <- radii * sin(angles)

  candidates <- cbind(
    presence_xy[source_idx, 1] + noise_x,
    presence_xy[source_idx, 2] + noise_y
  )

  # Density correction via KD-tree
  nn_result <- RANN::nn2(presence_xy, candidates,
                         k = min(50L, n_pres),
                         searchtype = "radius",
                         radius = noise_radius)
  neighbor_counts <- rowSums(nn_result$nn.dists < noise_radius)
  neighbor_counts[neighbor_counts == 0] <- 1
  weights <- 1 / neighbor_counts

  keep <- sample.int(n_candidates, n_background, prob = weights)
  candidates[keep, , drop = FALSE]
}
```

## Pairing with environmental data + dropping no-data cells

Use `oversample_factor = 5` (built into the generator above) so that after
NA drop you still hit `n_background`. Pseudocode:

```r
# 1. Generate candidates (5x oversample)
candidates <- generate_background_points(presence_xy, n_background = 5000)

# 2. Extract env at each candidate (e.g. with terra::extract on a raster
#    stack, or any other per-coordinate lookup)
pts <- terra::vect(candidates, crs = "EPSG:4326")
env_raw <- terra::extract(env_raster_stack, pts)[, -1, drop = FALSE]
env_raw <- as.matrix(env_raw)

# 3. Apply documented NA fills (env-source-specific; e.g. CHELSA has fill
#    values for snow vars in non-snow regions). Skip if irrelevant.
env_raw <- apply_known_na_fills(env_raw)

# 4. Drop ocean / true no-data: rows with any remaining NA
land <- complete.cases(env_raw)
env_raw <- env_raw[land, , drop = FALSE]
candidates <- candidates[land, , drop = FALSE]

# 5. Trim to exactly n_background
if (nrow(env_raw) > n_background) {
  keep_idx <- sample.int(nrow(env_raw), n_background)
  env_raw    <- env_raw[keep_idx, , drop = FALSE]
  candidates <- candidates[keep_idx, , drop = FALSE]
}

# 6. (Optional) Standardize env, subset to the variables you care about
env_std <- sweep(env_raw, 2, env_mean, "-")
env_std <- sweep(env_std, 2, env_sd,   "/")
```

Step 4 is the only place ocean / no-data filtering happens. With
`oversample_factor = 5`, ~80% NA-drop rate still yields enough
land bg to hit `n_background`. If your domain has very high ocean
fraction (e.g. coastal species), bump `oversample_factor` to 10.

## Choosing `radius_multiplier`

Each multiplier scales the noise radius proportionally to the presence
bbox-diagonal. Empirical bg-bbox area ratios (data point: a North American
salamander with 1100 presences over a 314 deg² bbox):

| multiplier | bg bbox (deg²) | ratio | character |
|---|---|---|---|
| 0.05 | 399 | 1.27× | bg hugs the presences (most discriminative bg) |
| 0.25 | 864 | 2.76× | typical default |
| 0.50 | 1660 | 5.29× | substantially wider (more bg outside species range) |
| 0.75 | 2763 | 8.81× | very wide |

Larger multipliers add bg from environments the species would actually
tolerate (potentially misleading negatives). Tighter multipliers strip
out the most discriminative env contrast. The "right" choice depends on
what you're testing; running multiple variants is informative.

## Smoke-testing a multiplier

Before committing to a full pipeline run, verify the bg-bbox change on
one species (~5 sec):

```r
bbox_area <- function(xy) diff(range(xy[, 1])) * diff(range(xy[, 2]))

cat(sprintf("presence: %.0f deg^2\n", bbox_area(presence_xy)))
for (m in c(0.05, 0.25, 0.5, 0.75)) {
  bg <- generate_background_points(presence_xy, 5000, radius_multiplier = m)
  cat(sprintf("multiplier %.2f: %.0f deg^2 (%.2fx)\n",
              m, bbox_area(bg), bbox_area(bg) / bbox_area(presence_xy)))
}
```

Eyeball the ratios — they're stable across species because the multiplier
is bbox-relative.

## Visualizing the bg ring

To eyeball whether the buffer feels right for a species, overlay
presences and bg in geographic space, panel-faceted across a few species
covering small / medium / large ranges. Zoom each panel to
`presence_bbox + buffer_radius * 1.1` so the entire bg ring is visible.

```r
library(ggplot2); library(dplyr); library(tidyr)

plot_df <- bind_rows(
  data.frame(x = presence_xy[, 1], y = presence_xy[, 2], type = "presence"),
  data.frame(x = bg_xy[, 1],       y = bg_xy[, 2],       type = "background")
)

ggplot(plot_df, aes(x, y, color = type, shape = type, size = type)) +
  geom_point(alpha = 0.6) +
  scale_color_manual(values = c(presence = "black", background = "#1F78B4")) +
  scale_shape_manual(values = c(presence = 17, background = 16)) +
  scale_size_manual(values = c(presence = 1.0, background = 0.5)) +
  coord_fixed() +
  theme_minimal()
```

For multi-species panel plots with country borders, see
`R/functions_evaluation.R` and the corresponding viz scripts in this
project — but the principle generalizes.

## Performance: parallelize across species, ship small chunks

For thousands of species, do not naively map the whole pipeline serially.
Two patterns matter, depending on framework:

### `targets` + `crew` (this project's pattern)

Group species into batches (~50 per batch) and make the bg-generation
target dynamically branched **per batch**:

```r
tar_target(uniform_eval_species_batches,
           split_into_batches(species_names, batch_size = 50L)),

tar_target(
  bg_data_per_batch,
  prepare_eval_batch(uniform_eval_species_batches, ...),  # processes one batch
  pattern = map(uniform_eval_species_batches),
  iteration = "list"
)
```

**Critical:** every downstream target in the chain must also use
`pattern = map(...)` and `iteration = "list"`, so the data is stored as
small (~25 MB) per-batch RDS files rather than a single 1 GB list. If a
downstream target stores the whole list as one file, every per-branch
dispatch re-loads that 1 GB, and pipelines that should take 20 min take
~12 hr. Empirically a ~30× slowdown was observed in this project when the
intermediate dataset target wasn't branched.

If your downstream MaxEnt-fit step is per-species, wrap it in a
batch-scoped function that maps over species inside one branch and
returns one tibble per batch — that way each crew worker handles ~50
species at a time without re-loading 1 GB per dispatch.

### Plain R / `furrr` / `future`

Same principle: chunk species into ~50-species batches and `furrr::future_map`
over batches, not over species. The cost of loading the env raster per
worker amortizes across the batch.

## Detached launch (survives terminal / session crashes)

R sessions on shared HPC nodes (especially those connected via VS Code's
remote-server) crash occasionally — memory pressure, network blips,
client-side updates. To make a long pipeline survive:

```bash
setsid nohup Rscript path/to/run.R > logs/run.log 2>&1 < /dev/null &
disown
```

- `setsid` puts the R process in its own session — SIGHUP from a dead
  controlling terminal won't propagate.
- `nohup` is belt-and-suspenders for the same.
- `disown` removes the job from the parent shell's job table.

Then poll the log file from a fresh shell whenever you want a status
check. The pipeline runs to completion regardless of session state.

## Validation checklist

After a run, confirm:

```r
# 1. Each species got exactly n_background bg points
bg_per_species_counts <- table(bg_xy_per_species$species)
stopifnot(all(bg_per_species_counts == n_background))

# 2. No NA env values in the final bg matrix
stopifnot(!any(is.na(bg_env)))

# 3. The bg cloud envelops the presences as expected for the multiplier.
#    Spot-check one species:
plot(bg_xy[idx, ], pch = 16, col = "skyblue", cex = 0.4)
points(pres_xy[idx, ], pch = 17, col = "black", cex = 0.7)
# bg should form a roughly circular cloud around the presences,
# scaled by `noise_radius`.
```

## Common pitfalls

- **Forgetting the `√U` in radius sampling** — without it, candidate
  density spikes at the centre of each disk and you get over-sampling
  near presences.
- **Dropping NA cells before extraction** — saves no time and breaks the
  density-correction step (it operates on the candidate cloud
  pre-extraction). Drop after.
- **Setting `oversample_factor` too low** in heavily ocean-bordered
  domains — you'll fall short of `n_background` on coastal species.
- **Per-record dispatch in parallel runs** — see performance section.
  Batch.
- **Picking `radius_multiplier` once and never sanity-checking** — it
  scales with bbox-diagonal, so very small or very large species can
  yield surprisingly extreme bg extents. Smoke-test on representatives
  from several range-size quantiles.

## Related: per-row "in range polygon" diagnostic (optional, requires polygons)

If your downstream analysis benefits from knowing whether each bg point
falls inside the species's mapped range, add a boolean column via
`sf::st_intersects` per species. This step **does** require polygon data
and is otherwise independent of the bg generation. See
`add_in_range_polygon()` in `R/functions_evaluation.R` for an
implementation that disables s2 (GEOS planar is more forgiving of malformed
IUCN polygons), runs species-by-species, and wraps each species in
`tryCatch` so a single bad polygon doesn't kill the run.
