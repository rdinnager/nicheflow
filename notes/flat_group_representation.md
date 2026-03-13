# Flat Group Representation for GeoEncoder Training

## The Problem: R's Garbage Collector

The geoencoder training data consists of ~1.74 million "groups" — each group is a set of (lat, lon) coordinates for one species in one observation set. In the original representation, this was stored as a **list of lists**:

```r
train_groups <- list(
  list(species = "Sp_001", coords = matrix(c(1.2, 3.4, 5.6, 7.8), ncol = 2)),
  list(species = "Sp_002", coords = matrix(c(9.0, 1.1, 2.3, 4.5, 6.7, 8.9), ncol = 2)),
  # ... 1,740,022 groups total
)
```

Each group has:
- A `species` string
- A `coords` matrix (variable number of rows, always 2 columns: lat/lon)

This creates approximately **10+ million R objects** in memory:
- 1.74M outer list elements
- 1.74M inner lists (each with 2 elements)
- 1.74M species strings
- 1.74M coordinate matrices
- Plus internal SEXP structures for each

R's garbage collector (GC) is a **mark-and-sweep** collector. During the "mark" phase, it must **visit every reachable R object** to determine what's still in use. With 10M+ objects, each GC pause takes multiple seconds. Since GC triggers frequently during training (torch tensor allocation and deallocation creates pressure), these pauses dominate training time.

**Evidence**: gdb stack sampling during training showed 3/3 samples in `R_gc_internal()`. The profiler benchmark hid this because it only ran 50 timed batches — not enough to trigger frequent GC. But even there, the mean batch time was 9.6x the median (4.28s vs 0.45s), with periodic multi-second stalls.

## The Solution: Flat Representation

Instead of 1.74M list elements, we store the same data in **just 4 R objects**:

```r
train_flat <- list(
  coords_all   = matrix(...),   # Single matrix: N_total × 2 (all coordinates concatenated)
  group_species = character(...), # Character vector: length 1,740,022
  group_start   = integer(...),  # Integer vector: length 1,740,022 (row offsets into coords_all)
  group_length  = integer(...),  # Integer vector: length 1,740,022 (number of rows per group)
  n_groups      = 1740022L       # Scalar: total number of groups
)
```

### How the Data is Laid Out

Imagine 3 groups:
- Group 1: species "A", 3 coordinates
- Group 2: species "B", 2 coordinates
- Group 3: species "A", 4 coordinates

**Original (list of lists)**: 3 lists × 2 elements each = 9 R objects

**Flat representation**:

```
coords_all (9 × 2 matrix):         group_species:   group_start:   group_length:
  row 1: [lat1, lon1]  ← Group 1     "A"              1              3
  row 2: [lat2, lon2]  ← Group 1     "B"              4              2
  row 3: [lat3, lon3]  ← Group 1     "A"              6              4
  row 4: [lat4, lon4]  ← Group 2
  row 5: [lat5, lon5]  ← Group 2
  row 6: [lat6, lon6]  ← Group 3
  row 7: [lat7, lon7]  ← Group 3
  row 8: [lat8, lon8]  ← Group 3
  row 9: [lat9, lon9]  ← Group 3
```

To access group `i`'s coordinates:
```r
start <- group_start[i]
len   <- group_length[i]
rows  <- start:(start + len - 1)
coords <- coords_all[rows, ]
```

### R Object Count Comparison

| Representation | R Objects | GC Mark Phase |
|---------------|-----------|---------------|
| List of lists | ~10,000,000 | Multi-second pauses |
| Flat          | ~5 | Negligible |

The key insight: a single `matrix(nrow=95000000, ncol=2)` is **one R object** internally (one SEXP pointing to a contiguous numeric array). R's GC visits it once, regardless of how many rows it has. The same applies to the character and integer vectors.

## How Batch Creation Works

The training loop needs to create batches of `batch_size` groups. Here's how `collate_geoencoder_batch_flat()` works:

### Step 1: Select Groups

The epoch shuffles group indices `1:n_groups` and iterates through them in chunks of `batch_size`. For a given batch, we have a vector `idx` of group indices (e.g., `idx = c(42, 7891, 500123, ...)`).

### Step 2: Get Raw Lengths

```r
raw_lengths <- flat$group_length[idx]  # How many coordinates each selected group has
```

This is a simple integer vector subset — no list traversal needed.

### Step 3: Apply Coordinate Dropout (optional)

During training, we randomly drop some coordinates to improve generalization:

```r
if (dropout_frac > 0) {
  kept_lengths <- pmax(1L, as.integer(rbinom(batch_size, raw_lengths, 1 - dropout_frac)))
} else {
  kept_lengths <- raw_lengths
}
```

Each group independently keeps a random subset of its coordinates, with at least 1 coordinate always kept.

### Step 4: Build Padded Arrays

The transformer needs fixed-size tensors, so we pad to the maximum sequence length in the batch:

```r
max_len <- max(kept_lengths)
coords_array <- array(0, dim = c(batch_size, max_len, 2))  # [B, T, 2]
mask_mat <- matrix(TRUE, nrow = batch_size, ncol = max_len) # [B, T] padding mask
target_mat <- matrix(0, nrow = batch_size, ncol = embed_dim) # [B, D] target embeddings
```

### Step 5: Fill in Each Group

For each group `i` in the batch:

```r
gi    <- idx[i]                                    # Global group index
start <- flat$group_start[gi]                      # Where this group's coords start
raw_n <- raw_lengths[i]                            # Original number of coordinates
n     <- kept_lengths[i]                           # After dropout
rows  <- start:(start + raw_n - 1)                 # Row indices into coords_all

# If dropout, randomly select which coordinates to keep
if (dropout_frac > 0 && n < raw_n) {
  rows <- rows[sample.int(raw_n, n)]
}

# Extract coordinates from the flat matrix
coord_mat <- flat$coords_all[rows, , drop = FALSE]  # n × 2

# Add jitter (optional spatial noise for augmentation)
if (jitter_sd > 0) {
  coord_mat <- coord_mat + matrix(rnorm(n * 2, sd = jitter_sd), nrow = n, ncol = 2)
}

# Place into the padded array
coords_array[i, 1:n, ] <- coord_mat
mask_mat[i, 1:n] <- FALSE  # FALSE = real data, TRUE = padding

# Look up the target embedding for this species
sp_id <- species_map[flat$group_species[gi]]
target_mat[i, ] <- target_embeddings[sp_id, ]
```

### Step 6: Convert to Torch Tensors

```r
list(
  coords     = torch_tensor(coords_array, device = device),     # [B, T, 2]
  mask       = torch_tensor(mask_mat, dtype = torch_bool(), device = device),  # [B, T]
  target_emb = torch_tensor(target_mat, device = device)        # [B, D]
)
```

These tensors go directly into the transformer model:
- `coords` → positional encoding → transformer encoder
- `mask` → attention mask (so the model ignores padding positions)
- `target_emb` → the training target (the species' precomputed embedding from the species embedding model)

## Why This Fixes the Speed Issue

The old `collate_geoencoder_batch()` did essentially the same work per batch — the batch creation itself was never the bottleneck. The bottleneck was **between** batches, when R's GC would fire and spend 3+ seconds scanning 10M objects.

By reducing to ~5 R objects, GC pauses become negligible. The training loop spends its time on actual GPU computation instead of waiting for R to finish garbage collection.

### Expected Speedup

On L4 (batch_size=96):
- **Before (list of lists)**: ~3.5s/batch (GC-dominated)
- **After (flat)**: ~0.45s/batch (GPU-compute-dominated, matching benchmark)
- **Speedup**: ~7-8x

This has not yet been verified in a live training run — the current smoke test is running the pre-edit code. A new smoke test with the flat representation is the next step.

## The `flatten_groups()` Function

This is a one-time conversion run after loading the training data:

```r
flatten_groups <- function(groups) {
  n_groups <- length(groups)
  group_species <- character(n_groups)
  group_length <- integer(n_groups)

  # First pass: collect species names and lengths
  for (i in seq_len(n_groups)) {
    group_species[i] <- groups[[i]]$species
    group_length[i] <- nrow(groups[[i]]$coords)
  }

  # Compute offsets
  n_total <- sum(group_length)
  group_start <- c(1L, cumsum(group_length[-n_groups]) + 1L)

  # Second pass: copy all coordinates into one matrix
  coords_all <- matrix(0, nrow = n_total, ncol = 2L)
  for (i in seq_len(n_groups)) {
    rows <- group_start[i]:(group_start[i] + group_length[i] - 1L)
    coords_all[rows, ] <- groups[[i]]$coords
  }

  list(
    coords_all = coords_all,
    group_species = group_species,
    group_start = group_start,
    group_length = group_length,
    n_groups = n_groups
  )
}
```

After flattening, the original list is removed with `rm(train_groups_list); gc()` to immediately free the 10M objects.
