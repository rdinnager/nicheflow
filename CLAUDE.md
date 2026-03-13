# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NicheFlow is a species distribution modeling (SDM) framework using progressive VAE-IWAE-SUMO training. It learns species niches from environmental data (CHELSA-BIOCLIM) and occurrence records using probabilistic deep learning in R with the torch package.

## Common Commands

### Run Data Pipeline (targets)
```r
targets::tar_make()                    # Run full pipeline
targets::tar_make(target_name)         # Run specific target
targets::tar_visnetwork()              # Visualize pipeline DAG
```

### Train Model
```bash
Rscript train.R --train_file data/processed/nichencoder_train.csv \
                --val_file data/processed/nichencoder_val.csv \
                --phase_config config/phases.yml

# With dual GPU (async validation)
Rscript train.R --train_device cuda:0 --val_device cuda:1 --async_validation TRUE ...
```

### Run Experiments
```bash
Rscript run_experiments.R --config_yaml experiments.yml
```

### Evaluate Model
```bash
Rscript evaluate.R --model_path output/model.pt \
                   --config_path output/config.rds \
                   --test_file data/processed/nichencoder_test.csv
```

### Submit to HPC (SLURM)
```bash
sbatch run.sh
```

## Architecture

### Training Flow
```
Raw Data → [_targets.R pipeline] → Preprocessed CSVs
                                        ↓
                              [train.R with phases.yml]
                                        ↓
                    VAE → IWAE-10 → IWAE-30 → IWAE-50 → SUMO-2 → SUMO-4
                                        ↓
                              Trained Model (.pt + .rds)
```

### Core Modules (R/)
- **model.R**: VAE architecture (encoder, decoder, species embeddings, loss functions for VAE/IWAE/SUMO)
- **training.R**: Training loops, phase management, gradient accumulation, checkpoint saving
- **async_validation.R**: Asynchronous validation on separate GPU via mirai
- **data.R**: Dataset preparation, torch dataloaders, feature masking
- **evaluation.R**: Model evaluation, per-species metrics, predictions
- **utils.R**: Batch size computation, logging, plateau detection for phase transitions
- **visualization.R**: Training progress plots

### Key Design Patterns
- **Progressive Training**: Automatically transitions phases when learning plateaus (patience + slope detection)
- **Gradient Accumulation**: Simulates larger batch sizes across mini-batches
- **Async Validation**: Second GPU validates while primary GPU trains
- **JADE Correction**: Sampling bias correction via Jacobian-adjusted density estimation (see notes/jade_sampling_procedure.md)

## Configuration

### experiments.yml
Defines experiment variants with shared parameters and per-experiment overrides (latent_dim, hidden_dims, noise_scale, etc.)

### config/phases.yml
Defines 6 training phases:
1. VAE (K=1)
2. IWAE-10, IWAE-30, IWAE-50
3. SUMO-2, SUMO-4 (truncated mixture)

Each phase specifies: mode, K samples, learning rate, max epochs, patience

## Key Dependencies

R packages: torch, targets, crew, tidyverse, sf, terra, arrow, mirai

## Data Structure

Training data CSVs contain:
- Environmental features (standardized BIOCLIM variables)
- Missing value indicators (`na_ind_*` columns)
- Species IDs for embedding lookup
- Ground truth labels

Large data files (data/) and pipeline outputs (_targets/) are gitignored.

## Memory/Resource Constraints

- `_targets.R` sets 16GB RAM limit via `unix::rlimit_as()`
- 8 parallel workers for targets pipeline
- Gradient accumulation used when batch size exceeds GPU memory

## ⚠️ CRITICAL: NEVER USE tar_destroy() ⚠️

**THIS IS A COMMANDMENT FOR USING TARGETS - READ THIS CAREFULLY:**

### NEVER, EVER use `tar_destroy()`

The `_targets/` directory contains the **CACHE** of all computed results from the pipeline. Destroying it means:
- **Losing hours of computation time** (ANOVA permutation tests take 2+ hours)
- **Wasting energy and contributing to environmental impact**
- **Defeating the entire purpose of using targets**

**Why destroying the cache is almost never the solution:**
- Functions are defined in `R/` files and sourced fresh each time - they are NOT stored in the cache
- If functions aren't updating, the problem is with how they're being sourced, not the cache
- Almost all problems can be solved with `tar_invalidate()` on specific targets

**What to do instead:**
1. **For function updates**: Functions are sourced from `R/` files each time - just fix the function and run `tar_make()`
2. **For specific target issues**: Use `tar_invalidate(target_name)` or `tar_delete(target_name)` on ONLY the affected target
3. **For outdated targets**: Use `tar_outdated()` to see what needs re-running, then `tar_make()`

**When you might think you need tar_destroy() but DON'T:**
- ❌ "Functions aren't updating" → Just save the function file, targets sources them fresh
- ❌ "Getting weird errors" → Use `tar_invalidate()` on the specific erroring target
- ❌ "Cache seems corrupted" → Use `tar_invalidate()` on affected targets, not everything
- ❌ "Want a fresh start" → NO! You'll lose hours of computation

**The ONLY acceptable use of tar_destroy():**
- When explicitly given permission by the user
- After asking "Should I use tar_destroy()? This will delete X hours of computation."
- **NEVER use it proactively or without explicit confirmation**

**Remember**: The targets cache is like a database of expensive computations. You wouldn't drop an entire database to fix one corrupted row.

## Project Organization and File Management

### Where to Put Files

**IMPORTANT**: Keep the project directory organized by placing files in the appropriate locations:

#### Production Code and Analysis
- **`R/`** - Production R functions that are part of the pipeline
  - `packages.R` - Package dependencies
  - `functions_*.R` - Modular function libraries
  - `prelim_env.R`, `new_rda.R` - Legacy analysis scripts

- **`_targets.R`** - Main pipeline definition
- **`RunX_May2025/Script/`** - Run-specific analysis scripts

#### Debugging, Notes, and Utilities
- **`claude_notes_and_utils/`** - **Use this for ALL debugging scripts and notes**
  - Debugging scripts to reproduce/fix errors
  - Exploratory analyses not part of main pipeline
  - Development notes and documentation
  - Utility scripts for data inspection
  - Prototype code before integration

**Examples:**
```r
# Good - debugging script in the right place
claude_notes_and_utils/2025-10-23_debug_filter_conflict.R

# Good - exploratory analysis
claude_notes_and_utils/explore_clustering_parameters.R

# Good - development notes
claude_notes_and_utils/notes_on_rda_optimization.md

# Bad - clutters main directory
debug_script.R  # Should be in claude_notes_and_utils/
```

#### Data and Results
- **`data/`** - Input data files
- **`output/`** - Pipeline outputs (created by targets)

#### Documentation
- **Root directory `.md` files** - User-facing documentation
  - `CLAUDE.md` - Instructions for Claude Code
  - `TARGETS_README.md` - Pipeline documentation
  - `README.md` - Project overview (if exists)

### File Naming Best Practices

**Debugging scripts and notes:**
```
claude_notes_and_utils/YYYY-MM-DD_descriptive_name.R
claude_notes_and_utils/YYYY-MM-DD_notes_on_topic.md
```

**Production functions:**
```
R/functions_descriptive_name.R
```

### Cleanup Policy

- **`claude_notes_and_utils/`** - Periodically review and delete old files
- **`output/`** - Managed by targets, can be regenerated
- **`_targets/`** - **NEVER DELETE THIS DIRECTORY!** See critical warning below

### Git Considerations

The `claude_notes_and_utils/` folder can be committed to Git to sync notes and utilities across machines.

Recommended `.gitignore` entries:
```
_targets/
output/
*.Rproj.user
.Rhistory
.RData
```

**Note**: `claude_notes_and_utils/` should be in `.gitignore` and files in it should never be tracked or committed or pushed.


## Modern Coding Practices

### Targets Pipeline
The project now uses the `targets` package for workflow management. This provides:
- **Reproducibility**: Full dependency tracking between analysis steps
- **Efficiency**: Only re-runs outdated targets when inputs change
- **Parallelization**: Dynamic branching allows concurrent analysis of 4 reference genomes
- **Documentation**: Self-documenting pipeline structure

See [TARGETS_README.md](TARGETS_README.md) for complete documentation.

### Tidyverse Best Practices

**IMPORTANT**: Use modern tidyverse functions:

- **DEPRECATED**: `map_dfr()` and `map_dfc()`
- **USE INSTEAD**: `map() |> list_rbind()` and `map() |> list_cbind()`

Example:
```r
# DON'T USE (deprecated)
results <- map_dfr(data_list, process_function)

# USE THIS INSTEAD
results <- map(data_list, process_function) |> list_rbind()
```

This applies to all purrr mapping functions that return data frames.

### Resolving Function Name Conflicts

**IMPORTANT**: Use the `conflicted` package to resolve name conflicts:

When you encounter function conflicts (e.g., `filter` from both `dplyr` and `stats`), **use `conflicts_prefer()`** in your packages.R file rather than using `::` notation throughout your code.

```r
# RECOMMENDED: Use conflicts_prefer() in packages.R
library(conflicted)
conflicts_prefer(dplyr::select)
conflicts_prefer(dplyr::filter)

# Then use functions normally in your code
data |> filter(x > 0)

# DON'T: Use :: everywhere (reduces readability)
data |> dplyr::filter(x > 0)
```

**When to declare preferences:**
- In `R/packages.R` at the top level of your targets workflow
- At the beginning of standalone scripts
- Whenever you load packages that mask common functions

**Common conflicts to resolve:**
```r
conflicts_prefer(dplyr::select)   # vs MASS::select
conflicts_prefer(dplyr::filter)   # vs stats::filter
conflicts_prefer(dplyr::lag)      # vs stats::lag
```

## R Coding Style Preferences

### Error Handling

**Minimal error handling is preferred.** The `targets` package handles errors elegantly, so excessive `try()`, `tryCatch()`, or null checks make code less readable without adding value.

```r
# DON'T: Excessive error handling
run_analysis <- function(data) {
  if (is.null(data)) return(NULL)
  tryCatch({
    result <- analyze(data)
    if (is.null(result)) return(NULL)
    return(result)
  }, error = function(e) {
    message("Error: ", e$message)
    return(NULL)
  })
}

# DO: Let targets handle errors
run_analysis <- function(data) {
  analyze(data)
}
```

**When to use error handling:**
- Only when you need to provide a more informative error message
- When you need to recover from expected, recoverable errors
- In interactive scripts (not targets pipelines)

### Anonymous Functions

**Use modern anonymous function syntax:**

1. **Tidyverse style** (preferred for simple operations with map/walk):
   ```r
   map(data, ~ .x + 1)
   walk(files, ~ write_csv(.x, paste0(.x$name, ".csv")))
   ```

2. **New R style** (preferred for complex multi-line operations):
   ```r
   map(data, \(x) {
     processed <- process(x)
     filtered <- filter(processed, value > 0)
     filtered
   })
   ```

**NEVER use old-style `function(x)` syntax:**
```r
# DON'T USE (old style)
map(data, function(x) x + 1)

# USE INSTEAD (tidyverse style)
map(data, ~ .x + 1)

# OR (new R style)
map(data, \(x) x + 1)
```

**When to use each style:**
- **Tidyverse `~ .x`**: Single-line operations, accessing single argument
- **New R `\(x)`**: Multi-line operations, multiple arguments, clearer variable names
- **Named function**: Reusable logic, complex operations that need documentation

Examples:
```r
# Simple transformation - use tidyverse style
map(numbers, ~ .x^2)

# Multiple arguments - use new R style
map2(x, y, \(a, b) a * b + 1)

# Complex operation - use new R style
map(clusters, \(clust) {
  center <- mean(clust$values)
  spread <- sd(clust$values)
  tibble(center = center, spread = spread)
})

# Reusable complex logic - use named function
calculate_cluster_stats <- function(clust) {
  center <- mean(clust$values)
  spread <- sd(clust$values)
  tibble(center = center, spread = spread)
}
map(clusters, calculate_cluster_stats)
```

## Working with the Targets Pipeline

### Quick Start

```r
# Install required packages (first time only)
source("install_packages.R")

# Visualize the pipeline
library(targets)
tar_visnetwork()

# Run the entire pipeline
tar_make()

# Run with parallel execution (faster)
tar_make_future(workers = 4)

# Load results
cluster_model <- tar_read(cluster_params_model)
cluster_empirical <- tar_read(cluster_params_empirical)
```

### Running the Pipeline

**First time setup:**
```r
# 1. Install required packages
source("install_packages.R")

# 2. Verify the pipeline structure
library(targets)
tar_manifest()  # View all targets
tar_visnetwork()  # Interactive dependency graph
```

**Running targets:**
```r
# Sequential execution (simplest)
tar_make()

# Parallel execution (faster for independent branches)
Still use tar_make() but use the crew package.
```

**Selective execution:**
```r
# Run only specific targets
tar_make(names = c("env_data", "rda"))

# Run everything except certain targets
tar_make(names = tar_manifest()$name[!tar_manifest()$name %in% c("env_pairplot")])

# Invalidate specific target to force re-run
tar_invalidate(rda)
tar_make()
```

### Monitoring Pipeline Progress

**Check pipeline status:**
```r
# See what's outdated and needs to run
tar_outdated()

# View progress during execution
tar_progress()

# Live visualization (updates every 10 seconds)
tar_watch(seconds = 10)

# See metadata about all targets
tar_meta() |> View()

# Check which targets have errors
tar_meta(fields = c(name, error), complete_only = TRUE) |>
  filter(!is.na(error))
```

**Understanding target status:**
- **dispatched**: Target started running
- **completed**: Target finished successfully with ✔
- **errored**: Target failed with ✖
- **skipped**: Target is up-to-date, not re-run
- **canceled**: Target canceled due to upstream error

**Inspecting results:**
```r
# Load a specific target's output
tar_read(cluster_params_model)

# Load multiple targets
tar_load(c(env_data, rda))

# Load all targets into environment
tar_load_everything()

# Read target without loading (useful for large objects)
tar_meta(rda)
```

### Debugging Errors

**Step 1: Identify the error**
```r
# Find targets with errors
tar_meta(fields = c(name, error), complete_only = TRUE) |>
  filter(!is.na(error))

# View detailed error traceback
tar_meta(fields = error, complete_only = TRUE)$error[[1]]
```

**Step 2: Load the workspace**
```r
# Targets saves a workspace snapshot when errors occur
tar_workspace(metadata)  # Load workspace for failed target

# Now debug interactively
ls()  # See what objects were available when error occurred
```

**Step 3: Debug interactively**
```r
# Run the target's command manually
tar_manifest() |> filter(name == "metadata")

# Copy the command and run it step-by-step
metadata_path <- tar_read(metadata_file)
contaminated_samples <- tar_read(contaminated_samples)
load_metadata(metadata_path, contaminated_samples)
```

**Step 4: Fix and re-run**
```r
# After fixing the code:
tar_make()  # Only outdated/errored targets will run
```

### Common Error Patterns & Solutions

**1. Function conflicts**
```
Error: [conflicted] filter found in 2 packages
```
**Solution:** Add `conflicts_prefer()` to `R/packages.R`:
```r
conflicts_prefer(dplyr::filter)
conflicts_prefer(dplyr::select)
```

**2. Missing dependencies**
```
Error: could not load dependency X of target Y
```
**Solution:** The upstream target X failed. Check its error:
```r
tar_meta(fields = error, complete_only = TRUE) |>
  filter(name == "X")
```

**3. Stale cache issues**
```
Error: invalid 'description' argument
```
**Solution:** **DO NOT use tar_destroy()!** Instead, invalidate only the affected target:
```r
# Find which target is corrupted
tar_meta(fields = error, complete_only = TRUE)

# Invalidate ONLY that specific target
tar_invalidate(target_name)
tar_make()

# If absolutely necessary and you have USER PERMISSION:
# tar_destroy()  # ⚠️ NEVER USE WITHOUT EXPLICIT PERMISSION - loses hours of computation!
```

**4. Function not found**
```
Error: could not find function "run_rda"
```
**Solution:** Functions not sourced properly. Check `_targets.R`:
```r
# Should have this line:
lapply(list.files("R", pattern = "^functions.*\\.R$", full.names = TRUE), source)
```

**5. File format errors**
```
Error: Cannot open file 'output.csv': No such file or directory
```
**Solution:** File target paths must be absolute or use `format = "file"`:
```r
tar_target(
  output_file,
  "output/results.csv",
  format = "file"  # Tells targets this is a file path
)
```

**6. Memory issues with large targets**
```
Error: cannot allocate vector of size X Gb
```
**Solution:** Use `tar_target(..., memory = "transient")` for large intermediate objects:
```r
tar_target(
  large_matrix,
  generate_huge_matrix(),
  memory = "transient"  # Don't keep in memory after downstream targets finish
)
```

### Best Practices for Pipeline Development

**1. Start small, build incrementally**
```r
# Test individual functions first
source("R/functions_rda.R")
test_data <- read_csv("RunA_May2025/RDA_results/imputed_genotype_matrix.csv")
run_rda(test_data, env_data)  # Test before adding to pipeline
```

**2. Use `tar_make()` frequently**
```r
# After adding each new target
tar_make()
```

**3. Check dependencies with visualization**
```r
# Verify dependency structure looks correct
tar_visnetwork()
```

**4. Use informative target names**
```r
# GOOD
tar_target(env_data_transformed, transform_env(env_raw))

# BAD
tar_target(data2, transform_env(data1))
```

**5. Keep targets modular and focused**
```r
# GOOD: Each target does one thing
tar_target(env_raw, extract_env_vars(metadata))
tar_target(env_reduced, remove_correlated_vars(env_raw))

# BAD: One target does everything
tar_target(env_processed, extract_env_vars(metadata) |>
           remove_correlated_vars())
```

If a target needs more than 1 line of code to specify, it should just call a function and the function should be defined in a file prefixed by "function_" in the R directory.

**6. Document complex targets**
```r
# Add comments explaining what the target does
tar_target(
  # Calculate ANOVA for each RDA to identify significant axes
  # Uses permutation tests (may be slow for large datasets)
  anova_results,
  calculate_anovas(rda),
  pattern = map(rda)
)
```

### Pipeline Maintenance

**⚠️ CRITICAL WARNING: Read the tar_destroy() section at the top of this file FIRST ⚠️**

**Clean up old targets:**
```r
# Remove targets no longer in _targets.R (safe)
tar_prune()

# Invalidate specific outdated targets (safe)
tar_invalidate(target_name)

# Delete specific target (use with caution, ask permission)
tar_delete(target_name)

# ⚠️ NEVER DO THIS without explicit user permission ⚠️
# tar_destroy()  # Deletes ALL computed results - hours of work lost!
# Only use tar_destroy() if explicitly instructed by the user
# Always ask: "Should I use tar_destroy()? This will delete X hours of computation."
```

**Validate pipeline:**
```r
# Check for issues
tar_validate()

# Test without running
tar_glimpse()
```

**Update dependencies:**
```r
# Update packages
update.packages()

# Re-run pipeline with new package versions
tar_invalidate_all()
tar_make()
```

### Advanced Targets Patterns

**Dynamic branching with list returns:**

When a target function returns a list and you use `pattern = map()`, targets has two behaviors:

```r
# WITHOUT iteration = "list" (default)
# Targets automatically unpacks the list into sub-targets
tar_target(
  my_target,
  my_function(),  # Returns list(a = data1, b = data2)
  pattern = map(input)
)
# Creates: my_target_branch1_a, my_target_branch1_b, my_target_branch2_a, etc.

# WITH iteration = "list"
# Targets keeps the list intact as a single object per branch
tar_target(
  my_target,
  my_function(),  # Returns list(a = data1, b = data2)
  pattern = map(input),
  iteration = "list"  # Keep list structure
)
# Creates: my_target_branch1, my_target_branch2, etc.
# Each branch contains the full list

# Then aggregate with:
tar_target(
  combined_a,
  map(my_target, "a") |> list_rbind()
)

tar_target(
  combined_b,
  map(my_target, "b") |> list_rbind()
)
```

**When to use `iteration = "list"`:**
- Your function returns a list with multiple named elements
- You want to aggregate those elements separately later
- You don't want targets to create sub-targets for each list element

**Data types in dynamic branching:**

Some R packages require specific data types (e.g., matrix vs data frame). Even if your data is numeric, certain functions will fail if given a data frame instead of a matrix:

```r
# Example: movMF package requires matrices
my_function <- function(clustering_model, data) {
  # This will fail if data is a data frame
  predict(clustering_model, data)

  # Fix: convert to matrix
  predict(clustering_model, as.matrix(data))
}

# Common cases requiring matrices:
# - Matrix multiplication: A %*% B
# - Some predict() methods (e.g., movMF)
# - Some modeling functions

# Common cases accepting data frames:
# - Most tidyverse functions
# - Base R functions like mahalanobis(), cov()
```

**Debugging matrix/data frame errors:**

If you see errors like `"requires numeric/complex matrix/vector arguments"`:
1. Check the class of your data: `class(data)` and `is.matrix(data)`
2. Convert to matrix if needed: `as.matrix(data)`
3. Note: Conversion preserves row/column names

### Best Practices for New Targets Projects

**IMPORTANT: Default `iteration = "list"` for dynamic branching**

For future projects, it's recommended to use `iteration = "list"` as the default for dynamically branched targets that return data frames or tibbles. This provides more predictable behavior and easier aggregation:

```r
# RECOMMENDED for new projects
tar_target(
  my_target,
  my_function(),
  pattern = map(input),
  iteration = "list"  # Add this by default
)

# Then aggregate easily
tar_target(
  combined_results,
  my_target |> list_rbind()
)
```

**Why this is better:**
- More predictable: targets doesn't automatically unpack complex return values
- Easier to work with `list_rbind()` and `list_cbind()`
- Avoids unexpected target proliferation when functions return lists
- Consistent with modern tidyverse patterns


### Targets Invalidation and Regeneration

**CRITICAL: Always invalidate before expecting changes**

When you update function code in `R/` files, targets will automatically detect changes and re-run affected targets **on the next `tar_make()`**. However, for plot files and other file targets, you may need to explicitly invalidate:

```r
# Function updates are detected automatically
# Just edit R/functions_*.R and run:
tar_make()  # Will re-run targets that depend on changed functions

# For file targets that you want to force regeneration:
tar_invalidate(target_name)
tar_make()

# Check what will run before running:
tar_outdated()  # Shows what needs updating
```

**Common mistake:**
```r
# Edit R/functions_manhattan.R
# Run tar_make()
# Target says "completed" but output looks the same

# Problem: File target wasn't invalidated!
# Solution:
tar_invalidate(manhattan_plots)
tar_make()
```

Modern R Development Guide
This document captures current best practices for R development, emphasizing modern tidyverse patterns, performance, and style. Last updated: August 2025

Core Principles
Use modern tidyverse patterns - Prioritize dplyr 1.1+ features, native pipe, and current APIs
Profile before optimizing - Use profvis and bench to identify real bottlenecks
Write readable code first - Optimize only when necessary and after profiling
Follow tidyverse style guide - Consistent naming, spacing, and structure
Modern Tidyverse Patterns
Pipe Usage (|> not %>%)
Always use native pipe |> instead of magrittr %>%
R 4.3+ provides all needed features
# Good - Modern native pipe
data |> 
  filter(year >= 2020) |>
  summarise(mean_value = mean(value))

# Avoid - Legacy magrittr pipe  
data %>% 
  filter(year >= 2020) %>%
  summarise(mean_value = mean(value))
Join Syntax (dplyr 1.1+)
Use join_by() instead of character vectors for joins
Support for inequality, rolling, and overlap joins
# Good - Modern join syntax
transactions |> 
  inner_join(companies, by = join_by(company == id))

# Good - Inequality joins
transactions |>
  inner_join(companies, join_by(company == id, year >= since))

# Good - Rolling joins (closest match)
transactions |>
  inner_join(companies, join_by(company == id, closest(year >= since)))

# Avoid - Old character vector syntax
transactions |> 
  inner_join(companies, by = c("company" = "id"))
Multiple Match Handling
Use multiple and unmatched arguments for quality control
# Expect 1:1 matches, error on multiple
inner_join(x, y, by = join_by(id), multiple = "error")

# Allow multiple matches explicitly  
inner_join(x, y, by = join_by(id), multiple = "all")

# Ensure all rows match
inner_join(x, y, by = join_by(id), unmatched = "error")
Data Masking and Tidy Selection
Understand the difference between data masking and tidy selection
Use {{}} (embrace) for function arguments
Use .data[[]] for character vectors
# Data masking functions: arrange(), filter(), mutate(), summarise()
# Tidy selection functions: select(), relocate(), across()

# Function arguments - embrace with {{}}
my_summary <- function(data, group_var, summary_var) {
  data |>
    group_by({{ group_var }}) |>
    summarise(mean_val = mean({{ summary_var }}))
}

# Character vectors - use .data[[]]
for (var in names(mtcars)) {
  mtcars |> count(.data[[var]]) |> print()
}

# Multiple columns - use across()
data |> 
  summarise(across({{ summary_vars }}, ~ mean(.x, na.rm = TRUE)))
Modern Grouping and Column Operations
Use .by for per-operation grouping (dplyr 1.1+)
Use pick() for column selection inside data-masking functions
Use across() for applying functions to multiple columns
Use reframe() for multi-row summaries
# Good - Per-operation grouping (always returns ungrouped)
data |>
  summarise(mean_value = mean(value), .by = category)

# Good - Multiple grouping variables
data |>
  summarise(total = sum(revenue), .by = c(company, year))

# Good - pick() for column selection
data |>
  summarise(
    n_x_cols = ncol(pick(starts_with("x"))),
    n_y_cols = ncol(pick(starts_with("y")))
  )

# Good - across() for applying functions
data |>
  summarise(across(where(is.numeric), mean, .names = "mean_{.col}"), .by = group)

# Good - reframe() for multi-row results
data |>
  reframe(quantiles = quantile(x, c(0.25, 0.5, 0.75)), .by = group)

# Avoid - Old persistent grouping pattern
data |>
  group_by(category) |>
  summarise(mean_value = mean(value)) |>
  ungroup()
Modern rlang Patterns for Data-Masking
Core Concepts
Data-masking allows R expressions to refer to data frame columns as if they were variables in the environment. rlang provides the metaprogramming framework that powers tidyverse data-masking.

Key rlang Tools
Embracing {{}} - Forward function arguments to data-masking functions
Injection !! - Inject single expressions or values
Splicing !!! - Inject multiple arguments from a list
Dynamic dots - Programmable ... with injection support
Pronouns .data/.env - Explicit disambiguation between data and environment variables
Function Argument Patterns
Forwarding with {{}}
Use {{}} to forward function arguments to data-masking functions:

# Single argument forwarding
my_summarise <- function(data, var) {
  data |> dplyr::summarise(mean = mean({{ var }}))
}

# Works with any data-masking expression
mtcars |> my_summarise(cyl)
mtcars |> my_summarise(cyl * am)
mtcars |> my_summarise(.data$cyl)  # pronoun syntax supported
Forwarding ... (No Special Syntax Needed)
# Simple dots forwarding
my_group_by <- function(.data, ...) {
  .data |> dplyr::group_by(...)
}

# Works with tidy selections too
my_select <- function(.data, ...) {
  .data |> dplyr::select(...)
}

# For single-argument tidy selections, wrap in c()
my_pivot_longer <- function(.data, ...) {
  .data |> tidyr::pivot_longer(c(...))
}
Names Patterns with .data
Use .data pronoun for programmatic column access:

# Single column by name
my_mean <- function(data, var) {
  data |> dplyr::summarise(mean = mean(.data[[var]]))
}

# Usage - completely insulated from data-masking
mtcars |> my_mean("cyl")  # No ambiguity, works like regular function

# Multiple columns with all_of()
my_select_vars <- function(data, vars) {
  data |> dplyr::select(all_of(vars))
}

mtcars |> my_select_vars(c("cyl", "am"))
Injection Operators
When to Use Each Operator
Operator	Use Case	Example
{{ }}	Forward function arguments	summarise(mean = mean({{ var }}))
!!	Inject single expression/value	summarise(mean = mean(!!sym(var)))
!!!	Inject multiple arguments	group_by(!!!syms(vars))
.data[[]]	Access columns by name	mean(.data[[var]])
Advanced Injection with !!
# Create symbols from strings
var <- "cyl"
mtcars |> dplyr::summarise(mean = mean(!!sym(var)))

# Inject values to avoid name collisions
df <- data.frame(x = 1:3)
x <- 100
df |> dplyr::mutate(scaled = x / !!x)  # Uses both data and env x

# Use data_sym() for tidyeval contexts (more robust)
mtcars |> dplyr::summarise(mean = mean(!!data_sym(var)))
Splicing with !!!
# Multiple symbols from character vector
vars <- c("cyl", "am")
mtcars |> dplyr::group_by(!!!syms(vars))

# Or use data_syms() for tidy contexts
mtcars |> dplyr::group_by(!!!data_syms(vars))

# Splice lists of arguments
args <- list(na.rm = TRUE, trim = 0.1)
mtcars |> dplyr::summarise(mean = mean(cyl, !!!args))
Dynamic Dots Patterns
Using list2() for Dynamic Dots Support
my_function <- function(...) {
  # Collect with list2() instead of list() for dynamic features
  dots <- list2(...)
  # Process dots...
}

# Enables these features:
my_function(a = 1, b = 2)           # Normal usage
my_function(!!!list(a = 1, b = 2))  # Splice a list
my_function("{name}" := value)      # Name injection
my_function(a = 1, )               # Trailing commas OK
Name Injection with Glue Syntax
# Basic name injection
name <- "result"
list2("{name}" := 1)  # Creates list(result = 1)

# In function arguments with {{
my_mean <- function(data, var) {
  data |> dplyr::summarise("mean_{{ var }}" := mean({{ var }}))
}

mtcars |> my_mean(cyl)        # Creates column "mean_cyl"
mtcars |> my_mean(cyl * am)   # Creates column "mean_cyl * am"

# Allow custom names with englue()
my_mean <- function(data, var, name = englue("mean_{{ var }}")) {
  data |> dplyr::summarise("{name}" := mean({{ var }}))
}

# User can override default
mtcars |> my_mean(cyl, name = "cylinder_mean")
Pronouns for Disambiguation
.data and .env Best Practices
# Explicit disambiguation prevents masking issues
cyl <- 1000  # Environment variable

mtcars |> dplyr::summarise(
  data_cyl = mean(.data$cyl),    # Data frame column
  env_cyl = mean(.env$cyl),      # Environment variable
  ambiguous = mean(cyl)          # Could be either (usually data wins)
)

# Use in loops and programmatic contexts
vars <- c("cyl", "am")
for (var in vars) {
  result <- mtcars |> dplyr::summarise(mean = mean(.data[[var]]))
  print(result)
}
Programming Patterns
Bridge Patterns
Converting between data-masking and tidy selection behaviors:

# across() as selection-to-data-mask bridge
my_group_by <- function(data, vars) {
  data |> dplyr::group_by(across({{ vars }}))
}

# Works with tidy selection
mtcars |> my_group_by(starts_with("c"))

# across(all_of()) as names-to-data-mask bridge  
my_group_by <- function(data, vars) {
  data |> dplyr::group_by(across(all_of(vars)))
}

mtcars |> my_group_by(c("cyl", "am"))
Transformation Patterns
# Transform single arguments by wrapping
my_mean <- function(data, var) {
  data |> dplyr::summarise(mean = mean({{ var }}, na.rm = TRUE))
}

# Transform dots with across()
my_means <- function(data, ...) {
  data |> dplyr::summarise(across(c(...), ~ mean(.x, na.rm = TRUE)))
}

# Manual transformation (advanced)
my_means_manual <- function(.data, ...) {
  vars <- enquos(..., .named = TRUE)
  vars <- purrr::map(vars, ~ expr(mean(!!.x, na.rm = TRUE)))
  .data |> dplyr::summarise(!!!vars)
}
Error-Prone Patterns to Avoid
Don't Use These Deprecated/Dangerous Patterns
# Avoid - String parsing and eval (security risk)
var <- "cyl" 
code <- paste("mean(", var, ")")
eval(parse(text = code))  # Dangerous!

# Good - Symbol creation and injection
!!sym(var)  # Safe symbol injection

# Avoid - get() in data mask (name collisions)
with(mtcars, mean(get(var)))  # Collision-prone

# Good - Explicit injection or .data
with(mtcars, mean(!!sym(var)))  # Safe
# or
mtcars |> summarise(mean(.data[[var]]))  # Even safer
Common Mistakes
# Don't use {{ }} on non-arguments
my_func <- function(x) {
  x <- force(x)  # x is now a value, not an argument
  quo(mean({{ x }}))  # Wrong! Captures value, not expression
}

# Don't mix injection styles unnecessarily
# Pick one approach and stick with it:
# Either: embrace pattern
my_func <- function(data, var) data |> summarise(mean = mean({{ var }}))
# Or: defuse-and-inject pattern  
my_func <- function(data, var) {
  var <- enquo(var)
  data |> summarise(mean = mean(!!var))
}
Package Development with rlang
Import Strategy
# In DESCRIPTION:
Imports: rlang

# In NAMESPACE, import specific functions:
importFrom(rlang, enquo, enquos, expr, !!!, :=)

# Or import key functions:
#' @importFrom rlang := enquo enquos
Documentation Tags
#' @param var <[`data-masked`][dplyr::dplyr_data_masking]> Column to summarize
#' @param ... <[`dynamic-dots`][rlang::dyn-dots]> Additional grouping variables  
#' @param cols <[`tidy-select`][dplyr::dplyr_tidy_select]> Columns to select
Testing rlang Functions
# Test data-masking behavior
test_that("function supports data masking", {
  result <- my_function(mtcars, cyl)
  expect_equal(names(result), "mean_cyl")
  
  # Test with expressions
  result2 <- my_function(mtcars, cyl * 2)
  expect_true("mean_cyl * 2" %in% names(result2))
})

# Test injection behavior
test_that("function supports injection", {
  var <- "cyl"
  result <- my_function(mtcars, !!sym(var))
  expect_true(nrow(result) > 0)
})
This modern rlang approach enables clean, safe metaprogramming while maintaining the intuitive data-masking experience users expect from tidyverse functions.

Performance Best Practices
Performance Tool Selection Guide
When to Use Each Performance Tool
Profiling Tools Decision Matrix
Tool	Use When	Don't Use When	What It Shows
profvis	Complex code, unknown bottlenecks	Simple functions, known issues	Time per line, call stack
bench::mark()	Comparing alternatives	Single approach	Relative performance, memory
system.time()	Quick checks	Detailed analysis	Total runtime only
Rprof()	Base R only environments	When profvis available	Raw profiling data
Step-by-Step Performance Workflow
# 1. Profile first - find the actual bottlenecks
library(profvis)
profvis({
  # Your slow code here
})

# 2. Focus on the slowest parts (80/20 rule)
# Don't optimize until you know where time is spent

# 3. Benchmark alternatives for hot spots
library(bench)
bench::mark(
  current = current_approach(data),
  vectorized = vectorized_approach(data),
  parallel = map(data, in_parallel(func))
)

# 4. Consider tool trade-offs based on bottleneck type
When Each Tool Helps vs Hurts
Parallel Processing (in_parallel())

# Helps when:
✓ CPU-intensive computations
✓ Embarassingly parallel problems  
✓ Large datasets with independent operations
✓ I/O bound operations (file reading, API calls)

# Hurts when:
✗ Simple, fast operations (overhead > benefit)
✗ Memory-intensive operations (may cause thrashing)
✗ Operations requiring shared state
✗ Small datasets

# Example decision point:
expensive_func <- function(x) Sys.sleep(0.1) # 100ms per call
fast_func <- function(x) x^2                 # microseconds per call

# Good for parallel
map(1:100, in_parallel(expensive_func))  # ~10s -> ~2.5s on 4 cores

# Bad for parallel (overhead > benefit)  
map(1:100, in_parallel(fast_func))       # 100μs -> 50ms (500x slower!)
vctrs Backend Tools

# Use vctrs when:
✓ Type safety matters more than raw speed
✓ Building reusable package functions
✓ Complex coercion/combination logic
✓ Consistent behavior across edge cases

# Avoid vctrs when:
✗ One-off scripts where speed matters most
✗ Simple operations where base R is sufficient  
✗ Memory is extremely constrained

# Decision point:
simple_combine <- function(x, y) c(x, y)           # Fast, simple
robust_combine <- function(x, y) vec_c(x, y)      # Safer, slight overhead

# Use simple for hot loops, robust for package APIs
Data Backend Selection

# Use data.table when:
✓ Very large datasets (>1GB)
✓ Complex grouping operations
✓ Reference semantics desired
✓ Maximum performance critical

# Use dplyr when:
✓ Readability and maintainability priority
✓ Complex joins and window functions
✓ Team familiarity with tidyverse
✓ Moderate sized data (<100MB)

# Use base R when:
✓ No dependencies allowed
✓ Simple operations
✓ Teaching/learning contexts
Profiling Best Practices
# 1. Profile realistic data sizes
profvis({
  # Use actual data size, not toy examples
  real_data |> your_analysis()
})

# 2. Profile multiple runs for stability
bench::mark(
  your_function(data),
  min_iterations = 10,  # Multiple runs
  max_iterations = 100
)

# 3. Check memory usage too
bench::mark(
  approach1 = method1(data), 
  approach2 = method2(data),
  check = FALSE,  # If outputs differ slightly
  filter_gc = FALSE  # Include GC time
)

# 4. Profile with realistic usage patterns
# Not just isolated function calls
Performance Anti-Patterns to Avoid
# Don't optimize without measuring
# ✗ "This looks slow" -> immediately rewrite
# ✓ Profile first, optimize bottlenecks

# Don't over-engineer for performance  
# ✗ Complex optimizations for 1% gains
# ✓ Focus on algorithmic improvements

# Don't assume - measure
# ✗ "for loops are always slow in R"
# ✓ Benchmark your specific use case

# Don't ignore readability costs
# ✗ Unreadable code for minor speedups
# ✓ Readable code with targeted optimizations
Backend Tools for Performance
Consider lower-level tools when speed is critical
Use vctrs, rlang backends when appropriate
Profile to identify true bottlenecks
# For packages - consider backend tools
# vctrs for type-stable vector operations
# rlang for metaprogramming
# data.table for large data operations
When to Use vctrs
Core Benefits
Type stability - Predictable output types regardless of input values
Size stability - Predictable output sizes from input sizes
Consistent coercion rules - Single set of rules applied everywhere
Robust class design - Proper S3 vector infrastructure
Use vctrs when:
Building Custom Vector Classes
# Good - vctrs-based vector class
new_percent <- function(x = double()) {
  vec_assert(x, double())
  new_vctr(x, class = "pkg_percent")
}

# Automatic data frame compatibility, subsetting, etc.
Type-Stable Functions in Packages
# Good - Guaranteed output type
my_function <- function(x, y) {
  # Always returns double, regardless of input values
  vec_cast(result, double())
}

# Avoid - Type depends on data
sapply(x, function(i) if(condition) 1L else 1.0)
Consistent Coercion/Casting
# Good - Explicit casting with clear rules
vec_cast(x, double())  # Clear intent, predictable behavior

# Good - Common type finding
vec_ptype_common(x, y, z)  # Finds richest compatible type

# Avoid - Base R inconsistencies  
c(factor("a"), "b")  # Unpredictable behavior
Size/Length Stability
# Good - Predictable sizing
vec_c(x, y)  # size = vec_size(x) + vec_size(y)
vec_rbind(df1, df2)  # size = sum of input sizes

# Avoid - Unpredictable sizing
c(env_object, function_object)  # Unpredictable length
vctrs vs Base R Decision Matrix
Use Case	Base R	vctrs	When to Choose vctrs
Simple combining	c()	vec_c()	Need type stability, consistent rules
Custom classes	S3 manually	new_vctr()	Want data frame compatibility, subsetting
Type conversion	as.*()	vec_cast()	Need explicit, safe casting
Finding common type	Not available	vec_ptype_common()	Combining heterogeneous inputs
Size operations	length()	vec_size()	Working with non-vector objects
Implementation Patterns
Basic Vector Class
# Constructor (low-level)
new_percent <- function(x = double()) {
  vec_assert(x, double())
  new_vctr(x, class = "pkg_percent")
}

# Helper (user-facing)
percent <- function(x = double()) {
  x <- vec_cast(x, double())
  new_percent(x)
}

# Format method
format.pkg_percent <- function(x, ...) {
  paste0(vec_data(x) * 100, "%")
}
Coercion Methods
# Self-coercion
vec_ptype2.pkg_percent.pkg_percent <- function(x, y, ...) {
  new_percent()
}

# With double
vec_ptype2.pkg_percent.double <- function(x, y, ...) double()
vec_ptype2.double.pkg_percent <- function(x, y, ...) double()

# Casting
vec_cast.pkg_percent.double <- function(x, to, ...) {
  new_percent(x)
}
vec_cast.double.pkg_percent <- function(x, to, ...) {
  vec_data(x)
}
Performance Considerations
When vctrs Adds Overhead
Simple operations - vec_c(1, 2) vs c(1, 2) for basic atomic vectors
One-off scripts - Type safety less critical than speed
Small vectors - Overhead may outweigh benefits
When vctrs Improves Performance
Package functions - Type stability prevents expensive re-computation
Complex classes - Consistent behavior reduces debugging
Data frame operations - Robust column type handling
Repeated operations - Predictable types enable optimization
Package Development Guidelines
Exports and Dependencies
# DESCRIPTION - Import specific functions
Imports: vctrs

# NAMESPACE - Import what you need
importFrom(vctrs, vec_assert, new_vctr, vec_cast, vec_ptype_common)

# Or if using extensively
import(vctrs)
Testing vctrs Classes
# Test type stability
test_that("my_function is type stable", {
  expect_equal(vec_ptype(my_function(1:3)), vec_ptype(double()))
  expect_equal(vec_ptype(my_function(integer())), vec_ptype(double()))
})

# Test coercion
test_that("coercion works", {
  expect_equal(vec_ptype_common(new_percent(), 1.0), double())
  expect_error(vec_ptype_common(new_percent(), "a"))
})
Don't Use vctrs When:
Simple one-off analyses - Base R is sufficient
No custom classes needed - Standard types work fine
Performance critical + simple operations - Base R may be faster
External API constraints - Must return base R types
The key insight: vctrs is most valuable in package development where type safety, consistency, and extensibility matter more than raw speed for simple operations.

Modern purrr Patterns
Use map() |> list_rbind() instead of superseded map_dfr()
Use walk() for side effects (file writing, plotting)
Use in_parallel() for scaling across cores
# Modern data frame row binding (purrr 1.0+)
models <- data_splits |> 
  map(\(split) train_model(split)) |>
  list_rbind()  # Replaces map_dfr()

# Column binding  
summaries <- data_list |> 
  map(\(df) get_summary_stats(df)) |>
  list_cbind()  # Replaces map_dfc()

# Side effects with walk()
plots <- walk2(data_list, plot_names, \(df, name) {
  p <- ggplot(df, aes(x, y)) + geom_point()
  ggsave(name, p)
})

# Parallel processing (purrr 1.1.0+)
library(mirai)
daemons(4)
results <- large_datasets |> 
  map(in_parallel(expensive_computation))
daemons(0)
String Manipulation with stringr
Use stringr over base R string functions
Consistent str_ prefix and string-first argument order
Pipe-friendly and vectorized by design
# Good - stringr (consistent, pipe-friendly)
text |>
  str_to_lower() |>
  str_trim() |>
  str_replace_all("pattern", "replacement") |>
  str_extract("\\d+")

# Common patterns
str_detect(text, "pattern")     # vs grepl("pattern", text)
str_extract(text, "pattern")    # vs complex regmatches()
str_replace_all(text, "a", "b") # vs gsub("a", "b", text)
str_split(text, ",")            # vs strsplit(text, ",")
str_length(text)                # vs nchar(text)
str_sub(text, 1, 5)             # vs substr(text, 1, 5)

# String combination and formatting
str_c("a", "b", "c")            # vs paste0()
str_glue("Hello {name}!")       # templating
str_pad(text, 10, "left")       # padding
str_wrap(text, width = 80)      # text wrapping

# Case conversion  
str_to_lower(text)              # vs tolower()
str_to_upper(text)              # vs toupper()
str_to_title(text)              # vs tools::toTitleCase()

# Pattern helpers for clarity
str_detect(text, fixed("$"))    # literal match
str_detect(text, regex("\\d+")) # explicit regex
str_detect(text, coll("é", locale = "fr")) # collation

# Avoid - inconsistent base R functions
grepl("pattern", text)          # argument order varies
regmatches(text, regexpr(...))  # complex extraction
gsub("a", "b", text)           # different arg order
Vectorization and Performance
# Good - vectorized operations
result <- x + y

# Good - Type-stable purrr functions
map_dbl(data, mean)    # always returns double
map_chr(data, class)   # always returns character

# Avoid - Type-unstable base functions
sapply(data, mean)     # might return list or vector

# Avoid - explicit loops for simple operations
result <- numeric(length(x))
for(i in seq_along(x)) {
  result[i] <- x[i] + y[i]
}
Function Writing Best Practices
Structure and Style
# Good function structure
rescale01 <- function(x) {
  rng <- range(x, na.rm = TRUE, finite = TRUE)
  (x - rng[1]) / (rng[2] - rng[1])
}

# Use type-stable outputs
map_dbl()   # returns numeric vector
map_chr()   # returns character vector  
map_lgl()   # returns logical vector
Naming and Arguments
# Good naming: snake_case for variables/functions
calculate_mean_score <- function(data, score_col) {
  # Function body
}

# Prefix non-standard arguments with .
my_function <- function(.data, ...) {
  # Reduces argument conflicts
}
Style Guide Essentials
Object Names
Use snake_case for all names
Variable names = nouns, function names = verbs
Avoid dots except for S3 methods
# Good
day_one
calculate_mean  
user_data

# Avoid
DayOne
calculate.mean
userData
Spacing and Layout
# Good spacing
x[, 1]
mean(x, na.rm = TRUE)
if (condition) {
  action()
}

# Pipe formatting
data |>
  filter(year >= 2020) |>
  group_by(category) |>
  summarise(
    mean_value = mean(value),
    count = n()
  )
Common Anti-Patterns to Avoid
Legacy Patterns
# Avoid - Old pipe
data %>% function()

# Avoid - Old join syntax  
inner_join(x, y, by = c("a" = "b"))

# Avoid - Implicit type conversion
sapply()  # Use map_*() instead

# Avoid - String manipulation in data masking
mutate(data, !!paste0("new_", var) := value)  
# Use across() or other approaches instead
Performance Anti-Patterns
# Avoid - Growing objects in loops
result <- c()
for(i in 1:n) {
  result <- c(result, compute(i))  # Slow!
}

# Good - Pre-allocate
result <- vector("list", n)
for(i in 1:n) {
  result[[i]] <- compute(i)
}

# Better - Use purrr
result <- map(1:n, compute)
Object-Oriented Programming
S7: Modern OOP for New Projects
S7 combines S3 simplicity with S4 structure
Formal class definitions with automatic validation
Compatible with existing S3 code
# S7 class definition
Range <- new_class("Range",
  properties = list(
    start = class_double,
    end = class_double
  ),
  validator = function(self) {
    if (self@end < self@start) {
      "@end must be >= @start"
    }
  }
)

# Usage - constructor and property access
x <- Range(start = 1, end = 10)
x@start  # 1
x@end <- 20  # automatic validation

# Methods
inside <- new_generic("inside", "x")
method(inside, Range) <- function(x, y) {
  y >= x@start & y <= x@end
}
OOP System Decision Matrix
S7 vs vctrs vs S3/S4 Decision Tree
Start here: What are you building?

1. Vector-like objects (things that behave like atomic vectors)
Use vctrs when:
✓ Need data frame integration (columns/rows)
✓ Want type-stable vector operations  
✓ Building factor-like, date-like, or numeric-like classes
✓ Need consistent coercion/casting behavior
✓ Working with existing tidyverse infrastructure

Examples: custom date classes, units, categorical data
2. General objects (complex data structures, not vector-like)
Use S7 when:
✓ NEW projects that need formal classes
✓ Want property validation and safe property access (@)
✓ Need multiple dispatch (beyond S3's double dispatch)
✓ Converting from S3 and want better structure
✓ Building class hierarchies with inheritance
✓ Want better error messages and discoverability

Use S3 when:
✓ Simple classes with minimal structure needs
✓ Maximum compatibility and minimal dependencies  
✓ Quick prototyping or internal classes
✓ Contributing to existing S3-based ecosystems
✓ Performance is absolutely critical (minimal overhead)

Use S4 when:
✓ Working in Bioconductor ecosystem
✓ Need complex multiple inheritance (S7 doesn't support this)
✓ Existing S4 codebase that works well
Detailed S7 vs S3 Comparison
Feature	S3	S7	When S7 wins
Class definition	Informal (convention)	Formal (new_class())	Need guaranteed structure
Property access	$ or attr() (unsafe)	@ (safe, validated)	Property validation matters
Validation	Manual, inconsistent	Built-in validators	Data integrity important
Method discovery	Hard to find methods	Clear method printing	Developer experience matters
Multiple dispatch	Limited (base generics)	Full multiple dispatch	Complex method dispatch needed
Inheritance	Informal, NextMethod()	Explicit super()	Predictable inheritance needed
Migration cost	-	Low (1-2 hours)	Want better structure
Performance	Fastest	~Same as S3	Performance difference negligible
Compatibility	Full S3	Full S3 + S7	Need both old and new patterns
Practical Guidelines
Choose S7 when you have:
# Complex validation needs
Range <- new_class("Range",
  properties = list(start = class_double, end = class_double),
  validator = function(self) {
    if (self@end < self@start) "@end must be >= @start"
  }
)

# Multiple dispatch needs  
method(generic, list(ClassA, ClassB)) <- function(x, y) ...

# Class hierarchies with clear inheritance
Child <- new_class("Child", parent = Parent)
Choose vctrs when you need:
# Vector-like behavior in data frames
percent <- new_vctr(0.5, class = "percentage") 
data.frame(x = 1:3, pct = percent(c(0.1, 0.2, 0.3)))  # works seamlessly

# Type-stable operations
vec_c(percent(0.1), percent(0.2))  # predictable behavior
vec_cast(0.5, percent())          # explicit, safe casting
Choose S3 when you have:
# Simple classes without complex needs
new_simple <- function(x) structure(x, class = "simple")
print.simple <- function(x, ...) cat("Simple:", x)

# Maximum performance needs (rare)
# Existing S3 ecosystem contributions
Migration Strategy
S3 → S7: Usually 1-2 hours work, keeps full compatibility
S4 → S7: More complex, evaluate if S4 features are actually needed
Base R → vctrs: For vector-like classes, significant benefits
Combining approaches: S7 classes can use vctrs principles internally
Package Development Decision Guide
Dependency Strategy
When to Add Dependencies vs Base R
# Add dependency when:
✓ Significant functionality gain
✓ Maintenance burden reduction
✓ User experience improvement
✓ Complex implementation (regex, dates, web)

# Use base R when:
✓ Simple utility functions
✓ Package will be widely used (minimize deps)
✓ Dependency is large for small benefit
✓ Base R solution is straightforward

# Example decisions:
str_detect(x, "pattern")    # Worth stringr dependency
length(x) > 0              # Don't need purrr for this
parse_dates(x)             # Worth lubridate dependency  
x + 1                      # Don't need dplyr for this
Tidyverse Dependency Guidelines
# Core tidyverse (usually worth it):
dplyr     # Complex data manipulation
purrr     # Functional programming, parallel
stringr   # String manipulation
tidyr     # Data reshaping

# Specialized tidyverse (evaluate carefully):
lubridate # If heavy date manipulation
forcats   # If many categorical operations  
readr     # If specific file reading needs
ggplot2   # If package creates visualizations

# Heavy dependencies (use sparingly):
tidyverse # Meta-package, very heavy
shiny     # Only for interactive apps
API Design Patterns
Function Design Strategy
# Modern tidyverse API patterns

# 1. Use .by for per-operation grouping
my_summarise <- function(.data, ..., .by = NULL) {
  # Support modern grouped operations
}

# 2. Use {{ }} for user-provided columns  
my_select <- function(.data, cols) {
  .data |> select({{ cols }})
}

# 3. Use ... for flexible arguments
my_mutate <- function(.data, ..., .by = NULL) {
  .data |> mutate(..., .by = {{ .by }})
}

# 4. Return consistent types (tibbles, not data.frames)
my_function <- function(.data) {
  result |> tibble::as_tibble()
}
Input Validation Strategy
# Validation level by function type:

# User-facing functions - comprehensive validation
user_function <- function(x, threshold = 0.5) {
  # Check all inputs thoroughly
  if (!is.numeric(x)) stop("x must be numeric")
  if (!is.numeric(threshold) || length(threshold) != 1) {
    stop("threshold must be a single number")
  }
  # ... function body
}

# Internal functions - minimal validation  
.internal_function <- function(x, threshold) {
  # Assume inputs are valid (document assumptions)
  # Only check critical invariants
  # ... function body
}

# Package functions with vctrs - type-stable validation
safe_function <- function(x, y) {
  x <- vec_cast(x, double())
  y <- vec_cast(y, double())
  # Automatic type checking and coercion
}
Error Handling Patterns
# Good error messages - specific and actionable
if (length(x) == 0) {
  cli::cli_abort(
    "Input {.arg x} cannot be empty.",
    "i" = "Provide a non-empty vector."
  )
}

# Include function name in errors
validate_input <- function(x, call = caller_env()) {
  if (!is.numeric(x)) {
    cli::cli_abort("Input must be numeric", call = call)
  }
}

# Use consistent error styling
# cli package for user-friendly messages
# rlang for developer tools
When to Create Internal vs Exported Functions
Export Function When:
✓ Users will call it directly
✓ Other packages might want to extend it
✓ Part of the core package functionality
✓ Stable API that won't change often

# Example: main data processing functions
export_these <- function(.data, ...) {
  # Comprehensive input validation
  # Full documentation required
  # Stable API contract
}
Keep Function Internal When:
✓ Implementation detail that may change
✓ Only used within package
✓ Complex implementation helpers
✓ Would clutter user-facing API

# Example: helper functions
.internal_helper <- function(x, y) {
  # Minimal documentation
  # Can change without breaking users
  # Assume inputs are pre-validated
}
Testing and Documentation Strategy
Testing Levels
# Unit tests - individual functions
test_that("function handles edge cases", {
  expect_equal(my_func(c()), expected_empty_result)
  expect_error(my_func(NULL), class = "my_error_class")
})

# Integration tests - workflow combinations  
test_that("pipeline works end-to-end", {
  result <- data |> 
    step1() |> 
    step2() |>
    step3()
  expect_s3_class(result, "expected_class")
})

# Property-based tests for package functions
test_that("function properties hold", {
  # Test invariants across many inputs
})
Documentation Priorities
# Must document:
✓ All exported functions
✓ Complex algorithms or formulas
✓ Non-obvious parameter interactions
✓ Examples of typical usage

# Can skip documentation:
✗ Simple internal helpers
✗ Obvious parameter meanings
✗ Functions that just call other functions
Migration Notes
From Base R to Modern Tidyverse
# Data manipulation
subset(data, condition)          -> filter(data, condition)
data[order(data$x), ]           -> arrange(data, x)
aggregate(x ~ y, data, mean)    -> summarise(data, mean(x), .by = y)

# Functional programming
sapply(x, f)                    -> map(x, f)  # type-stable
lapply(x, f)                    -> map(x, f)

# String manipulation  
grepl("pattern", text)          -> str_detect(text, "pattern")
gsub("old", "new", text)        -> str_replace_all(text, "old", "new")
substr(text, 1, 5)              -> str_sub(text, 1, 5)
nchar(text)                     -> str_length(text)
strsplit(text, ",")             -> str_split(text, ",")
paste0(a, b)                    -> str_c(a, b)
tolower(text)                   -> str_to_lower(text)
From Old to New Tidyverse Patterns
# Pipes
data %>% function()             -> data |> function()

# Grouping (dplyr 1.1+)
group_by(data, x) |> 
  summarise(mean(y)) |> 
  ungroup()                     -> summarise(data, mean(y), .by = x)

# Column selection
across(starts_with("x"))        -> pick(starts_with("x"))  # for selection only

# Joins
by = c("a" = "b")              -> by = join_by(a == b)

# Multi-row summaries
summarise(data, x, .groups = "drop") -> reframe(data, x)

# Data reshaping
gather()/spread()               -> pivot_longer()/pivot_wider()

# String separation (tidyr 1.3+)
separate(col, into = c("a", "b")) -> separate_wider_delim(col, delim = "_", names = c("a", "b"))
extract(col, into = "x", regex)   -> separate_wider_regex(col, patterns = c(x = regex))
Performance Migrations
# Old -> New performance patterns
for loops for parallelizable work -> map(data, in_parallel(f))
Manual type checking             -> vec_assert() / vec_cast()
Inconsistent coercion           -> vec_ptype_common() / vec_c()

# Superseded purrr functions (purrr 1.0+)
map_dfr(x, f)                   -> map(x, f) |> list_rbind()
map_dfc(x, f)                   -> map(x, f) |> list_cbind()
map2_dfr(x, y, f)               -> map2(x, y, f) |> list_rbind()
pmap_dfr(list, f)               -> pmap(list, f) |> list_rbind()
imap_dfr(x, f)                  -> imap(x, f) |> list_rbind()

# For side effects
walk(x, write_file)             # instead of for loops
walk2(data, paths, write_csv)   # multiple arguments

---

## Lessons Learned (Do Not Forget)

**IMPORTANT**: Add critical lessons learned during development here. Keep entries concise - only include things that are project-specific or easy to forget.

### Targets Best Practices

1. **Separate targets for different data types (e.g., 16S vs ITS)**: Don't use dynamic branching when you'll later need to treat them differently. Targets can run independent targets in parallel automatically without dynamic branching.

2. **Individual config parameter targets**: Instead of one `config_params` list, break each parameter into its own target. This way, changing one parameter only invalidates targets that depend on that specific parameter.
   ```r
   # GOOD: Individual parameters
   tar_target(swarm_d, 1)
   tar_target(filter_alpha, 0.1)

   # BAD: Single config list (changing any param invalidates all dependents)
   tar_target(config_params, list(swarm_d = 1, filter_alpha = 0.1, ...))
   ```

3. **Don't use `tar_invalidate()` for failed targets**: Failed targets are automatically re-run when you call `tar_make()`. The whole point of targets is to intelligently manage invalidation automatically. Only use `tar_invalidate()` in special circumstances (e.g., when you need to force a re-run of a target that appears up-to-date but should be re-run for external reasons).

4. **Check function signatures before using external packages**: When using functions from external packages (like DECIPHER's `LearnTaxa`), always check the actual function signature first with `args(function_name)` or `?function_name`. Don't assume argument names based on documentation from other sources or memory.

5. **Bioconductor packages create many function conflicts**: When adding Biostrings/DECIPHER or other Bioconductor packages, expect multiple conflicts with base R and tidyverse functions. Resolve proactively:
   ```r
   # In R/packages.R - add all at once to avoid iterative failures
   conflicts_prefer(Biostrings::collapse)  # vs dplyr::collapse
   conflicts_prefer(base::intersect)       # vs Biostrings/generics
   conflicts_prefer(base::setdiff)         # vs Biostrings/generics
   conflicts_prefer(base::union)           # vs Biostrings/generics
   conflicts_prefer(base::setequal)        # vs Biostrings/generics
   ```

### Script Execution Best Practices

6. **Always run R scripts in the background**: When running Rscripts (especially long-running ones), always use background execution and monitor progress periodically. R's `message()` output can be heavily buffered, so poll the output file regularly to track progress:
   ```bash
   # Run in background
   Rscript my_script.R 2>&1 &

   # Monitor progress
   tail -f output.log
   ```
   Keep the user informed as the script runs by periodically checking output.

7. **Use small sample sizes in spot tests**: For spot tests on cropped data, use very small sample sizes (e.g., 5000 instead of 1M). The goal is to verify the code works correctly, not to get production-quality results. Large sample sizes on small test data can cause disproportionate slowdowns.

