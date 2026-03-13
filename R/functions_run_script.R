# functions_run_script.R
# Lightweight script runner with parameter injection and output capture.
# Replaces guildai for running R scripts with parameter overrides.
#
# Scripts use comment flags to mark parameters, outputs, and tracked data:
#   #| param    — next line is a parameter assignment with a default value
#   #| output   — next line is an output variable to capture and return
#   #| data key — declares a key to extract from stdout (e.g., "loss: 0.54")
#
# Usage:
#   run_script("train.R")                                    # defaults
#   run_script("train.R", list(latent_dim = 64, lr = 0.001)) # overrides
#   Rscript run.R train.R latent_dim=64 lr=0.001             # CLI

parse_script_flags <- function(lines) {
  param_output_pattern <- "^\\s*#\\|\\s*(param|output)\\s*$"
  data_pattern <- "^\\s*#\\|\\s*data\\s+(.+?)\\s*$"
  assign_pattern <- "^\\s*([a-zA-Z_.][a-zA-Z0-9_.]*?)\\s*(<-|=)\\s*(.+)$"

  params <- list()
  outputs <- character(0)
  data_keys <- character(0)

  # Handle #| data declarations (key names on the same line)
  # Supports: #| data loss acc `kl loss`
  data_lines <- which(grepl(data_pattern, lines))
  for (dl in data_lines) {
    raw <- sub(data_pattern, "\\1", lines[dl])
    # Extract backtick-quoted keys and bare words
    tokens <- regmatches(
      raw, gregexpr("`[^`]+`|\\S+", raw)
    )[[1]]
    # Strip backticks from quoted tokens
    tokens <- gsub("^`|`$", "", tokens)
    data_keys <- c(data_keys, tokens)
  }

  # Handle #| param and #| output (tag the next line)
  flag_lines <- which(grepl(param_output_pattern, lines))

  for (fl in flag_lines) {
    flag_type <- sub(param_output_pattern, "\\1", trimws(lines[fl]))

    # Find the next non-blank, non-comment line
    target_line <- fl + 1
    while (target_line <= length(lines) &&
           (trimws(lines[target_line]) == "" ||
            grepl("^\\s*#", lines[target_line]))) {
      target_line <- target_line + 1
    }
    if (target_line > length(lines)) next

    m <- regmatches(lines[target_line],
                    regexec(assign_pattern, lines[target_line]))[[1]]
    if (length(m) == 0) {
      warning("Flag #| ", flag_type, " on line ", fl,
              " is not followed by an assignment (line ", target_line, ")")
      next
    }

    var_name <- m[2]

    if (flag_type == "param") {
      params[[length(params) + 1]] <- list(
        name = var_name,
        line_number = target_line,
        default_expr = m[4]
      )
    } else if (flag_type == "output") {
      outputs <- c(outputs, var_name)
    }
  }

  list(params = params, outputs = outputs, data_keys = data_keys)
}

inject_params <- function(lines, parsed, overrides) {
  assign_pattern <- "^(\\s*[a-zA-Z_.][a-zA-Z0-9_.]*?\\s*(?:<-|=)\\s*)(.+)$"

  for (p in parsed$params) {
    if (p$name %in% names(overrides)) {
      value_str <- deparse1(overrides[[p$name]])
      lines[p$line_number] <- sub(
        assign_pattern,
        paste0("\\1", value_str),
        lines[p$line_number],
        perl = TRUE
      )
    }
  }

  lines
}

coerce_cli_value <- function(x) {
  if (tolower(x) %in% c("true", "false")) return(as.logical(toupper(x)))
  if (grepl("^-?[0-9]+L$", x)) return(as.integer(sub("L$", "", x)))
  num <- suppressWarnings(as.numeric(x))
  if (!is.na(num)) return(num)
  parsed <- tryCatch(eval(parse(text = x)), error = \(e) NULL)
  if (!is.null(parsed)) return(parsed)
  x
}

parse_captured_output <- function(log_lines, data_keys) {
  escape_regex <- function(s) {
    chars <- strsplit(s, "")[[1]]
    special <- c(".", "\\", "|", "(", ")", "[", "]", "{", "}", "^", "$", "*", "+", "?")
    paste0(ifelse(chars %in% special, paste0("\\", chars), chars), collapse = "")
  }
  result <- lapply(data_keys, \(key) {
    escaped_key <- escape_regex(key)
    pattern <- paste0("^\\s*", escaped_key, "\\s*:\\s*(.+)$")
    matches <- regmatches(log_lines, regexec(pattern, log_lines))
    values <- vapply(matches, \(m) if (length(m) >= 2) m[2] else NA_character_,
                     character(1))
    values <- values[!is.na(values)]
    suppressWarnings(as.numeric(trimws(values)))
  })
  names(result) <- data_keys
  # Drop keys with no matches
  result[vapply(result, \(x) length(x) > 0 && !all(is.na(x)), logical(1))]
}

run_script <- function(script,
                       params = list(),
                       run_dir = getwd(),
                       env = new.env(parent = globalenv()),
                       verbose = TRUE) {

  script_path <- if (grepl("^(/|[A-Za-z]:)", script)) script
                 else file.path(run_dir, script)

  if (!file.exists(script_path)) {
    stop("Script not found: ", script_path)
  }

  lines <- readLines(script_path)
  parsed <- parse_script_flags(lines)

  param_names <- vapply(parsed$params, \(p) p$name, character(1))

  if (verbose) {
    if (length(param_names) > 0) {
      message("Parameters: ", paste(param_names, collapse = ", "))
    }
    if (length(parsed$outputs) > 0) {
      message("Outputs: ", paste(parsed$outputs, collapse = ", "))
    }
    if (length(parsed$data_keys) > 0) {
      message("Data keys: ", paste(parsed$data_keys, collapse = ", "))
    }
  }

  # Warn about unknown overrides
  unknown <- setdiff(names(params), param_names)
  if (length(unknown) > 0) {
    warning("Unknown parameters (not flagged with #| param): ",
            paste(unknown, collapse = ", "))
  }

  if (verbose && length(params) > 0) {
    matched <- intersect(names(params), param_names)
    if (length(matched) > 0) {
      message("Overriding: ",
              paste(matched, "=", vapply(params[matched], deparse1, character(1)),
                    collapse = ", "))
    }
  }

  modified_lines <- inject_params(lines, parsed, params)

  tmp_script <- tempfile(fileext = ".R")
  on.exit(unlink(tmp_script), add = TRUE)
  writeLines(modified_lines, tmp_script)

  old_wd <- getwd()
  on.exit(setwd(old_wd), add = TRUE)
  setwd(run_dir)

  # Set up stdout capture if data keys are declared
  has_data <- length(parsed$data_keys) > 0
  if (has_data) {
    tmp_log <- tempfile(fileext = ".log")
    on.exit(unlink(tmp_log), add = TRUE)
    sink(tmp_log, split = TRUE)
  }

  # Run the script, capturing any error
  script_error <- tryCatch(
    { source(tmp_script, local = env); NULL },
    error = \(e) e
  )

  # Close sink before anything else (must happen even on error)
  if (has_data) {
    sink()
    log_lines <- readLines(tmp_log)
    captured_data <- parse_captured_output(
      log_lines, parsed$data_keys
    )
  }

  if (!is.null(script_error)) {
    message("Script error: ", conditionMessage(script_error))
    message("Returning partial results.")
  }

  # Collect whatever outputs exist so far
  output_list <- lapply(parsed$outputs, \(nm) {
    if (exists(nm, envir = env, inherits = FALSE)) {
      get(nm, envir = env, inherits = FALSE)
    } else {
      NULL
    }
  })
  names(output_list) <- parsed$outputs

  if (has_data && length(captured_data) > 0) {
    output_list$.data <- captured_data
  }

  if (!is.null(script_error)) {
    output_list$.error <- script_error
  }

  output_list
}
