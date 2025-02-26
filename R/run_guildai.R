run_model_guildai <- function(script, label, tag, flags, comment = "",
                              GUILD_HOME = "/blue/rdinnage.fiu/rdinnage.fiu/Projects/nicheflow/.guild") {
  assignInNamespace("find_guild", guildai:::find_r_guildai_guild, ns = "guildai")
  Sys.setenv(GUILD_HOME = GUILD_HOME)
  
  guild_run(script, label = label, tag = tag, as_job = FALSE,
            run_dir = file.path(".guild", "runs", label),
            flags = flags, comment = comment#, test_sourcecode = TRUE
  )
  
  run_info <- runs_info(label = label)
  #model_file <- file.path(run_info$dir, "output", "trained_ivae.to")
  
  return(run_info)
  
}