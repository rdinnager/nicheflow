#!/bin/bash
#SBATCH --job-name=nicheflow_targets_get_samples_1
#SBATCH --mail-user=rdinnage@fiu.edu
#SBATCH --mail-type=FAIL,END
#SBATCH --account=rdinnage.fiu
#SBATCH --qos=rdinnage.fiu
##SBATCH --partition=gpu
##SBATCH --gres=gpu:a100:1

# Where to put the outputs: %j expands into the job number (a unique identifier for this job)
#SBATCH --output logs/job-%x-%j.out
#SBATCH --error logs/job-%x-%j.err

# Number of nodes to use
#SBATCH --nodes=1

# Number of tasks (usually translate to processor cores) to use: important! this means the number of mpi ranks used, useless if you are not using Rmpi)
#SBATCH --ntasks=1

#number of cores to parallelize with:
#SBATCH --cpus-per-task=9
##SBATCH --mem=64G
# Memory per cpu core. Default is megabytes, but units can be specified with M
# or G for megabytes or Gigabytes.
#SBATCH --mem-per-cpu=16G

# Job run time in [DAYS]
# HOURS:MINUTES:SECONDS
# [DAYS] are optional, use when it is convenient
#SBATCH --time=72:00:00

## activate conda
export PATH=/blue/rdinnage.fiu/rdinnage.fiu/.conda/envs/rstudio-gpu/bin:$PATH

# Save some useful information to the "output" file
date;hostname;pwd

# Load R and run a command
#Rscript R/squamate_train_nichePFN.R
Rscript -e "targets::tar_make(c(bias_samples, ground_truth_samples))"
