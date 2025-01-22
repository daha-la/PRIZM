#!/bin/bash

# Set the model name and indices
MODEL="unirep"
FIRST_INDEX=0  # Replace as needed
LAST_INDEX=1   # Replace as needed

# Logfiles directory
LOG_DIR="/home/dahala/mnt/ZeroShot/submissions/logfiles"

# Output files based on model and indices, saved in the logfiles directory
OUTFILE="${LOG_DIR}/${MODEL}_${FIRST_INDEX}-${LAST_INDEX}_$(date +'%Y%m%d_%H%M%S').out"
ERRFILE="${LOG_DIR}/${MODEL}_${FIRST_INDEX}-${LAST_INDEX}_$(date +'%Y%m%d_%H%M%S').err"

# Print start date
date

# Set environment variables
export LANG=C.UTF-8
export first_index=$FIRST_INDEX
export last_index=$LAST_INDEX

# Commands
cd /home/dahala/mnt/ZeroShot/ProteinGym_code/scripts/scoring_DMS_zero_shot/

# Run the script with nohup, redirecting both stdout and stderr to the logfiles
nohup bash evotune_UniRep_substitutions.sh $first_index $last_index > "$OUTFILE" 2> "$ERRFILE" &

# Capture the process ID
echo "Process running with PID: $!"

# Print end date
date
