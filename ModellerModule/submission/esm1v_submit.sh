#!/bin/bash

# Set the model name and indices
MODEL="esm1v"
FIRST_INDEX=0  # Adjust as needed
LAST_INDEX=1   # Adjust as needed

# Output files based on model and indices, saved in the existing logfiles directory
OUTFILE="/home/dahala/mnt/ZeroShot/submissions/logfiles/${MODEL}_${FIRST_INDEX}-${LAST_INDEX}_$(date +'%Y%m%d_%H%M%S').out"
ERRFILE="/home/dahala/mnt/ZeroShot/submissions/logfiles/${MODEL}_${FIRST_INDEX}-${LAST_INDEX}_$(date +'%Y%m%d_%H%M%S').err"

# Print start date
date

# Set environment variables
export LANG=C.UTF-8
export first_index=$FIRST_INDEX
export last_index=$LAST_INDEX

# Commands
cd /home/dahala/mnt/ZeroShot/ProteinGym_code/scripts/scoring_DMS_zero_shot/

# Run the script with nohup, redirecting both stdout and stderr to the logfiles
nohup bash scoring_ESM1v_substitutions.sh $first_index $last_index > "$OUTFILE" 2> "$ERRFILE" &

# Capture the process ID
echo "Process running with PID: $!"

# Print end date
date
