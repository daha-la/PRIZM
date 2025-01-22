#!/bin/bash

# Set the model name and indices
MODEL="rita"
FIRST_INDEX=59  # Adjust as needed
LAST_INDEX=59   # Adjust as needed

# Output files based on model and indices, saved in the existing logfiles directory
OUTFILE="/home/dahala/mnt/ZeroShot/submissions/logfiles/${MODEL}_${FIRST_INDEX}-${LAST_INDEX}_$(date +'%Y%m%d_%H%M%S').out"
ERRFILE="/home/dahala/mnt/ZeroShot/submissions/logfiles/${MODEL}_${FIRST_INDEX}-${LAST_INDEX}_$(date +'%Y%m%d_%H%M%S').err"

# Print start date
date

# Set environment variables
export LANG=C.UTF-8
export first_index=$FIRST_INDEX
export last_index=$LAST_INDEX

# Commands to execute
cd /home/dahala/mnt/ZeroShot/ProteinGym_code/scripts/scoring_DMS_zero_shot/

# Run the scoring script with nohup, redirecting both stdout and stderr
nohup bash scoring_RITA_substitutions.sh $first_index $last_index > "$OUTFILE" 2> "$ERRFILE" &

# Capture the process ID
echo "Process running with PID: $!"

# Print end date
date

