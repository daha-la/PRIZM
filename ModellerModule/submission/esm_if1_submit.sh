#!/bin/bash

# Set the model name and indices
MODEL="esm_if1"
FIRST_INDEX=173  # Adjust as needed
LAST_INDEX=173   # Adjust as needed

# Output files based on model and indices, saved in the existing logfiles directory
OUTFILE="logfiles/${MODEL}_${FIRST_INDEX}-${LAST_INDEX}_$(date +'%Y%m%d_%H%M%S').out"
ERRFILE="logfiles/${MODEL}_${FIRST_INDEX}-${LAST_INDEX}_$(date +'%Y%m%d_%H%M%S').err"

# Print start date
date

# Set environment variables
export first_index=$FIRST_INDEX
export last_index=$LAST_INDEX

# Change to the scoring script directory
cd ../proteingym/scripts/scoring_DMS_zero_shot/

# Run the scoring script with nohup, redirecting both stdout and stderr
nohup bash scoring_ESM_IF1_substitutions.sh $first_index $last_index > "$OUTFILE" 2> "$ERRFILE" &

# Capture the process ID
echo "Process running with PID: $!"

# Print end date
date

