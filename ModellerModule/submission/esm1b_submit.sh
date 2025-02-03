#!/bin/bash

# Set the model name and indices
MODEL="esm1b"
FIRST_INDEX=0  # Replace as needed
LAST_INDEX=1   # Replace as needed

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

# Run the script with nohup, redirecting both stdout and stderr to the logfiles
nohup bash scoring_ESM1b_substitutions.sh $first_index $last_index > "$OUTFILE" 2> "$ERRFILE" &

# Capture the process ID
echo "Process running with PID: $!"

# Print end date
date
