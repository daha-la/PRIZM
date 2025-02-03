# This file has all general filepaths and directories used in the scoring pipeline. The individual scripts may have 
# additional parameters specific to each method 

PRIZM_PATH="INSERT_PATH_TO_PRIZM"
export PRIZM_PATH
# DMS zero-shot parameters

# Folder containing the dataset csvs
export DMS_data_folder_subs="$PRIZM_PATH/data/lowN"
#export DMS_data_folder_subs="$PRIZM_PATH/data/validation"

# Folders containing MSA and MSA weights
export DMS_MSA_data_folder="$PRIZM_PATH/data/protein_information/msa/files"
export DMS_MSA_weights_folder="$PRIZM_PATH/data/protein_information/msa/weights"

# Path to the reference file
export DMS_reference_file_path_subs="$PRIZM_PATH/ModellerModule/reference_files/FlA_reference.csv"
#export DMS_reference_file_path_subs=/home/dahala/mnt/ZeroShot/ProteinGym_code/reference_files/DMS_substitutions.csv
#export DMS_reference_file_path_subs=/home/dahala/mnt/ZeroShot/ProteinGym_code/reference_files/DMS_substitutions_validation.csv

# Folders where zero-shot scores are saved 
export DMS_output_score_folder_subs="$PRIZM_PATH/results"

# Folder containing EVE models
export DMS_EVE_model_folder="$PRIZM_PATH/finetuned_models/EVE"

# Folder containing protein structures
export DMS_structure_folder="$PRIZM_PATH/data/protein_information/structure"
