# This file has all general filepaths and directories used in the scoring pipeline. The individual scripts may have 
# additional parameters specific to each method 

# Please adapt the following variables to fit your folder structure.
export PRIZM_PATH="/path/to/PRIZM" # Path to the PRIZM directory, please adapt this to reflect your folder structure.
export checkpoint_folder="$PRIZM_PATH/ModellerModule/checkpoints" # Checkpoint folder, adapt this to reflect your folder structure. We recommen using "$PRIZM_PATH/ModellerModule/checkpoints"/data/checkpoints
export installation_folder="$PRIZM_PATH/ModellerModule/installation" # Installation folder, adapt this to reflect your folder structure. We recommend using ""/data/installation

# Please adapt the following variables to fit your objectives
export data_location="validation" # Data location, can be "lowN", "validation" or "insilico_libraries"
export reference_file="DMS_substitutions_ProteinGym.csv" # Name of reference file. For validation analysis, please use "DMS_substitutions_ProteinGym.csv"

# Please do not change the following variables unless you want to adapt the folder structure for input/output files

# Folder containing the dataset csvs.
export DMS_data_folder_subs="$PRIZM_PATH/data/$data_location"

# Folders containing MSA and MSA weights
export DMS_MSA_data_folder="$PRIZM_PATH/data/protein_information/msa/files"
export DMS_MSA_weights_folder="$PRIZM_PATH/data/protein_information/msa/weights"

# Path to the reference file.
export DMS_reference_file_path_subs="$PRIZM_PATH/ModellerModule/reference_files/$reference_file"

# Folders where zero-shot scores are saved 
export DMS_output_score_folder_subs="$PRIZM_PATH/results"

# Folder containing EVE models
export DMS_EVE_model_folder="$PRIZM_PATH/finetuned_models/EVE"

# Folder containing protein structures
export DMS_structure_folder="$PRIZM_PATH/data/protein_information/structure"
