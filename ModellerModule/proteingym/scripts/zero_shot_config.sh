# This file has all general filepaths and directories used in the scoring pipeline. The individual scripts may have 
# additional parameters specific to each method 

# DMS zero-shot parameters

# Folders containing the csvs with the variants for each DMS assay
export DMS_data_folder_subs="/home/dahala/mnt/ZeroShot/DMS_data/DMS_ProteinGym_substitutions"
# export DMS_data_folder_indels="Folder containing DMS indel csvs"

# Folders containing multi ple sequence alignments and MSA weights for all DMS assays
export DMS_MSA_data_folder="/home/dahala/mnt/ZeroShot/DMS_msa_files"
export DMS_MSA_weights_folder="/home/dahala/mnt/ZeroShot/DMS_msa_weights"

# Reference files for substitution and indel assays
export DMS_reference_file_path_subs=/home/dahala/mnt/ZeroShot/ProteinGym_code/reference_files/DMS_substitutions_wInhouse.csv
#export DMS_reference_file_path_subs=/home/dahala/mnt/ZeroShot/ProteinGym_code/reference_files/DMS_substitutions_validation.csv
#export DMS_reference_file_path_subs=/home/dahala/mnt/ZeroShot/ProteinGym_code/reference_files/DMS_substitutions_diverse.csv
#export DMS_reference_file_path_subs=/home/dahala/mnt/ZeroShot/ProteinGym_code/reference_files/SakSTAR_reference.csv
export DMS_reference_file_path_indels=/home/dahala/mnt/ZeroShot/ProteinGym_code/reference_files/DMS_indels.csv

# Folders where fitness predictions for baseline models are saved 
export DMS_output_score_folder_subs="/home/dahala/mnt/ZeroShot/results/DMS_scores_subs"
# export DMS_output_score_folder_indels="folder for DMS indel scores"

# Folder containing EVE models for each DMS assay
export DMS_EVE_model_folder="/home/dahala/mnt/ZeroShot/results/DMS_EVE_models"

# Folders containing merged score files for each DMS assay
export DMS_merged_score_folder_subs="/home/dahala/mnt/ZeroShot/results/DMS_merged_scores_subs"
# export DMS_merged_score_folder_indels="folder for merged score for DMS indels"

# Folders containing predicted structures for the DMSs 
export DMS_structure_folder="/home/dahala/mnt/ZeroShot/ProteinGym_AF2_structures"
