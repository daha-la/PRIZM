source ../zero_shot_config.sh
#source activate prosst

# All models can be found at https://huggingface.co/AI4Protein
# ProSST models: ProSST-20 ProSST-128 ProSST-512 ProSST-1024 ProSST-2048 ProSST-4096 ProSST-3di
# export model_name=AI4Protein/ProSST-20 AI4Protein/ProSST-128 AI4Protein/ProSST-512 AI4Protein/ProSST-1024 AI4Protein/ProSST-2048 AI4Protein/ProSST-4096 AI4Protein/ProSST-3di

# the structure pdb files can be found in ProtSSN: https://github.com/tyang816/ProtSSN
# please download and unzip the following files to a folder: https://drive.google.com/file/d/1lSckfPlx7FhzK1FX7EtmmXUOrdiMRerY/view?usp=sharing
export DMS_folder="$PRIZM_PATH/data/special_folders/ProSST"
#export DMS_residue_folder="${DMS_folder}/residue_sequence"
export DMS_structure_seq_folder="$PRIZM_PATH/data/special_folders/VenusREM/struc_seq"
#export DMS_data_folder_subs="${DMS_folder}/substitutions"
#  3di
cd ../../

for ((i=$1; i<=$2; i++))
do
    echo "Evaluating DMS index $i"
    export DMS_index=$i
    for model_hyp in 20 128 512 1024 2048 4096; do
        
        start_time=$(date +%s.%N)
        
        export model_name="AI4Protein/ProSST-$model_hyp"

        echo "Using Model: $model_name"

        export DMS_output_folder="${DMS_output_score_folder_subs}/ProSST/$model_hyp"
        
        python baselines/prosst/compute_fitness.py \
        --model_name ${model_name} \
        --base_dir ${DMS_folder} \
        --mutant_dir $DMS_data_folder_subs \
        --struc_seq_dir $DMS_structure_seq_folder \
        --pdb_dir $DMS_structure_folder \
        --DMS_reference_file_path $DMS_reference_file_path_subs \
        --DMS_index $DMS_index \
        --output_scores_folder ${DMS_output_folder}
        

        end_time=$(date +%s.%N)
        elapsed_time=$(awk "BEGIN {print $end_time - $start_time}")
        echo "Time taken for $model_hyp: $elapsed_time seconds"

    done
done