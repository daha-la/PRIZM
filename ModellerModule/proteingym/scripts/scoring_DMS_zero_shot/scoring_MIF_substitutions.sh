source ../zero_shot_config.sh


export model_name="mif"
export model_path="$checkpoint_folder/MIF/mif.pt"
export DMS_output_score_folder=${DMS_output_score_folder_subs}/MIF
export performance_file='MIF_performance.csv'

cd ../../

for ((i=$1; i<=$2; i++))
do
    echo "Evaluating DMS index $i"
    export DMS_index=$i

    start_time=$(date +%s.%N)

    python baselines/carp_mif/compute_fitness.py \
                --model_name ${model_name} \
                --model_path ${model_path} \
                --DMS_reference_file_path ${DMS_reference_file_path_subs} \
                --DMS_data_folder ${DMS_data_folder_subs} \
                --DMS_index $DMS_index \
                --output_scores_folder ${DMS_output_score_folder} \
                --performance_file ${performance_file} \
                --structure_data_folder ${DMS_structure_folder}

    end_time=$(date +%s.%N)
    elapsed_time=$(awk "BEGIN {print $end_time - $start_time}")
    echo "Time taken for $i with MIF: $elapsed_time seconds"

done