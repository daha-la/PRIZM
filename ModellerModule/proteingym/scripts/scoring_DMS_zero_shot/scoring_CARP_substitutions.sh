source ../zero_shot_config.sh

cd ../../

export model_path="$checkpoint_folder/CARP"

for ((i=$1; i<=$2; i++))
do
    echo "Evaluating DMS index $i"
    export DMS_index=$i

    start_time=$(date +%s.%N)

    export DMS_output_folder=${DMS_output_score_folder_subs}/CARP/600K

    export model_name="carp_600k" #[carp_600k|carp_38M|carp_76M|carp_640M]
    export performance_file='CARP_600K_performance.csv'

    python baselines/carp_mif/compute_fitness.py \
                --model_name ${model_name} \
                --model_path ${model_path} \
                --DMS_reference_file_path ${DMS_reference_file_path_subs} \
                --DMS_data_folder ${DMS_data_folder_subs} \
                --DMS_index $DMS_index \
                --output_scores_folder ${DMS_output_folder} \
                --performance_file ${performance_file} 

    end_time=$(date +%s.%N)
    elapsed_time=$(awk "BEGIN {print $end_time - $start_time}")
    echo "Time taken for $i with 600K: $elapsed_time seconds"

    start_time=$(date +%s.%N)

    export DMS_output_folder=${DMS_output_score_folder_subs}/CARP/38M

    export model_name="carp_38M" #[carp_600k|carp_38M|carp_76M|carp_640M]
    export performance_file='CARP_38M_performance.csv'

    python baselines/carp_mif/compute_fitness.py \
                --model_name ${model_name} \
                --model_path ${model_path} \
                --DMS_reference_file_path ${DMS_reference_file_path_subs} \
                --DMS_data_folder ${DMS_data_folder_subs} \
                --DMS_index $DMS_index \
                --output_scores_folder ${DMS_output_folder} \
                --performance_file ${performance_file}

    end_time=$(date +%s.%N)
    elapsed_time=$(awk "BEGIN {print $end_time - $start_time}")
    echo "Time taken for $i with 38M: $elapsed_time seconds"

    start_time=$(date +%s.%N)

    export DMS_output_folder=${DMS_output_score_folder_subs}/CARP/76M

    export model_name="carp_76M" #[carp_600k|carp_38M|carp_76M|carp_640M]
    export performance_file='CARP_76M_performance.csv'

    python baselines/carp_mif/compute_fitness.py \
                --model_name ${model_name} \
                --model_path ${model_path} \
                --DMS_reference_file_path ${DMS_reference_file_path_subs} \
                --DMS_data_folder ${DMS_data_folder_subs} \
                --DMS_index $DMS_index \
                --output_scores_folder ${DMS_output_folder} \
                --performance_file ${performance_file}

    end_time=$(date +%s.%N)
    elapsed_time=$(awk "BEGIN {print $end_time - $start_time}")
    echo "Time taken for $i with 76M: $elapsed_time seconds"

    start_time=$(date +%s.%N)

    export DMS_output_folder=${DMS_output_score_folder_subs}/CARP/640M

    export model_name="carp_640M" #[carp_600k|carp_38M|carp_76M|carp_640M]
    export performance_file='CARP_640M_performance.csv'

    python baselines/carp_mif/compute_fitness.py \
                --model_name ${model_name} \
                --model_path ${model_path} \
                --DMS_reference_file_path ${DMS_reference_file_path_subs} \
                --DMS_data_folder ${DMS_data_folder_subs} \
                --DMS_index $DMS_index \
                --output_scores_folder ${DMS_output_folder} \
                --performance_file ${performance_file}

    end_time=$(date +%s.%N)
    elapsed_time=$(awk "BEGIN {print $end_time - $start_time}")

    echo "Time taken for $i with 640M: $elapsed_time seconds"

done