import numpy as np
import pandas as pd
import os

def hit_rate(true_bin: np.ndarray, predicted_labels: np.ndarray, k: int = 10) -> float:
    """
    Calculate the hit rate based on the top k predicted labels.
    
    Args:
        true_bin (np.ndarray): Binary labels indicating hits (1 for hit, 0 for miss).
        predicted_labels (np.ndarray): Model-predicted scores used to rank the top k entries.
        k (int): The number of top entries to consider.
        
    Returns:
        float: The hit rate as the ratio of hits in the top k entries.
    """

    # Get the top k positions in the predicted_labels array
    top_k_positions = np.argsort(predicted_labels)[-k:]
    
    # Select elements in true_bin based on positions, not index labels
    hits_in_top_k = true_bin.take(top_k_positions)
    
    # Calculate and return hit rate
    return hits_in_top_k.sum() / k

def extract_mutant_info(mutant_str: str) -> pd.Series:
    """
    Function for extracting the wild-type amino acid, mutated amino acid, and position from a mutant string.

    Args:
        mutant_str (str): A string representing a mutant in the format "WTposMut".

    Returns:
        pd.Series: A pandas Series with the wild-type amino acid, mutated amino acid, and position.
    """

    WT, Mut, Pos, WTnPos = [], [], [], []
    mutations = mutant_str.split(':')  # Split the string by colon
    for m in mutations:
        WT.append(m[0])               # Wild-type amino acid
        Pos.append(int(m[1:-1]))      # Position
        Mut.append(m[-1])             # Mutated amino acid
    return pd.Series([WT, Mut, Pos])

def varnumb(n: int,k: int) -> int:
    """
    Function for calculating the number of combinations of k elements from a set of n elements.

    Args:
        n (int): The total number of elements in the set.
        k (int): The number of elements to select.
    """

    combi = np.math.factorial(n)/(np.math.factorial(k)*np.math.factorial(n-k))
    numb=combi*(20**k-1)
    return numb

def reference_builder(numb_prot: list[str], protein_name: list[str], wt_sequence: list[str], DMS_binarization_cutoff: list[float], MSA_name: list[str], MSA_num_seqs: list[float], pdb_file: list[str],
                      reference_name: list[str], custom_identifier: list[str]):
    """
    This function builds a reference file that contains all relevant information about the proteins of interest. This reference file
    is used to pass information to the other functions in the pipeline.

    Args:
        numb_prot (list[str]): The number of the proteins
        protein_name (list[str]): The names of the proteins of interest
        wt_sequence (list[str]): The wild-type sequences of the proteins of interest
        DMS_binarization_cutoff (list[float]): The cutoffs for binarizing DMS data, often just WT experimental value
        MSA_name (list[str]): The names of the MSA files without the file extension
        MSA_num_seqs (list[float]): The numbers of sequences in the MSAs
        pdb_file (list[str]): The names of the pdb files
        custom_identifier (list[str]): Custom identifiers for the proteins of interest
        reference_name (str): The name of the reference file.
    
    Returns:
        reference_df (pd.DataFrame): The reference file in a pandas DataFrame format
    """

    # Initialize dictionary to store reference information
    reference = {}

    # Iterate over proteins to build reference dictionary
    for i in range(numb_prot):

        # If custom identifiers are provided, use them, otherwise only use protein names
        if custom_identifier[i] is not None:
            DMS_id = protein_name[i] + "_" + custom_identifier[i]
        else:
            DMS_id = protein_name[i]

        # Initializing and save weights for MSA if they do not already exist. Will be changed in future pipelines: 
        msa_ = MSA_name[i]
        if f"{msa_}_weights.npy" not in os.listdir("../data/protein_information/msa/weights/"):
            weights = np.ones(len(wt_sequence[i]))
            np.save(f"../data/protein_information/msa/weights/{msa_}_weights.npy", weights)
        
        # Build reference dictionary
        reference[i] = {
            'DMS_id': DMS_id,
            'DMS_filename': f'{DMS_id}.csv',
            'target_seq': wt_sequence[i],
            'seq_len': len(wt_sequence[i]),
            'DMS_binarization_cutoff': DMS_binarization_cutoff[i],
            'MSA_filename': MSA_name[i]+'.a2m',
            'MSA_start': 1,
            'MSA_end': len(wt_sequence[i]),
            'MSA_len': len(wt_sequence[i]),
            'MSA_num_seqs': MSA_num_seqs[i],
            'weight_file_name': f"{msa_}_weights.npy",
            'pdb_file': pdb_file[i]
        }

    reference_df = pd.DataFrame.from_dict(reference, orient='index')
    reference_df.to_csv(f"../ModellerModule/reference_files/"+reference_name, index=False)
    return reference_df
