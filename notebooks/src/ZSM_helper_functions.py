import numpy as np
import pandas as pd

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
