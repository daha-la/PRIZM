import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpl_patches
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from sklearn.utils import resample
from typing import Callable, Dict, List, Tuple
import yaml
import seaborn as sns
import itertools
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score, average_precision_score, ndcg_score

# Import the necessary functions from PRIZM_helper_functions.py
from .PRIZM_helper_functions import extract_mutant_info, varnumb, hit_rate

# Define the amino acids, their order, and a dictionary to map amino acids to indices
amino_acids = ['A','R','N','D','C','E','Q','G','H','I','L','K','M','F','P','S','T','W','Y','V']
aa_order = ['G', 'P', 'A', 'V', 'L', 'I', 'M', 'F', 'Y', 'W', 'K', 'R', 'D', 'E', 'C', 'S', 'T', 'N', 'Q', 'H']
aa_to_index = {aa: idx for idx, aa in enumerate(aa_order)}

# Define the metrics dictionary
metric_dict = {
    "pearsonr": pearsonr,
    "spearmanr": spearmanr,
    "auc_roc": roc_auc_score,
    "ndcg": ndcg_score,
    "hit_rate": hit_rate,
    'average_precision': average_precision_score
}

class ZeroShotModellerModule:
    """
    A class for evaluating zero-shot models based on experimental data.
    Can be used to load data, calculate correlation metrics, and determine the best and worst models based on these metrics.
    Also includes functions for Single Mutant Landscapes, mutant sorting, and in silico saturation mutagenesis.
    """

    def __init__(self, config_path: str):
        """
        Initialize the ZeroShotModellerModule class.

        Args:
            config_path (str): The path to the YAML model configuration file.
        """

        self.config = self._load_config(config_path)  # Load YAML config file
        self.metric_dict = metric_dict  # Dictionary of metric functions

        # Initialize the dictionaries to store the data
        self.name_list = {} # Dictionary of model names and their variants
        self.model_list = {}  # Dictionary of models to be evaluated
        self.datasets = {}  # Dictionary to hold dataset specific scores
        self.wildtype = {}  # Dictionary to hold wildtype scores
        self.metric_results = {}  # Dictionary for storing metric results
        self.threshold = {}  # Dictionary for storing thresholds
        self.best_variants = {}  # Dictionary for storing best variants
        self.best_models = {}  # Dictionary for storing best models
        self.worst_models = {}  # Dictionary for storing worst models
        self.landscape = {}  # Dictionary for storing landscapes


    def _load_config(self, file_path: str) -> Dict:
        """
        Load the YAML model configuration file.

        Args:
            file_path (str): The path to the YAML model configuration file.

        Returns:
            Dict: A dictionary containing the loaded model configuration.
        """

        with open(file_path, 'r') as file:
            return yaml.safe_load(file)

        
    def get_name_list(self, model_list: list) -> List[str]:
        """
        Get a list of model names and their variants based on the model list.

        Args:
            model_list (list): A list of model names to consider.

        Returns:
            List[str]: A list of model names and their variants.
        """

        name_list = []

         # Iterate over all models in the config
        for model in self.config['models']:
            
            # Check if the model is in the list
            if model['name'] in model_list:

                # Check if the model has variants
                if 'variants' in model:
                    for variant in model['variants']:
                        name_list.append(f"{model['name']}_{variant['name']}")
                
                # Add single-variant models directly
                else:
                    name_list.append(model['name'])
        return name_list

    def set_threshold(self, dataset_name: str, threshold: float) -> None:
        """
        Set a threshold for a specific dataset. The threshold is used to binarize the experimental data.
        
        Args:
            dataset_name (str): The name of the dataset.
            threshold (float): The threshold value to set.
        """
        self.threshold[dataset_name] = threshold

    def load_specific_models(self, model_names: List[str] = None, categories: List[str] = None) -> List:
        """
        Load only specific models from the config file based on model names or categories.

        Args:
            model_names (List[str]): A list of model names to load. If None, all models are considered.
            categories (List[str]): A list of categories to filter by. If None, all categories are considered.

        Returns:
            List[Dict]: A list of models (and their variants) that match the specified criteria.
        """
        filtered_models = []

        # Iterate over all models in the config
        for model in self.config['models']:

            # Check if model matches specified categories
            if (categories is None or model['category'] in categories):
                
                # Check if the model has variants
                if 'variants' in model:
                    for variant in model['variants']:
                        variant_full_name = f"{model['name']}_{variant['name']}"

                        # Check if the variant matches the specified model names
                        if (model_names is None or variant_full_name in model_names):
                            filtered_models.append(variant_full_name)
                
                # Add single-variant models directly
                else:

                    # Check if the model matches the specified model names
                    if (model_names is None or model['name'] in model_names):
                        filtered_models.append(model['name'])

        return filtered_models

    def dataload(self, path: str, dataset_name: str) -> pd.DataFrame:
        """
        Load a dataset from a specified path and return a DataFrame.

        Args:
            path (str): The path to the dataset.
            dataset_name (str): The name of the dataset.

        Returns:
            pd.DataFrame: A DataFrame containing the loaded dataset.
        """

        # Check if the dataset is EVE (and not TranceptEVE), as the first row is the WT
        if 'EVE' in path and not 'Trancept' in path:
            start = 1
        else:
            start = 0

        # Load the dataset and return it as a DataFrame
        df_path = f'{path}/{dataset_name}.csv'
        df = pd.read_csv(df_path)
        df = df.iloc[start:,:].reset_index(drop=True)
        return df

    def load_scores(self, dataset_name: str, model_list: list, results_path: str = '../results/', wt = False, dict_name: str = None) -> None:
        """
        Load scores for a specific dataset and models and save them in the class.

        Args:
            dataset_name (str): The name of the dataset to process.
            model_list (list): A list of model names to consider.
            results (str): The path to the results folder. The default is '../results/'
            wt (bool): Whether to include the wildtype score. Default is False
            dict_name (str): The name of the dictionary to store the scores in. If None, the dataset name is used. Default is None
        """

        # Check if a dictionary name is provided. If not, use the dataset name
        if dict_name is None:
            dict_name = dataset_name

        # Initialize the dictionaries to store the data
        self.model_list[dict_name] = model_list
        self.name_list[dict_name] = self.get_name_list(model_list)
        self.datasets[dict_name] = {}
        self.wildtype[dict_name] = {}

        # Load the experimental data and set the threshold
        if dict_name not in self.threshold:
            raise ValueError("Threshold not set for dataset.")
        thresh = self.threshold[dict_name]
        exp_data_ = self.dataload(results_path+'ESM1b', dataset_name)[['mutant','mutated_sequence','DMS_score']] # ESM1b is used for experimental data, maybe change this
        exp_data_['DMS_score_bin'] = (exp_data_['DMS_score'] >= thresh).astype(int)
        exp_data_[['WT', 'Mut', 'Pos']] = exp_data_['mutant'].apply(extract_mutant_info)

        # Store the experimental data in the class
        self.datasets[dict_name]['exp_data'] = exp_data_[:-1] if wt else exp_data_
        
        # If wildtype is included, store the wildtype data in the class
        if wt:
            self.wildtype[dict_name]['exp_data'] = exp_data_.iloc[-1,:]


        # Load the scores for each model and store them in the class
        # First, iterate over all models in the config
        for model in self.config['models']:

            # Check if the model is in the list
            if model['name'] in self.model_list[dict_name]:

                # Check if the model has variants
                if 'variants' in model:
                    for variant in model['variants']:
                        model_path = variant['path']
                        model_name = f"{model['name']}_{variant['name']}" # Full model name with variant

                        # Load the model data
                        data_ = self.dataload(results_path+model_path, dataset_name)

                        # If the model outputs a "mutant" column, use it to match the experimental data
                        if 'mutant' in data_.columns:
                            data_ = data_[['mutant',variant['score_column']]]
                            data_unique = data_.drop_duplicates('mutant').set_index('mutant')
                            data_ = data_unique.loc[exp_data_.mutant.to_list()].reset_index()

                        # Else, if the model outputs a "mutated_sequence" column, use it to match the experimental data
                        elif 'mutated_sequence' in self.dataload(results_path+model_path, dataset_name).columns:
                            data_ = data_[['mutated_sequence',variant['score_column']]]
                            data_unique = data_.drop_duplicates('mutated_sequence').set_index('mutated_sequence')
                            data_ = data_unique.loc[exp_data_.mutated_sequence.to_list()].reset_index()

                        # If neither "mutant" nor "mutated_sequence" columns are found, raise an error
                        else:
                            raise ValueError("No mutant or mutated_sequence column found in the data.")
                        
                        # Rename the score column to the model name
                        data_ = data_.rename(columns={variant['score_column']: model_name})

                        # If wildtype is included, store the wildtype score in the class and remove it from the dataset
                        if wt:
                            self.wildtype[dict_name][model_name] = data_.iloc[-1,:]
                            data_ = data_.iloc[:-1,:]

                        # Store the model data in the class
                        self.datasets[dict_name][model_name] = data_

                # Add single-variant models directly
                else:
                    model_path = model['path']
                    model_name = model['name']

                    # Load the model data
                    data_ = self.dataload(results_path+model_path, dataset_name)

                    # If the model outputs a "mutant" column, use it to match the experimental data
                    if 'mutant' in self.dataload(results_path+model_path, dataset_name).columns:
                        data_ = data_[['mutant',model['score_column']]]
                        data_unique = data_.drop_duplicates('mutant').set_index('mutant')
                        data_ = data_unique.loc[exp_data_.mutant.to_list()].reset_index()

                    # Else, if the model outputs a "mutated_sequence" column, use it to match the experimental data
                    elif 'mutated_sequence' in self.dataload(results_path+model_path, dataset_name).columns:
                        data_ = self.dataload(results_path+model_path, dataset_name)[['mutated_sequence',model['score_column']]]
                        data_unique = data_.drop_duplicates('mutated_sequence').set_index('mutated_sequence')
                        data_ = data_unique.loc[exp_data_.mutated_sequence.to_list()].reset_index()

                    # If neither "mutant" nor "mutated_sequence" columns are found, raise an error
                    else:
                        raise ValueError("No mutant or mutated_sequence column found in the data.")
                    
                    # Rename the score column to the model name
                    data_ = data_.rename(columns={model['score_column']: model['name']})

                    # If wildtype is included, store the wildtype score in the class and remove it from the dataset
                    if wt:
                        self.wildtype[dict_name][model_name] = data_.iloc[-1,:]
                        data_ = data_.iloc[:-1,:]

                    # Store the model data in the class
                    self.datasets[dict_name][model_name] = data_
        
    def metric_val_func(self, metric_name: str, true_labels: np.ndarray, true_bin: np.ndarray, predicted_labels: np.ndarray, k: int = 10) -> float:
        """
        Calculate a specific metric value based on the true and predicted labels.

        Args:
            metric_name (str): The name of the metric to calculate.
            true_labels (np.ndarray): The true labels.
            true_bin (np.ndarray): The true binary labels.
            predicted_labels (np.ndarray): The predicted labels.
            k (int): The number of top entries to consider for hit rate. Default is 10.

        Returns:
            float: The calculated metric value.
        """

        # Get the metric function from the metric dictionary
        metric_fn = self.metric_dict.get(metric_name)

        # Define actions for different metrics
        metric_actions = {
            "pearsonr": lambda: metric_fn(true_labels, predicted_labels)[0],
            "spearmanr": lambda: metric_fn(true_labels, predicted_labels)[0],
            "ndcg": lambda k: metric_fn([(true_labels - true_labels.min()) / (true_labels.max() - true_labels.min())], [predicted_labels], k=k), # Normalize true_labels to [0, 1]
            "auc_roc": lambda: metric_fn(true_bin, predicted_labels) if not self.check_class(true_bin)[0] else np.nan, # Skip AUC-ROC if all elements are the same
            "hit_rate": lambda: metric_fn(true_bin, predicted_labels, k=k),
            "average_precision": lambda: metric_fn(true_bin, predicted_labels) if not self.check_class(true_bin)[0] else np.nan # Skip average precision if all elements are the same
        }

        # Return the calculated metric value if the metric is in the dictionary, else return NaN
        return metric_actions[metric_name]() if metric_name in metric_actions else np.nan

    def check_class(self, y_true: np.ndarray) -> Tuple[bool, int]:
        """
        Check if all elements in an array are identical and return the class.

        Args:
            y_true (np.ndarray): The array to check.

        Returns:
            Tuple[bool, int]: A tuple with a boolean indicating if all elements are identical and the class.
        """

        y_true = np.array(y_true)
        y_true_identical = np.all(y_true == y_true[0])

        # Identify the class
        y_true_class = y_true[0]

        return y_true_identical, y_true_class

    def bootstrap_score(self, true: np.ndarray, true_bin: np.ndarray, pred: np.ndarray, metric_type: str,
                        n_bootstrap: int = 1000, pre_reverse = None) -> Tuple[float, float, bool]:
        """
        A function to bootstrap the metric scores for a specific metric type. The function will return the overall value and standard deviation of the metric scores,
        as well as a boolean indicating whether the scores should be reversed.
        
        Args:
            true (np.ndarray): The true labels.
            true_bin (np.ndarray): The true binary labels.
            pred (np.ndarray): The predicted labels.
            metric_type (str): The metric type to calculate.
            n_bootstrap (int): The number of bootstrap iterations. Default is 1000.
            pre_reverse (bool): A boolean indicating whether the scores should be reversed. Default is None.

        Returns:
            Tuple[float, float, bool]: A tuple containing the overall metric score, the standard deviation of the metric scores, and a boolean indicating whether the scores should be reversed.
        """

        # If pre_reverse is provided, calculate the original score using the pre-defined order
        if pre_reverse is not None:
            reverse = pre_reverse
            original_score = self.metric_val_func(metric_type, true,true_bin, pred)

            # If reverse is True, calculate the score with the negative predictions
            if reverse:
                original_score = self.metric_val_func(metric_type, true,true_bin, -pred)

        # Else, calculate the original score and the score with the negative predictions and determine the order based on the best score
        else:
            score_with_pred = self.metric_val_func(metric_type, true,true_bin, pred)
            score_with_neg_pred = self.metric_val_func(metric_type, true,true_bin, -pred)
            
            # If the score with the predicted labels is better, set the original score to the score with the predicted labels and set reverse to False
            if score_with_pred >= score_with_neg_pred:
                original_score = score_with_pred
                reverse = False

            # If the score with the negative predicted labels is better, set the original score to the score with the negative predicted labels and set reverse to True
            else:
                original_score = score_with_neg_pred
                reverse = True
    
        # Initialize an array to store the scores
        scores = np.zeros(n_bootstrap)

        # Bootstrap the scores
        np.random.seed(1) # Set seed for reproducibility

        # Iterate over the number of bootstrap iterations
        for i in range(n_bootstrap):
            sample_index = resample(np.arange(len(pred)), stratify=true_bin) # Stratification is used to maintain the class distribution
            
            # Get the new index based on the sample index, and extract the sampled true, true_bin, and predicted labels
            new_index = pred.index[sample_index]
            pred_bootstrap, true_bootstrap, true_bin_bootstrap = pred[new_index], true[new_index], true_bin[new_index]

            # Calculate the metric score and store it in the scores array
            scores[i] = self.metric_val_func(metric_type, true_bootstrap, true_bin_bootstrap, pred_bootstrap if not reverse else -pred_bootstrap)

        return original_score, scores.std(), reverse

    def compute_metrics(self, dataset_name: str, n_bootstrap: int = 1000, metrics: list = ['spearmanr', 'average_precision'],
                        previous_metrics: pd.DataFrame = None, best_metrics: list = ['spearmanr', 'average_precision'], best_met_extremas: list = [[0,0],[1,1]]) -> None:
        """
        A function to compute metrics for a specific dataset and save results in a structured DataFrame within the class.
        Also determines the best variants for each model based on the normalized metrics.

        Args:
            dataset_name (str): The name of the dataset to process.
            n_bootstrap (int): The number of bootstrap iterations. Default is 1000.
            metrics (list): The metrics to calculate. Default is ['spearmanr', 'average_precision'].
            previous_metrics (pd.DataFrame): A DataFrame containing the previous metrics. Used to extract pre-determined order. Default is None.
            best_metrics (list): The best metrics to consider when selecting the best variants of a model. Default is ['spearmanr', 'average_precision'].
            best_met_extremas (list): The extremas for the best metrics. First list is the minimums, second list is the maximums. Default is [[0,0],[1,1]].
        """
        
        # Retrieve experimental data and score list for the specified dataset
        exp_score = self.datasets[dataset_name]['exp_data'].DMS_score  # Experimental data
        exp_bin = self.datasets[dataset_name]['exp_data'].DMS_score_bin  # Binary data
        score_list = self.datasets[dataset_name] # Predicted scores for each model

        
        # Define MultiIndex for columns
        index_tuples = [(met_name, stat) for met_name in metrics for stat in ['avg', 'std', 'reverse']]
        multi_index = pd.MultiIndex.from_tuples(index_tuples, names=['metric', 'stat'])
        
        # Initialize the DataFrame for storing results
        metric_results_df = pd.DataFrame(index=self.name_list[dataset_name], columns=multi_index)

        # Iterate over all metrics
        for met_name in metrics:

            # Initialize lists to store the metric scores
            corr_list_ = []
            std_list_ = []
            reverse_list_ = []

            # Iterate over all models initialized for the dataset
            for name in self.name_list[dataset_name]:

                # Check if previous metrics are provided and extract the pre_reverse order. Else, set it to None
                pre_reverse = previous_metrics.iloc[:,previous_metrics.columns.get_level_values('stat') == 'reverse'][met_name].mean(axis=1)[name] > 0.5 if previous_metrics is not None else None

                # Calculate the metric with bootstrap, capturing overall, std, and reverse order boolean
                corr_, std_, reverse = self.bootstrap_score(exp_score,exp_bin, score_list[name].iloc[:,-1], met_name, n_bootstrap=n_bootstrap, pre_reverse=pre_reverse)
                
                # Append results to lists
                corr_list_.append(corr_)
                std_list_.append(std_)
                reverse_list_.append(reverse)
    
            # Store results in the DataFrame
            metric_results_df[(met_name, 'avg')] = corr_list_
            metric_results_df[(met_name, 'std')] = std_list_
            metric_results_df[(met_name, 'reverse')] = reverse_list_

        # Save the DataFrame in the class for the specified dataset
        self.metric_results[dataset_name] = metric_results_df

        # Compute the best variants for the dataset
        self.best_variants[dataset_name] = self.get_best_variants(dataset_name, best_metrics, best_met_extremas)

    def compute_metrics_iterations(self, dataset_name: str, n_bootstrap: int = 1000, metrics: list = ['spearmanr', 'average_precision'], iterations: int = 10, ite_size: int = 40,
                                   best_metrics: list = ['spearmanr', 'average_precision'], best_met_extremas: list = [[0,0],[1,1]]) -> None:
        """
        A function to compute metrics for a specific dataset and save results in a structured DataFrame within the class.
        Also determines the best variants for each model based on the normalized metrics.
        This function is used for datasets with multiple iterations.

        Args:
            dataset_name (str): The name of the dataset to process.
            n_bootstrap (int): The number of bootstrap iterations. Default is 1000.
            metrics (list): The metrics to calculate. Default is ['spearmanr', 'average_precision'].
            iterations (int): The number of iterations for the dataset. Default is 10.
            ite_size (int): The data size of each iteration. Default is 40.
            best_metrics (list): The best metrics to consider when selecting the best variants of a model. Default is ['spearmanr', 'average_precision'].
            best_met_extremas (list): The extremas for the best metrics. First list is the minimums, second list is the maximums. Default is [[0,0],[1,1]].
        """
        
        # Retrieve experimental data and score list for the specified dataset
        exp_score = self.datasets[dataset_name]['exp_data'].DMS_score  # Experimental data
        exp_bin = self.datasets[dataset_name]['exp_data'].DMS_score_bin  # Binary data
        score_list = self.datasets[dataset_name] # Predicted scores for each model

        # Define MultiIndex for columns
        index_tuples = [(met_name, f'ite{i}', stat) for met_name in metrics for i in range(10) for stat in ['avg', 'std', 'reverse']]
        multi_index = pd.MultiIndex.from_tuples(index_tuples, names=['metric', 'iteration', 'stat'])
        
        # Initialize the DataFrame for storing results
        metric_results_df = pd.DataFrame(index=self.name_list[dataset_name], columns=multi_index)

        # Iterate over all metrics
        for met_name in metrics:

            # Iterate over all iterations
            for i in range(iterations):
                corr_list_ = []
                std_list_ = []
                reverse_list_ = []

                # Iterate over all models initialized for the dataset
                for name in self.name_list[dataset_name]:

                    # Calculate the metric with bootstrap, capturing overall, std, and reverse order boolean
                    corr_, std_, reverse = self.bootstrap_score(exp_score.iloc[i * ite_size:i * ite_size + ite_size],exp_bin.iloc[i * ite_size:i * ite_size + ite_size], score_list[name].iloc[i * ite_size:i * ite_size + ite_size,-1],
                                                                met_name, n_bootstrap=n_bootstrap)
                    
                    # Append results to lists
                    corr_list_.append(corr_)
                    std_list_.append(std_)
                    reverse_list_.append(reverse)
                    
                # Store results in the DataFrame
                metric_results_df[(met_name, f'ite{i}', 'avg')] = corr_list_
                metric_results_df[(met_name, f'ite{i}', 'std')] = std_list_
                metric_results_df[(met_name, f'ite{i}', 'reverse')] = reverse_list_

        # Save the DataFrame in the class for the specified dataset
        self.metric_results[dataset_name] = metric_results_df
        self.best_variants[dataset_name] = self.get_best_variants_iteration(dataset_name = dataset_name, metrics = best_metrics, met_extremas = best_met_extremas)


    def get_best_variants(self, dataset_name: str,
                          metrics: list = ['spearmanr', 'average_precision'], met_extremas: list = [[0,0],[1,1]]) -> List:
        """
        Select the best variant for each model based on normalized metrics and save them in the class.
        
        Args:
            dataset_name (str): The name of the dataset to process.
            metrics (list): The metrics to use for determining the best and worst models. Default is ['spearmanr', 'average_precision'].
            met_extremas (list): The extremas for the metrics. First list is the minimums, second list is the maximums. Default is [[0,0],[1,1]].

        Returns:
            List: A list of the best variants for each model.
        """

        best_variants = []

        # Retrieve the metric results for the specified dataset
        metric_list = self.metric_results[dataset_name]

        # Correct the minimum for average precision, as a random model would have an average precision equal to the mean of the binary scores
        if 'average_precision' == metrics[0]:
            ap_random = self.datasets[dataset_name]['exp_data'].DMS_score_bin.mean()
            met_extremas[0] = [ap_random,0]
        if 'average_precision' == metrics[1]:
            ap_random = self.datasets[dataset_name]['exp_data'].DMS_score_bin.mean()
            met_extremas[0] = [0,ap_random]

        # Iterate over all models initialized for the dataset
        for model in self.model_list[dataset_name]:

            # Check if the model has multiple variants
            if any(m in model for m in [m['name'] for m in self.config['models'] if 'variants' in m]):
               
                # Get the base model name without variant suffix
                base_model = model.split('_')[0]
                df_ = metric_list.iloc[metric_list.index.str.contains(base_model)]
                
                # Extract the avg and std metrics for the variants
                met1_avg = df_[(metrics[0], 'avg')]
                met2_avg = df_[(metrics[1], 'avg')]

                # Normalize the metrics
                met1_min, met2_min = met_extremas[0]
                met1_max, met2_max = met_extremas[1]
                met1_norm = (met1_avg - met1_min) / (met1_max - met1_min)
                met2_norm = (met2_avg - met2_min) / (met2_max - met2_min)

                # Combine the normalized metrics and select the best variant
                combined_avg = met1_norm + met2_norm
                best_variant = combined_avg.idxmax()

                # Ensure the best variant is added only once
                if best_variant not in best_variants:
                    best_variants.append(best_variant)
            
            # Add single-variant models directly
            else: 
                best_variants.append(model)
        
        return best_variants
    
    def get_best_variants_iteration(self, dataset_name: str, iterations: int = 10,
                          metrics: list = ['spearmanr', 'average_precision'], met_extremas: list = [[0,0],[1,1]]) -> dict:
        """
        Select the best variant for each model based on normalized metrics and save them in the class.
        The function is used for datasets with multiple iterations.

        Args:
            dataset_name (str): The name of the dataset to process.
            iterations (int): The number of iterations for the dataset. Default is 10.
            metrics (list): The metrics to use for determining the best and worst models. Default is ['spearmanr', 'average_precision'].
            met_extremas (list): The extremas for the metrics. First list is the minimums, second list is the maximums. Default is [[0,0],[1,1]].

        Returns:
            dict: A dictionary of the best variants for each iteration.
        """

        best_variants = {}

        # Retrieve the metric results for the specified dataset
        metric_list = self.metric_results[dataset_name]

        # Initialize the dictionary to store the best variants
        for i in range(iterations):
            best_ite = []

            # Correct the minimum for average precision, as a random model would have an average precision equal to the mean of the binary scores
            if 'average_precision' == metrics[0]:
                ap_random = self.datasets[dataset_name]['exp_data'].iloc[i*40:(i+1)*40].DMS_score_bin.mean()
                met_extremas[0] = [ap_random,0]
            if 'average_precision' == metrics[1]:
                ap_random = self.datasets[dataset_name]['exp_data'].iloc[i*40:(i+1)*40].DMS_score_bin.mean()
                met_extremas[0] = [0,ap_random]

            # Iterate over all models initialized for the dataset
            for model in self.model_list[dataset_name]:

                # Check if the model has multiple variants
                if any(m in model for m in [m['name'] for m in self.config['models'] if 'variants' in m]):

                    # Get the base model name without variant suffix
                    base_model = model.split('_')[0]
                    df_ = metric_list.iloc[metric_list.index.str.contains(base_model)]

                    # Extract the avg and std metrics for the variants
                    met1_avg = df_[(metrics[0], f'ite{i}', 'avg')]
                    met2_avg = df_[(metrics[1], f'ite{i}', 'avg')]

                    # Normalize the metrics
                    met1_min, met2_min = met_extremas[0]
                    met1_max, met2_max = met_extremas[1]
                    met1_norm = (met1_avg - met1_min) / (met1_max - met1_min)
                    met2_norm = (met2_avg - met2_min) / (met2_max - met2_min)

                    # Combine the normalized metrics and select the best variant
                    combined_avg = met1_norm + met2_norm
                    best_variant = combined_avg.idxmax()

                    # Ensure the best variant is added only once
                    if best_variant not in best_variants:
                        best_ite.append(best_variant)

                # Add single-variant models directly
                else:
                    best_ite.append(model)

            # Save the best variants for the iteration
            best_variants[f'ite{i}'] = best_ite

        return best_variants
    
    def process_models(self, dataset_name: str, categories = True,
                        metrics: list = ['spearmanr', 'average_precision'], met_extremas: list = [[0,0],[1,1]]) -> None:
        '''
        A function to process the models and determine the best and worst models for each category and overall.
        The function will save the best and worst models in the class.

        Args:
            dataset_name (str): The name of the dataset to process.
            categories (bool): A boolean indicating whether to process the models for each category. Default is True.
            metrics (list): The metrics to use for determining the best and worst models. Default is ['spearmanr', 'average_precision'].
            met_extremas (list): The extremas for the metrics to normalize the scores. Default is [[0,0],[1,1]].
        '''

        # Retrieve the metric results for the specified dataset and the best variants, and initialize the dictionaries to store the best and worst models
        df = self.metric_results[dataset_name]
        best_variants = self.best_variants[dataset_name]
        best_models = {}
        worst_models = {}

        # Extract the avg and std metrics for the best variants
        met1_avg = df[(metrics[0], 'avg')].loc[best_variants]
        met2_avg = df[(metrics[1], 'avg')].loc[best_variants]

        # Correct the minimum for average precision, as a random model would have an average precision equal to the mean of the binary scores
        if 'average_precision' == metrics[0]:
            ap_random = self.datasets[dataset_name]['exp_data'].DMS_score_bin.mean()
            met_extremas[0] = [ap_random,0]
        if 'average_precision' == metrics[1]:
            ap_random = self.datasets[dataset_name]['exp_data'].DMS_score_bin.mean()
            met_extremas[0] = [0,ap_random]

        # Normalize the metrics and combine them
        met1_min, met2_min = met_extremas[0]
        met1_max, met2_max = met_extremas[1]
        met1_norm = (met1_avg - met1_min) / (met1_max - met1_min)
        met2_norm = (met2_avg - met2_min) / (met2_max - met2_min)
        combined_avg = met1_norm + met2_norm

        # Determine the indices of the best and worst models
        best_model_idx = combined_avg.idxmax()
        worst_model_idx = combined_avg.idxmin()
        best_models['overall'] = best_model_idx
        worst_models['overall'] = worst_model_idx

        # Determine best and worst models for all three categories if categories is True
        if categories:
            for cat in ['sequence', 'structure', 'MSA']:

                # Load the models for the specific category
                models_cat = self.load_specific_models(model_names=self.name_list[dataset_name],categories=[cat])

                # Skip the category if no models are found
                if models_cat == []:
                    print(f'No models found for {cat} for {dataset_name}.')
                    continue

                # Extract the avg and std metrics for the models in the category. Currently does not consider best variants.
                met1_avg = df[(metrics[0], 'avg')].loc[models_cat]
                met2_avg = df[(metrics[1], 'avg')].loc[models_cat]

                # Normalize the metrics and combine them
                met1_norm = (met1_avg - met1_min) / (met1_max - met1_min)
                met2_norm = (met2_avg - met2_min) / (met2_max - met2_min)
                combined_avg = met1_norm + met2_norm

                # Determine the indices of the best and worst models
                best_model_idx = combined_avg.idxmax()
                worst_model_idx = combined_avg.idxmin()
                best_models[cat] = best_model_idx
                worst_models[cat] = worst_model_idx

        self.best_models[dataset_name] = best_models
        self.worst_models[dataset_name] = worst_models

    def process_models_iteration(self, dataset_name: str, metrics: list = ['spearmanr', 'average_precision'], met_extremas: list = [[0,0],[1,1]]) -> None:
        '''
        A function to process the models and determine the best and worst models for each category and overall.
        The function will save the best and worst models in the class.
        This function is used for datasets with multiple iterations.

        Args:
            dataset_name (str): The name of the dataset to process.
            metrics (list): The metrics to use for determining the best and worst models. Default is ['spearmanr', 'average_precision'].
            met_extremas (list): The extremas for the metrics to normalize the scores. Default is [[0,0],[1,1]].
        '''

        # Retrieve the metric results for the specified dataset, and initialize the dictionaries to store the best and worst models
        df = self.metric_results[dataset_name]
        best_models = {}
        worst_models = {}
        
        # Initialize the extremas for the metrics
        met1_min, met2_min = met_extremas[0]
        met1_max, met2_max = met_extremas[1]

        # Iterate over all iterations
        for i in range(10):

            # Correct the minimum for average precision, as a random model would have an average precision equal to the mean of the binary scores
            if 'average_precision' == metrics[0]:
                ap_random = self.datasets[dataset_name]['exp_data'].DMS_score_bin.mean()
                met_extremas[0] = [ap_random,0]
            if 'average_precision' == metrics[1]:
                ap_random = self.datasets[dataset_name]['exp_data'].DMS_score_bin.mean()
                met_extremas[0] = [0,ap_random]
            
            # Initialize the dictionaries to store the best and worst models for the iteration, and retrieve the best variants
            best_models[f'ite{i}'] = {}
            worst_models[f'ite{i}'] = {}
            best_variants = self.best_variants[dataset_name][f'ite{i}']

            # Extract the avg and std metrics for the best variants
            met1_avg = df[(metrics[0], f'ite{i}', 'avg')].loc[best_variants]
            met2_avg = df[(metrics[1], f'ite{i}', 'avg')].loc[best_variants]

            # Normalize the metrics and combine them
            met1_norm = (met1_avg - met1_min) / (met1_max - met1_min)
            met2_norm = (met2_avg - met2_min) / (met2_max - met2_min)
            combined_avg = met1_norm + met2_norm

            # Determine the indices of the best and worst models
            best_model_idx = combined_avg.idxmax()
            worst_model_idx = combined_avg.idxmin()
            best_models[f'ite{i}']['overall'] = best_model_idx
            worst_models[f'ite{i}']['overall'] = worst_model_idx


            # Determine best and worst models for all three categories if categories is True
            for cat in ['sequence', 'structure', 'MSA']:

                # Load the models for the specific category
                models_cat = self.load_specific_models(model_names=self.name_list[dataset_name],categories=[cat])

                # Skip the category if no models are found
                if models_cat == []:
                    if i == 0:
                        print(f'No models found for {cat} for {dataset_name}.')
                    continue

                # Extract the avg and std metrics for the models in the category. Currently does not consider best variants.
                met1_avg = df[(metrics[0], f'ite{i}', 'avg')].loc[models_cat]
                met2_avg = df[(metrics[1], f'ite{i}', 'avg')].loc[models_cat]

                # Normalize the metrics and combine them
                met1_norm = (met1_avg - met1_min) / (met1_max - met1_min)
                met2_norm = (met2_avg - met2_min) / (met2_max - met2_min)
                combined_avg = met1_norm + met2_norm

                # Determine the indices of the best and worst models
                best_model_idx = combined_avg.idxmax()
                worst_model_idx = combined_avg.idxmin()
                best_models[f'ite{i}'][cat] = best_model_idx
                worst_models[f'ite{i}'][cat] = worst_model_idx

        self.best_models[dataset_name] = best_models
        self.worst_models[dataset_name] = worst_models

    def sm_landscape(self, dataset_name: str, model_name: str, wt: bool = False):
        """
        Generate a single mutant landscape for a specific model and dataset. The landscape will be stored in the class as a DataFrame.

        Args:
            dataset_name (str): The name of the dataset to process.
            model_name (str): The name of the model to generate the landscape for.
            wt (bool): Whether to include the wildtype sequence in the landscape. Default is False.
        """

        # Initialize the landscape dictionary if it does not exist
        if dataset_name not in self.landscape:
            self.landscape[dataset_name] = {}

        # Define the protein length, number of mutations
        prot_len = len(self.wildtype[dataset_name]['exp_data']['mutated_sequence'])
        num_mutations = len(aa_order)
        
        # Initialize the matrix to store the landscape
        pred_matrix = np.full((prot_len, num_mutations), np.nan) # Initialize with NaNs

        # Define the wildtype score for the model and dataset if wt is True else set it to NaN
        wt_score = self.wildtype[dataset_name][model_name][model_name] if wt else np.nan

        # Populate the matrix using the wildtype data
        for i,aa in enumerate(self.wildtype[dataset_name]['exp_data']['mutated_sequence']):
            pred_matrix[i, aa_to_index[aa]] = wt_score
        
        # Populate the matrix using the predicted data
        for mutant,score in zip(self.datasets[dataset_name]['exp_data']['mutant'] ,self.datasets[dataset_name][model_name][model_name]):
            pos = int(mutant[1:-1])  # Extract position (e.g., "M1A" -> 1)
            mut = mutant[-1]         # Extract mutated residue (e.g., "M1A" -> "A")
            pred_matrix[pos - 1, aa_to_index[mut]] = score  # Position is 1-indexed
        
        # Create a DataFrame from the matrix and store it in the class
        landscape = pd.DataFrame(pred_matrix, columns=aa_order, index=range(1, prot_len + 1))
        self.landscape[dataset_name][model_name] = landscape
    
    def sorting_mutants(self, dataset_name: str, model_names: list, sort_by_model: str, output_path: str, wt: bool = False,
                        previous_metrics: pd.DataFrame = None, previous_metric: str = 'spearmanr') -> pd.DataFrame:
        """
        Sort the experimental data and the predicted data for a specific dataset and model based on the predicted scores.
        The function will save the sorted data in a CSV and Excel file, as well as returning the sorted DataFrame.

        Args:
            dataset_name (str): The name of the dataset to process.
            model_names (list): The names of the models to sort the data for.
            sort_by_model (str): The model to sort the data by.
            output_path (str): The path to save the sorted data.
            wt (bool): Whether to include a binary column based on the wildtype score. Default is False.
            previous_metrics (pd.DataFrame): A DataFrame containing the previous metrics. Used for determining the order. Default is None.
            previous_metric (str): The metric to use determining the order. Default is 'spearmanr'.

        Returns:
            pd.DataFrame: A DataFrame with the sorted data.
        """

        # Initialize the DataFrame to store the sorted data and a list to store the model names
        df = self.datasets[dataset_name]['exp_data'].copy().drop(columns=['DMS_score','DMS_score_bin'])
        names = []

        # Iterate over specific models
        for model_name in model_names:

            # Extract the model name and the predicted scores
            name = model_name.split('_')[0]
            names.append(name)
            model_res = self.datasets[dataset_name][model_name][model_name]
            df[name] = model_res

            # Determine the order based on the previous metrics. If no previous metrics are provided, set reverse to False
            reverse_ = previous_metrics.iloc[:,previous_metrics.columns.get_level_values('stat') == 'reverse'][previous_metric].mean(axis=1)[model_name] > 0.5 if previous_metrics is not None else False
            
            # Rank the predicted scores and add a binary column if wt is True
            rankings = model_res.rank(ascending=reverse_)
            df[f'{name}_rank'] = rankings
            if wt:
                binary = (model_res >= self.wildtype[dataset_name][model_name][model_name]).astype(int)
                df[f'{name}_bin'] = binary

        # Sort the DataFrame based on the specified model
        # First, determine the order of the modelbased on the previous metrics. If no previous metrics are provided, set reverse to False
        reverse_ = previous_metrics.iloc[:,previous_metrics.columns.get_level_values('stat') == 'reverse'][previous_metric].mean(axis=1)[sort_by_model] > 0.5 if previous_metrics is not None else False
        df = df.sort_values(by=sort_by_model,ascending=reverse_)

        # Save the sorted DataFrame in a CSV and Excel file using the model names as part of the file name
        names = '&'.join(names)
        df.to_csv(output_path+f'{dataset_name}_sorted_{names}.csv',index=False)
        df.to_excel(output_path+f'{dataset_name}_sorted_{names}.xlsx',index=False)
        return df

    def insilico_library(self,dataset_name: str, numb_muts: List[int], wt_seq: str = None, exclude_aa: List[str] = None, mut_sites: List[int] = None,
                         exclude_sites: List[int] = None, ignore_large: bool = False, wt: bool = True, save_path: str = '../data/insilico_libraries/', custom_name: str = None) -> pd.DataFrame:
        """
        Generate a list of mutated sequences by substituting amino acids at specified mutation sites. The function will save the list in a CSV file and return the DataFrame.

        Args:
            dataset_name (str): The name of the dataset to process.
            numb_muts (List): A list of the number of mutations to apply. For example, [1, 2] will both generate single and double mutants.
            wt_seq (str): The wildtype sequence. If not provided, the function will use the experimental data.
            exclude_aa (List): A list of amino acids to exclude from the mutations. Default is None.
            mut_sites (List): A list of mutation sites. If not provided, the function will use all sites. Default is None.
            exclude_sites (List): A list of sites to exclude from the mutations. Default is None.
            ignore_large (bool): A boolean indicating whether to ignore large variant numbers. Default is False.
            wt (bool): Whether to include the wildtype sequence in the library. Default is True.
            save_path (str): The path to save the mutated sequences. Default is '../data/insilico_libraries/'.
            custom_name (str): A custom name to append to the saved file. Default is None.

        Returns:
            pd.DataFrame: A dataframe with mutated sequences and IDs.
        """

        # Define the wildtype sequence. If not provided, use the experimental data. If also not provided, raise an error.
        if wt_seq is not None:
            wildtype = wt_seq
        elif 'exp_data' in self.wildtype[dataset_name]:
            wildtype = self.wildtype[dataset_name]['exp_data'].mutated_sequence
        else:
            raise ValueError('No wildtype sequence provided')
        
        # Convert the wildtype sequence to an array
        wildtype_array = np.array(list(wildtype))

        # If exclude_aa and mut_sites are not provided, set them to empty lists
        if exclude_aa is None:
            exclude_aa = []
        if exclude_sites is None:
            exclude_sites = []
        
        # Determine possible amino acid mutations and mutation sites
        aa_list = [aa for aa in amino_acids if aa not in exclude_aa]
        mut_sites = list(range(1, len(wildtype) + 1)) if mut_sites is None else mut_sites
        mut_sites = [site for site in mut_sites if site not in exclude_sites]
        
        # Initialize lists to store the mutated sequences and IDs
        mut_list = []
        mut_ID_list = []

        # Calculate the number of variants and raise an error if the number is very large and ignore_large is False
        var_numb = 0
        for numb in numb_muts:
            var_numb += varnumb(len(mut_sites),numb)
        if var_numb > 1000000 and not ignore_large:
            raise ValueError(f'The number of variants is very large ({int(var_numb)}). Please reduce the number of mutations/sites, or set "ignore_large" to True.')

        # Iterate over the number of mutations.
        for numb in numb_muts:
            # Generate site combinations and amino acid substitutions
            site_combinations = list(itertools.combinations(mut_sites, r=numb))
            aa_combinations = list(itertools.product(aa_list, repeat=numb))
            
            # Iterate over the site and amino acid combinations
            for sites in site_combinations:
                for mut_ in aa_combinations:

                    # Ensure that mutant is not the same as the wildtype
                    if any(wt_seq[site - 1] == aa for site, aa in zip(sites, mut_)):
                        continue  # Skip this mutation combination

                    # Copy the wildtype sequence
                    mut_seq = wildtype_array.copy()
                    mut_ID = []
                    
                    # Apply mutations
                    for site, aa in zip(sites, mut_):
                        original_aa = wildtype[site - 1]
                        mut_seq[site - 1] = aa
                        mut_ID.append(f"{original_aa}{site}{aa}")
                    
                    # Create mutated sequence
                    mut_seq_str = ''.join(mut_seq)

                    # Only add the mutated sequence and ID if the sequence is different from the wildtype
                    if mut_ID:
                        mut_list.append(mut_seq_str)
                        mut_ID_list.append(':'.join(mut_ID))
        
        # Create a DataFrame with the results
        df = pd.DataFrame({
            'mutant': mut_ID_list,
            'mutated_sequence': mut_list,
            'DMS_score': np.zeros(len(mut_list))
        }).drop_duplicates(subset='mutant').reset_index(drop=True)

        # Add the wildtype sequence to the DataFrame if wt is True
        if wt:
            wt_row = [wildtype[0]+'1'+wildtype[0],wildtype,0]
            df.iloc[-1,:] = wt_row


        # Save the file; inserting a custom specifier for the file if custom_name is provided.
        if custom_name is not None:
            df.to_csv(save_path+f'{dataset_name}_{custom_name}_insilico_library.csv',index=False)
        else:
            df.to_csv(save_path+f'{dataset_name}_insilico_library.csv',index=False)

        return df
        
class PlottingModule:
    """
    A class used for plotting the results of the ZeroShotModeller class.
    Contains static functions for comparative scatter plots, metric visualizations, and heatmaps
    """

    # Dictionary to define the shapes for the different categories
    shape_dict = {'sequence':'o','MSA':'s','structure':'^'}

    @staticmethod
    def zs_vs_exp_scatter(zsm, dataset_name: str, model_name: str, custom_zsname: str = None, custom_expname: str = None,
                          metrics: list = ['spearmanr', 'average_precision'], met1_corr_decimal: int = 2, met1_std_decimal: int = 2, met2_corr_decimal: int = 2, met2_std_decimal: int = 2,
                      color: str = '#df9966', figsize: Tuple = (9,9),size: int = 300, save_path: str = '../figures/scatter/', dpi: int = 300, format: str = 'png'):
        """
        Generate a scatter plot comparing the experimental and predicted scores for a specific model and dataset.
        The plot will be saved as a PNG file if save_path is provided.

        Args:
            zsm (ZeroShotModeller): The ZeroShotModeller object.
            dataset_name (str): The name of the dataset to process.
            model_name (str): The name of the model to generate the scatter plot for.
            custom_zsname (str): A custom name for the predicted score. Default is None.
            custom_expname (str): A custom name for the experimental score. Default is None.
            metrics (list): The metrics to display in the legend. Default is ['spearmanr', 'average_precision'].
            met1_corr_decimal (int): The number of decimal places for the first metric correlation. Default is 2.
            met1_std_decimal (int): The number of decimal places for the first metric standard deviation. Default is 2.
            met2_corr_decimal (int): The number of decimal places for the second metric correlation. Default is 2.
            met2_std_decimal (int): The number of decimal places for the second metric standard deviation. Default is 2.
            color (str): The color of the scatter points. Default is '#df9966' (persian orange).
            figsize (Tuple): The figure size. Default is (9,9).
            size (int): The size of the scatter points. Default is 300.
            save_path (str): The path to save the scatter plot. Default is None.
            dpi (int): The resolution of the saved image. Default is 300.
            format (str): The format of the saved image. Default is 'png'.
        """

        # Extract the experimental and predicted scores
        exp_ = zsm.datasets[dataset_name]['exp_data'].DMS_score
        zs_score_ = zsm.datasets[dataset_name][model_name][model_name]
        
        # Create the scatter plot
        fig,ax = plt.subplots(figsize=figsize)
        ax.set_box_aspect(1)
        ax.scatter(exp_,zs_score_,color=color,edgecolor='k',s=size,alpha=0.8)

        # Create a list with two empty handles used for the legend
        handles = [mpl_patches.Rectangle((0, 0), 1, 1, fc="white", ec="white", lw=0, alpha=0),mpl_patches.Rectangle((0, 0), 1, 1, fc="white", ec="white", lw=0, alpha=0)]
        labels = []
        
        # Create a regression line for the scatter plot
        sns.regplot(x=exp_,y=zs_score_,color='darkgrey',scatter_kws={'s':0},order=2,ci=50)

        # Extract the metrics and add them to the legend
        corr = zsm.metric_results[dataset_name][(metrics[0],'avg')][model_name]
        std = zsm.metric_results[dataset_name][(metrics[0],'std')][model_name]
        label = f"Met1 = {corr:.{met1_corr_decimal}f}"+r'$\pm$'+f'{std:.{met1_std_decimal}f}'
        labels.append(label)
        corr = zsm.metric_results[dataset_name][(metrics[1],'avg')][model_name]
        std = zsm.metric_results[dataset_name][(metrics[1],'std')][model_name]
        label = f"Met2 = {corr:.{met2_corr_decimal}f}"+r'$\pm$'+f'{std:.{met2_std_decimal}f}'
        labels.append(label)

        # Create the legend, supressing the blank space of the empty line symbol and the
        # padding between symbol and label by setting handlelength and handletextpad
        ax.legend(handles, labels, loc='best', fontsize=30, 
                fancybox=True, framealpha=0.7, 
                handlelength=0, handletextpad=0)
        
        # Set the axis labels and tick parameters
        ax.tick_params(axis='both', which='major', labelsize=25)
        if custom_expname is not None:
            ax.set_xlabel(custom_expname,fontsize=30)
        else:
            ax.set_xlabel('Experimental Score',fontsize=30)
        if custom_zsname is not None:
            ax.set_ylabel(custom_zsname+' Score',fontsize=30)
        else:
            ax.set_ylabel(model_name+' Score',fontsize=30)
        plt.tight_layout()

        # Save the scatter plot
        plt.savefig(save_path+f'{dataset_name}_{model_name}_scatter.{format}',dpi=dpi)
        plt.show()

    @staticmethod
    def metrics(zsm, dataset_name: str, save_path: str = '../figures/metrics/', dpi: int = 300, custom_name: str = None,
                     metrics: list = ['average_precision','spearmanr'], met_labels = ['Average Precision','Absolute Spearman Correlation'], format: str = 'png'):
        """
        Generate a scatter plot with errorbars comparing two metrics for the different models. The plot will be saved as a PNG file.

        Args:
            zsm (ZeroShotModeller): The ZeroShotModeller object.
            dataset_name (str): The name of the dataset to process.
            save_path (str): The path to save the scatter plot. Default is '../figures/metrics/'.
            dpi (int): The resolution of the saved image. Default is 300.
            custom_name (str): A custom name for the saved file. Default is None.
            metrics (list): The metrics to display in the scatter plot. Default is ['average_precision','spearmanr'].
            met_labels (list): The labels for the metrics. Default is ['Average Precision','Absolute Spearman Correlation'].
            format (str): The format of the saved image. Default is 'png'.
        """

        # Initialize the figure and axes
        fig, ax = plt.subplots(figsize=(12,9))

        ebar_list = []
        legends = []
        
        # Extract the full metric results and the best variants
        data_df = zsm.metric_results[dataset_name]
        metx_avg = data_df[(metrics[0], 'avg')]
        metx_std = data_df[(metrics[0], 'std')]
        mety_avg = data_df[(metrics[1], 'avg')]
        mety_std = data_df[(metrics[1], 'std')]
        best_variants = zsm.best_variants[dataset_name]

        # Iterate over the best variants
        for model_name in best_variants:

            # Extract the model name, shape, and color
            for model in zsm.config['models']:
                if model['name'] == model_name.split('_')[0]: # Remove variant suffix
                    legends.append(model['name'])
                    c = model['color']
                    shape = PlottingModule.shape_dict[model['category']]
                    break

            # Create the scatter plot with errorbars
            ebar_ = ax.errorbar(
                np.absolute(metx_avg.loc[model_name]),
                mety_avg.loc[model_name],
                xerr=metx_std.loc[model_name],
                yerr=mety_std.loc[model_name],
                markersize=20,
                ecolor='lightgrey',
                elinewidth=1,
                marker=shape,
                lw=0,
                capsize=5,
                color=c,
                markeredgecolor='k',
            )

            # Append the errorbar to the list to use in the legend
            ebar_list.append(ebar_)

        # Set figure properties
        ax.set_box_aspect(1)
        ax.grid(True, color='lightgrey', ls=':')
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.set_xlabel(met_labels[0], fontsize=30)
        ax.set_ylabel(met_labels[1], fontsize=30)

        # Create the legend and place it outside the plot
        fig.legend(ebar_list,legends, loc='center left', bbox_to_anchor=(0.85, 0.5),fontsize=25)

        plt.tight_layout()

        # Save the scatter plot, using the custom name if provided
        if custom_name is not None:
            output_path = save_path+f'{dataset_name}_{custom_name}_corr.{format}'
        else:
            output_path = save_path+f'{dataset_name}_corr.{format}'
        plt.savefig(output_path, dpi=dpi)
        plt.show()
    
    @staticmethod
    def metrics_iteration(zsm, dataset_name: str, custom_name: str = None, save_path: str = '../figures/metrics/', dpi: int = 300,
                               metrics: list = ['average_precision','spearmanr'], met_labels = ['Average Precision','Absolute Spearman Correlation'],format: str = 'png'):
        """
        Generate a scatter plot with errorbars comparing two metrics for the different models for each iteration. The plot will be saved as a PNG file.
        Currently only supports 10 iterations.
        Args:
            zsm (ZeroShotModeller): The ZeroShotModeller object.
            dataset_name (str): The name of the dataset to process.
            save_path (str): The path to save the scatter plot. Default is '../figures/metrics/'.
            dpi (int): The resolution of the saved image. Default is 300.
            custom_name (str): A custom name for the saved file. Default is None.
            metrics (list): The metrics to display in the scatter plot. Default is ['average_precision','spearmanr'].
            met_labels (list): The labels for the metrics. Default is ['Average Precision','Absolute Spearman Correlation'].
            format (str): The format of the saved image. Default is 'png'.
        """
        
        # Initialize the figure and axes
        fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(30, 12),sharey=True,sharex=True)
        axes = axes.flatten()
        ebar_list = []
        legends = []
        
        # Iterate over the 10 iterations
        for i in range(10):
            ax = axes[i]

            # Extract the metric results and the best variants for the specific iteration
            data_df = zsm.metric_results[dataset_name]
            metx_avg = data_df[(metrics[0], f'ite{i}', 'avg')]
            metx_std = data_df[(metrics[0], f'ite{i}', 'std')]
            mety_avg = data_df[(metrics[1], f'ite{i}', 'avg')]
            mety_std = data_df[(metrics[1], f'ite{i}', 'std')]
            best_variants = zsm.best_variants[dataset_name][f'ite{i}']

            # Iterate over the best variants
            for model_name in best_variants:
                skip = False # Skip if the model has already been added to the legend

                # Extract the model name, shape, and color
                for model in zsm.config['models']:
                    if model['name'] == model_name.split('_')[0]:
                        if model['name'] not in legends:
                            legends.append(model['name'])
                        else:
                            skip = True
                        c = model['color']
                        shape = PlottingModule.shape_dict[model['category']]
                        break
                
                # Create the scatter plot with errorbars
                ebar_ = ax.errorbar(
                    np.absolute(metx_avg.loc[model_name]),
                    mety_avg.loc[model_name],
                    xerr=metx_std.loc[model_name],
                    yerr=mety_std.loc[model_name],
                    markersize=20,
                    ecolor='lightgrey',
                    elinewidth=1,
                    marker=shape,
                    lw=0,
                    capsize=5,
                    color=c,
                    markeredgecolor='k',
                )

                # Append the errorbar to the list to use in the legend
                if not skip:
                    ebar_list.append(ebar_)

            # Set figure properties
            ax.set_box_aspect(1)
            ax.grid(True, color='lightgrey', ls=':')
            ax.tick_params(axis='both', which='major', labelsize=20)
            ax.set_title(f'Iteration {i+1}', fontsize=25)

        # Set the axis labels and legend
        axes[7].set_xlabel(met_labels[0], fontsize=30)
        axes[5].set_ylabel(met_labels[1], fontsize=30)
        axes[5].yaxis.set_label_coords(-0.15, 1.05)

        # Create the legend and place it outside the plot
        fig.legend(ebar_list,legends, loc='center left', bbox_to_anchor=(1, 0.5),fontsize=30)

        plt.tight_layout()

        # Save the scatter plot, using the custom name if provided
        if custom_name is not None:
            output_path = save_path+f'{dataset_name}_{custom_name}_corr.{format}'
        else:
            output_path = save_path+f'{dataset_name}_corr.{format}'
        plt.savefig(output_path, dpi=dpi)
        plt.show()
    
    @staticmethod
    def metrics_cat(zsm, dataset_name: str, custom_name: str = None, save_path: str = '../figures/metrics/', dpi: int = 300, format: str = 'png',
                         best_color: str = '#df9966', worst_color: str = '#516f84', metrics: list = ['average_precision','spearmanr'], met_labels = ['Average Precision','Absolute Spearman Correlation']):
        """
        Generate a scatter plot with errorbars comparing two metrics for the different models, coloring the best and worst models for each category.
        The plot will be saved as a PNG file.

        Args:
            zsm (ZeroShotModeller): The ZeroShotModeller object.
            dataset_name (str): The name of the dataset to process.
            save_path (str): The path to save the scatter plot. Default is '../figures/metrics/'.
            dpi (int): The resolution of the saved image. Default is 300.
            format (str): The format of the saved image. Default is 'png'.
            custom_name (str): A custom name for the saved file. Default is None.
            best_color (str): The color for the best models. Default is '#df9966' (persian orange).
            worst_color (str): The color for the worst models. Default is '#516f84' (slate gray).
            metrics (list): The metrics to display in the scatter plot. Default is ['average_precision','spearmanr'].
            met_labels (list): The labels for the metrics. Default is ['Average Precision','Absolute Spearman Correlation'].
        """

        # Initialize the figure and axes
        fig, ax = plt.subplots(figsize=(12, 9))

        # Initialize the lists to store the best and worst models and the legends
        ebar_best_list = []
        ebar_worst_list = []
        legends_best = []
        legends_worst = []

        # Extract the full metric results and the best variants
        data_df = zsm.metric_results[dataset_name]
        metx_avg = data_df[(metrics[0], 'avg')]
        metx_std = data_df[(metrics[0], 'std')]
        mety_avg = data_df[(metrics[1], 'avg')]
        mety_std = data_df[(metrics[1], 'std')]
        best_variants = zsm.best_variants[dataset_name]

        # Iterate over the best variants
        for model_name in best_variants:

            # Extract the model name and shape
            for model in zsm.config['models']:
                if model['name'] == model_name.split('_')[0]:
                    shape = PlottingModule.shape_dict[model['category']]
                    break
            
            # Create the scatter plot with errorbars, coloring all models in light grey
            ax.errorbar(
                np.absolute(metx_avg.loc[model_name]),
                mety_avg.loc[model_name],
                xerr=metx_std.loc[model_name],
                yerr=mety_std.loc[model_name],
                markersize=20,
                ecolor='lightgrey',
                elinewidth=1,
                marker=shape,
                lw=0,
                capsize=5,
                color='#dee3e8ff',
                markeredgecolor='k',
                zorder=1
            )

        # Iterate over the categories
        for cat in ['sequence','MSA','structure']:

            # Extract the shape, best model, and worst model for the category
            shape = PlottingModule.shape_dict[cat]
            best_model = zsm.best_models[dataset_name][cat]
            worst_model = zsm.worst_models[dataset_name][cat]

            # Create the scatter plot with errorbars for the best (green) and worst (red) models
            ebar_best = ax.errorbar(
                np.absolute(metx_avg.loc[best_model]),
                mety_avg.loc[best_model],
                xerr=metx_std.loc[best_model],
                yerr=mety_std.loc[best_model],
                markersize=20,
                ecolor='lightgrey',
                elinewidth=1,
                marker=shape,
                lw=0,
                capsize=5,
                color=best_color,
                markeredgecolor='k',
                zorder=1
            )
            ebar_worst = ax.errorbar(
                np.absolute(metx_avg.loc[worst_model]),
                mety_avg.loc[worst_model],
                xerr=metx_std.loc[worst_model],
                yerr=mety_std.loc[worst_model],
                markersize=20,
                ecolor='lightgrey',
                elinewidth=1,
                marker=shape,
                lw=0,
                capsize=5,
                color=worst_color,
                markeredgecolor='k',
                zorder=1
            )

            # Append the errorbars to the lists to use in the legend
            ebar_best_list.append(ebar_best)
            legends_best.append(best_model.split('_')[0])
            ebar_worst_list.append(ebar_worst)
            legends_worst.append(worst_model.split('_')[0])

        # Set figure properties
        ax.set_box_aspect(1)
        ax.grid(True, color='lightgrey', ls=':')
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.set_xlabel(met_labels[0], fontsize=30)
        ax.set_ylabel(met_labels[1], fontsize=30)

        # Create the legend and set the axis labels
        ax.legend(ebar_best_list+ebar_worst_list,legends_best+legends_worst, loc=0,fontsize=15)

        plt.tight_layout()

        # Save the scatter plot, using the custom name if provided
        if custom_name is not None:
            output_path = save_path+f'corr_cat_{dataset_name}_{custom_name}.{format}'
        else:
            output_path = save_path+'corr_cat_'+dataset_name+f'.{format}'
        plt.savefig(output_path, dpi=dpi)
        plt.show()

    @staticmethod
    def metrics_cat_iteration(zsm, dataset_name: str, custom_name: str = None, save_path: str = '../figures/metrics/', dpi: int = 300, format: str = 'png',
                                   best_color: str = '#df9966', worst_color: str = '#516f84',metrics: list = ['average_precision','spearmanr'],
                                   met_labels = ['Average Precision','Absolute Spearman Correlation']):
        """
        Generate a scatter plot with errorbars comparing two metrics for the different models for each iteration, coloring the best and worst models for each category.
        The plot will be saved as a PNG file.
        Currently only supports 10 iterations.

        Args:
            zsm (ZeroShotModeller): The ZeroShotModeller object.
            dataset_name (str): The name of the dataset to process.
            custom_name (str): A custom name for the saved file. Default is None.
            save_path (str): The path to save the scatter plot. Default is '../figures/metrics/'.
            dpi (int): The resolution of the saved image. Default is 300.
            format (str): The format of the saved image. Default is 'png'.
            best_color (str): The color for the best models. Default is '#df9966' (persian orange).
            worst_color (str): The color for the worst models. Default is '#516f84' (slate gray).
            metrics (list): The metrics to display in the scatter plot. Default is ['average_precision','spearmanr'].
            met_labels (list): The labels for the metrics. Default is ['Average Precision','Absolute Spearman Correlation'].
        """

        # Initialize the figure and axes
        fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(30, 12),sharey=True,sharex=True)
        axes = axes.flatten()
        
        # Iterate over the 10 iterations
        for i in range(10):
            ebar_best_list = []
            ebar_worst_list = []
            legends_best = []
            legends_worst = []
            ax = axes[i]

            # Extract the metric results and the best variants for the specific iteration
            data_df = zsm.metric_results[dataset_name]
            metx_avg = data_df[(metrics[0], f'ite{i}', 'avg')]
            metx_std = data_df[(metrics[0], f'ite{i}', 'std')]
            mety_avg = data_df[(metrics[1], f'ite{i}', 'avg')]
            mety_std = data_df[(metrics[1], f'ite{i}', 'std')]
            best_variants = zsm.best_variants[dataset_name][f'ite{i}']

            # Iterate over the best variants
            for model_name in best_variants:

                # Extract the model name and shape
                for model in zsm.config['models']:
                    if model['name'] == model_name.split('_')[0]:
                        shape = PlottingModule.shape_dict[model['category']]
                        break

                # Create the scatter plot with errorbars, coloring all models in light grey
                ax.errorbar(
                    np.absolute(metx_avg.loc[model_name]),
                    mety_avg.loc[model_name],
                    xerr=metx_std.loc[model_name],
                    yerr=mety_std.loc[model_name],
                    markersize=20,
                    ecolor='lightgrey',
                    elinewidth=1,
                    marker=shape,
                    lw=0,
                    capsize=5,
                    color='#dee3e8ff',
                    markeredgecolor='k',
                    zorder=1
                )

            # Iterate over the categories
            for cat in ['sequence','MSA','structure']:

                # Extract the shape, best model, and worst model for the category
                shape = PlottingModule.shape_dict[cat]
                best_model = zsm.best_models[dataset_name][f'ite{i}'][cat]
                worst_model = zsm.worst_models[dataset_name][f'ite{i}'][cat]

                # Create the scatter plot with errorbars for the best (green) and worst (red) models
                ebar_best = ax.errorbar(
                    np.absolute(metx_avg.loc[best_model]),
                    mety_avg.loc[best_model],
                    xerr=metx_std.loc[best_model],
                    yerr=mety_std.loc[best_model],
                    markersize=20,
                    ecolor='lightgrey',
                    elinewidth=1,
                    marker=shape,
                    lw=0,
                    capsize=5,
                    color=best_color,
                    markeredgecolor='k',
                    zorder=1
                )
                ebar_worst = ax.errorbar(
                    np.absolute(metx_avg.loc[worst_model]),
                    mety_avg.loc[worst_model],
                    xerr=metx_std.loc[worst_model],
                    yerr=mety_std.loc[worst_model],
                    markersize=20,
                    ecolor='lightgrey',
                    elinewidth=1,
                    marker=shape,
                    lw=0,
                    capsize=5,
                    color=worst_color,
                    markeredgecolor='k',
                    zorder=1
                )

                # Append the errorbars to the lists to use in the legend
                ebar_best_list.append(ebar_best)
                legends_best.append(best_model.split('_')[0])
                ebar_worst_list.append(ebar_worst)
                legends_worst.append(worst_model.split('_')[0])

            # Set figure properties
            ax.set_box_aspect(1)
            ax.grid(True, color='lightgrey', ls=':')
            ax.tick_params(axis='both', which='major', labelsize=20)
            ax.set_title(f'Iteration {i+1}', fontsize=25)

            # Set the axis legend
            ax.legend(ebar_best_list+ebar_worst_list,legends_best+legends_worst, loc=0,fontsize=15)

        # Set the figure labels
        axes[7].set_xlabel(met_labels[0], fontsize=30)
        axes[5].set_ylabel(met_labels[1], fontsize=30)
        axes[5].yaxis.set_label_coords(-0.15, 1.05)
        

        plt.tight_layout()

        # Save the scatter plot, using the custom name if provided
        if custom_name is not None:
            output_path = save_path+f'corr_cat_ite_{dataset_name}_{custom_name}.{format}'
        else:
            output_path = save_path+'corr_cat_ite_'+dataset_name+f'.{format}'
        plt.savefig(output_path, dpi=dpi)
        plt.show()

    @staticmethod
    def bar_metrics(zsm,dataset_names: list, metric: str, ylabel: str, xlabel: str, alternative_names: list = None,
                    cat: str='overall', ylimit: float = 1.135, figsize: Tuple = (12,9), best_color: str = '#df9966', worst_color: str = '#516f84',
                    save_path: str = '../figures/metrics/', dpi: int = 300, custom_name: str = None, format: str = 'png'):
        """
        Generate a bar plot comparing the full metrics of the best and worst models from the 10 iterations for different datasets.
        Require that the iteration datasets are named 'dataset_name_selection'.
        Will use the best and worst models from the 'overall' category to extract the metrics from the metric_results dictionary of the full dataset.
        Currently only supports 10 iterations.
        The plot will be saved as a PNG file.

        Args:
            zsm (ZeroShotModeller): The ZeroShotModeller object.
            dataset_names (list): The names of the datasets to process.
            metric (str): The metric to display in the bar plot.
            ylabel (str): The label for metric property used to compare the models.
            xlabel (str): The label for the x-axis, denoting the property used to group the datasets.
            alternative_names (list): The alternative names for the datasets. Default is None.
            cat (str): The category to extract the best and worst models from. Default is 'overall'.
            ylimit (float): The limit for the y-axis. Default is 1.135.
            figsize (Tuple): The figure size. Default is (12,9).
            best_color (str): The color for the best models. Default is '#df9966' (persian orange).
            worst_color (str): The color for the worst models. Default is '#516f84' (slate gray).
            save_path (str): The path to save the bar plot. Default is '../figures/metrics/'.
            dpi (int): The resolution of the saved image. Default is 300.
            custom_name (str): A custom name for the saved file. Default is None.
            format (str): The format of the saved image. Default is 'png'.
        """

        # Initialize the figure and axes
        fig,ax = plt.subplots(figsize=figsize)

        np.random.seed(42) # Set seed for reproducibility

        # Iterate over the datasets
        for i,name in enumerate(dataset_names):
            scatter_best = np.zeros(10)
            scatter_worst = np.zeros(10)

            # Iterate over the 10 iterations
            for j in range(10):
                best_model = zsm.best_models[name+'_selection'][f'ite{j}'][cat]
                worst_model = zsm.worst_models[name+'_selection'][f'ite{j}'][cat]
                scatter_best[j] = zsm.metric_results[name][(f'{metric}','avg')][best_model]
                scatter_worst[j] = zsm.metric_results[name][(f'{metric}','avg')][worst_model]

            # Use the average metric over all iterations to create the bar plot with errorbars, coloring the best models in dark blue and the worst models in light brown
            # For the first dataset, add the legend
            if i == 0:
                ax.bar(i-0.2,scatter_best.mean(),yerr=scatter_best.std(),color=best_color,edgecolor='k',capsize=15,width=0.4,alpha=1,label='Best Models')
                ax.bar(i+0.2,scatter_worst.mean(),yerr=scatter_worst.std(),color=worst_color,edgecolor='k',capsize=15,width=0.4,alpha=1,label='Worst Models')

                # Add scatter points of the metrics for the best and worst models from the 10 iterations, with some random noise to avoid overlap
                ax.scatter(i-0.2+np.random.uniform(low=-1,high=1,size=10)/10,scatter_best,color='darkgray',alpha=0.8,s=100,edgecolors='k')
                ax.scatter(i+0.2+np.random.uniform(low=-1,high=1,size=10)/10,scatter_worst,color='darkgray',alpha=0.8,s=100,edgecolors='k')
            else:
                ax.bar(i-0.2,scatter_best.mean(),yerr=scatter_best.std(),color=best_color,edgecolor='k',capsize=15,width=0.4,alpha=1)
                ax.bar(i+0.2,scatter_worst.mean(),yerr=scatter_worst.std(),color=worst_color,edgecolor='k',capsize=15,width=0.4,alpha=1)

                # Add scatter points of the metrics for the best and worst models from the 10 iterations, with some random noise to avoid overlap
                ax.scatter(i-0.2+np.random.uniform(low=-1,high=1,size=10)/10,scatter_best,color='darkgray',alpha=0.8,s=100,edgecolors='k')
                ax.scatter(i+0.2+np.random.uniform(low=-1,high=1,size=10)/10,scatter_worst,color='darkgray',alpha=0.8,s=100,edgecolors='k')
            

        # Set the x-ticks and labels, using the alternative names if provided
        x_ticks = np.arange(len(dataset_names))
        if alternative_names is not None:
            x_ticks_labels = alternative_names
        else:
            x_ticks_labels = dataset_names
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_ticks_labels,fontsize=25)

        # Set the figure properties
        ax.set_ylabel(ylabel,fontsize=30)
        ax.set_xlabel(xlabel,fontsize=30,labelpad=17)
        ax.set_ylim(0,ylimit)
        ax.legend(fontsize=20,loc='upper right')
        ax.tick_params(axis='both', which='major', labelsize=25)
        plt.tight_layout()

        # Save the bar plot, using the custom name if provided
        if custom_name is not None:
            suffix = custom_name
        else:
            suffix = '_'.join(dataset_names)
        plt.savefig(save_path+f'bars_{metric}_{suffix}.{format}',dpi=dpi)
        plt.show()

    @staticmethod
    def heatmap(zsm, dataset_name: str, model_name: str, rows: str, cbar_label: str,
                save_path: str = '../figures/landscapes/', figsize: Tuple = (60, 25), colors: list = ['#516f84', 'white', '#df9966'], normalize = False,
                 alpha: float = 0.5, dpi: int = 300, format: str = 'png'):
        """
        Generate a heatmap of the landscape for a specific model and dataset. The plot will be saved as a PNG file.

        Args:
            zsm (ZeroShotModeller): The ZeroShotModeller object.
            dataset_name (str): The name of the dataset to process.
            model_name (str): The name of the model to generate the heatmap for.
            rows (int): The number of rows to split the protein sequence into.
            cbar_label (str): The label for the colorbar.
            save_path (str): The path to save the heatmap. Default is '../figures/landscapes'.
            figsize (Tuple): The figure size. Default is (60, 25).
            colors (list): The colors for the custom colormap. Default is ['#516f84', 'white', '#df9966'].
            normalize (bool): Whether to normalize the colors of the heatmap. Default is False.
            alpha (float): The transparency of the heatmap. Default is 0.5.
            dpi (int): The resolution of the saved image. Default is 300.
            format (str): The format of the saved image. Default is 'png'.
        """

        # Create the custom colormap
        diverging_cmap = LinearSegmentedColormap.from_list("custom_diverging", colors)

        # Extract the landscape and wildtype score
        landscape = zsm.landscape[dataset_name][model_name]
        wt_score = zsm.wildtype[dataset_name][model_name][model_name]

        # Initialize the figure and axes
        fig, axes = plt.subplots(figsize=figsize, nrows=rows, ncols=1)
        seq = zsm.wildtype[dataset_name]['exp_data'].mutated_sequence

        # Split the protein sequence into the specified number of rows
        prot_len = len(seq)
        end_points = [np.ceil(prot_len/rows)*i for i in range(1, rows+1)]

        # Iterate over the rows
        for i, end_point in enumerate(end_points):

            # Set the end point for the last row
            if end_point > prot_len:
                end_point = prot_len
            
            # Plot heatmap
            if normalize:
                norm = TwoSlopeNorm(vmin=np.min(landscape), vcenter=wt_score, vmax=np.max(landscape))
                hm = sns.heatmap(landscape.T.iloc[:, int(i*np.ceil(prot_len/rows)):int(end_point)], cmap=diverging_cmap, alpha=alpha, 
                            ax=axes[i], cbar=False, norm=norm)
            else:
                hm = sns.heatmap(landscape.T.iloc[:, int(i*np.ceil(prot_len/rows)):int(end_point)], cmap=diverging_cmap, center=wt_score, alpha=alpha, 
                            vmin=np.min(landscape), vmax=np.max(landscape), ax=axes[i], cbar=False)
            
            # Set x-ticks and labels
            axes[i].set_xticks(np.arange(0,end_point-int(i*np.ceil(prot_len/rows))) + 0.5)
            axes[i].set_xticklabels(list(seq[int(i*np.ceil(prot_len/rows)):int(end_point)]), rotation=0, fontsize=25)
            
            # Create twin x-axis for the position numbers
            ax2 = axes[i].twiny()
            ax2.xaxis.set_ticks_position("bottom")
            ax2.xaxis.set_label_position("bottom")
            ax2.spines["bottom"].set_position(("axes", -0.06))
            ax2.set_frame_on(True)
            ax2.patch.set_visible(False)
            for sp in ax2.spines.values():
                sp.set_visible(False)
            ax2.spines["bottom"].set_visible(True)

            # Set x-ticks and labels for positions
            ax2.set_xticks(np.arange(0,end_point-int(i*np.ceil(prot_len/rows)))[4::5] + 0.5)
            ax2.set_xlim(axes[i].get_xlim())
            ax2.set_xticklabels(np.arange(int(i*np.ceil(prot_len/rows)+1),int(end_point+1))[4::5],
                                rotation=0, fontsize=25)
            
            # Set y-ticks and labels for amino acids
            axes[i].set_yticks(np.arange(0.5, 20.5, 1))
            axes[i].set_yticklabels(landscape.columns,rotation=0,fontsize=25)
            axes[i].tick_params(labelbottom=True, labeltop=False, labelleft=True, labelright=True,
                                bottom=True, top=False, left=True, right=True)
            
            # Add grid lines
            for j in range(0, 20):
                axes[i].axhline(y=j, color='k', linestyle=':', lw=0.5)
            for j in range(0, int(end_point-np.ceil(prot_len/rows)+1)):
                axes[i].axvline(x=j, color='k', linestyle=':', lw=0.3)

        # Add colorbar
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        cbar = fig.colorbar(hm.get_children()[0], cax=cbar_ax)
        cbar.set_label(cbar_label, fontsize=50, rotation=270, labelpad=75)
        cbar_ax.tick_params(labelsize=40)

        # Add figure labels and adjust y-axis label position
        axes[rows-1].set_xlabel('WT/Position', fontsize=50, labelpad=55)
        axes[rows-2].set_ylabel('Mutant', fontsize=50)
        if (rows-2) % 2 == 0:
            axes[0].yaxis.set_label_coords(-0.015, -0.1)


        # Save the heatmap
        output_name = save_path+f'landscape_{dataset_name}_{model_name}.{format}'
        plt.savefig(output_name, dpi=dpi, bbox_inches='tight')
        plt.show()

    @staticmethod
    def best_models_scatter(zsm, dataset_name: str, model_list: list,reverse: bool = False,figsize: Tuple = (9,9),
                            labels: str = None, pos_interest: list = None, pos_catalytic: list = None, colors = ['#516f84', 'white', '#df9966'], alpha: float = 0.5,
                            save_path: str = '../figures/model_comp/', dpi: int = 300, format: str = 'png'):
        """
        Generate a scatter plot comparing the scores of three models for a dataset. The plot will be saved as a PNG file.

        Args:
            zsm (ZeroShotModeller): The ZeroShotModeller object.
            dataset_name (str): The name of the dataset to process.
            model_list (list): The list of model names to compare.
            reverse (bool): Whether to reverse the order of the scatter plot. Default is False.
            figsize (Tuple): The figure size. Default is (9,9).
            labels (list): The labels for the x-axis, y-axis, and colorbar. Default is None.
            pos_interest (list): The positions of interest to highlight. Default is None.
            pos_catalytic (list): The catalytic positions to highlight. Default is None.
            colors (list): The colors for the custom colormap. Default is ['#516f84', 'white', '#df9966'].
            alpha (float): The transparency of the scatter plot. Default is 0.5.
            save_path (str): The path to save the scatter plot. Default is '../figures/model_comp/'.
            dpi (int): The resolution of the saved image. Default is 300.
            format (str): The format of the saved image. Default is 'png'.
        
        """

        # Create the colormap
        diverging_cmap = LinearSegmentedColormap.from_list("custom_diverging", colors)

        # Extract the models and wildtype scores
        model1 = zsm.datasets[dataset_name][model_list[0]][model_list[0]]
        model2 = zsm.datasets[dataset_name][model_list[1]][model_list[1]]
        model3 = zsm.datasets[dataset_name][model_list[2]][model_list[2]]
        wts = [zsm.wildtype[dataset_name][model][model] for model in model_list] # This plot requires the wildtype scores

        # Sort the models based on the third model to ensure that the positive mutants are more visible
        sort_idx = np.argsort(model3)
        if reverse: # Reverse the order if specified
            sort_idx = sort_idx[::-1]
        model1 = model1[sort_idx]
        model2 = model2[sort_idx]
        model3 = model3[sort_idx]

        # Set the colorbar limits based on the minimum and maximum values of the third model
        # Center the colorbar around the wildtype score
        if np.absolute(model3.max()-wts[2]) < np.absolute(wts[2]-model3.min()):
            model3_max = 2*wts[2]-model3.min()
            model3_min = model3.min()
        else:
            model3_max = model3.max()
            model3_min = 2*wts[2]-model3.max()

        # Initialize the figure and axes
        fig,ax = plt.subplots(figsize=figsize)
        ax.set_box_aspect(1)

        # Create the scatter plot
        sc = ax.scatter(model1,model2,s=50,alpha=alpha,edgecolors='k',c=model3,zorder=0,cmap=diverging_cmap,
                        vmin=model3_min,vmax=model3_max)
        
        # Highlight the positions of interest and catalytic positions
        positions = np.array(zsm.datasets[dataset_name]['exp_data'].Pos.to_list())[:,0] # Extract the positions
        if pos_interest is not None:
            for pos in pos_interest:

                # Find the index of the position in the dataset and plot the scatter point
                idx = zsm.datasets[dataset_name]['exp_data'].index[positions==pos]
                ax.scatter(model1.loc[idx],model2.loc[idx],s=50,alpha=1,marker='o',edgecolors='k',color='#D22B2B')
        if pos_catalytic is not None:
            for pos in pos_catalytic:

                # Find the index of the position in the dataset and plot the scatter point
                idx = zsm.datasets[dataset_name]['exp_data'].index[positions==pos]
                ax.scatter(model1.loc[idx],model2.loc[idx],s=50,alpha=1,marker='*',edgecolors='k',color='k')

        # Add the wildtype scores through a vline, hline, and colored scatter point
        ax.scatter(wts[0],wts[1],s=100,alpha=1,edgecolors='k',c='white',zorder=3)
        ax.axvline(wts[0], color='k', linestyle='--',zorder=1)
        ax.axhline(wts[1], color='k', linestyle='--',zorder=1)

        # Set the colorbar and figure properties
        ax.tick_params(labelsize=20)
        cb = fig.colorbar(sc,ax=ax,fraction=0.04569, pad=0.04)
        cb.ax.tick_params(labelsize=18)
        cb.ax.set_ylim((model3.min(),model3.max()))

        # Set the labels for the x-axis, y-axis, and colorbar, using the model names if labels are not provided
        if labels is not None:
            ax.set_xlabel(labels[0],fontsize=25,labelpad=5)
            ax.set_ylabel(labels[1],fontsize=25,labelpad=5)
            cb.set_label(labels[2],fontsize=25,rotation=270,labelpad=30)
        else:
            ax.set_xlabel(model_list[0],fontsize=25,labelpad=5)
            ax.set_ylabel(model_list[1],fontsize=25,labelpad=5)
            cb.set_label(model_list[2],fontsize=25,rotation=270,labelpad=30)

        # Save the scatter plot
        outfile = save_path + 'scatter_' + dataset_name + '_' + '_'.join(model_list)

        # If positions of interest or catalytic positions are provided, add the suffix to the filename
        if pos_interest is not None or pos_catalytic is not None:
            outfile = outfile + f'_pos.{format}'
        else:
            outfile = outfile + f'.{format}'
        plt.savefig(outfile, bbox_inches='tight', dpi=dpi)
        plt.show()