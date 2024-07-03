import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

def generate_kde_indep(nums, sample_num):
    """
    Generate numerical synthetic data based on independent KDE of each column.
    nums = Original numerical data as a dataframe with multiple columns, can have NaNs
    sample_num = Number of samples to generate

    Returns a dataframe with the synthetic data
    """
    samples = sample_num    
    synth_data = pd.DataFrame() # Make a new dataframe to hold the synthetic data
    
    for column in nums.columns:

        nan_counts = nums[column].isnull().sum()    # Work out how many NaNs are in the column
        kde = gaussian_kde(nums[column].dropna())   # Fit the KDE to the non-NaN data
        
        # Sample from the relevant KDE
        synthetic_data = kde.resample(samples)[0]

        # Cap GCS_Total values at 15 and ensure all values are non-negative
        if column == 'GCS Total':
            synthetic_data = np.clip(synthetic_data, a_min=0, a_max=15)
        else:
            synthetic_data = np.clip(synthetic_data, a_min=0, a_max=None)
        
        # Introduce NaNs based on the original NaN distribution
        nan_indices = np.random.choice(samples, nan_counts, replace=False)
        synthetic_data[nan_indices] = np.nan
        
        synth_data[column] = synthetic_data
    
    return synth_data

# Needs testing:
def generate_kde_multi(nums, sample_num):
    """
    Generate numerical synthetic data based on the KDE of the entire data
    nums = Original numerical data as a dataframe with multiple columns, can have NaNs
    sample_num = Number of samples to generate

    Returns a dataframe with the synthetic data
    """

    samples = sample_num    
    nan_counts = nums.isnull().sum()    # Work out how many NaNs are in each column
    kde = gaussian_kde(dataset = nums.dropna().T)   # Fit the KDE to the non-NaN data as a multivariate KDE

    synthetic_data = kde.resample(samples)
    synthetic_data_transposed = synthetic_data.T
    synth_data = pd.DataFrame(synthetic_data_transposed, columns=nums.columns)

    # Repopulate NaNs based on original distribution
    for column in nums.columns:
        nan_count = nan_counts[column]
        if nan_count > 0:  # Only proceed if there are NaNs to repopulate
            nan_indices = np.random.choice(samples, nan_count*samples/len(nums[column]), replace=False)  # Select random indices for NaNs
            synth_data.loc[nan_indices, column] = np.nan  # Assign NaNs to these indices
    
    return synth_data
