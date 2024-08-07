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
        
        kde = gaussian_kde(nums[column].dropna())   # Fit the KDE to the non-NaN data

        clip_min = nums[column].min()  # Set the minimum value to the minimum value in the original data
        clip_max = nums[column].max()  # Set the maximum value to the maximum value in the original data
        
        # Sample from the relevant KDE
        synthetic_data = kde.resample(samples)[0]
        synthetic_data = np.clip(synthetic_data, a_min=clip_min, a_max=clip_max)  # Clip the synthetic data to the original data range
                
        # Calculate the proportion of NaNs in the original data
        nan_proportion = np.isnan(nums[column]).mean()

        # Determine the number of NaNs to introduce in the synthetic data
        nan_counts = int(nan_proportion * sample_num)

        # Introduce NaNs based on the calculated proportion
        nan_indices = np.random.choice(sample_num, nan_counts, replace=False)
        synthetic_data.flat[nan_indices] = np.nan
        
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
        # Calculate the proportion of NaNs in the original data
        nan_proportion = nums[column].isna().mean()

        # Determine the number of NaNs to introduce in the synthetic data
        nan_counts = int(nan_proportion * sample_num)

        if nan_counts > 0:  # Only proceed if there are NaNs to repopulate
            # Introduce NaNs based on the calculated proportion
            nan_indices = np.random.choice(sample_num, nan_counts, replace=False)
            synth_data.loc[nan_indices, column] = np.nan  # Assign NaNs to these indices
    
    return synth_data
