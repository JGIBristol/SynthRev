# Code for benchmarking, series of functions that take in original and synthetic data and output metrics and plots

# Statistical metrics:
# Compare distributions of original and synthetic data independently

import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import numpy as np

def correlation_heatmaps(original_data: pd.DataFrame, synthetic_data: pd.DataFrame):
    """
    Calculate the correlation matrices for the original and synthetic data, and the difference between them.
    """
    OG_Corr = original_data.corr()
    Synth_Corr = synthetic_data.corr()
    Corr_Diff = abs(OG_Corr - Synth_Corr)

    return OG_Corr, Synth_Corr, Corr_Diff

def Corr_Diff_Sum(Corr_Diff: pd.DataFrame):
    """
    Sum the absolute values of the difference between the correlation matrices of the original and synthetic data.
    Scale by the number of columns squared.
    """
    return Corr_Diff.sum().sum()/Corr_Diff.shape[0]**0.5

# Still need to think on this one
def correlation_plot(original_data: pd.DataFrame, synthetic_data: pd.DataFrame):
    """
    Plot the correlation plots for the original and synthetic data overlayed in a grid
    """
    
    columns = original_data.columns.intersection(synthetic_data.columns)  # Assuming both dataframes have the same columns
    n = len(columns)
    fig, axs = plt.subplots(n, n, figsize=(16, 16), squeeze=False)

    for i in range(n):
        for j in range(n):
            ax = axs[i, j]
            if i == j:
                # Diagonal: Plot histograms
                original_data[columns[i]].plot(kind='hist', ax=ax, color='blue', alpha=0.3, density=True)
                synthetic_data[columns[i]].plot(kind='hist', ax=ax, color='red', alpha=0.3, density=True)
            else:
                # Off-diagonal: Plot scatter
                ax.scatter(original_data[columns[j]], original_data[columns[i]], alpha=0.3, color='blue', s=10)
                ax.scatter(synthetic_data[columns[j]], synthetic_data[columns[i]], alpha=0.3, color='red', s=10)
            
            # Hide axis labels and ticks if not on the edge
            if i < n - 1: ax.set_xticks([])
            if j > 0: ax.set_yticks([])
            if j == 0: ax.set_ylabel(columns[i])
            if i == n - 1: ax.set_xlabel(columns[j])

    plt.show()
    return None

# This one needs testing!
def nearest_record(original_data: pd.DataFrame, synthetic_data: pd.DataFrame):
    """
    For each record in the synthetic data, find the nearest record in the original data.
    """
    # for row in real_data.iterrows():
    return 
