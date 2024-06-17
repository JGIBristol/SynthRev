import numpy as np
import pandas as pd

def cat_synth_gen(data: pd.DataFrame, column: str, N: int) -> pd.Series:
    """
    Generate synthetic data of size N for a given column in a dataset.
    data = your original dataset (pandas dataframe)
    column = the name of column you want to generate synthetic data for (string)
    N = the number of synthetic data points you want to generate (int)
    """
    # Replace NaNs with a placeholder
    placeholder = "ReallyUnlikelyStringToOccur3"
    input_data = data[column].copy()
    input_data = input_data.replace(np.nan, placeholder)

    # Calculate the probabilities for each unique value
    probabilities = input_data.value_counts(normalize=True).values

    # Get the unique values
    unique_values = input_data.value_counts(normalize=True).index

    # Generate synthetic data
    synth = np.random.choice(unique_values, N, p=probabilities)

    # Replace the placeholder with NaNs
    synth = pd.Series(synth).replace(placeholder, np.nan)

    # Rename the series
    synth.name = f"Synth_{column}"

    return synth


# Could add some tests for this function to see how it handles NaNs, ints and floats
