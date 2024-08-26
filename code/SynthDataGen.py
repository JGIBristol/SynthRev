# Synthetic Data Generation code using different options.

# Import libraries
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt

from scipy import stats
from scipy.stats import gaussian_kde
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import RMSprop, SGD
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization

import warnings
warnings.filterwarnings('ignore')

# 1. Statistical Sampling using statistical properties

def process_and_generate_synthetic_data(data):
    # Calculate mean, variance, and NaN counts for each column
    def calculate_stats(data):
        nan_counts = data.isnull().sum()
        smean = data.mean()
        svar = data.var()
        return nan_counts, smean, svar

    # Fit different distributions to the data and select the best fit
    def fit_distributions(data):
        distributions = {
            'norm': stats.norm,
            'lognorm': stats.lognorm,
            'expon': stats.expon,
            'gamma': stats.gamma,
            'beta': stats.beta,
            'weibull_min': stats.weibull_min,
            't': stats.t,
            'f': stats.f,
        }
        results = {}
        test_results = {}
        for column in data.columns:
            column_data = data[column].dropna()
            _, smean, svar = calculate_stats(column_data)  # Extract only mean and variance

            best_fit = None
            best_p_value = -1
            test_result = []

            for name, distribution in distributions.items():
                try:
                    params = distribution.fit(column_data)
                    ks_stat, p_value = stats.kstest(column_data, name, args=params)
                    test_result.append((name, ks_stat, p_value))

                    if p_value > best_p_value:
                        best_fit = (name, params)
                        best_p_value = p_value
                except Exception as e:
                    print(f"Could not fit {name} distribution to {column}: {e}")

            results[column] = best_fit
            test_results[column] = test_result

        return results, test_results

    # Generate synthetic data based on the best-fitting distributions
    def generate_synthetic_data(data, fits):
        samples = 15000
        synth_data = pd.DataFrame()

        for column in data.columns:
            dist_name, params = fits[column]
            distribution = getattr(stats, dist_name)
            synthetic_data = distribution.rvs(*params[:-2], loc=params[-2], scale=params[-1], size=samples)

            # Ensure all values are non-negative and cap columns appropriately
            if column == 'GCS Total':
                synthetic_data = np.clip(synthetic_data, a_min=0, a_max=15)
            elif column == 'anchor_age':
                synthetic_data = np.clip(synthetic_data, a_min=18, a_max=100)
            elif column == 'ART BP Systolic ':
                synthetic_data = np.clip(synthetic_data, a_min=0, a_max=251)
            elif column == 'Heart Rate':
                synthetic_data = np.clip(synthetic_data, a_min=0, a_max=186)
            elif column == 'O2 saturation pulseoxymetry':
                synthetic_data = np.clip(synthetic_data, a_min=0, a_max=100)
            elif column == 'Respiratory Rate':
                synthetic_data = np.clip(synthetic_data, a_min=0, a_max=65)
            elif column == 'Temperature Celsius':
                synthetic_data = np.clip(synthetic_data, a_min=0, a_max=46.5)
            else:
                synthetic_data = np.clip(synthetic_data, a_min=0, a_max=None)

            # Introduce NaNs based on the original NaN distribution
            nan_counts = data[column].isnull().sum()
            if nan_counts > samples:
                nan_counts = samples
            nan_indices = np.random.choice(samples, nan_counts, replace=False)
            synthetic_data[nan_indices] = np.nan

            synth_data[column] = synthetic_data

            '''In case you use the whole dataset as your sample data ie samples = len(data), use this code instead:
            # Introduce NaNs based on the original NaN distribution
                nan_counts = data[column].isnull().sum()
                nan_indices = np.random.choice(samples, nan_counts, replace=False)
                synthetic_data[nan_indices] = np.nan
                
                synth_data[column] = synthetic_data
            
            return synth_data
    '''

        return synth_data

    # Compare statistics of real and synthetic data
    def compare_stats(real_data, synth_data):
        real_nan_counts, real_mean, real_var = calculate_stats(real_data)
        synth_nan_counts, synth_mean, synth_var = calculate_stats(synth_data)

        print("Comparison of Real and Synthetic Data:\n")
        for column in real_data.columns:
            print(f"Column: {column}")

            print(f"Real Mean: {real_mean[column]}")
            print(f"Synthetic Mean: {synth_mean[column]}")
            print(f"Mean Difference: {real_mean[column] - synth_mean[column]}\n")

            print(f"Real Variance: {real_var[column]}")
            print(f"Synthetic Variance: {synth_var[column]}")
            print(f"Variance Difference: {real_var[column] - synth_var[column]}\n")

            print(f"Real NaN Count: {real_nan_counts[column]}")
            print(f"Synthetic NaN Count: {synth_nan_counts[column]}")
            print(f"NaN Count Difference: {real_nan_counts[column] - synth_nan_counts[column]}\n")
            print("-" * 50)

    # Print the results of the distribution tests and the best choice
    def print_test_results(test_results):
        for column, results in test_results.items():
            print(f"Column: {column}")
            for name, ks_stat, p_value in results:
                print(f"Distribution: {name}, KS Statistic: {ks_stat}, P-Value: {p_value}")

            best_fit = max(results, key=lambda item: item[2])
            print(f"Best Fit: {best_fit[0]}, KS Statistic: {best_fit[1]}, P-Value: {best_fit[2]}")
            print("-" * 50)

    # Visualize the combined scatter matrix of real and synthetic data
    def visualize_combined_scatter_matrix(real_data, synth_data):
        real_data['Type'] = 'Real'
        synth_data['Type'] = 'Synthetic'
        combined_data = pd.concat([real_data, synth_data], ignore_index=True)

        pairplot = sns.pairplot(combined_data, hue='Type', palette={'Real': 'blue', 'Synthetic': 'red'}, plot_kws={'alpha':0.6, 's':50}, diag_kws={'fill': True})
        pairplot.fig.suptitle(f'Combined Scatter Matrix for Real and Synthetic Data', y=1.02, fontsize=16)
        pairplot.fig.set_size_inches(14, 12)
        pairplot._legend.set_title('Data Type')

        plt.show()

    # Process the data and generate synthetic data
    fits, test_results = fit_distributions(data)
    synth_data = generate_synthetic_data(data, fits)

    # Compare statistics of real and synthetic data
    compare_stats(data, synth_data)

    # Print the results of the distribution tests
    print_test_results(test_results)

    # Visualize the combined scatter matrix
    visualize_combined_scatter_matrix(data, synth_data)

    return synth_data, test_results

# 2. KDE Sampling (Independent & Multivariate)

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

# 3. WGANs Synthetic Data

def train_wgan_and_generate_samples(data, epochs=25001, latent_dim=100, n_critic=20, clip_value=0.01, batch_size=512):
    data = data.values 
    data = data[~np.isnan(data).any(axis=1)]
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)

    # Function to randomly select a batch of samples from the dataset for training
    def sample_data(n=batch_size):
        indices = np.random.randint(0, data.shape[0], n)
        return data[indices]

    # Generator Network
    def generator(Z, output_dim=4, hsize=[32]):
        model = tf.keras.Sequential()
        model.add(Dense(hsize[0], input_dim=Z.shape[1], kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.02)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(output_dim, activation='tanh'))
        return model

    # Critic Network
    def critic(X, hsize=[32], output_dim=1, activation='linear'):
        model = tf.keras.Sequential()
        model.add(Dense(hsize[0], input_dim=X.shape[1], kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.02)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(output_dim, activation=activation))
        return model

    # Wasserstein Loss Function
    def wasserstein_loss(y_true, y_pred):
        return tf.keras.backend.mean(y_true * y_pred)

    # Custom training step with gradient clipping
    @tf.function
    def train_step(real_data, generator_model, critic_model, latent_dim, batch_size, clip_value, generator_optimizer, critic_optimizer):
        z = tf.random.normal((batch_size, latent_dim))
        
        with tf.GradientTape() as tape:
            fake_data = generator_model(z, training=True)
            real_output = critic_model(real_data, training=True)
            fake_output = critic_model(fake_data, training=True)
            d_loss_real = wasserstein_loss(-tf.ones_like(real_output), real_output)
            d_loss_fake = wasserstein_loss(tf.ones_like(fake_output), fake_output)
            d_loss = 0.5 * (d_loss_real + d_loss_fake)

        gradients = tape.gradient(d_loss, critic_model.trainable_variables)
        clipped_gradients = [tf.clip_by_value(grad, -clip_value, clip_value) for grad in gradients]
        critic_optimizer.apply_gradients(zip(clipped_gradients, critic_model.trainable_variables))

        with tf.GradientTape() as tape:
            generated_data = generator_model(z, training=True)
            validity = critic_model(generated_data, training=True)
            g_loss = wasserstein_loss(-tf.ones_like(validity), validity)
        
        gradients = tape.gradient(g_loss, generator_model.trainable_variables)
        generator_optimizer.apply_gradients(zip(gradients, generator_model.trainable_variables))
        
        return d_loss, g_loss

    # Function to visualize samples
    def visualize_samples(real_data, generated_data, epoch):
        real_df = pd.DataFrame(real_data, columns=['anchor_age', 'Heart Rate', 'O2 saturation pulseoxymetry', 'Respiratory Rate'])
        real_df['type'] = 'Real'
        
        generated_df = pd.DataFrame(generated_data, columns=['anchor_age', 'Heart Rate', 'O2 saturation pulseoxymetry', 'Respiratory Rate'])
        generated_df['type'] = 'Generated'
        
        combined_df = pd.concat([real_df, generated_df])
        
        sns.set(style="darkgrid")
        
        pairplot = sns.pairplot(combined_df, hue='type', palette={'Real': 'blue', 'Generated': 'red'}, markers=["o", "o"], plot_kws={'alpha':0.6, 's':50}, diag_kws={'shade': True})
        pairplot.fig.suptitle(f'Generated vs Real Data at Epoch {epoch}', y=1.02, fontsize=16)
        pairplot.fig.set_size_inches(14, 12)
        pairplot._legend.set_title('Data Type')

        plt.show()

    # Function to generate samples from the generator model
    def generate_samples(generator_model, latent_dim, num_samples=10000, batch_size=512):
        generated_data = []
        num_batches = num_samples // batch_size
        for _ in range(num_batches):
            z = np.random.normal(0, 1, (batch_size, latent_dim))
            batch_data = generator_model.predict(z)
            generated_data.append(batch_data)
        return np.vstack(generated_data)

    # Training parameters
    critic_model = critic(data)
    critic_optimizer = SGD(learning_rate=0.000015)

    generator_model = generator(np.random.normal(0, 1, (batch_size, latent_dim)))
    generator_optimizer = RMSprop(learning_rate=0.000012)

    # Training loop
    for epoch in range(epochs):
        for _ in range(n_critic):
            real_data = sample_data(batch_size)
            d_loss, g_loss = train_step(real_data, generator_model, critic_model, latent_dim, batch_size, clip_value, generator_optimizer, critic_optimizer)
        
        # Print the progress every 1000 epochs
        if epoch % 1000 == 0:
            print(f"{epoch} [D loss: {d_loss.numpy()}] [G loss: {g_loss.numpy()}]")
            
            # Generate and visualize samples
            z = np.random.normal(0, 1, (batch_size, latent_dim))
            generated_samples = generator_model.predict(z)
            visualize_samples(real_data, generated_samples, epoch)

    # After the final epoch, generate 10,000 samples and save to a CSV file
    num_samples = 10000
    generated_samples = generate_samples(generator_model, latent_dim, num_samples, batch_size)

    # Inverse transform the generated samples to original format
    generated_samples = scaler.inverse_transform(generated_samples)

    # Convert the generated samples to a DataFrame
    generated_samples_df = pd.DataFrame(generated_samples, columns=['anchor_age', 'Heart Rate', 'O2 saturation pulseoxymetry', 'Respiratory Rate'])

    return generated_samples_df

