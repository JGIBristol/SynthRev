import os
import json
import pandas as pd
import numpy as np

from scipy import stats
from scipy.stats import expon, lognorm, kstest, weibull_min, gamma

import seaborn as sns
import matplotlib.pyplot as plt
# Load the data
'''
# Load the configuration file
config_file_path = 'config.json'

with open(config_file_path, 'r') as config_file:
    config = json.load(config_file)

# Get the paths from the configuration
icu_data_path = config.get('icu_data_path')
patient_data_path = config.get('patient_data_path')
admission_data_path = config.get('admission_data_path')
diagnosis_data_path = config.get('diagnosis_data_path')
diagnosis_patient_info_path = config.get('diagnosis_patient_info_path')
treatment_data_path = config.get('treatment_data_path')
treatment_patient_info_path = config.get("treatment_patient_info_path")
'''
# icu-stays

#icu_data = pd.read_csv(icu_data_path, usecols=['subject_id', 'hadm_id','los'])
icu_data = pd.read_csv(os.path.join('icustays.csv', 'icustays.csv'), usecols=['subject_id', 'hadm_id','los'])

icu_data.head()

#icu_data.shape

#patients
patient_data = pd.read_csv(os.path.join('patients.csv', 'patients.csv'))

patient_data.head()

#patient_data.shape

#admissions
admission_data = pd.read_csv(os.path.join('admissions.csv', 'admissions.csv'), usecols=['subject_id','hadm_id', 'admission_location', 'discharge_location', 'marital_status', 'race'])

admission_data.head()

#admission_data.shape

# ICD-10 Diagnosis
diag_data = pd.read_csv(os.path.join('d_icd_diagnoses.csv', 'd_icd_diagnoses.csv'))

diag_data = diag_data[diag_data['icd_version'] == 10]

diag_data.head()

# ICD-10 Patient Info
diag_patient = pd.read_csv('diagnoses_icd.csv', usecols= ['subject_id', 'hadm_id','icd_code', 'icd_version'])

diag_patient = diag_patient[diag_patient['icd_version'] == 10]

diag_patient.head()

# ICD - 10 Procedures 
procedure_data = pd.read_csv(os.path.join('d_icd_procedures.csv', 'd_icd_procedures.csv'))

procedure_data.head()

#ICD - 10 Treatment info
treat_patient = pd.read_csv('procedures_icd.csv', usecols= ['subject_id', 'hadm_id','icd_code', 'icd_version'])

treat_patient = treat_patient[treat_patient['icd_version'] == 10]

treat_patient.head()

# Creating the consolidated table

# 1. Combine the two diagnoses tables and procedures table
diagnosis = pd.merge(diag_patient, diag_data, on = ['icd_code', 'icd_version'])

diagnosis.rename(columns = {'long_title': 'diagnosis', 'icd_code': 'icd_code_d'}, inplace= True)

diagnosis.head()

treatment = pd.merge(treat_patient, procedure_data, on = ['icd_code', 'icd_version'])

treatment.rename(columns = {'long_title': 'treatment', 'icd_code': 'icd_code_p'}, inplace= True)

treatment.head()

# Combine all other tables
data = icu_data.merge(patient_data, on = 'subject_id').merge(admission_data, on =  ['subject_id', 'hadm_id'], how= 'inner').merge(diagnosis, on =  ['subject_id', 'hadm_id'], how= 'inner').merge(treatment, on =  ['subject_id', 'hadm_id'], how= 'inner')

data.head(30)

data.shape

data.nunique()
'''data1 = pd.merge(icu_data, patient_data, on = 'subject_id', how= 'inner')

data1.head(10)

data2 = pd.merge(icu_data, admission_data, on =  ['subject_id', 'hadm_id'], how= 'inner')

data2.head(10)

data3 = pd.merge(icu_data, diagnosis, on =  ['subject_id', 'hadm_id'], how= 'inner')

data3.head(10)'''

# Table with diagnosis information
data4 = icu_data.merge(patient_data, on = 'subject_id').merge(admission_data, on =  ['subject_id', 'hadm_id'], how= 'inner').merge(diagnosis, on =  ['subject_id', 'hadm_id'], how= 'inner')

data4.head(40)

data4.shape

data4.nunique()

data4.info()

data4.describe()

#Table with treatment information
data44 = icu_data.merge(patient_data, on = 'subject_id').merge(admission_data, on =  ['subject_id', 'hadm_id'], how= 'inner').merge(treatment, on =  ['subject_id', 'hadm_id'], how= 'inner')

data44.head(30)

data44.shape

data44.nunique()

# Analyze the distribution of each attribute in the data
'''for column in data4.columns:
    sns.histplot(data4[column], kde= True)
    plt.title(f'Distribution of {column}')
    plt.show()'''
# Numerical columns ie age and los
numerical = ['anchor_age', 'los']

# 1. Check how values are distributed 
for column in numerical:
    sns.histplot(data4[column], kde = True)
    plt.title(f'Distribution of {column}')
    plt.show()

# 2. Get the descriptive statistics 
for column in numerical:
    print(data4[column].describe())


# 3. Check if the columns follow a normal distribution
for column in numerical:
    stats.probplot(data4[column], dist = 'norm', plot = plt)
    plt.title(f'Distribution of {column}')
    plt.show()

# From the graphs it can be concluded that the data does not follow a normal distribution.

for column in numerical:
    params = expon.fit(data4[column])
    plt.hist(data4[column], bins = 30, density = True, alpha = 0.6, color= 'g')
    x = np.linspace(data4[column].min(), data4[column].max(), 100)
    plt.plot(x, expon.pdf(x, *params), 'r-', lw = 2)
    plt.title(f'Exponential Distribution of {column}')
    plt.show()

# Length of stay column follows an exponential distribution with a heavy skew on the right meaning more lower values

shape, loc , scale = lognorm.fit(data4['anchor_age'], floc = 0)
x = np.linspace(data4['anchor_age'].min(), data4['anchor_age'].max(), 1000)
pdf = lognorm.pdf(x, shape, loc, scale)
plt.hist(data4['anchor_age'], bins = 30, density = True, alpha = 0.6, color= 'g')
plt.plot(x, pdf, 'r-', label='Log-Normal PDF')
plt.show()

# The log-normal distribution fits the age column pretty well, age peaks around 60-70 and skews to the right

# 4. Goodness of fit test for the distributions
loc_exp, scale_exp = expon.fit(data4['los'], floc = 0)

log_normal_ks_stat, log_normal_ks_pvalue = kstest(data4['anchor_age'], 'lognorm', args=(shape, loc, scale))
print(f'Log-Normal K-S test statistic: {log_normal_ks_stat}, p-value: {log_normal_ks_pvalue}')

exp_ks_stat, exp_ks_pvalue = kstest(data4['los'], 'expon', args=(loc_exp, scale_exp))
print(f'Exponential K-S test statistic: {exp_ks_stat}, p-value: {exp_ks_pvalue}')

# Fit a Weibull distribution to the data
weibull_params = weibull_min.fit(data4['anchor_age'])

# Perform the K-S test for the Weibull distribution
weibull_ks_stat, weibull_ks_pvalue = kstest(data4['anchor_age'], 'weibull_min', args=weibull_params)
print(f'Weibull K-S test statistic: {weibull_ks_stat}, p-value: {weibull_ks_pvalue}')

# From the k-s test statistic I get conflicting results from the initial graphs ie age falls under weibull distribution and los does not fall under exponential distribution.

# Generate synthetic data for the age column based on the log-normal distribution

# Calculate the parameters for the log-normal distribution ie mean and standard deviation
log_params = {}

log_data = np.log(data4['anchor_age'])

mean_log = log_data.mean()

std_log = log_data.std()

log_params = {'mean_log': mean_log, 'std_log': std_log}

print(log_params)

# Generate the data
samples = 100

synthetic_age = np.random.lognormal(log_params['mean_log'], log_params['std_log'], samples)

synthetic_age = pd.DataFrame({column: synthetic_age})

synthetic_age.rename(columns = {'los': 'synthetic_age'}, inplace = True)

synthetic_age

# Check the correctness of the random generated data.
synthetic_age.describe()

'''shape, loc , scale = lognorm.fit(synthetic_age['synthetic_age'], floc = 0)
x = np.linspace(synthetic_age['synthetic_age'].min(), synthetic_age['synthetic_age'].max(), 1000)
pdf = lognorm.pdf(x, shape, loc, scale)
plt.hist(synthetic_age['synthetic_age'], bins = 30, density = True, alpha = 0.6, color= 'g')
plt.plot(x, pdf, 'r-', label='Log-Normal PDF')
plt.show()'''