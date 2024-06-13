import os
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt
# Load the data

# icu-stays
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
# 1. Age


