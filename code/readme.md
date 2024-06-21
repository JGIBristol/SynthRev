Note what each bit of code does and who wrote it :)

# gztocsv.ipynb (Huw)
- Used for converting .csv.gz files that MIMIC IV comes in into bog standard .csv files

# CatSynthGen.py (Huw)
- Module for generating synthetic versions of categorical variables.
- Designed to integrate with pandas dataframes.
- Future ideas make it more compatible with NaN values and maybe add some testing for the functions.

# CatData.ipynb (Chakaya)
- Code for creating a consolidated dataset based on categorical features.
- Creates two seperate datasets for diagnosis and treatments.
- Generates the csv files for these two datasets, Combined_Diagnosis.csv, Combined_Treatments.csv
- Data files used: icustays.csv, patients.csv, admissions.csv, d_icd_diagnoses.csv, diagnoses.csv, d_icd_procedures.csv, procedures_icd.csv

# NumericalData.ipynb (Chakaya)
- This code extracts numerical data based on specified numerical variables such as age and heart rate.
- Generates the NumericalData.csv file
- Data files used: d_items.csv, chartevents.csv, patients.csv

# NumericalSyntheticData (Chakaya)
- _Under construction_
