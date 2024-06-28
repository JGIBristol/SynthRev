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

# NumericalDataOpt.ipynb (Chakaya)
- This code is used to create numerical datasets.
- Three options are used:
  - Option 1: Using the earliest chart time for each hadm_id
  - Option 2: Picking the charttime with the fewest NaNs
  - Option 3: Picking the first reading within the hour (from the start of the first recorded time)
- Generates three csv files NumOp1.csv, NumOp2.csv and NumOp3.csv
- Data files used: d_items.csv, chartevents.csv, patients.csv
- Filtered items extracted in d_items.csv include 22021: Respiratory Rate, 220277: O2 Saturation pulseoxymetry, 225309:ART BP Systolic, 220045: Heart Rate, 20739: GCS - Eye Opening, 223900: GCS - Verbal Response, 223901: GCS - Motor Response, 223762: Temperature Celsius
- **_Note: Appropriate option for data cleaning will be identified and code will be updated accordingly_**	

  # NumericalSythenticData.ipynb (Chakaya)
  - Code to generate synthetic data based off statistical properties of the data (i.e Mean and Varaiance) and a gaussian distribution.
  - Data files used: NumOp1.csv, NumOp2.csv, NumOp3.csv
