# SynthRev

Synthetic Data for Electronic Health Records, Generation and Benchmarking methods intern project summer 2024 (June 4th - August 23rd)
Code produced for this project should be stored in this repository. This repository should not be used to store any data.

## Project Info

### Project Steps

1.	Get to grips with the project problem (~1-2 weeks)
- [x]	Watch this review of Synthetic Data generation methods: https://youtu.be/KMJodqsvvtE?si=NcByvF51XbH-2WZm 
-	[x] Read this review paper: SyntheticDataReview.pdf
https://www.sciencedirect.com/science/article/pii/S0925231222004349?casa_token=PctLs_5KiZYAAAAA:VvibS1nKYZ6uZHuYKwkprs2Aah4C33lY-riaS0bwX801IyNP9pZ7Pw_rR__9quz0hp0HTe_0vQ
-	[x] Access MIMIC IV electronic health records dataset: https://physionet.org/content/mimiciv/2.2/
-	[x] Read ALSPAC Seedcorn Blog/Survey responses
(Huw will send these over when they’re available)
2.	Basic data exploration and low fidelity synthetic data generation + benchmarking (~2-4 weeks)
-	[x] Pick some subset of the dataset (decided to use ICU records initially as they're smaller than hospital records), identify some particular variables of interest and make it into a usable form. Maybe use the ALSPAC Seedcorn blog to guide you here.
-	[x] Get patient: Age, sex, ethnicity, location (home, GP, hospital/other treatment location), ICD Diagnosis Codes (suggest ICD-10 only), OPCS - 4 treatment codes
For each one, we looked at the data type and see how easy it will be to replicate the distribution of this dataset. 
- [x] For the categorical data, Huw's going to make some simple code that takes a categorical column and randomly sample from that column's values to produce a synthetic version of the categories.
- [x] Chakaya's next step is to look into numerical data for MIMIC and extract time series data with the following columns:
subject_id, hadm_id, age, chart_time, Respiratory Rate, O2 saturation pulseoxymetry, ART BP Systolic, Heart Rate, GCS Total, Temperature Celsius
If this is too difficult/computationally intensive, she could just extract the first reading for each of these variables for each hadm_id
- [ ] Chakaya will then look into writing some basic synthetic generation based on independently sampling from each column. A few suggested options below:
    1) Work out the sample mean and variance of each column individually. And then randomly sample from a Gaussian of the same mean and variance for your         synthetic.
    2) pick a collection of different continuous probability distributions (e.g. gaussian, log-normal, exponential...), set them to be the same mean and           variance as your data and pick which one fits best for each column, or least worst. Then randomly sample from that to synthesise new data.
    3) Generate a KDE for each variable and then sample from that (recommend looking at:                                     https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html)

-	[ ] Write some benchmarking code that compares the distribution between original and synthetic variables, the correlations between pairs of variables and maybe some other basic sanity checks.
    Some suggested options:
      1) Compare summary stats (mean, median, quartiles, range, outliers etc) for each variable between original and synthetic data
      2) Compare correlation coeffecients between each pair of variables in original and then in synthetic data (might want to output this as a heatmap?)
      
3.	Pick some more involved ML methods, maybe using pre-existing packages (one option is: https://github.com/vanderschaarlab/synthcity) or code your own. (Whatever time you have left!)
For each method:
- [ ] Use it generate some synthetic data
- [ ] Use some of your benchmarking methods to evaluate how good the synthesis methods are

### Other suggested acitivites 

-	[x] Attending some events for Bristol Data Week https://www.bristol.ac.uk/golding/events/data-week/
-	[ ] Attending some JGI Data Science Team Meetings (sent meeting invites on outlook)
-	[ ] Attending Data Ethics Club events https://dataethicsclub.com/ 
-	[ ] Attending (tickets are free, can attend online) HACA 2024 https://haca-conference.nhs.uk/
-	[ ] Engage with the Data Hazards project, maybe fill out a self assessment form for this sort of exercise https://datahazards.com/

### Expected Outputs

- [ ] Upload any of your code to this repo (once you finish this placement, we can make the repo public so you can share the code with other people) *This repository should not be used to store any data*
- [ ] Write a JGI blog post to talk about your experiences and your findings
- [ ] Present your work to some data scientists, perhaps from the JGI team and the JGI/iLabAfrica group!

