import pandas as pd 
from scipy.stats import pearsonr
import numpy as np 
from scipy.stats import ttest_ind
from ipdb import set_trace as bp
## effect size 

df = pd.read_csv('../data/master_dataset.csv')

## merge by pid with same dosage 

df = df[['pid', 'dosage', 'mit_age','taxonomy']]
df = df.groupby(['pid','dosage']).agg({'mit_age': 'mean', 'taxonomy': 'first'}).reset_index()
print(df.shape)
statistic, pvalue = pearsonr(df['mit_age'], df['dosage'])
print('When including all patients with known dosage (including zero dosage) (N=', df.shape[0], ')', ', the correlation is: ', np.round(statistic, 2), 'with p-value: ', np.round(pvalue, 2))
df = df[df['dosage'] > 0]
statistic, pvalue = pearsonr(df['mit_age'], df['dosage'])
print('When only including patients with known dosage > 0, (N=', df.shape[0], ')', ', the correlation is: ', np.round(statistic, 2), 'with p-value: ', np.round(pvalue, 2))

df = df.dropna(subset=['mit_age', 'dosage'])
df_drug_over_65 = df[df['mit_age'] >= 65]
df_drug_under_65 = df[df['mit_age'] < 65]
test_result = ttest_ind(df_drug_under_65['dosage'].values, df_drug_over_65['dosage'].values)
print(f'\n \n\n\nOverall t-test: {test_result.pvalue:.3e} {test_result.statistic:.3f}\n')
print(f"N={df_drug_under_65.shape[0]} vs {df_drug_over_65.shape[0]}, Mean={df_drug_under_65['dosage'].mean():.2f} under 65 vs {df_drug_over_65['dosage'].mean():.2f} over 65")
## version 2: for each drug, calculate the dosage under and over 65 years old
for drug in df['taxonomy'].unique():
    if ',' in drug:
        continue 
    df_drug = df[df['taxonomy'].str.contains(drug)].copy()
    
    df_drug_under_65 = df_drug[df_drug['mit_age'] < 65]
    df_drug_over_65 = df_drug[df_drug['mit_age'] >= 65]
    if df_drug_under_65.shape[0] <=1 or df_drug_over_65.shape[0] <=1:
        continue 
    print(drug)
    test_result = ttest_ind(df_drug_under_65['dosage'].values, df_drug_over_65['dosage'].values)
    print(f'{drug} t-test: {test_result.pvalue:.3e} {test_result.statistic:.3f}, N={df_drug_under_65.shape[0]} vs {df_drug_over_65.shape[0]}')