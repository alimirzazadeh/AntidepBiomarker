import pandas as pd 
from scipy.stats import pearsonr
import numpy as np 
## effect size 

df = pd.read_csv('../data/master_dataset.csv')

## merge by pid with same dosage 
df = df[['pid', 'dosage', 'mit_age']]
df = df.groupby(['pid','dosage']).agg({'mit_age': 'mean'}).reset_index()
print(df.shape)
statistic, pvalue = pearsonr(df['mit_age'], df['dosage'])
print('When including all patients with known dosage (including zero dosage) (N=', df.shape[0], ')', ', the correlation is: ', np.round(statistic, 2), 'with p-value: ', np.round(pvalue, 2))
df = df[df['dosage'] > 0]
statistic, pvalue = pearsonr(df['mit_age'], df['dosage'])
print('When only including patients with known dosage > 0, (N=', df.shape[0], ')', ', the correlation is: ', np.round(statistic, 2), 'with p-value: ', np.round(pvalue, 2))