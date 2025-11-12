import pandas as pd 
from ipdb import set_trace as bp
import numpy as np
import datetime
df = pd.read_csv('../data/master_dataset.csv')

df = df[['pid', 'taxonomy', 'pred', 'dataset', 'date', 'dosage','label']]
df = df[df['dataset'] == 'rf']
df = df[df['label'] == 1]
df['pred'] = 1 / (1 + np.exp(-df['pred']))
## dont aggreagte
def get_variability(group):
    return np.std(group['pred'])

def calculate_night_variability(group):
    # Sort by date to ensure proper ordering
    group = group.sort_values('date')
    
    # there might be missing nights, Calculate differences between consecutive nights, only if one night apart from each other 
    group_dates = pd.to_datetime(group['date'])
    ## use shift to get the previous night
    # if False:
    #     previous_night_dates = group_dates.shift(1)
    #     next_night_mask = (group_dates - previous_night_dates).dt.days == 1
    #     # print('valid nights: ', next_night_mask.sum(), 'total nights: ', len(group_dates))
    #     night_to_night_diff = group['pred'].diff()[next_night_mask]
    #     variability = night_to_night_diff.std()
    #     mean_diff = night_to_night_diff.abs().mean()
    # else:
    variability = group['pred'].std()
    mean_diff = group['pred'].abs().mean()
    
    # Calculate variance of these differences (excluding the first NaN)
    ## get the mean diff too 

    return variability, mean_diff

# Apply to each group
variability_results = df.groupby(['pid', 'taxonomy', 'dosage']).apply(lambda x: calculate_night_variability(x), include_groups=False).reset_index()

# Extract tuple values and create separate columns
# The result column is the one that's not in the grouping columns
grouping_cols = ['pid', 'taxonomy', 'dosage']
result_col = [col for col in variability_results.columns if col not in grouping_cols][0]
variability_df = variability_results.copy()
variability_df[['night_variability', 'mean_diff']] = pd.DataFrame(variability_results[result_col].tolist(), index=variability_df.index)
variability_df = variability_df.drop(columns=[result_col])

## pring the mean and std of the variability
print(f'Mean std: {variability_df["night_variability"].mean():.4f}, Std std: {variability_df["night_variability"].std():.4f}')
print(f'Median (IQR): {variability_df["night_variability"].median():.4f} ({variability_df["night_variability"].quantile(0.25):.4f}-{variability_df["night_variability"].quantile(0.75):.4f})')
print('done')
bp() 