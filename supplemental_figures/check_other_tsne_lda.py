import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from ipdb import set_trace as bp
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.patches import Patch

"""
    This script does a TSNE by medication type in the following method: 
    1. We filter data to only taking a single medication class
    2. We sample up to 10 samples per patient 
    3. We run LDA to project the latent space into 4 dimensions
    4. We run TSNE on the LDA projected space to get 2D embeddings
    5. We take one sample per patient and plot the TSNE embeddings by medication type
"""


df = pd.read_csv('../data/master_dataset.csv')
df = df[['taxonomy','filename','pid','fold','dataset'] + [col for col in df.columns if col.startswith('latent_')]].copy()

## classes to use are: TCA, SNRI, SSRI, Bupropion, Mirtazapine
df['med_type'] = df['taxonomy'].apply(lambda x: -1 if type(x) != str else -1 if 'C' in x else -1 if ',' in x else 0 if x=='NBxx' else 1 if x=='NMxx' else 2 if x.startswith('NN') else 3 if x.startswith('NS') else 4 if x.startswith('T') else -1)
df = df[df['med_type'] != -1]
## randomly sample up to 10 samples per pid, otherwise use all
sampled_dfs = []
for pid, group in df.groupby(['pid','fold']):
    n_samples = min(10, len(group))
    sampled_dfs.append(group.sample(n=n_samples, random_state=42))
df = pd.concat(sampled_dfs, ignore_index=True)

# X: shape (n_samples, latent_dim = 128)
# y: medication classes (0..4)
fig, ax = plt.subplots(1, 4, figsize=(18, 4))
for fold in range(4):
    df_fold = df[df['fold'] == fold].copy()
    ## randomly sample up to 10 samples per pid, otherwise use all
    X = df_fold[[col for col in df_fold.columns if col.startswith('latent_')]].values
    y = df_fold['med_type'].values
    ## create a mask that is a dot per pid 
    
    lda = LinearDiscriminantAnalysis(n_components=4)
    Z = lda.fit_transform(X, y)
    ## now do a tsne on the Z values
    tsne = TSNE(n_components=2, random_state=41, perplexity=30, init='pca')
    Z_tsne = tsne.fit_transform(Z)
    
    df_fold['Z_tsne_0'] = Z_tsne[:, 0]
    df_fold['Z_tsne_1'] = Z_tsne[:, 1]
    print(df_fold['dataset'].value_counts())
    df_fold = df_fold.groupby(['pid','med_type']).agg(lambda x: x.iloc[0]).reset_index()
    print(df_fold['dataset'].value_counts())
    ## now plot the Z_tsne values
    ax[fold].scatter(df_fold['Z_tsne_0'], df_fold['Z_tsne_1'], c=df_fold['med_type'], cmap='jet', s=16, alpha=0.5)
    
    ## add legend manually with patch and label as string 
    ## med_type mapping: 0=Bupropion, 1=Mirtazapine, 2=SNRI, 3=SSRI, 4=TCA
    ## Legend order: ['TCA', 'SNRI', 'SSRI', 'Bupropion', 'Mirtazapine']
    ## Map legend index to med_type value for correct color: TCA->4, SNRI->2, SSRI->3, Bupropion->0, Mirtazapine->1
    med_type_to_color = {0: 0.0, 1: 0.25, 2: 0.5, 3: 0.75, 4: 1.0}  # med_type -> normalized color value
    legend_labels = ['TCA', 'SNRI', 'SSRI', 'Bupropion', 'Mirtazapine']
    legend_med_types = [4, 2, 3, 0, 1]  # med_type values corresponding to legend order
    legend_elements = [Patch(facecolor=plt.cm.jet(med_type_to_color[med_type]), label=label) 
                       for med_type, label in zip(legend_med_types, legend_labels)]
    ax[fold].legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0), ncol=3, frameon=False, columnspacing=1.0)
    ## remove x and y ticks
    ax[fold].set_xticks([])
    ax[fold].set_yticks([])
    ax[fold].set_facecolor('whitesmoke')
    ax[fold].set_ylabel('Fold ' + str(fold), fontweight='bold')
plt.savefig('tsne_lda_by_medication_type.png', dpi=300, bbox_inches='tight')
