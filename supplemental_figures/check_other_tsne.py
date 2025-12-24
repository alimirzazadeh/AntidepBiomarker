import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ipdb import set_trace as bp 
import sys 
## note: missing cfs in the rem latency csv, need to address 
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
from sklearn.manifold import TSNE
from scipy.stats import ttest_ind
import pickle
from matplotlib.patches import Patch

# COLOR= 'whitesmoke'
COLOR = 'dimgray'

def sample_group(group):
    if len(group) > 10:
        return group.sample(n=10, replace=False)
    else:
        return group  # or use replace=True if you want to upsample small groups

def tsne_progression_plot(
    X,                      # (n, 2) t-SNE or UMAP coordinates
    y,                      # (n,) progression variable
    method="smooth",        # "points" | "grid" | "smooth"
    grid_size=200,          # resolution for grid/smooth
    point_frac=0.2,         # fraction of points to plot for "points"
    min_count=3,            # mask grid cells with < this many points
    smooth_sigma=1.2,       # Gaussian sigma (in grid cells) for "smooth"
    cmap="viridis",
    overlay_points_frac=0.02,  # add a tiny scatter on top for context
    s=2,                    # scatter size for overlays
    alpha=0.25,             # alpha for scatter overlays
    contour=False,          # draw contours over grid/smooth
    contour_levels=10,
    ax=None,
    random_state=0,
    rasterized=True,
    ticks=None,
    discrete_y=False,
):
    """
    Returns (ax, im) where im is the image/mesh handle (None for pure scatter).
    """
    X = np.asarray(X); y = np.asarray(y)
    X = X.astype(float, copy=False)
    
    ## randomize the order of the points
    idx = np.random.permutation(len(y))
    X = X[idx]
    y = y[idx]
    
    if discrete_y:
        pass
    else:
        y = y.astype(float, copy=False)
        y_min, y_max = np.percentile(y, 5), np.percentile(y, 95)
        y[y > y_max] = y_max
        y[y < y_min] = y_min
    assert X.ndim == 2 and X.shape[1] == 2 and y.shape[0] == X.shape[0]

    rng = np.random.default_rng(random_state)
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5), dpi=150)

    # common bounds
    x_min, x_max = X[:,0].min(), X[:,0].max()
    y_min, y_max = X[:,1].min(), X[:,1].max()
    extent = (x_min, x_max, y_min, y_max)

    im = None

    if method == "points":
        # Subsample + alpha so dense regions saturate
        n = X.shape[0]
        k = max(1, int(point_frac * n))
        idx = rng.choice(n, size=k, replace=False)
        if discrete_y:
            im = ax.scatter(X[idx,0], X[idx,1], c=pd.Categorical(y[idx]).codes, s=s, alpha=max(0.05, alpha),
           cmap=cmap, linewidths=0, rasterized=rasterized)
            
            y_cat = pd.Categorical(y[idx])
            unique_labels = y_cat.categories
            colors = plt.cm.get_cmap(cmap)(np.linspace(0, 1, len(unique_labels)))
            
            
            legend_elements = [Patch(facecolor=colors[i], label=label) 
                            for i, label in enumerate(unique_labels)]
            # ax.legend(handles=legend_elements, loc='center left', 
            #         bbox_to_anchor=(0.8, 1.0), frameon=False)
            ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0), ncol=3, frameon=False, columnspacing=1.0)
        else:
            tsne_1 = X[idx,0]
            tsne_2 = X[idx,1]
            tsne_y = y[idx]
            sorted_order = np.argsort(tsne_y)
            sorted_idx = idx[sorted_order]
            im = ax.scatter(tsne_1[sorted_idx], tsne_2[sorted_idx], c=tsne_y[sorted_idx], s=s, alpha=max(0.05, alpha),
                    cmap=cmap, linewidths=0, rasterized=rasterized)
            

    else:
        # Build a grid
        gx = np.linspace(x_min, x_max, grid_size+1)
        gy = np.linspace(y_min, y_max, grid_size+1)
        # histogram of counts
        H, _, _ = np.histogram2d(X[:,0], X[:,1], bins=[gx, gy])
        # histogram of weighted sums for y
        Hw, _, _ = np.histogram2d(X[:,0], X[:,1], bins=[gx, gy], weights=y)

        if method == "grid":
            with np.errstate(invalid='ignore', divide='ignore'):
                mean_grid = Hw / H
            # mask sparse cells
            mean_grid = np.where(H >= min_count, mean_grid, np.nan)
            # imshow expects pixel centers; transpose because histogram2d returns [x, y]
            img = np.flipud(mean_grid.T)
            im = ax.imshow(img, extent=extent, aspect="equal",
                           cmap=cmap, interpolation="nearest", rasterized=rasterized)

        elif method == "smooth":
            # Smooth numerator and denominator separately; then take ratio
            Hs  = gaussian_filter(H,  smooth_sigma, mode="constant")
            Hws = gaussian_filter(Hw, smooth_sigma, mode="constant")
            with np.errstate(invalid='ignore', divide='ignore'):
                smooth_grid = Hws / Hs
            # light mask of ultra-empty regions
            smooth_grid = np.where(Hs > 1e-6, smooth_grid, np.nan)
            img = np.flipud(smooth_grid.T)
            im = ax.imshow(img, extent=extent, aspect="equal",
                           cmap=cmap, interpolation="bilinear", rasterized=rasterized)

        else:
            raise ValueError("method must be 'points', 'grid', or 'smooth'")

        if contour and im is not None:
            # Build a contourable array (nan-safe by using np.nan_to_num + mask)
            Z = img.copy()
            mask = np.isnan(Z)
            Zc = np.nan_to_num(Z, nan=np.nanmin(Z[~mask]) if np.any(~mask) else 0.0)
            CS = ax.contour(
                np.linspace(x_min, x_max, Zc.shape[1]),
                np.linspace(y_min, y_max, Zc.shape[0]),
                Zc, levels=contour_levels, linewidths=0.6
            )
            ax.clabel(CS, inline=True, fontsize=6, fmt="%.2g")

        # tiny point overlay for context (edges of clusters etc.)
        if overlay_points_frac and overlay_points_frac > 0:
            n = X.shape[0]
            k = max(1, int(overlay_points_frac * n))
            idx = rng.choice(n, size=k, replace=False)
            ax.scatter(X[idx,0], X[idx,1], s=s, c="k", alpha=0.15,
                       linewidths=0, rasterized=rasterized)

    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(x_min, x_max); ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal", adjustable="datalim")
    if im is not None and not discrete_y:
        # plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="", ticks=ticks)
        sm = plt.cm.ScalarMappable(norm=im.norm, cmap=im.cmap)
        plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, label="", ticks=ticks)
    return ax, im



# df = pd.read_csv('../data/inference_v6emb_3920_all.csv')
# df['filename'] = df['filename'].apply(lambda x: x.split('/')[-1])
# df_latency = pd.read_csv('../data/figure_draft_v16_rem_latency.csv')
# df_powers = pd.read_csv('../data/so_beta_powers_sleep_only.csv')
# df_powers['filename'] = df_powers['concat_names'].apply(lambda x: x.split('/')[-1])
df = pd.read_csv('../data/master_dataset.csv')
# df.dropna(subset=['filepath'], inplace=True)
# df_b = pd.read_csv('../data/inference_v6emb_3920_all.csv')
# ## get the row order using filepath column, and rearrange df as such without merging
# df_b_order = df_b['filepath'].tolist()
# df['order_idx'] = df['filepath'].apply(lambda x: df_b_order.index(x))
# df = df.sort_values(by='order_idx')
# df.drop(columns=['order_idx'], inplace=True)

from enum import Enum
class NIGHT_AGGREGATION(Enum):
    MEAN = 'mean'
    SUBSAMPLE = 'subsample'
    SINGLE_NIGHT = 'single_night'
    ALL_NIGHTS = 'all_nights'


necessary_columns = ['filename', 'pred', 'label', 'rem_latency_gt', 'sws_duration_gt', 'rem_duration_gt', 'sleep_efficiency_gt','fold','dataset','pid', 'dosage', 'taxonomy','mit_age']
necessary_columns.extend([col for col in df.columns if col.startswith('latent_')])
df = df[necessary_columns]
# df = df[df['dataset'].isin(['shhs','hchs','wsc','cfs'])].copy() 

aggregation_method = NIGHT_AGGREGATION.SUBSAMPLE
## merge all by pid , keep the columns still as pid and fold
if aggregation_method == NIGHT_AGGREGATION.MEAN:
    df = df.groupby(['pid', 'fold','taxonomy']).agg(lambda x: x.mean() if pd.api.types.is_numeric_dtype(x) else x.iloc[0]).reset_index()
elif aggregation_method == NIGHT_AGGREGATION.SUBSAMPLE:
    df = df.groupby(['pid', 'fold', 'taxonomy'], group_keys=False).apply(sample_group).reset_index(drop=True)
    # df = df.groupby(['pid', 'fold','taxonomy']).agg(lambda x: x.sample(n=10, replace=True).).reset_index()
elif aggregation_method == NIGHT_AGGREGATION.SINGLE_NIGHT:
    df = df.groupby(['pid', 'fold','taxonomy']).agg(lambda x: x.iloc[0]).reset_index()
elif aggregation_method == NIGHT_AGGREGATION.ALL_NIGHTS:
    # df = df.groupby(['pid', 'fold','taxonomy']).agg(lambda x: x.values).reset_index()
    pass 

print(df.shape)
def get_x_y(fold, df, clip=True):
    df = df.copy() 
    # df_latency = df_latency.copy() 
    # df_powers = df_powers.copy() 
    if fold != -1:
        df = df[(df['fold'] == fold) ] # * (df['label'] == 1)]
    else:
        df = df

    df_ready = df

    y2 = df_ready['pred']
    y2 = 1 / (1 + np.exp(-y2))
    dataset = df_ready['dataset']
    y3 = df_ready['label']

    # y = np.log(df_ready['rem_latency_gt'])
    y = df_ready['rem_latency_gt'] / 2
    sws_duration = df_ready['sws_duration_gt'] / 2
    rem_duration = df_ready['rem_duration_gt'] / 2
    sleep_efficiency = df_ready['sleep_efficiency_gt'] 
    dose = df_ready['dosage']
    dose[dose > 3] = 3 ## clip at 3
    taxonomy = df_ready['taxonomy'].apply(lambda x: 'SSRI' if x.startswith('NS') else 'SNRI' if x.startswith('NN') else 'TCA' if x.startswith('T') else 'Bupropion' if x.startswith('NB') else 'Mirtazapine' if x.startswith('NM') else '')
    mit_age = df_ready['mit_age']
    if clip:
        ## use .clip for 99 and 1 percentiles
        clip_percentages = [5, 95]
        # y = y.clip(np.percentile(y, clip_percentages[0]), np.percentile(y, clip_percentages[1]))
        y = y.clip(60, 180)
        sws_duration = sws_duration.clip(np.percentile(sws_duration, clip_percentages[0]), np.percentile(sws_duration, clip_percentages[1]))
        rem_duration = rem_duration.clip(np.percentile(rem_duration, clip_percentages[0]), np.percentile(rem_duration, clip_percentages[1]))
        sleep_efficiency = sleep_efficiency.clip(np.percentile(sleep_efficiency, clip_percentages[0]), np.percentile(sleep_efficiency, clip_percentages[1]))
        
        # under_60_mask = df_ready['rem_latency_gt'] < 120
        # df_ready.loc[under_60_mask, 'rem_latency_gt'] = 120
        # over_120_mask = df_ready['rem_latency_gt'] > 360
        # df_ready.loc[over_120_mask, 'rem_latency_gt'] = 360
    x = df_ready[[item for item in df_ready.columns if item.startswith('latent_')]]

    return x, y, y2, y3, dataset, sws_duration, rem_duration, sleep_efficiency, dose, taxonomy, mit_age


def mantel_correlation(x, y):
    x = np.array(x)
    y = np.array(y)
    d_tsne = squareform(pdist(x))
    d_var = squareform(pdist(np.array(y).reshape(-1, 1)))
    mantel_r, p_value = pearsonr(d_tsne.ravel(), d_var.ravel())
    return mantel_r, p_value

def pls_correlation(X, y):
    from sklearn.cross_decomposition import PLSRegression

    # Fit PLS with, say, 10 components
    pls = PLSRegression(n_components=10)
    pls.fit(X, y)

    # Transform X into PLS space
    X_pls = pls.transform(X)  # (n_samples, n_components)

    # Check how well PLS predicts y
    y_pred = pls.predict(X)
    r2 = np.corrcoef(y.ravel(), y_pred.ravel())[0, 1] ** 2
    print("R² with PLS:", r2)
    return X_pls

def load_tsne_model(fold):
    with open(f'../data/tsne_model_fold_{fold}.pkl', 'rb') as f:
        tsne = pickle.load(f)
    return tsne
# fig, ax = plt.subplots(4, 2, figsize=(8, 10))
fig2, ax2 = plt.subplots(4, 5, figsize=(20,16))
fig3, ax3 = plt.subplots(4, 4, figsize=(16,16))
## now run a tsne on the latent space 



for fold in range(4):

    x, y, y2, y3, dataset, sws_duration, rem_duration, sleep_efficiency, dose, taxonomy, mit_age = get_x_y(fold, df, clip=True)
    # Distance matrices
    tsne = TSNE(n_components=2, random_state=41, perplexity=300, init='pca')
    x_tsne = tsne.fit_transform(x)
    pos_mask = y3 == 1
    tsne_pos = TSNE(n_components=2, random_state=41, perplexity=30, init='pca')
    x_tsne_2 = tsne_pos.fit_transform(x[pos_mask])
    
    plot_columns = {'Model Score': y2, 'REM Latency': y, 'SWS Duration': sws_duration, 'REM Duration': rem_duration, 'Sleep Efficiency': sleep_efficiency, 'Dose': dose, 'Taxonomy': taxonomy, 'Dataset': dataset} #'Age': mit_age}
    
    for i, col_name in enumerate(['Model Score', 'REM Latency', 'SWS Duration', 'REM Duration', 'Sleep Efficiency']): #'Age'
        col = plot_columns[col_name]

        ax2[0, i].set_title(col_name, fontweight='bold') ## bold the title
        if col_name == 'Dataset': 
            valid_mask = ~(col.isin(['','C']) | col.str.contains(','))
            col = col[valid_mask]
            x_tsne_valid = x_tsne[valid_mask]
            
            dot_size = 6
            tsne_progression_plot(x_tsne_valid, col, ax=ax2[fold, i], method='points', cmap='jet', s=dot_size, point_frac=1.0, alpha=0.2, overlay_points_frac=0.0, discrete_y=True)
        
            ax2[fold, i].set_facecolor(COLOR)
        else:
            
            valid_mask = ~np.isnan(col) ## change to handle strings too 
            col = col[valid_mask]
            x_tsne_valid = x_tsne[valid_mask]
            # tsne_progression_plot(x_tsne_valid, col, ax=ax2[fold, i], method='smooth', cmap='magma')
            tsne_progression_plot(x_tsne_valid, col, ax=ax2[fold, i], method='points', cmap='magma', s=8, point_frac=1.0, alpha=0.6, overlay_points_frac=0.0)
            
            ax2[fold, i].set_facecolor(COLOR)

    i = 0 
    for col_name in ['Model Score', 'Dose', 'Taxonomy', 'Dataset']:
        col = plot_columns[col_name]
        ax3[0, i].set_title(col_name, fontweight='bold') ## bold the title
        col = col[pos_mask]
        
        if col_name in ['Model Score', 'Dose']:
            valid_mask = ~np.isnan(col) & (col > 0)
        else:
            valid_mask = ~(col.isin(['','C']) | col.str.contains(','))
        col = col[valid_mask]
        x_tsne_valid = x_tsne_2[valid_mask]

        if col_name == 'Dose':
            tsne_progression_plot(x_tsne_valid, col, ax=ax3[fold, i], method='points', cmap='magma', s=40, point_frac=1.0, alpha=1.0, overlay_points_frac=0.0)
        elif col_name == 'Taxonomy' or col_name == 'Dataset':
            dot_size = 16 
            tsne_progression_plot(x_tsne_valid, col, ax=ax3[fold, i], method='points', cmap='jet', s=dot_size, point_frac=1.0, alpha=0.5, overlay_points_frac=0.0, discrete_y=True)
        else:
            # tsne_progression_plot(x_tsne_valid, col, ax=ax3[fold, i], method='smooth', cmap='magma')
            tsne_progression_plot(x_tsne_valid, col, ax=ax3[fold, i], method='points', cmap='magma', s=10, point_frac=1.0, alpha=0.6, overlay_points_frac=0.0)
        ax3[fold, i].set_facecolor(COLOR)
        i += 1
    
ax2[0, 0].set_ylabel('Fold 0', fontweight='bold')
ax2[1, 0].set_ylabel('Fold 1', fontweight='bold')
ax2[2, 0].set_ylabel('Fold 2', fontweight='bold')
ax2[3, 0].set_ylabel('Fold 3', fontweight='bold')
ax3[0, 0].set_ylabel('Fold 0', fontweight='bold')
ax3[1, 0].set_ylabel('Fold 1', fontweight='bold')
ax3[2, 0].set_ylabel('Fold 2', fontweight='bold')
ax3[3, 0].set_ylabel('Fold 3', fontweight='bold')

## set the background color of all subplots to gray 
fig2.savefig('tsne_ablation_v4.png', dpi=300, bbox_inches='tight')
# fig3.savefig('tsne_ablation_v4_pos.png', dpi=300, bbox_inches='tight')

# plt.show()
sys.exit()

