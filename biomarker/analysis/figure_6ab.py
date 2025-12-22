import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
    label=None,
):
    """
    Returns (ax, im) where im is the image/mesh handle (None for pure scatter).
    """
    X = np.asarray(X); y = np.asarray(y)
    X = X.astype(float, copy=False)
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
        ax.scatter(X[idx,0], X[idx,1], c=y[idx], s=s, alpha=max(0.05, alpha),
                   cmap=cmap, linewidths=0, rasterized=rasterized)
        #im = None
        ## add a colorbar to the subplot
        sm = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=np.min(y), vmax=np.max(y)), cmap=cmap)
        plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, label=label, ticks=ticks)

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
            im = ax.imshow(img, extent=extent, origin="lower", aspect="equal",
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
            im = ax.imshow(img, extent=extent, origin="lower", aspect="equal",
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
    if im is not None:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="", ticks=ticks)
    return ax, im



df = pd.read_csv('../../data/inference_v6emb_3920_all.csv')
df['filename'] = df['filename'].apply(lambda x: x.split('/')[-1])
df_latency = pd.read_csv('../../data/figure_draft_v16_rem_latency.csv')
df_powers = pd.read_csv('../../data/so_beta_powers_sleep_only.csv')
df_powers['filename'] = df_powers['concat_names'].apply(lambda x: x.split('/')[-1])


def get_power_pred_correlation(df):
    df = df.copy() 
    df = df[['filename', 'pred', 'label']]
    df_mages = pd.read_csv('all_mages.csv')
    df_mages['filename'] = df_mages['filename'].apply(lambda x: x.split('/')[-1])
    df_mages = df_mages.merge(df, on='filename', how='inner')
    y2 = df_mages['pred']
    y2 = 1 / (1 + np.exp(-y2)) 
    fig, ax = plt.subplots(3,figsize=(10, 5))
    for line in ['sleep']: #,'nrem',  'rem' , 'all']:
        subset_cols = [col for col in df_mages.columns if col.startswith(line+'_')]
        pearons_arr = np.zeros(256)
        for col in subset_cols:
            idx = int(col.split('_')[-1])
            pearons_arr[idx] = pearsonr(df_mages[col], y2)[0]
            
        ax[0].plot(pearons_arr, label=line)
        
        pearons_arr_control = np.zeros(256)
        pearons_arr_antidep = np.zeros(256)
        control_mask = df_mages['label'] == 0
        antidep_mask = df_mages['label'] == 1
        for col in subset_cols:
            idx = int(col.split('_')[-1])
            na_mask = df_mages[col].isna()
            pearons_arr_control[idx] = pearsonr(df_mages[col][control_mask & ~na_mask], y2[control_mask & ~na_mask])[0]
            pearons_arr_antidep[idx] = pearsonr(df_mages[col][antidep_mask & ~na_mask], y2[antidep_mask & ~na_mask])[0]
            
        ax[1].plot(pearons_arr_control, label=line)
        ax[2].plot(pearons_arr_antidep, label=line)
    
    ## calculate the mean so and beta in sleep and get the correlation of their sum with the prediction 
    so_cols = [col for col in df_mages.columns if col.startswith('sleep_') and int(col.split('_')[-1]) < 8]
    beta_cols = [col for col in df_mages.columns if col.startswith('sleep_') and int(col.split('_')[-1]) > 28 * 8]
    so_mean = np.mean(df_mages[so_cols], axis=1)
    beta_mean = np.mean(df_mages[beta_cols], axis=1)
    sum_mean = so_mean + beta_mean
    for i, mask in enumerate([None, control_mask, antidep_mask]):
        mask_name = ['Overall', 'Control', 'Antidepressant'][i]
        if mask is None:
            mask = np.ones(len(y2), dtype=bool) 
        else:
            mask = mask.astype(bool)
        pearons_arr_so = pearsonr(so_mean[mask], y2[mask])[0]
        pearons_arr_beta = pearsonr(beta_mean[mask], y2[mask])[0]
        pearons_arr_sum = pearsonr(sum_mean[mask], y2[mask])[0]
        print(f"{mask_name} SO mean correlation: {pearons_arr_so:.3f}")
        print(f"{mask_name} Beta mean correlation: {pearons_arr_beta:.3f}")
        print(f"{mask_name} Sum mean correlation: {pearons_arr_sum:.3f}")
    
    ax[0].set_title('Overall')
    ax[1].set_title('Control')
    ax[2].set_title('Antidepressant')
    ax[0].legend()
    
    
    ttest_control_antidep_so = ttest_ind(so_mean[control_mask], so_mean[antidep_mask])
    ttest_control_antidep_beta = ttest_ind(beta_mean[control_mask], beta_mean[antidep_mask])
    print(f"Control vs Antidepressant SO mean t-test: {ttest_control_antidep_so.pvalue:.3e} {ttest_control_antidep_so.statistic:.3f}")
    print(f"Control vs Antidepressant Beta mean t-test: {ttest_control_antidep_beta.pvalue:.3e} {ttest_control_antidep_beta.statistic:.3f}")
    return df_mages

def plot_binned_boxplots(x, y, min_x, max_x, n_bins=8, ax=None):
    FONT_SIZE = 9
    x = np.array(x)
    y = np.array(y)
    bins = np.linspace(min_x, max_x, n_bins)
    bins = np.insert(bins, 0, 0)
    bins = np.append(bins, 1000000)
    boxplots = []
    for i in range(len(bins) - 1):
        mask = (x >= bins[i]) & (x < bins[i+1])
        y_bin = y[mask]
        boxplots.append(list(y_bin))
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    # ax.boxplot(boxplots, showfliers=False)
    sns.boxplot(data=boxplots, ax=ax, showfliers=False, palette='Greens')
    pval = pearsonr(x, y)[1]
    pval = f'p<1e-10' if pval < 1e-10 else f'p = {pval:.2e}'
    label=f'Pearson r: {pearsonr(x, y)[0]:.3f} \n {pval}'
    ax.text(0.02, 0.8, label, ha='left', va='bottom', transform=ax.transAxes)
    y_pos = np.zeros(len(boxplots))
    for i in range(len(boxplots)):
        y_pos[i] = np.median(boxplots[i])
    for i in range(len(boxplots)):
        ax.text(i, y_pos[i], f'N={len(boxplots[i])}', ha='center', va='bottom', fontsize=FONT_SIZE-1)
    ax.set_xticks(np.arange(0, len(bins) - 1))
    ax.set_xticklabels([f'{bins[i]:.0f}-{bins[i+1]:.0f}' if i < len(bins) - 2 and i != 0 else f'{bins[i]:.0f}+' if i == len(bins) - 2 else f'< {bins[i+1]:.0f}' for i in range(len(bins) - 1)], fontsize=FONT_SIZE)
    ax.set_xlabel('REM Latency (min)')
    ax.set_ylabel('Average Model Score')
    ax.set_ylim(0, 1)
    return ax

def get_x_y(fold, df, df_latency, df_powers, clip=True):
    df = df.copy() 
    df_latency = df_latency.copy() 
    df_powers = df_powers.copy() 
    if fold != -1:
        df_0 = df[(df['fold'] == fold) ] # * (df['label'] == 1)]
    else:
        df_0 = df
    df_latency = df_latency[['filename', 'rem_latency_gt']]
    df_powers = df_powers[['filename', 'concat_beta_powers', 'concat_so_powers']]
    df_latency.dropna(inplace=True)
    df_powers.dropna(inplace=True)
    print('Shape of df_latency: ', df_latency.shape)
    df_ready = df_latency.merge(df_0, on='filename', how='inner')
    print('Shape of df_ready: ', df_ready.shape)
    df_ready = df_ready.merge(df_powers, on='filename', how='inner')
    print('Shape of df_ready: ', df_ready.shape)

    y2 = df_ready['pred']
    y2 = 1 / (1 + np.exp(-y2))
    dataset = df_ready['dataset']
    y3 = df_ready['label']
    if clip:
        under_60_mask = df_ready['rem_latency_gt'] < 120
        df_ready.loc[under_60_mask, 'rem_latency_gt'] = 120
        over_120_mask = df_ready['rem_latency_gt'] > 360
        df_ready.loc[over_120_mask, 'rem_latency_gt'] = 360

    # y = np.log(df_ready['rem_latency_gt'])
    y = df_ready['rem_latency_gt'] / 2
    x = df_ready[[item for item in df_ready.columns if item.startswith('latent_')]]

    return x, y, y2, y3, dataset, df_ready['concat_beta_powers'], df_ready['concat_so_powers']


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

# fig, ax = plt.subplots(4, 2, figsize=(8, 10))

## now run a tsne on the latent space 

x, y, y2, y3, dataset, beta_powers, so_powers = get_x_y(-1, df, df_latency, df_powers, clip=False)

# print('Control vs Antidepressant REM Latency: ', ttest_ind(y[y3==0], y[y3==1]))

# table_data = {
#     'Metric': ['Control', 'Antidep-\nressant', 'Overall'],
#     'Pearson \n Correlation': [pearsonr(y[y3 == 0], y2[y3 == 0])[0], pearsonr(y[y3 == 1], y2[y3 == 1])[0], pearsonr(y, y2)[0]],
#     'p-value': [pearsonr(y[y3 == 0], y2[y3 == 0])[1], pearsonr(y[y3 == 1], y2[y3 == 1])[1], pearsonr(y, y2)[1]],
# }
# table = pd.DataFrame(table_data)
# table['Pearson \n Correlation'] = table['Pearson \n Correlation'].apply(lambda x: f"{x:.3f}")
# table['p-value'] = table['p-value'].apply(lambda x: f"{x:.2e}")


print(f"Control correlation with rem latency: {pearsonr(y[y3 == 0], y2[y3 == 0])[0]:.3f}")
print(f"Antidep correlation with rem latency: {pearsonr(y[y3 == 1], y2[y3 == 1])[0]:.3f}")
print(f"Overall correlation with rem latency: {pearsonr(y, y2)[0]:.3f}")

print(f"Control correlation with beta powers: {pearsonr(y2[y3 == 0], beta_powers[y3 == 0])[0]:.3f}")
print(f"Antidep correlation with beta powers: {pearsonr(y2[y3 == 1], beta_powers[y3 == 1])[0]:.3f}")
print(f"Overall correlation with beta powers: {pearsonr(y2, beta_powers)[0]:.3f}")

print(f"Control correlation with so powers: {pearsonr(y2[y3 == 0], so_powers[y3 == 0])[0]:.3f}")
print(f"Antidep correlation with so powers: {pearsonr(y2[y3 == 1], so_powers[y3 == 1])[0]:.3f}")
print(f"Overall correlation with so powers: {pearsonr(y2, so_powers)[0]:.3f}")

print('--------------------------------')

SINGLE_FOLD = True
COLOR = 'dimgray'

if SINGLE_FOLD:
    ## make the height 6 
    fig2 = plt.figure(figsize=(6.5, 4.5))
    gs = gridspec.GridSpec(2, 2, figure=fig2)
    ax2 = np.empty((2, 2), dtype=object)
    ax2[0, 0] = fig2.add_subplot(gs[0, 0])
    ax2[0, 1] = fig2.add_subplot(gs[0, 1])
    ax2[1, 0] = fig2.add_subplot(gs[1, :])
    ax2[1, 1] = ax2[1, 0]  # Same axis spanning both columns
else:
    fig2, ax2 = plt.subplots(4, 3, figsize=(24, 24))

for fold in range(4):
    if SINGLE_FOLD and fold != 0:
        continue
    if SINGLE_FOLD:
        plot_binned_boxplots(y, y2, 30, 240, ax=ax2[1, 0])
    else:
        plot_binned_boxplots(y, y2, 30, 240, ax=ax2[fold, 2])

for fold in range(4):
    if SINGLE_FOLD and fold != 0:
        continue
    x, y, y2, y3, dataset, beta_powers, so_powers = get_x_y(fold, df, df_latency, df_powers)
    # Distance matrices
    mantel_r, p_value = mantel_correlation(x, y)
    print(f"Rem Latency Mantel correlation: {mantel_r:.3f}, p-value: {p_value:.3f}")

    tsne = TSNE(n_components=2, random_state=42, perplexity=300, init='pca')
    # x = pls_correlation(x, y)
    x_tsne = tsne.fit_transform(x)
    # print(f"R² with PLS: {r2:.3f}")
    if SINGLE_FOLD:
        tsne_progression_plot(x_tsne, y, ax=ax2[0, 1], method='points', cmap='magma', s=8, point_frac=1.0, alpha=0.6, overlay_points_frac=0.0, label='Minutes')
        ax2[0, 1].set_facecolor(COLOR)
        tsne_progression_plot(x_tsne, y2, ax=ax2[0, 0], method='points', cmap='magma', s=8, point_frac=1.0, alpha=0.6, overlay_points_frac=0.0, label='Score')
        ax2[0, 0].set_facecolor(COLOR)
        # tsne_progression_plot(x_tsne, y, ax=ax2[0, 1], method='smooth', cmap='magma')
        # tsne_progression_plot(x_tsne, y2, ax=ax2[0, 0], method='smooth', cmap='magma', ticks = np.arange(0, 1, 0.2))
    else:
        tsne_progression_plot(x_tsne, y, ax=ax2[fold, 1], method='smooth', cmap='magma')
        tsne_progression_plot(x_tsne, y2, ax=ax2[fold, 0], method='smooth', cmap='magma', ticks = np.arange(0, 1, 0.2))

plt.tight_layout()
plt.savefig('tsne_remlatency_overlay_v3.png', dpi=300, bbox_inches='tight')

