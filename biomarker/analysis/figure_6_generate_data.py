"""
Data generation for figure_6.py.

generate_6ab_data() -- runs locally (latent_ cols are in inference CSV).
generate_6c_data()  -- requires the server (reads MAGE .npz files from
                       /data/netmit/...). Run this part on the cluster.

Outputs three small CSVs to biomarker/analysis/figure_6_data/:
    figure_6ab_tsne.csv     -- t-SNE coords + score + latency + label
    figure_6ab_boxplot.csv  -- all-patient REM latency + score
    figure_6c_power.csv     -- frequency, % power diff, 95% CI

Usage (from repo root):
    # locally (6a/b only):
    python biomarker/analysis/figure_6_generate_data.py --skip-6c

    # on server (all):
    python biomarker/analysis/figure_6_generate_data.py
"""

import os
import random
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.manifold import TSNE
from tqdm import tqdm

HERE     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, '../../data')
OUT_DIR  = os.path.join(HERE, 'figure_6_data')
os.makedirs(OUT_DIR, exist_ok=True)


# ── Utilities shared with figure_6c ──────────────────────────────────────────

def process_stages(stages):
    stages = stages.copy()
    stages[stages < 0] = 0
    stages[stages > 5] = 0
    stages = stages.astype(int)
    mapping = np.array([0, 1, 2, 3, 3, 4, 0, 0, 0, 0, 0], np.int64)
    return mapping[stages]

def get_dataset(file):
    if 'cfs'   in file: return 'cfs'
    if 'mros'  in file: return 'mros1' if 'visit1' in file else 'mros2'
    if 'wsc'   in file: return 'wsc'
    if 'shhs1' in file: return 'shhs1'
    if 'shhs2' in file: return 'shhs2'
    return ''

ALIGNED_STATS = {
    'bwh':   {'mean': -119.816, 'std': 9.759},
    'ccshs': {'mean': -121.279, 'std': 9.403},
    'cfs':   {'mean': -121.192, 'std': 9.592},
    'chat1': {'mean': -113.508, 'std': 9.082},
    'chat2': {'mean': -114.893, 'std': 9.717},
    'chat3': {'mean': -116.877, 'std': 9.435},
    'mesa':  {'mean': -122.687, 'std': 9.602},
    'mgh':   {'mean': -119.628, 'std': 9.872},
    'mgh2':  {'mean': -120.553, 'std': 9.842},
    'mros1': {'mean': -120.982, 'std': 9.363},
    'mros2': {'mean': -118.224, 'std': 10.073},
    'p18c':  {'mean': -120.749, 'std': 10.055},
    'shhs1': {'mean': -121.102, 'std': 9.976},
    'shhs2': {'mean': -121.540, 'std': 9.873},
    'sof':   {'mean': -119.194, 'std': 10.182},
    'stages':{'mean': -118.392, 'std': 9.337},
    'wsc':   {'mean': -119.986, 'std': 9.550},
}

def naive_power_post_onset(mage, stage, minutes=1_000_000, which_stage=None):
    if which_stage is None:
        which_stage = [1, 2, 3, 4]
    if mage is None:
        return None
    mage, stage = mage.copy(), stage.copy()
    is_stage = np.zeros_like(stage, dtype=bool)
    for s in which_stage:
        is_stage |= (stage == s)
    if np.sum(stage > 0) == 0:
        return None
    onset_idx = np.argwhere(stage > 0)[0][0]
    end_idx   = min(len(stage), onset_idx + 2 * minutes)
    mage[:, ~is_stage] = np.nan
    return np.nanmean(mage[:, onset_idx:end_idx], 1)

def get_mage_stage(filename, dataset, fold=0):
    filename    = filename.split('/')[-1]
    STAGE_DIR   = f'/data/netmit/wifall/ADetect/data/{dataset}/stage/'
    MAGE_DIR    = f'/data/netmit/sleep_lab/filtered/MAGE/{dataset}_new/mage/cv_{fold}/'
    GT_DIR      = f'/data/netmit/sleep_lab/filtered/MAGE/{dataset}/c4_m1_multitaper'
    if not os.path.exists(GT_DIR):
        GT_DIR = GT_DIR.replace(f'/{dataset}/', f'/{dataset}_new/')
    if not os.path.exists(GT_DIR):
        return None, None, None
    try:
        mage_gt = np.load(os.path.join(GT_DIR, filename))['data']
        stage   = np.load(os.path.join(STAGE_DIR, filename))
        stage   = stage['data'][::int(30 * stage['fs'])]
        stage   = process_stages(stage)
        mage    = np.load(os.path.join(MAGE_DIR, filename))['pred']
    except (FileNotFoundError, KeyError):
        return None, None, None
    if mage.shape[1] < 4 * 60 * 2 or len(stage) < 4 * 60 * 2:
        return None, None, None
    if len(stage) < mage.shape[1]:
        return None, None, None
    stage = stage[:mage.shape[1]]
    return mage, mage_gt[:, :mage.shape[1]], stage

def mean_percent_difference(observed, expected):
    b75  = np.nanpercentile(expected, 75, 0)
    b25  = np.nanpercentile(expected, 25, 0)
    bmin = b25 - 1.5 * 1.5 * (b75 - b25)
    bmax = b75 + 1.5 * 1.5 * (b75 - b25)
    a = (observed - bmin) / (bmax - bmin)
    b = (expected - bmin) / (bmax - bmin)
    return (np.nanmean(a, 0) - np.nanmean(b, 0)) / np.nanmean(b, 0) * 100

def bootstrap_percent_difference(observed, expected, n_bootstrap=1000, ci=95):
    diffs = np.zeros((n_bootstrap, observed.shape[1]))
    for i in tqdm(range(n_bootstrap), desc='Bootstrap 6c'):
        obs = observed[np.random.choice(observed.shape[0], observed.shape[0], replace=True)]
        exp = expected[np.random.choice(expected.shape[0], expected.shape[0], replace=True)]
        diffs[i] = mean_percent_difference(obs, exp)
    lower = np.percentile(diffs, (100 - ci) / 2,       axis=0)
    upper = np.percentile(diffs, 100 - (100 - ci) / 2, axis=0)
    return mean_percent_difference(observed, expected), lower, upper


# ── 6a/b: t-SNE + REM latency ─────────────────────────────────────────────────

def generate_6ab_data(fold=0):
    print(f'\n[6a/b] Loading data (fold {fold})...')
    df        = pd.read_csv(os.path.join(DATA_DIR, 'inference_v6emb_3920_all.csv'))
    df_lat    = pd.read_csv(os.path.join(DATA_DIR, 'figure_draft_v16_rem_latency.csv'))
    df_powers = pd.read_csv(os.path.join(DATA_DIR, 'so_beta_powers_sleep_only.csv'))

    df['filename']        = df['filename'].apply(lambda x: x.split('/')[-1])
    df_powers['filename'] = df_powers['concat_names'].apply(lambda x: x.split('/')[-1])

    latent_cols = [c for c in df.columns if c.startswith('latent_')]
    assert latent_cols, 'No latent_ columns found — run on server with full inference CSV'

    df_lat    = df_lat[['filename', 'rem_latency_gt']].dropna()
    df_powers = df_powers[['filename', 'concat_beta_powers', 'concat_so_powers']].dropna()

    def prep(fold_val):
        sub = df if fold_val == -1 else df[df['fold'] == fold_val]
        merged = df_lat.merge(sub, on='filename', how='inner')
        merged = merged.merge(df_powers, on='filename', how='inner')
        score  = 1 / (1 + np.exp(-merged['pred']))
        latency_min = merged['rem_latency_gt'].copy()
        return merged[latent_cols], latency_min, score, merged['label']

    # Fold-specific: for t-SNE scatter (clip latency)
    x, lat_min, score, label = prep(fold)
    lat_clipped = lat_min.copy()
    lat_clipped[lat_clipped < 120] = 120
    lat_clipped[lat_clipped > 360] = 360

    print(f'[6a/b] Running t-SNE on {len(x)} points...')
    tsne   = TSNE(n_components=2, random_state=42, perplexity=300, init='pca')
    x_tsne = tsne.fit_transform(x.values)

    tsne_df = pd.DataFrame({
        'tsne1':           x_tsne[:, 0],
        'tsne2':           x_tsne[:, 1],
        'model_score':     score.values,
        'rem_latency_min': lat_clipped.values,
        'label':           label.values,
    })
    tsne_path = os.path.join(OUT_DIR, 'figure_6ab_tsne.csv')
    tsne_df.to_csv(tsne_path, index=False)
    print(f'  Saved {tsne_path}  ({len(tsne_df)} rows)')

    # All folds: for boxplot (unclipped)
    _, lat_all, score_all, _ = prep(-1)
    box_df = pd.DataFrame({
        'rem_latency_min': lat_all.values,
        'model_score':     score_all.values,
    })
    box_path = os.path.join(OUT_DIR, 'figure_6ab_boxplot.csv')
    box_df.to_csv(box_path, index=False)
    print(f'  Saved {box_path}  ({len(box_df)} rows)')


# ── 6c: MAGE power spectrum ───────────────────────────────────────────────────

def generate_6c_data():
    print('\n[6c] Loading patient list...')
    datasets = ['mros', 'wsc', 'shhs', 'cfs']
    df = pd.read_csv(os.path.join(DATA_DIR, 'master_dataset.csv'))
    df = df[['filename', 'label', 'dataset', 'fold']]
    df = df[df['dataset'].isin(datasets)].groupby('filename').agg('first').reset_index()

    all_controls = df[df['label'] == 0]['filename'].tolist()
    all_antideps = df[df['label'] == 1]['filename'].tolist()

    def collect(files, desc):
        pwr_list = []
        for file in tqdm(files, desc=desc):
            ds   = get_dataset(file)
            fold = int(df[df['filename'] == file]['fold'].values[0]) \
                   if ds != 'wsc' else random.randint(0, 3)
            mg, _, st = get_mage_stage(file, ds, fold)
            if mg is None:
                continue
            pwr = naive_power_post_onset(mg, st)
            if pwr is not None and not np.any(np.isnan(pwr)) and not np.any(np.isinf(pwr)):
                pwr_list.append(pwr)
        return np.stack(pwr_list) if pwr_list else None

    antidep_pwr = collect(all_antideps, 'Antidep MAGE')
    control_pwr = collect(all_controls, 'Control MAGE')

    if antidep_pwr is None or control_pwr is None:
        print('[6c] No MAGE data found — skipping.')
        return

    print(f'[6c] Bootstrap ({antidep_pwr.shape[0]} antidep, {control_pwr.shape[0]} control)...')
    mean_diff, lower, upper = bootstrap_percent_difference(antidep_pwr, control_pwr)

    freq = np.linspace(0, 32, 256)
    power_df = pd.DataFrame({
        'freq_hz':   freq,
        'pct_diff':  mean_diff,
        'lower_ci':  lower,
        'upper_ci':  upper,
    })
    power_path = os.path.join(OUT_DIR, 'figure_6c_power.csv')
    power_df.to_csv(power_path, index=False)
    print(f'  Saved {power_path}')


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    skip_6c  = '--skip-6c'  in sys.argv
    skip_6ab = '--skip-6ab' in sys.argv
    if not skip_6ab:
        generate_6ab_data(fold=0)
    else:
        print('\n[6a/b] Skipped (--skip-6ab).')
    if not skip_6c:
        generate_6c_data()
    else:
        print('\n[6c] Skipped (--skip-6c). Run on server to generate figure_6c_power.csv.')
    print('\nAll done.')
