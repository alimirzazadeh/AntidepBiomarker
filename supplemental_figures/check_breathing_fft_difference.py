"""
Compare FFT of breathing signal during sleep between antidepressant and control cohorts.
Breathing: 10 Hz. Stage: 1/30 Hz (one value per 30 s). Aligned: 300 breathing samples per stage epoch.
"""
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats
from scipy.signal import stft
from ipdb import set_trace as bp

# --- Config ---
datasets = ['mros', 'wsc', 'shhs', 'cfs']
BREATHING_FS = 10   # Hz
FREQ_RESOLUTION = 0.001  # Hz per bin -> nperseg = fs / df for STFT
NPERSEG = int(BREATHING_FS / FREQ_RESOLUTION)  # 10000 samples = 1000 s window
STAGE_EPOCH_SEC = 30
SAMPLES_PER_STAGE_EPOCH = BREATHING_FS * STAGE_EPOCH_SEC  # 300
STAGE_PREFIX = '/data/netmit/wifall/ADetect/data/{dataset}/stage/'
BREATHING_PREFIX = '/data/netmit/wifall/ADetect/data/{dataset}/thorax/'  # adjust if needed
SLEEP_STAGES = [1, 2, 3, 4]  # N1, N2, N3, REM (during sleep, exclude wake/unknown)
MIN_EPOCHS = 4 * 60 * 2   # at least 4 hours (same as original)

df = pd.read_csv('../data/master_dataset.csv')
df = df[['filename', 'label', 'dataset', 'fold']]
df = df[df['dataset'].isin(datasets)].groupby('filename').agg('first').reset_index()

all_controls = df[df['label'] == 0]['filename'].tolist()
all_antideps = df[df['label'] == 1]['filename'].tolist()


def get_dataset(file):
    if 'cfs' in file:
        return 'cfs'
    if 'mros' in file:
        return 'mros1' if 'visit1' in file else 'mros2'
    if 'wsc' in file:
        return 'wsc'
    if 'shhs1' in file:
        return 'shhs1'
    if 'shhs2' in file:
        return 'shhs2'
    return ''


def process_stages(stages):
    stages = np.asarray(stages, dtype=float)
    stages[stages < 0] = 0
    stages[stages > 5] = 0
    stages = stages.astype(int)
    mapping = np.array([0, 1, 2, 3, 3, 4, 0, 0, 0, 0, 0], dtype=np.int64)
    return mapping[stages]


def load_breathing_and_stage(filename, dataset):
    """Load breathing (10 Hz) and stage (1/30 Hz). Return (breathing_during_sleep, n_epochs_used) or (None, 0)."""
    fname = filename.split('/')[-1]
    stage_path = os.path.join(STAGE_PREFIX.format(dataset=dataset), fname)
    breath_path = os.path.join(BREATHING_PREFIX.format(dataset=dataset), fname)

    if not os.path.exists(stage_path) or not os.path.exists(breath_path):
        return None, 0

    stage_obj = np.load(stage_path, allow_pickle=True)
    stage = stage_obj['data'][::int(30 * stage_obj['fs'])]
    stage = process_stages(stage)

    breath_obj = np.load(breath_path, allow_pickle=True)
    breathing = breath_obj['data'].flatten() if hasattr(breath_obj, 'keys') and 'data' in breath_obj.files else np.asarray(breath_obj).flatten()

    n_epochs = min(len(stage), len(breathing) // SAMPLES_PER_STAGE_EPOCH)
    if n_epochs < MIN_EPOCHS:
        return None, 0

    stage = stage[:n_epochs]
    breathing = breathing[:n_epochs * SAMPLES_PER_STAGE_EPOCH]

    is_sleep = np.isin(stage, SLEEP_STAGES)
    if not np.any(is_sleep):
        return None, 0

    # Index breathing by stage: epoch i -> breathing[i*300 : (i+1)*300]
    sleep_breathing = []
    for i in range(n_epochs):
        if is_sleep[i]:
            start = i * SAMPLES_PER_STAGE_EPOCH
            sleep_breathing.append(breathing[start : start + SAMPLES_PER_STAGE_EPOCH])
    sleep_breathing = np.concatenate(sleep_breathing, axis=0)
    return sleep_breathing, np.sum(is_sleep)


def compute_spectrum_stft(signal, fs=BREATHING_FS, nperseg=NPERSEG):
    """
    STFT with fixed frequency resolution (df = fs/nperseg), then average magnitude over time.
    Returns (freqs, mag) with mag shape (n_freq,) and freqs in Hz.
    """
    if len(signal) < nperseg:
        return None, None
    f, t, Zxx = stft(signal, fs=fs, nperseg=nperseg, window='hann')
    mag = np.abs(Zxx).mean(axis=1)  # average over time -> one spectrum
    return f, mag


def mean_percent_difference(observed, expected):
    """Percent difference (observed - expected) / expected * 100, per frequency bin."""
    mean_obs = np.nanmean(observed, axis=0)
    mean_exp = np.nanmean(expected, axis=0)
    denom = np.abs(mean_exp)
    denom[denom < 1e-12] = np.nan
    return (mean_obs - mean_exp) / denom * 100


def bootstrap_percent_difference(observed, expected, n_bootstrap=1000, ci=95):
    n_freq = observed.shape[1]
    boot_diffs = np.zeros((n_bootstrap, n_freq))
    for i in range(n_bootstrap):
        so = observed[np.random.choice(observed.shape[0], size=observed.shape[0], replace=True)]
        se = expected[np.random.choice(expected.shape[0], size=expected.shape[0], replace=True)]
        boot_diffs[i] = mean_percent_difference(so, se)
    mean_diff = mean_percent_difference(observed, expected)
    lower = np.nanpercentile(boot_diffs, (100 - ci) / 2, axis=0)
    upper = np.nanpercentile(boot_diffs, 100 - (100 - ci) / 2, axis=0)
    return mean_diff, lower, upper


# --- Collect FFT spectra per cohort ---
control_fft = []
antidep_fft = []
freq_axis = None

for file in tqdm(all_antideps, desc='Antidep'):
    dataset = get_dataset(file)
    if not dataset:
        continue
    fold = int(df[df['filename'] == file]['fold'].values[0]) if dataset != 'wsc' else random.randint(0, 3)
    breath_sleep, _ = load_breathing_and_stage(file, dataset)
    if breath_sleep is None or len(breath_sleep) < NPERSEG:
        continue
    freqs, mag = compute_spectrum_stft(breath_sleep)
    if mag is None:
        continue
    mask = (freqs >= 0.1) & (freqs <= 1)
    freqs, mag = freqs[mask], mag[mask]
    if freq_axis is None:
        freq_axis = freqs
    antidep_fft.append(mag)

for file in tqdm(all_controls, desc='Control'):
    dataset = get_dataset(file)
    if not dataset:
        continue
    fold = int(df[df['filename'] == file]['fold'].values[0]) if dataset != 'wsc' else random.randint(0, 3)
    breath_sleep, _ = load_breathing_and_stage(file, dataset)
    if breath_sleep is None or len(breath_sleep) < NPERSEG:
        continue
    freqs, mag = compute_spectrum_stft(breath_sleep)
    if mag is None:
        continue
    mask = (freqs >= 0.1) & (freqs <= 1)
    freqs, mag = freqs[mask], mag[mask]
    control_fft.append(mag)

if not antidep_fft or not control_fft:
    raise SystemExit('No subjects with valid breathing + stage. Check BREATHING_PREFIX and STAGE_PREFIX paths.')

antidep_fft = np.stack(antidep_fft)
control_fft = np.stack(control_fft)

assert antidep_fft.shape[1] == control_fft.shape[1] == len(freq_axis)

# --- Cohort comparison ---
mean_diff, lower, upper = bootstrap_percent_difference(antidep_fft, control_fft)

# --- Plot ---
fig, ax = plt.subplots(1, 1, figsize=(6.5, 5))
ax.plot(freq_axis, mean_diff, color='black', ls='dashed', label='Sleep (antidep − control)')
ax.fill_between(freq_axis, lower, upper, alpha=0.2, color='black')
ax.axhline(0, color='gray', ls='dotted', alpha=0.8)
# X-axis: 0–1 Hz, tick labels in BPM (1 Hz = 60 BPM)
bpm_ticks = np.arange(10, 61, 10)  # 0, 10, 20, ..., 60 BPM
ax.set_xticks(bpm_ticks / 60.0)   # Hz = BPM / 60
ax.set_xticklabels([str(b) for b in bpm_ticks])
ax.set_xlim(0.1666665, 1)
ax.set_xlabel('Breathing rate (BPM)', fontsize=12)
ax.set_ylabel('Percent difference in FFT magnitude\n(Antidepressants − Controls)', fontsize=12)
ax.set_ylim(-15, 30)
plt.tight_layout()
plt.savefig('check_breathing_fft_difference.png', dpi=300, bbox_inches='tight')
plt.close()
print('Saved check_breathing_fft_difference.png')
print('Antidep n=%d, Control n=%d' % (antidep_fft.shape[0], control_fft.shape[0]))
