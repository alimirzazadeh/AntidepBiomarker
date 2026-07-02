"""
Figure 4 (a–e) — consolidated exact p-values.
Run from the project root: python biomarker/analysis/figure_4_pvalues.py

Statistical test             : Welch's independent samples t-test (equal_var=False)
Directionality               : Two-sided
Multiple comparisons correction : None
Effect size                  : Cohen's d = (mean_a − mean_b) / sqrt((var_a + var_b) / 2)
                               Positive d → group a has the higher model score
"""
import numpy as np
import pandas as pd
import os
import sys
sys.path.insert(0, '.')
from scipy.stats import ttest_ind as _scipy_ttest, pearsonr

CSV = 'data/'
INFERENCE_FILE = os.path.join(CSV, 'inference_v6emb_3920_all.csv')
TAXONOMY_FILE  = os.path.join(CSV, 'antidep_taxonomy_all_datasets_v6.csv')
OUT = 'biomarker/analysis/figure_4_all_pvalues.txt'

# ── helpers ───────────────────────────────────────────────────────────────────
lines = []
def w(s=''): lines.append(str(s))
def sep(c='─'): w(c * 72)

def welch(a, b):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    b = np.asarray(b, float); b = b[np.isfinite(b)]
    _, p = _scipy_ttest(a, b, equal_var=False)
    sp = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    d  = (np.mean(a) - np.mean(b)) / sp if sp > 0 else np.nan
    return p, d, len(a), len(b)

def row(label, p, d, na, nb):
    w(f'  {label}')
    w(f'    N = {na} vs {nb}    p = {p:.6e}    Cohen\'s d = {d:+.4f}')
    w()

# ── shared loaders ────────────────────────────────────────────────────────────
def _base_inference():
    df = pd.read_csv(INFERENCE_FILE)
    df = df[df['dataset'].isin(['mros', 'wsc', 'rf'])].copy()
    df['filename'] = df['filepath'].apply(lambda x: x.split('/')[-1])
    df = df.groupby('filename').agg(
        lambda x: x.mean() if pd.api.types.is_numeric_dtype(x) else x.iloc[0])
    df['pred'] = 1 / (1 + np.exp(-df['pred']))
    return df

def _with_taxonomy(df):
    tax = pd.read_csv(TAXONOMY_FILE)[['filename', 'taxonomy']]
    return pd.merge(df, tax, on='filename', how='inner')


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 4a — model score by psychotropic medication type
# ═════════════════════════════════════════════════════════════════════════════
def stats_4a():
    w(); sep('═')
    w('FIGURE 4a — Model score by psychotropic medication type')
    w('Each medication group compared to Antidepressants (Welch\'s t-test, two-sided)')
    sep('═'); w()

    from biomarker.analysis.figure_4a import (
        process_mros_medications, process_wsc_medications, process_mit_medications)

    df = _base_inference()
    df['pid'] = df.apply(
        lambda x: x['pid'][1:] if x['dataset'] in ['wsc', 'mros'] else x['pid'], axis=1)
    df = _with_taxonomy(df)
    meds = pd.concat([process_mros_medications(CSV),
                      process_wsc_medications(CSV),
                      process_mit_medications(CSV)])
    df = df.merge(meds, on='filename', how='inner')
    df = df.groupby(['pid', 'label', 'benzos', 'antipsycho', 'convuls',
                     'hypnotics', 'anticholinergics', 'stimulants', 'dataset']
                    ).agg({'pred': 'mean'}).reset_index()

    antidep = df[df['label'] == 1]['pred'].values
    groups  = {
        'No Psychotropic (Control)': df[
            (df['label'] == 0) & ~df['benzos'] & ~df['antipsycho'] &
            ~df['convuls'] & ~df['hypnotics'] & ~df['anticholinergics']]['pred'].values,
        'Benzodiazepines':   df[(df['label'] == 0) & df['benzos']]['pred'].values,
        'Antipsychotics':    df[(df['label'] == 0) & df['antipsycho']]['pred'].values,
        'Anticonvulsants':   df[(df['label'] == 0) & df['convuls']]['pred'].values,
        'Hypnotics':         df[(df['label'] == 0) & df['hypnotics']]['pred'].values,
        'Anticholinergics':  df[(df['label'] == 0) & df['anticholinergics']]['pred'].values,
    }
    for name, vals in groups.items():
        p, d, na, nb = welch(vals, antidep)
        row(f'{name}  vs  Antidepressants', p, d, na, nb)


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 4b — model score by antidepressant co-therapy
# ═════════════════════════════════════════════════════════════════════════════
def stats_4b():
    w(); sep('═')
    w('FIGURE 4b — Model score by antidepressant co-therapy')
    w('Each cohort compared to Controls (Welch\'s t-test, two-sided)')
    sep('═'); w()

    from biomarker.analysis.figure_4a import (
        process_mros_medications, process_wsc_medications, process_mit_medications)

    df = _base_inference()
    df['pid'] = df.apply(
        lambda x: x['pid'][1:] if x['dataset'] in ['shhs', 'mros', 'wsc'] else x['pid'], axis=1)
    df = _with_taxonomy(df)
    meds = pd.concat([process_mros_medications(CSV),
                      process_wsc_medications(CSV),
                      process_mit_medications(CSV)])
    df = df.merge(meds, on='filename', how='inner')
    df = df.groupby(['taxonomy', 'pid', 'label', 'benzos', 'antipsycho',
                     'convuls', 'hypnotics', 'stimulants', 'dataset']
                    ).agg({'pred': 'mean'}).reset_index()

    controls = df[df['label'] == 0]['pred'].values
    cohorts  = {
        'Single Antidepressant':    df[(~df['taxonomy'].str.contains(',')) & (df['label'] != 0)]['pred'].values,
        'Multi-Antidepressant':     df[df['taxonomy'].str.contains(',')]['pred'].values,
        'Antidep + Benzodiazepine': df[(df['label'] == 1) & df['benzos']]['pred'].values,
        'Antidep + Anticonvulsant': df[(df['label'] == 1) & df['convuls']]['pred'].values,
        'Antidep + Antipsychotic':  df[(df['label'] == 1) & df['antipsycho']]['pred'].values,
        'Antidep + Hypnotic':       df[(df['label'] == 1) & df['hypnotics']]['pred'].values,
    }
    for name, vals in cohorts.items():
        if len(vals) == 0:
            w(f'  {name}: no data'); w(); continue
        p, d, na, nb = welch(vals, controls)
        row(f'{name}  vs  Controls', p, d, na, nb)


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 4c — model score by OSA severity
# ═════════════════════════════════════════════════════════════════════════════
def stats_4c():
    w(); sep('═')
    w('FIGURE 4c — Model score by OSA severity')
    sep('═'); w()

    df = _base_inference()
    df['pid'] = df.apply(
        lambda x: x['pid'][1:] if x['dataset'] in ['shhs', 'mros', 'wsc'] else x['pid'], axis=1)
    df = _with_taxonomy(df)
    for col, pfx, sub in [('is_tca','T',',T'), ('is_ntca','N',',N'),
                           ('is_snri','NN',',NN'), ('is_ssri','NS',',NS')]:
        df[col] = df['taxonomy'].apply(lambda x: 1 if x.startswith(pfx) or sub in x else 0)
    df = df[(df['is_snri']==1)|(df['is_ssri']==1)|(df['is_ntca']==1)|(df['is_tca']==1)|(df['label']==0)]

    ahis = pd.read_csv(os.path.join(CSV, 'shhs_mros_cfs_wsc_ahi.csv'))
    df = df.merge(ahis, on='filename', how='inner')
    df = df[df['pred'].notna()]
    df['osa_group'] = pd.cut(df['ahi'], bins=[0, 5, 15, 30, np.inf],
                             labels=['Normal', 'Mild OSA', 'Moderate OSA', 'Severe OSA'])
    df['Group'] = df['label'].apply(lambda x: 'Control' if x == 0 else 'Antidepressant')

    osa_labels = ['Normal', 'Mild OSA', 'Moderate OSA', 'Severe OSA']
    osa_ctrl = {}; osa_anti = {}

    w('  Primary: Control vs Antidepressant per OSA bin'); sep()
    for lbl in osa_labels:
        c = df[(df['osa_group']==lbl)&(df['Group']=='Control')]['pred'].dropna().values
        a = df[(df['osa_group']==lbl)&(df['Group']=='Antidepressant')]['pred'].dropna().values
        osa_ctrl[lbl] = c; osa_anti[lbl] = a
        if len(c) and len(a):
            p, d, na, nb = welch(c, a)
            row(f'{lbl}: Control vs Antidepressant', p, d, na, nb)

    w(); w('  Secondary: Within Antidepressant group (vs Normal Antidep)'); sep()
    for lbl in ['Mild OSA', 'Moderate OSA', 'Severe OSA']:
        if len(osa_anti[lbl]) and len(osa_anti['Normal']):
            p, d, na, nb = welch(osa_anti[lbl], osa_anti['Normal'])
            row(f'{lbl} Antidep vs Normal Antidep', p, d, na, nb)

    w(); w('  Secondary: Antidepressant OSA bins vs Normal Control'); sep()
    for lbl in ['Mild OSA', 'Moderate OSA', 'Severe OSA']:
        if len(osa_anti[lbl]) and len(osa_ctrl['Normal']):
            p, d, na, nb = welch(osa_anti[lbl], osa_ctrl['Normal'])
            row(f'{lbl} Antidep vs Normal Control', p, d, na, nb)


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 4d — model score by BMI
# ═════════════════════════════════════════════════════════════════════════════
def stats_4d():
    w(); sep('═')
    w('FIGURE 4d — Model score by BMI')
    sep('═'); w()

    df = _base_inference()
    df['pid'] = df.apply(
        lambda x: x['pid'][1:] if x['dataset'] in ['shhs', 'mros', 'wsc'] else x['pid'], axis=1)
    df = df.groupby(['pid', 'label'], as_index=False).agg(
        lambda x: x.mean() if pd.api.types.is_numeric_dtype(x) else x.iloc[0])

    bmi_labels = ['<18.5', '18.5-25', '25-30', '30-35', '35-40', '>40']
    df_bmi = df[df['mit_bmi'].notna()].copy()
    df_bmi['bmi_bin'] = pd.cut(df_bmi['mit_bmi'],
                                bins=[0, 18.5, 25, 30, 35, 40, 100],
                                labels=bmi_labels, include_lowest=True)
    df_bmi['Group'] = df_bmi['label'].apply(lambda x: 'Control' if x == 0 else 'Antidepressant')
    all_anti = df_bmi[df_bmi['Group']=='Antidepressant']['pred'].dropna().values
    all_ctrl = df_bmi[df_bmi['Group']=='Control']['pred'].dropna().values

    w('  Primary: Control vs Antidepressant per BMI bin'); sep()
    for lbl in bmi_labels:
        c = df_bmi[(df_bmi['bmi_bin']==lbl)&(df_bmi['Group']=='Control')]['pred'].dropna().values
        a = df_bmi[(df_bmi['bmi_bin']==lbl)&(df_bmi['Group']=='Antidepressant')]['pred'].dropna().values
        if len(c) and len(a):
            p, d, na, nb = welch(c, a)
            row(f'BMI {lbl}: Control vs Antidepressant', p, d, na, nb)

    w(); w('  Secondary: Each BMI bin Control vs ALL Antidepressants'); sep()
    for lbl in bmi_labels:
        c = df_bmi[(df_bmi['bmi_bin']==lbl)&(df_bmi['Group']=='Control')]['pred'].dropna().values
        if len(c):
            p, d, na, nb = welch(c, all_anti)
            row(f'BMI {lbl} Control vs ALL Antidepressants', p, d, na, nb)

    w(); w('  Secondary: Each BMI bin Antidepressant vs ALL Controls'); sep()
    for lbl in bmi_labels:
        a = df_bmi[(df_bmi['bmi_bin']==lbl)&(df_bmi['Group']=='Antidepressant')]['pred'].dropna().values
        if len(a):
            p, d, na, nb = welch(a, all_ctrl)
            row(f'BMI {lbl} Antidepressant vs ALL Controls', p, d, na, nb)


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE 4e — model score by depression severity (Zung Index, WSC only)
# ═════════════════════════════════════════════════════════════════════════════
def stats_4e():
    w(); sep('═')
    w('FIGURE 4e — Model score by depression severity (Zung Index, WSC cohort only)')
    sep('═'); w()

    wsc = pd.read_csv(os.path.join(CSV, 'wsc-dataset-0.7.0.csv'))
    wsc['filename'] = wsc.apply(
        lambda x: f'wsc-visit{x["wsc_vst"]}-{x["wsc_id"]}-nsrr.npz', axis=1)

    inf = pd.read_csv(INFERENCE_FILE)
    inf['filename'] = inf['filepath'].apply(lambda x: x.split('/')[-1])
    inf = inf.groupby('filename').agg(
        lambda x: x.mean() if pd.api.types.is_numeric_dtype(x) else x.iloc[0])
    inf['pred'] = 1 / (1 + np.exp(-inf['pred']))

    df = wsc[['zung_index', 'filename']].merge(inf, on='filename', how='inner')
    df.dropna(subset=['zung_index', 'pred'], inplace=True)
    df['group'] = df['label'].apply(lambda x: 'Control' if x == 0 else 'Antidepressant')

    zung_labels = ['<30', '30-40', '40-50', '50-60', '60+']
    df['zung_bin'] = pd.cut(df['zung_index'],
                             bins=[0, 30, 40, 50, 60, 200],
                             labels=zung_labels, include_lowest=True)
    all_pos = df[df['group']=='Antidepressant']['pred'].dropna().values
    all_neg = df[df['group']=='Control']['pred'].dropna().values

    w('  Primary: Control vs Antidepressant per Zung bin'); sep()
    for lbl in zung_labels:
        c = df[(df['zung_bin']==lbl)&(df['group']=='Control')]['pred'].dropna().values
        a = df[(df['zung_bin']==lbl)&(df['group']=='Antidepressant')]['pred'].dropna().values
        if len(c) and len(a):
            p, d, na, nb = welch(c, a)
            row(f'Zung {lbl}: Control vs Antidepressant', p, d, na, nb)

    w(); w('  Secondary: Each Zung bin Control vs ALL Antidepressants'); sep()
    for lbl in zung_labels:
        c = df[(df['zung_bin']==lbl)&(df['group']=='Control')]['pred'].dropna().values
        if len(c):
            p, d, na, nb = welch(c, all_pos)
            row(f'Zung {lbl} Control vs ALL Antidepressants', p, d, na, nb)

    w(); w('  Secondary: Each Zung bin Antidepressant vs ALL Controls'); sep()
    for lbl in zung_labels:
        a = df[(df['zung_bin']==lbl)&(df['group']=='Antidepressant')]['pred'].dropna().values
        if len(a):
            p, d, na, nb = welch(a, all_neg)
            row(f'Zung {lbl} Antidepressant vs ALL Controls', p, d, na, nb)

    w(); w('  Pearson correlations: Zung Index vs Model Score'); sep()
    df_neg = df[df['group']=='Control']
    df_pos = df[df['group']=='Antidepressant']
    for name, sub in [('Control', df_neg), ('Antidepressant', df_pos), ('Overall', df)]:
        r, p = pearsonr(sub['zung_index'], sub['pred'])
        w(f'  {name} (N={len(sub)}):  r = {r:+.4f},  p = {p:.6e}')
    w()


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
w('Figure 4 (a–e) — Consolidated exact p-values')
sep('═'); w()
w('Statistical test             : Welch\'s independent samples t-test (equal_var=False)')
w('Directionality               : Two-sided')
w('Multiple comparisons correction : None applied')
w('Effect size                  : Cohen\'s d = (mean_a − mean_b) / sqrt((var_a + var_b) / 2)')
w('                               Positive d → group a has the higher model score')

stats_4a()
stats_4b()
stats_4c()
stats_4d()
stats_4e()

with open(OUT, 'w') as fh:
    fh.write('\n'.join(lines) + '\n')

print(f'Saved {OUT}')
