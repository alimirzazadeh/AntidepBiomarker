import sys 
sys.path.append('./')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from supplemental_figures.osa_analysis_v2 import generate_osa_figure
from biomarker.analysis.figure_4d import generate_other_medications_figure
from supplemental_figures.check_mdd_confound_v2 import generate_mdd_confound_figure
from supplemental_figures.fairness_analysis import generate_fairness_analysis_figure
from supplemental_figures.cotherapy_analysis import generate_cotherapy_analysis_figure
plt.rcParams.update({
    "font.size": 12,              # Base font size
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 12,
})

## create a gridspec with 3 elements in top row and 2 elements in bottom row
gs = gridspec.GridSpec(2, 6, figure=plt.figure(figsize=(24, 12)))

## create a figure with the gridspec
fig = plt.figure(figsize=(24, 12))

## create 3 subplots in the top row (each spanning 2 columns)
ax00 = fig.add_subplot(gs[0, 0:2])  # top left
ax01 = fig.add_subplot(gs[0, 2:4])  # top middle
ax02 = fig.add_subplot(gs[0, 4:6])  # top right

## create 2 subplots in the bottom row (each spanning 3 columns)
ax10 = fig.add_subplot(gs[1, 0:3])  # bottom left
ax10.set_xlim(ax10.get_xlim()[0] - 1, ax10.get_xlim()[1])
ax11 = fig.add_subplot(gs[1, 3:6])  # bottom right

print('Generating mdd confound figure...')
generate_mdd_confound_figure(ax=ax11, save=False)
ax11.set_title('Robustness across Depression Severity Levels')
print('Generating fairness analysis figure...')
generate_fairness_analysis_figure(ax=ax10, save=False, age_sex=False, bmi=True)
ax10.set_title('Robustness across BMI Levels')
print('Generating cotherapy analysis figure...')
generate_cotherapy_analysis_figure(ax=ax01, save=False)
ax01.set_title('Robustness to Drug Co-Therapy')
print('Generating other medications figure...')
generate_other_medications_figure(ax=ax00, save=False)
ax00.set_title('Robustness to Psychotropic and Anticholinergic Medications')
print('Generating osa figure...')
generate_osa_figure(ax=ax02, save=False)
ax02.set_title('Robustness to Sleep Apnea')

plt.tight_layout()
plt.savefig('biomarker/analysis/figure_7_v3.png', dpi=600, bbox_inches='tight')
print('Done')
