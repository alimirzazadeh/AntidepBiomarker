import sys 
sys.path.append('./')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from biomarker.analysis.figure_4c import generate_osa_figure
from biomarker.analysis.figure_4a import generate_other_medications_figure
from biomarker.analysis.figure_4e import generate_mdd_confound_figure
from biomarker.analysis.figure_4d import generate_fairness_analysis_figure
from biomarker.analysis.figure_4b import generate_cotherapy_analysis_figure

font_size = 16
def reset_font_size():
    plt.rcParams.update({
        "font.size": font_size,              # Base font size
        "axes.titlesize": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "legend.fontsize": font_size,
        "figure.titlesize": font_size,
    })

## create a gridspec with 3 elements in top row and 2 elements in bottom row
gs = gridspec.GridSpec(3, 4, figure=plt.figure(figsize=(13, 9)))

## create a figure with the gridspec
fig = plt.figure(figsize=(13, 12))

## create 3 subplots in the top row (each spanning 2 columns)
ax00 = fig.add_subplot(gs[0, 0:2])  # top left
ax01 = fig.add_subplot(gs[0, 2:4])  # top middle
ax02 = fig.add_subplot(gs[1, 0:2])  # top right

## create 2 subplots in the bottom row (each spanning 3 columns)
ax10 = fig.add_subplot(gs[1, 2:4])  # bottom left
ax10.set_xlim(ax10.get_xlim()[0] - 1, ax10.get_xlim()[1])
ax11 = fig.add_subplot(gs[2, :])  # bottom right

reset_font_size()

print('Generating mdd confound figure...')
generate_mdd_confound_figure(ax=ax11, save=False)
ax11.set_title('Robustness across Depression Severity Levels', fontsize=font_size)

reset_font_size()
print('Generating fairness analysis figure...')
generate_fairness_analysis_figure(ax=ax10, save=False, age_sex=False, bmi=True)
ax10.set_title('Robustness across BMI Levels', fontsize=font_size)

reset_font_size()
print('Generating cotherapy analysis figure...')
generate_cotherapy_analysis_figure(ax=ax01, save=False)
ax01.set_title('Robustness to Drug Co-Therapy', fontsize=font_size)

reset_font_size()
print('Generating other medications figure...')
generate_other_medications_figure(ax=ax00, save=False)
ax00.set_title('Robustness to Psychotropic and Anticholinergic Medications', fontsize=font_size)

reset_font_size()
print('Generating osa figure...')
generate_osa_figure(ax=ax02, save=False)
ax02.set_title('Robustness to Sleep Apnea', fontsize=font_size)


plt.tight_layout()
plt.subplots_adjust(hspace=0.47, wspace=0.35) ## was 0.33 before 
plt.savefig('biomarker/analysis/anonymized_figure_4.png', dpi=600, bbox_inches='tight')
print('Done')
