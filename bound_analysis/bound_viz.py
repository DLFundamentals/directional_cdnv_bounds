import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.lines as mlines
from matplotlib.ticker import ScalarFormatter, NullLocator, FixedFormatter, FixedLocator
from scipy.stats import pearsonr

from bound_analysis.bound_core import compute_error_bound_for_m


def set_border(g):
    for spine in ['top', 'bottom', 'left', 'right']:
        g.spines[spine].set_color('black')
        g.spines[spine].set_linewidth(1)

def plot_xy(ssl_few_shot_accs_0, ssl_few_shot_accs_1, 
            nscl_few_shot_accs_0, nscl_few_shot_accs_1,
            ssl_linprob_train, ssl_linprob_test,
            nscl_linprob_train, nscl_linprob_test,
            N = [1, 5, 10, 20, 50, 100],
            output_path = None,
            dcl_cdnv_train=None, dcl_cdnv_test=None,
            dcl_dir_cdnv_train=None, dcl_dir_cdnv_test=None,
            nscl_cdnv_test=None, nscl_dir_cdnv_test=None,
            figsize=(14,11)):
    
    sns.set_theme(style="whitegrid", font_scale=3.0, rc={"xtick.bottom": True, "ytick.left": True})
    sns.set_context(rc={'patch.linewidth': 2.0})
    plt.figure(figsize=figsize)
    x = N

    error_bound_train = [compute_error_bound_for_m(dcl_dir_cdnv_train, dcl_cdnv_train, m) for m in N]
    error_bound_test = [compute_error_bound_for_m(dcl_dir_cdnv_test, dcl_cdnv_test, m) for m in N]

    inf_error_bound_train = [compute_error_bound_for_m(dcl_dir_cdnv_train, dcl_cdnv_train, 1e+6) for _ in N]
    inf_error_bound_test_dcl = [compute_error_bound_for_m(dcl_dir_cdnv_test, dcl_cdnv_test, 1e+6) for _ in N]
    inf_error_bound_test_nscl = [compute_error_bound_for_m(nscl_dir_cdnv_test, nscl_cdnv_test, 1e+6) for _ in N]

    # compute pearson coefficients\
    correlation1, p_value1 = pearsonr(error_bound_test, ssl_few_shot_accs_0)
    correlation2, p_value2 = pearsonr(error_bound_test, ssl_linprob_test)

    # Color palette
    ssl_color_0 = 'blue'
    ssl_color_1 = 'blue'
    nscl_color_0 = 'red'
    nscl_color_1 = 'red'
    ssl_linprob_color = 'green'
    nscl_linprob_color = 'orange'
    error_bound_color = 'black'
    inf_error_bound_color_dcl = 'red'
    inf_error_bound_color_nscl = 'black'

    # Plot lines
    # ================ NCCC ==================
    sns.lineplot(x=x, y=ssl_few_shot_accs_1, marker='o', alpha=1.0, color=ssl_color_1,
                 markersize=10, markeredgecolor="black", linewidth=2.0)
    
    sns.lineplot(x=x, y=ssl_few_shot_accs_0, marker='*', alpha=0.9, linestyle='--', color=ssl_color_0,
                 markersize=10, markeredgecolor="black", linewidth=2.0)

    # sns.lineplot(x=x, y=nscl_few_shot_accs_1, marker='o', alpha=1.0, color=nscl_color_1,
    #              markersize=10, markeredgecolor="black", linewidth=2.0)
    
    # sns.lineplot(x=x, y=nscl_few_shot_accs_0, marker='*', alpha=0.9, linestyle='--', color=nscl_color_0,
    #              markersize=10, markeredgecolor="black", linewidth=2.0)

    # ================ linear probe ==================
    sns.lineplot(x=x, y=ssl_linprob_train, marker='o', alpha=1.0, color=ssl_linprob_color,
                 markersize=10, markeredgecolor="black", linewidth=2.0)
    
    sns.lineplot(x=x, y=ssl_linprob_test, marker='*', alpha=0.9, linestyle='--', color=ssl_linprob_color,
                 markersize=10, markeredgecolor="black", linewidth=2.0)

    # sns.lineplot(x=x, y=nscl_linprob_train, marker='o', alpha=1.0, color=nscl_linprob_color,
    #              markersize=10, markeredgecolor="black", linewidth=2.0)
    
    # sns.lineplot(x=x, y=nscl_linprob_test, marker='*', alpha=0.9, linestyle='--', color=nscl_linprob_color,
    #              markersize=10, markeredgecolor="black", linewidth=2.0)
    
    # ================ error bounds ==================
    sns.lineplot(x=x, y=error_bound_train, marker='o', alpha=1.0, color=error_bound_color,
                 markersize=10, markeredgecolor="black", linewidth=2.0)
    
    sns.lineplot(x=x, y=error_bound_test, marker='*', alpha=0.9, linestyle='--', color=error_bound_color,
                 markersize=10, markeredgecolor="black", linewidth=2.0)
    
    # sns.lineplot(x=x, y=inf_error_bound_train, alpha=0.9, color=inf_error_bound_color,
    #              linewidth=2.0)
    
    sns.lineplot(x=x, y=inf_error_bound_test_dcl, alpha=0.9, linestyle='--', color=inf_error_bound_color_dcl,
                 linewidth=3.0)
    # sns.lineplot(x=x, y=inf_error_bound_test_nscl, alpha=0.9, linestyle='--', color=inf_error_bound_color_nscl,
    #              linewidth=3.0)

    set_border(plt.gca())

    # Custom legends
    handles = [
        # mlines.Line2D([], [], color=nscl_linprob_color, linestyle='-', label='NSCL, LP'),
        # mlines.Line2D([], [], color=nscl_color_1, linestyle='-', label="NSCL, NCCC"),
        mlines.Line2D([], [], color=ssl_color_1, linestyle='-', label="DCL, NCCC"),
        mlines.Line2D([], [], color=ssl_linprob_color, linestyle='-', label='DCL, LP'),
        mlines.Line2D([], [], color=error_bound_color, linestyle='-', label="Error bound"),
         mlines.Line2D([], [], color=inf_error_bound_color_dcl, linestyle='--', label="Lim bound"),
    ]
    style_handles = [
        mlines.Line2D([], [], color='black', linestyle='-', marker='o', label='Train', markersize=12),
        mlines.Line2D([], [], color='black', linestyle='--', marker='*', label='Test', markersize=15)
    ]

    legend1 = plt.legend(handles=handles, loc='upper right', fontsize=30)
    plt.gca().add_artist(legend1)

    # Annotate correlation
    corr_handles = [
        mlines.Line2D([], [], linestyle='none', marker='', label=r'$\rho$' + f', p-val (NCCC): {correlation1: 0.2f},{p_value1: 0.2f}'),
        mlines.Line2D([], [], linestyle='none', marker='', label=r'$\rho$' + f', p-val (LP): {correlation2: 0.2f},{p_value2: 0.2f}'),
    ]
    legend2 = plt.legend(handles=corr_handles, loc='lower left', fontsize=25, handlelength=0, handletextpad=0.4)
    legend2.get_frame().set_alpha(0.6)
    plt.gca().add_artist(legend2)

    plt.legend(handles=style_handles, bbox_to_anchor=(0.59, 1.0), fontsize=30)

    # Axes formatting
    plt.xscale('log')
    plt.yscale('log')

    # ✅ Only show ticks at desired locations
    plt.yticks([i for i in [0.01, 0.1, 0.5, 1, 10, 100]])
    plt.gca().xaxis.set_major_locator(FixedLocator(N))             # Show ticks at specific positions
    plt.gca().xaxis.set_major_formatter(FixedFormatter([str(n) for n in N]))  # Show actual numbers like "5", "20"
    
    # ✅ Remove minor grid lines and ticks that clutter x-axis
    plt.gca().xaxis.set_minor_locator(NullLocator())

    # ✅ Keep only y-axis major grid lines
    plt.grid(True, which='major', axis='y', linestyle='--', linewidth=0.8, alpha=0.9)

    # Keep regular numeric format on y-axis
    plt.gca().yaxis.set_major_formatter(ScalarFormatter())

    plt.xlabel(r"Number of shots")
    plt.ylabel(r"Error ($\log$ scale)")
    plt.tight_layout()

    plt.savefig(output_path, bbox_inches='tight')
    
    plt.show()
