import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.lines as mlines
from matplotlib.ticker import FuncFormatter, FixedLocator, FixedFormatter, NullLocator
from scipy.stats import pearsonr

def set_border(g):
    for spine in ['top', 'bottom', 'left', 'right']:
        g.spines[spine].set_color('black')
        g.spines[spine].set_linewidth(1)

def safe_log2(y):
    y = np.array(y, dtype=float)
    y[y <= 0] = np.nan
    return np.log2(y)

def plot_error_bound(train_error, test_error,
                     train_error_bound, test_error_bound,
                     old_train_error_bound=None,
                     old_test_error_bound=None,
                     m_values=[1, 5, 10, 20, 50, 100],
                     output_path=None,
                     figsize=(14, 11)):
    
    sns.set_theme(style="whitegrid", font_scale=3.0, rc={"xtick.bottom": True, "ytick.left": True})
    sns.set_context(rc={'patch.linewidth': 2.0})
    plt.figure(figsize=figsize)
    x = m_values

    inf_error_bound_train = [train_error_bound[int(1e6)] for _ in m_values]
    inf_error_bound_test = [test_error_bound[int(1e6)] for _ in m_values]

    train_error_bound.pop(int(1e6), None)
    test_error_bound.pop(int(1e6), None)

    train_error_bound = [train_error_bound[int(m)] for m in m_values]
    test_error_bound  = [test_error_bound[int(m)] for m in m_values]

    # compute pearson coefficients
    correlation1, p_value1 = pearsonr(train_error, train_error_bound)
    correlation2, p_value2 = pearsonr(test_error, test_error_bound)

    # Colors
    nccc_color = 'blue'
    error_bound_color = 'black'
    old_error_bound_color = 'purple'
    inf__color = 'red'

    # ✅ Transform all y-values to log2 space for uniform 2^i spacing
    train_error_log2 = safe_log2(train_error)
    test_error_log2 = safe_log2(test_error)
    train_error_bound_log2 = safe_log2(train_error_bound)
    test_error_bound_log2 = safe_log2(test_error_bound)
    inf_error_bound_train_log2 = safe_log2(inf_error_bound_train)
    inf_error_bound_test_log2 = safe_log2(inf_error_bound_test)

    # Plot
    sns.lineplot(x=x, y=train_error_log2, marker='o', color=nccc_color, linewidth=2.0, markeredgecolor="black")
    sns.lineplot(x=x, y=test_error_log2, marker='*', color=nccc_color, linestyle='--', linewidth=2.0, markeredgecolor="black")

    sns.lineplot(x=x, y=train_error_bound_log2, marker='o', color=error_bound_color, linewidth=2.0, markeredgecolor="black")
    sns.lineplot(x=x, y=test_error_bound_log2, marker='*', color=error_bound_color, linestyle='--', linewidth=2.0, markeredgecolor="black")

    sns.lineplot(x=x, y=inf_error_bound_train_log2, color=inf__color, linewidth=2.0)
    sns.lineplot(x=x, y=inf_error_bound_test_log2, color=inf__color, linestyle='--', linewidth=2.0)

    if old_train_error_bound is not None and old_test_error_bound is not None:
        old_train_error_bound_vals = [old_train_error_bound[int(m)] for m in m_values]
        old_test_error_bound_vals  = [old_test_error_bound[int(m)]  for m in m_values]
        sns.lineplot(x=x, y=safe_log2(old_train_error_bound_vals), marker='o', alpha=0.5, color=old_error_bound_color, linewidth=2.0)
        sns.lineplot(x=x, y=safe_log2(old_test_error_bound_vals), marker='*', alpha=0.5, color=old_error_bound_color, linestyle='--', linewidth=2.0)

    set_border(plt.gca())

    # ✅ Legends
    handles = [
        mlines.Line2D([], [], color=nccc_color, linestyle='-', label="NCCC"),
        mlines.Line2D([], [], color=error_bound_color, linestyle='-', label="Error bound"),
        mlines.Line2D([], [], color=inf__color, linestyle='--', label="Lim bound"),
    ]
    if old_train_error_bound is not None:
        handles.append(mlines.Line2D([], [], color=old_error_bound_color, linestyle='-', label="Old bound"))
    style_handles = [
        mlines.Line2D([], [], color='black', linestyle='-', marker='o', label='Train', markersize=12),
        mlines.Line2D([], [], color='black', linestyle='--', marker='*', label='Test', markersize=15)
    ]

    legend1 = plt.legend(handles=handles, loc='upper right', fontsize=30)
    plt.gca().add_artist(legend1)
    plt.legend(handles=style_handles, bbox_to_anchor=(0.59, 1.0), fontsize=30)

    # ✅ Correlation legend
    corr_handles = [
        mlines.Line2D([], [], linestyle='none', marker='', label=rf'$\rho$, p-val (Train): {correlation1:.2f}, {p_value1:.2f}'),
        mlines.Line2D([], [], linestyle='none', marker='', label=rf'$\rho$, p-val (Test): {correlation2:.2f}, {p_value2:.2f}')
    ]
    legend2 = plt.legend(handles=corr_handles, loc='lower left', fontsize=25, handlelength=0, handletextpad=0.4)
    legend2.get_frame().set_alpha(0.6)
    plt.gca().add_artist(legend2)

    # ✅ Axes formatting
    plt.xscale('log')
    plt.gca().xaxis.set_major_locator(FixedLocator(m_values))
    plt.gca().xaxis.set_major_formatter(FixedFormatter([str(n) for n in m_values]))
    plt.gca().xaxis.set_minor_locator(NullLocator())

    # y-axis: fixed powers of 2
    powers = np.arange(-7, 8)
    plt.gca().set_yticks(powers)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'$2^{{{int(y)}}}$'))
    plt.ylim(-7, 7)
    plt.grid(True, which='major', axis='y', linestyle='--', linewidth=0.8, alpha=0.9)

    plt.xlabel(r"Number of shots")
    plt.ylabel(r"Error ($2^i$ scale)")
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
    
    plt.show()