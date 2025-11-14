# Compute per-class stats and new bound, then plot
import math
import numpy as np

def compute_new_error_bound(pairwise_metrics, m, sel_classes):
    """
    Args:
        pairwise_metrics: dict[(i,j)] → {
            'Vtilde_ij': ...,
            'Vij': ...,
            'Theta_ij': ...,
            'vi': ...,
            'vj': ...,
            'd2': ...
        }
        m: int, number of samples per class
        num_classes: int, total number of classes (C')

    Returns:
        scalar: upper bound on err_NCC^{m,D}(f)
    """
    total = 0.0
    num_pairs = 0

    for i in sel_classes:
        for j in sel_classes:
            if i == j:
                continue
            vals = pairwise_metrics.get((i, j), None)
            if vals is None:
                continue
            Vtilde = vals['Vtilde_ij']
            V = vals['Vij']
            Theta = vals['Theta_ij']
            vi, vj = vals['vi'], vals['vj']
            d2 = vals['d2']

            # --- compute A_T, A_S, A_Q ---
            A_T = (4/m) * (V**2 + 0.25 * V)
            A_S = V / m
            A_Q = (Theta + 2*(m-1)*V**2) / (m**3)

            term2 = (math.sqrt(A_T) + math.sqrt(A_S) + math.sqrt(A_Q))**2
            denom_term = (1 + (vj - vi) / (m * d2))**2

            bound_ij = (4*Vtilde + term2)/denom_term
            total += bound_ij
            num_pairs += 1

    # average_bound = total / num_pairs
    average_bound = total / len(sel_classes) 
    return average_bound
