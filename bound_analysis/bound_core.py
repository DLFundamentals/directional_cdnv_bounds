import numpy as np

def compute_error_bound_for_m(alpha: float, beta: float, m: int) -> float:
    """
    Compute the theoretical upper bound on classification error (Proposition 1)
    for a given number of shots `m`.

    Args:
        alpha (float): Directional CDNV
        beta (float): CDNV
        m (int): Number of shots per class

    Returns:
        float: Upper bound on classification error
    """
    A = 2 + (2 ** 1.5) / m
    F = (alpha * A * m) / (np.sqrt(2) * beta)
    A_squared = A ** 2
    threshold = (8 * F) / 27

    if A_squared >= threshold:
        root_term = np.sqrt(A_squared - threshold)
        term1 = np.cbrt(4 * F * (2 * A + root_term))
        term2 = np.cbrt(4 * F * (2 * A - root_term))
        y_star = term1 + term2
    else:
        inner = (1 / 3) * np.arccos(3 * A * np.sqrt(3 / (8 * F)))
        y_star = 4 * np.sqrt((2 * F) / 3) * np.cos(inner)

    error_bound = (np.sqrt(2) * beta / (2 * A * m)) * (y_star + 2 * A) * (y_star + 4 * A)
    return error_bound
