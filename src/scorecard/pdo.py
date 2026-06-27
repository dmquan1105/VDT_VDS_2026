import numpy as np

def probability_to_pdo_score(p: np.ndarray | float, base_score: float = 600, pdo: float = 50) -> np.ndarray | float:
    """
    Converts raw/calibrated probability to a PDO scorecard score (0-1000).
    A higher probability of fraud results in a lower score.
    """
    p_clipped = np.clip(p, 1e-15, 1.0 - 1e-15)
    factor = pdo / np.log(2.0)
    score = base_score + factor * np.log((1.0 - p_clipped) / p_clipped)
    return np.clip(score, 0.0, 1000.0)
