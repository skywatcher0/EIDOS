import numpy as np

def quantize_y(y, delta_y):
    if not np.isfinite(delta_y) or delta_y <= 0:
        raise ValueError("delta_y must be finite and strictly positive")
    return np.floor(np.asarray(y, dtype=float) / delta_y + 0.5) * delta_y
