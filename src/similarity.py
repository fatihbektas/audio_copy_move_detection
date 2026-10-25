import numpy as np
from scipy.stats import pearsonr


def prepare_pitch(f0, target_length=50):
    """
    Geçerli F0 noktalarını ortak uzunluğa getirir.
    Eksik kareler doğrusal enterpolasyonla tahmin edilir.
    """
    f0 = np.asarray(f0, dtype=float)
    valid = np.isfinite(f0)

    if valid.sum() < 5:
        return None

    x = np.linspace(0, 1, len(f0))
    target_x = np.linspace(0, 1, target_length)

    return np.interp(target_x, x[valid], f0[valid])


def calculate_similarity(f0_a, f0_b, target_length=50):
    a = prepare_pitch(f0_a, target_length)
    b = prepare_pitch(f0_b, target_length)

    if a is None or b is None:
        return None

    if np.std(a) < 1e-8 or np.std(b) < 1e-8:
        return None

    correlation = float(np.corrcoef(a, b)[0, 1])

    # Ham Öklid uzaklığı (Hz)
    euclidean_hz = float(np.linalg.norm(a - b))

    # Ölçekten bağımsız karşılaştırma için
    # ortak referansla normalize edilmiş RMSE
    reference = max(float(np.mean(np.concatenate([a, b]))), 1e-8)

    normalized_distance = float(np.sqrt(np.mean((a - b) ** 2)) / reference)

    return {
        "correlation": correlation,
        "euclidean_hz": euclidean_hz,
        "normalized_distance": normalized_distance,
    }
