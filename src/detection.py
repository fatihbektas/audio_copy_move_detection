from itertools import combinations
import pandas as pd

from src.similarity import calculate_similarity


def detect_copy_move(
    pitch_results,
    min_correlation=0.95,
    max_distance=0.10,
    target_length=50,
    min_voiced_frames=5,
    min_voiced_ratio=0.25,
):
    matches = []

    for a, b in combinations(pitch_results, 2):

        if min(a["voiced_frames"], b["voiced_frames"]) < min_voiced_frames:
            continue

        if min(a["voiced_ratio"], b["voiced_ratio"]) < min_voiced_ratio:
            continue

        similarity = calculate_similarity(a["f0"], b["f0"], target_length)

        if similarity is None:
            continue

        corr = similarity["correlation"]
        distance = similarity["normalized_distance"]

        suspicious = corr >= min_correlation and distance <= max_distance

        matches.append(
            {
                "segment_a": a["word_id"],
                "segment_b": b["word_id"],
                "start_a": a["start"],
                "end_a": a["end"],
                "start_b": b["start"],
                "end_b": b["end"],
                "correlation": corr,
                "euclidean_hz": similarity["euclidean_hz"],
                "normalized_distance": distance,
                "suspicious": suspicious,
            }
        )

    columns = [
        "segment_a",
        "segment_b",
        "start_a",
        "end_a",
        "start_b",
        "end_b",
        "correlation",
        "euclidean_hz",
        "normalized_distance",
        "suspicious",
    ]

    df = pd.DataFrame(matches, columns=columns)

    if not df.empty:
        df = df.sort_values(
            ["suspicious", "correlation", "normalized_distance"],
            ascending=[False, False, True],
        ).reset_index(drop=True)

    return df
