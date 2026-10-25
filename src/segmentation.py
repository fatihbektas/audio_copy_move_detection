import librosa
import numpy as np


def segment_words(
    y, sr, top_db=25, min_duration=0.15, frame_length=512, hop_length=160
):
    """
    Sessizlik tabanlı otomatik konuşma segmentasyonu.
    Segmentler her zaman tek kelime olmayabilir.
    """
    intervals = librosa.effects.split(
        y, top_db=top_db, frame_length=frame_length, hop_length=hop_length
    )

    segments = []

    for start_sample, end_sample in intervals:
        duration = (end_sample - start_sample) / sr

        if duration < min_duration:
            continue

        segments.append(
            {
                "word_id": len(segments) + 1,
                "word": f"Segment_{len(segments) + 1}",
                "start": start_sample / sr,
                "end": end_sample / sr,
                "duration": duration,
                "audio": y[start_sample:end_sample].copy(),
            }
        )

    return segments
