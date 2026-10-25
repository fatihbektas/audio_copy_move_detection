import numpy as np
import librosa


def extract_pitch(audio, sr, fmin=65, fmax=500, frame_length=1024, hop_length=160):
    f0, voiced, voiced_prob = librosa.pyin(
        audio,
        sr=sr,
        fmin=fmin,
        fmax=fmax,
        frame_length=frame_length,
        hop_length=hop_length,
        center=True,
    )

    times = librosa.times_like(f0, sr=sr, hop_length=hop_length)

    return {"times": times, "f0": f0, "voiced": voiced, "voiced_prob": voiced_prob}


def extract_all_pitches(segments, sr, **kwargs):
    results = []

    for segment in segments:
        pitch = extract_pitch(segment["audio"], sr, **kwargs)

        f0 = pitch["f0"]
        valid = f0[np.isfinite(f0)]

        results.append(
            {
                "word_id": segment["word_id"],
                "word": segment["word"],
                "start": segment["start"],
                "end": segment["end"],
                "duration": segment["duration"],
                "times": pitch["times"],
                "f0": f0,
                "voiced": pitch["voiced"],
                "mean_f0": (float(np.mean(valid)) if len(valid) else np.nan),
                "voiced_frames": len(valid),
                "voiced_ratio": (len(valid) / len(f0) if len(f0) else 0.0),
            }
        )

    return results
