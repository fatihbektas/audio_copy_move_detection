import pandas as pd

import config as cfg

from src.audio_loader import load_audio
from src.segmentation import segment_words
from src.pitch_extraction import extract_all_pitches
from src.detection import detect_copy_move
from src.visualization import plot_segments, plot_pitch_results, plot_similarity_matrix


def main():
    cfg.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    cfg.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Ses dosyasını yükle
    y, sr = load_audio(cfg.AUDIO_PATH, cfg.SR)

    # 2. Otomatik segmentasyon
    segments = segment_words(
        y,
        sr,
        top_db=cfg.TOP_DB,
        min_duration=cfg.MIN_DURATION,
        frame_length=cfg.FRAME_LENGTH,
        hop_length=cfg.HOP_LENGTH,
    )

    print(f"\nBulunan segment sayısı: {len(segments)}")

    if len(segments) < 2:
        print("Karşılaştırma için en az iki segment gerekli.")
        return

    segment_table = pd.DataFrame(
        [
            {
                "segment_id": s["word_id"],
                "start": s["start"],
                "end": s["end"],
                "duration": s["duration"],
            }
            for s in segments
        ]
    )

    segment_table.to_csv(cfg.TABLES_DIR / "segments.csv", index=False)

    plot_segments(y, sr, segments, cfg.FIGURES_DIR / "segments.png")

    # 3. Pitch çıkarımı
    pitch_results = extract_all_pitches(
        segments,
        sr,
        fmin=cfg.FMIN,
        fmax=cfg.FMAX,
        frame_length=cfg.PITCH_FRAME_LENGTH,
        hop_length=cfg.PITCH_HOP_LENGTH,
    )

    pitch_table = pd.DataFrame(
        [
            {
                "segment_id": r["word_id"],
                "start": r["start"],
                "end": r["end"],
                "mean_f0": r["mean_f0"],
                "voiced_frames": r["voiced_frames"],
                "voiced_ratio": r["voiced_ratio"],
            }
            for r in pitch_results
        ]
    )

    pitch_table.to_csv(cfg.TABLES_DIR / "pitch_summary.csv", index=False)

    # Her kareye ait F0 değerlerini de kaydet
    frame_rows = []

    for r in pitch_results:
        for time, f0 in zip(r["times"], r["f0"]):
            frame_rows.append(
                {"segment_id": r["word_id"], "time": r["start"] + time, "f0_hz": f0}
            )

    pd.DataFrame(frame_rows).to_csv(cfg.TABLES_DIR / "pitch_frames.csv", index=False)

    plot_pitch_results(pitch_results, cfg.FIGURES_DIR / "pitch.png")

    # 4. Benzerlik ve sahtecilik tespiti
    matches = detect_copy_move(
        pitch_results,
        min_correlation=cfg.MIN_CORRELATION,
        max_distance=cfg.MAX_EUCLIDEAN,
        target_length=cfg.TARGET_LENGTH,
        min_voiced_frames=cfg.MIN_VOICED_FRAMES,
        min_voiced_ratio=cfg.MIN_VOICED_RATIO,
    )

    matches.to_csv(cfg.TABLES_DIR / "all_comparisons.csv", index=False)

    suspicious = matches[matches["suspicious"]]

    suspicious.to_csv(cfg.TABLES_DIR / "suspicious_matches.csv", index=False)

    plot_similarity_matrix(
        matches,
        [s["word_id"] for s in segments],
        cfg.FIGURES_DIR / "similarity_matrix.png",
    )

    print("\nEN YÜKSEK BENZERLİKLER")
    print("-" * 65)

    if matches.empty:
        print("Geçerli karşılaştırma bulunamadı.")
    else:
        print(matches.head(10).to_string(index=False))

    print(f"\nŞüpheli eşleşme sayısı: {len(suspicious)}")
    print(f"Sonuç klasörü: {cfg.RESULTS_DIR}")


if __name__ == "__main__":
    main()
