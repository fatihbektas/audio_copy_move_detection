from src.audio_loader import load_audio
from src.segmentation import segment_words
from src.pitch_extraction import extract_all_pitches


def main():

    audio_path = "data/tampered/tampered.wav"

    # 1. Ses dosyasını yükle
    y, sr = load_audio(audio_path)

    # 2. Otomatik segmentasyon
    segments = segment_words(y, sr, top_db=25, min_duration=0.15)

    print("\nOTOMATİK SEGMENTASYON")
    print("-" * 60)

    for segment in segments:
        print(
            f"{segment['word']} | "
            f"Başlangıç: {segment['start']:.2f} sn | "
            f"Bitiş: {segment['end']:.2f} sn | "
            f"Süre: {segment['duration']:.2f} sn"
        )

    # 3. Pitch çıkarımı
    results = extract_all_pitches(segments, sr)

    print("\nPITCH ANALİZİ")
    print("-" * 60)

    for item in results:
        print(
            f"{item['word']} | "
            f"Ortalama F0: {item['mean_f0']:.2f} Hz | "
            f"Ötümlü kare: {item['voiced_frames']}"
        )


if __name__ == "__main__":
    main()
