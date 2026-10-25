import librosa
import numpy as np


def segment_words(
    y, sr, top_db=25, min_duration=0.15, frame_length=512, hop_length=160
):
    """
    Ses sinyalini kelimelere ayırmak için kullanılan fonksiyon.

    Args:
        y (np.ndarray): Ses sinyali.
        sr (int): Örnekleme hızı.
        top_db (int): Enerji eşiği (desibel cinsinden) sessiz bölümleri belirlemek için.
        min_duration (float): Minimum kelime süresi (saniye cinsinden).
        frame_length (int): Çerçeve uzunluğu.
        hop_length (int): Çerçeve kaydırma uzunluğu.

    Returns:
        list: Her bir kelimenin başlangıç ve bitiş örnek indekslerini içeren bir liste.
    """

    # Sessiz bölgeleri tespit et
    intervals = librosa.effects.split(
        y, top_db=top_db, frame_length=frame_length, hop_length=hop_length
    )
    # Minimum süreyi karşılayan kelimeleri filtrele
    segments = []

    # Her bir kelimeyi segmentlere ayır.
    for start_sample, end_sample in intervals:
        start = start_sample / sr
        end = end_sample / sr
        duration = end - start

        if duration < min_duration:
            continue

        auidio_segment = y[start_sample:end_sample]

        segments.append(
            {
                "word_id": len(segments) + 1,
                "word": f"Segment_{len(segments) + 1}",
                "start": round(start, 3),
                "end": round(end, 3),
                "duration": round(duration, 3),
                "audio": auidio_segment,
            }
        )
    return segments
