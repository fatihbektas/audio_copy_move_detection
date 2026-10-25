import librosa


def load_audio(file_path, sr=16000):
    y, sr = librosa.load(str(file_path), sr=sr, mono=True)

    if len(y) == 0:
        raise ValueError("Ses dosyası boş.")

    duration = len(y) / sr

    print(f"Dosya: {file_path}")
    print(f"Örnekleme frekansı: {sr} Hz")
    print(f"Süre: {duration:.2f} saniye")

    return y, sr
