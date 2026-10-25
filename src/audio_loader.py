import librosa


def load_audio(file_path, sr=16000):
    """
    WAV dosyasını yükler.

    Returns:
        y: np.ndarray, ses sinyali
        sr: Örnekleme frekansı
    """
    y, sr = librosa.load(file_path, sr=sr, mono=True)

    duration = len(y) / sr

    print(
        f"Dosya: {file_path}, Süre: {duration:.2f} saniye, Örnekleme frekansı: {sr} Hz"
    )

    return y, sr
