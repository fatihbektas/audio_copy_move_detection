from src.audio_loader import load_audio


def main():
    audio_path = "data/tampered/tampered.wav"

    y, sr = load_audio(audio_path)

    print("Ses dosyası başarıyla yüklendi.")
    print(f"Toplam örnek sayısı: {len(y)}")


if __name__ == "__main__":
    main()
