
# Audio Copy-Move Detection

Kelime/konuşma parçası tabanlı pitch analiziyle
ses kayıtlarında kopyala-yapıştır sahteciliği tespiti.

## Yöntem

1. WAV dosyasını yükleme
2. Sessizlik tabanlı otomatik segmentasyon
3. pYIN ile F0 çıkarımı
4. Pitch dizilerini yeniden örnekleme
5. Pearson korelasyonu
6. Öklid uzaklığı
7. Şüpheli segment çiftlerinin belirlenmesi

## Kurulum

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

## Kullanım

Ses dosyasını şu konuma yerleştirin:

data/tampered/tampered.wav

Ardından:

python main.py

## Sonuçlar

results/tables/
- segments.csv
- pitch_summary.csv
- pitch_frames.csv
- all_comparisons.csv
- suspicious_matches.csv

results/figures/
- segments.png
- pitch.png
- similarity_matrix.png

## Sınırlılıklar

Sessizlik tabanlı segmentasyon her zaman kelime sınırlarını
belirleyemez. Bir segment birden fazla kelime içerebilir.

Yüksek pitch benzerliği tek başına sahtecilik kanıtı değildir.
Eşikler deneysel olarak ayarlanmalıdır.
