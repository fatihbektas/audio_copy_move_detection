from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

AUDIO_PATH = BASE_DIR / "data/tampered/tampered.wav"
RESULTS_DIR = BASE_DIR / "results"
TABLES_DIR = RESULTS_DIR / "tables"
FIGURES_DIR = RESULTS_DIR / "figures"

SR = 16000

# Segmentasyon
TOP_DB = 25
MIN_DURATION = 0.15
FRAME_LENGTH = 512
HOP_LENGTH = 160

# Pitch
FMIN = 65
FMAX = 500
PITCH_FRAME_LENGTH = 1024
PITCH_HOP_LENGTH = 160
TARGET_LENGTH = 50
MIN_VOICED_FRAMES = 5
MIN_VOICED_RATIO = 0.25

# Sahtecilik aday eşikleri (deneysel)
MIN_CORRELATION = 0.95
MAX_EUCLIDEAN = 0.10
