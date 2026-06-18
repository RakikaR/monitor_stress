"""
Konfigurasi terpusat untuk Stress Monitor.

Semua "angka ajaib" (magic numbers) dan parameter yang mungkin di-tweak
saat eksperimen diletakkan di sini, bukan tersebar di seluruh kode.
Tujuannya: kalau mau ganti threshold atau ukuran window, cukup edit
satu file ini, tidak perlu menelusuri seluruh codebase.
"""

from pathlib import Path

# ── Path & Penyimpanan ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILENAME = "face_landmarker.task"
MODEL_PATH = MODEL_DIR / MODEL_FILENAME
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)

DATASET_CSV = DATA_DIR / "dataset_wajah_stres.csv"

# ── Label Stres ─────────────────────────────────────────────────────
# Label ditentukan OPERATOR per sesi (lihat alasan di code review),
# bukan diturunkan otomatis dari ekspresi wajah, supaya tidak sirkular.
STRESS_LABELS = {
    "1": "Rileks",
    "2": "Stres_Ringan",
    "3": "Stres_Berat",
}
DEFAULT_LABEL = "Rileks"

# ── Indeks Landmark Wajah (MediaPipe Face Mesh, 478 titik) ─────────
# Indeks dikonfirmasi sesuai konvensi resmi MediaPipe:
# 33 = sudut luar mata kiri, 263 = sudut luar mata kanan, 1 = ujung hidung.
EYE_LEFT = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
EYE_RIGHT = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
EYEBROW_LEFT = [46, 53, 52, 65, 55, 70, 63, 105, 66, 107]
EYEBROW_RIGHT = [276, 283, 282, 295, 285, 300, 293, 334, 296, 336]
MOUTH_JAW = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

SELECTED_INDICES = EYE_LEFT + EYE_RIGHT + EYEBROW_LEFT + EYEBROW_RIGHT + MOUTH_JAW

# Titik acuan untuk normalisasi (lihat utils/normalization.py)
NASAL_TIP_IDX = 1
LEFT_EYE_OUTER_IDX = 33
RIGHT_EYE_OUTER_IDX = 263

# ── Parameter MediaPipe ─────────────────────────────────────────────
MIN_FACE_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
NUM_FACES = 1

# ── Threshold Klasifikasi Ekspresi (HANYA untuk overlay visual,
#    BUKAN dipakai sebagai label training — lihat code review) ─────
EXPR_THRESHOLDS = {
    "jaw_open": 0.3,
    "brow_inner_up": 0.3,
    "eye_blink": 0.4,
    "mouth_smile": 0.45,
    "mouth_frown": 0.1,
}

# ── Sliding Window untuk TCN ────────────────────────────────────────
WINDOW_SIZE = 90   # ~3 detik pada 30 FPS
STEP_SIZE = 30     # overlap 66%

# ── Tampilan ─────────────────────────────────────────────────────────
WINDOW_TITLE = "Stress Monitor — Akuisisi Data"
DRAW_ALL_LANDMARKS = False  # True = gambar 478 titik (berat), False = hanya titik terpilih
