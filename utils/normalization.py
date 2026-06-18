"""
Normalisasi koordinat landmark wajah.

Tujuan normalisasi: menghilangkan efek jarak partisipan ke kamera dan
posisi wajah di frame, supaya model TCN belajar pola EKSPRESI, bukan
pola "seberapa dekat orang duduk dari kamera".

Metode: titik referensi = ujung hidung (translasi), skala = jarak
inter-ocular (lebar mata kiri-kanan), karena jarak ini relatif stabil
antar individu dan tidak berubah karena ekspresi.

Logika ini dipisah dari loop kamera secara sengaja: fungsi murni
(pure function) seperti ini gampang ditulis unit test-nya tanpa
perlu mock kamera atau model MediaPipe.
"""

from dataclasses import dataclass

import numpy as np

from config import settings


@dataclass
class NormalizationResult:
    values: list  # flat list [x0, y0, z0, x1, y1, z1, ...]
    inter_ocular_dist: float


def compute_inter_ocular_distance(landmarks) -> float:
    """Menghitung jarak Euclidean 3D antara sudut luar mata kiri & kanan."""
    p_left = landmarks[settings.LEFT_EYE_OUTER_IDX]
    p_right = landmarks[settings.RIGHT_EYE_OUTER_IDX]

    dx = p_left.x - p_right.x
    dy = p_left.y - p_right.y
    dz = p_left.z - p_right.z
    return float(np.sqrt(dx**2 + dy**2 + dz**2))


def normalize_landmarks(landmarks, selected_indices=None) -> NormalizationResult:
    """
    Menormalisasi landmark terpilih relatif terhadap ujung hidung,
    diskalakan oleh jarak inter-ocular.

    Args:
        landmarks: list landmark MediaPipe (478 titik), masing-masing
            punya atribut .x .y .z (koordinat ternormalisasi MediaPipe).
        selected_indices: indeks landmark yang ingin diekstrak.
            Default ke SELECTED_INDICES di settings.

    Returns:
        NormalizationResult dengan values sepanjang len(selected_indices) * 3.
    """
    selected_indices = selected_indices or settings.SELECTED_INDICES
    nasal_tip = landmarks[settings.NASAL_TIP_IDX]

    inter_ocular_dist = compute_inter_ocular_distance(landmarks)
    if inter_ocular_dist < 1e-9:
        # Hindari pembagian dengan nol pada kasus landmark mata berhimpitan
        # (jarang terjadi, tapi bisa muncul pada deteksi yang sangat noisy).
        inter_ocular_dist = 1e-6

    values = []
    for idx in selected_indices:
        lm = landmarks[idx]
        values.append((lm.x - nasal_tip.x) / inter_ocular_dist)
        values.append((lm.y - nasal_tip.y) / inter_ocular_dist)
        values.append((lm.z - nasal_tip.z) / inter_ocular_dist)

    return NormalizationResult(values=values, inter_ocular_dist=inter_ocular_dist)
