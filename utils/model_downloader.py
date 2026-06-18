"""
Pengunduh model MediaPipe Face Landmarker.

Dipisah dari logika utama supaya:
1. Bisa diuji terpisah (unit test) tanpa membuka kamera.
2. Bisa dipanggil ulang di script lain (training, batch processing) tanpa duplikasi.
3. Kegagalan unduh (jaringan putus, URL berubah) gagal dengan pesan jelas,
   bukan menyebabkan urllib melempar traceback mentah ke pengguna.
"""

import urllib.error
import urllib.request

from config import settings


def ensure_model_available(model_path=None, model_url=None) -> bool:
    """
    Memastikan file model tersedia secara lokal. Mengunduh jika belum ada.

    Returns:
        True jika model tersedia (baik sudah ada atau berhasil diunduh).

    Raises:
        RuntimeError: jika unduhan gagal (jaringan, URL tidak valid, dsb).
    """
    model_path = model_path or settings.MODEL_PATH
    model_url = model_url or settings.MODEL_URL

    if model_path.exists():
        return True

    print(f"Model '{model_path.name}' belum ada. Mengunduh dari server MediaPipe...")
    try:
        urllib.request.urlretrieve(model_url, model_path)
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Gagal mengunduh model dari {model_url}.\n"
            f"Periksa koneksi internet Anda, atau unduh manual dan letakkan di {model_path}.\n"
            f"Detail error: {e}"
        ) from e

    if not model_path.exists() or model_path.stat().st_size == 0:
        raise RuntimeError(
            f"Unduhan model tampak gagal — file {model_path} tidak ditemukan atau kosong."
        )

    print(f"Model berhasil diunduh ke {model_path}")
    return True
