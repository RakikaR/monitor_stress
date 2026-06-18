#!/usr/bin/env python3
"""
Entry point: Akuisisi Data untuk Stress Monitoring berbasis TCN.

Alur:
  1. Pastikan model MediaPipe tersedia (unduh jika perlu).
  2. Minta ID partisipan & sesi dari operator.
  3. Buka kamera, rekam landmark wajah + label stres (dikontrol manual
     oleh operator via keyboard 1/2/3) sampai ESC ditekan.
  4. Post-processing: imputasi nilai hilang, bentuk sliding window,
     simpan tensor siap pakai untuk training TCN.

Jalankan dari root folder project:
    python main.py
"""

import sys

from config import settings
from core.capture import run_capture_session
from core.postprocessing import run_post_processing
from core.session import prompt_session_info
from utils.model_downloader import ensure_model_available


def main():
    try:
        ensure_model_available()
    except RuntimeError as e:
        print(f"[FATAL] {e}")
        sys.exit(1)

    participant_id, session_id = prompt_session_info()

    try:
        run_capture_session(participant_id, session_id)
    except RuntimeError as e:
        print(f"[FATAL] {e}")
        sys.exit(1)

    try:
        run_post_processing()
    except ValueError as e:
        print(f"[Peringatan] Post-processing tidak selesai: {e}")
        print(f"Data mentah tetap tersimpan di: {settings.DATASET_CSV}")


if __name__ == "__main__":
    main()
