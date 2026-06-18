#!/usr/bin/env python3
"""
Entry point: Akuisisi Data untuk Stress Monitoring berbasis TCN.

Alur:
  1. Pastikan model MediaPipe tersedia (unduh jika perlu).
  2. Minta ID partisipan & sesi dari operator.
  3. Pilih sumber: kamera live ATAU file video yang sudah ada.
  4. Rekam/proses landmark wajah + label stres sampai ESC ditekan
     atau video selesai.
  5. Post-processing: imputasi nilai hilang, bentuk sliding window,
     simpan tensor siap pakai untuk training TCN.

Jalankan dari root folder project:
    python main.py
"""

import sys

from config import settings
from core.capture import run_capture_session
from core.postprocessing import run_post_processing
from core.session import prompt_input_source, prompt_session_info
from utils.model_downloader import ensure_model_available


def main():
    try:
        ensure_model_available()
    except RuntimeError as e:
        print(f"[FATAL] {e}")
        sys.exit(1)

    participant_id, session_id = prompt_session_info()
    source, is_testing = prompt_input_source()

    default_label = settings.TESTING_LABEL if is_testing else None
    if is_testing:
        print(f"\n[Info] Mode testing — frame akan diberi label '{settings.TESTING_LABEL}'.")
        print("Data ini sebaiknya tidak dicampur ke dataset training final.\n")

    try:
        run_capture_session(
            participant_id, session_id,
            source=source,
            default_label=default_label,
        )
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
