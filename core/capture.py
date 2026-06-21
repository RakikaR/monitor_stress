"""
Loop akuisisi data utama: sumber video (kamera ATAU file) → MediaPipe
→ normalisasi → CSV.

Modul ini HANYA orkestrasi (memanggil modul lain secara berurutan).
Logika murni (normalisasi, klasifikasi ekspresi, penulisan CSV) sudah
dipisah ke modul masing-masing supaya:
  1. Bisa diuji unit test tanpa kamera fisik.
  2. Mudah dibaca: loop ini jadi pendek dan jelas alurnya.
  3. Bisa dipakai ulang untuk mode lain (kamera live maupun file
     video yang sudah ada) tanpa duplikasi logika.

CATATAN TIMESTAMP (penting, bukan cuma detail teknis):
MediaPipe VIDEO mode mewajibkan timestamp yang naik monoton.
- Kamera live: tidak ada timestamp intrinsik, jadi dipakai jam sistem
  (time.time()) sebagai pendekatan "kapan frame ini benar-benar diambil".
- File video: punya FPS tetap dan timeline sendiri yang sudah pasti.
  Memakai time.time() di sini SALAH, karena loop pemrosesan file bisa
  berjalan jauh lebih cepat/lambat dari kecepatan video aslinya (tidak
  menunggu seperti webcam) — ini bisa membuat timestamp tidak akurat,
  atau bahkan mundur jika frame diproses lebih cepat dari resolusi
  jam sistem. Maka dipakai timestamp_ms = frame_idx * (1000 / fps),
  diturunkan dari posisi frame itu sendiri, bukan jam dinding.
"""

import time
from pathlib import Path

import cv2
import mediapipe as mp

from config import settings
from core.dataset_writer import DatasetWriter
from core.session import SessionState, print_label_legend
from utils.normalization import normalize_landmarks

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
FaceLandmarker = mp.tasks.vision.FaceLandmarker


def build_face_landmarker():
    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(settings.MODEL_PATH)),
        running_mode=VisionRunningMode.VIDEO,
        num_faces=settings.NUM_FACES,
        min_face_detection_confidence=settings.MIN_FACE_DETECTION_CONFIDENCE,
        min_tracking_confidence=settings.MIN_TRACKING_CONFIDENCE,
        output_face_blendshapes=False, # <--- PERBAIKAN DI SINI (Ubah menjadi False)
    )
    return FaceLandmarker.create_from_options(options)


def draw_overlay(frame, landmarks, w, h, draw_all: bool):
    """Menggambar titik landmark. draw_all=False hanya menggambar titik terpilih
    (jauh lebih murah secara komputasi — relevan untuk perangkat low-resource)."""
    if draw_all:
        for idx, lm in enumerate(landmarks):
            x_px, y_px = int(lm.x * w), int(lm.y * h)
            is_selected = idx in settings.SELECTED_INDICES
            color = (0, 255, 255) if is_selected else (0, 255, 0)
            radius = 2 if is_selected else 1
            cv2.circle(frame, (x_px, y_px), radius, color, -1)
    else:
        for idx in settings.SELECTED_INDICES:
            lm = landmarks[idx]
            x_px, y_px = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (x_px, y_px), 2, (0, 255, 255), -1)


def open_source(source):
    """
    Membuka sumber video.

    Args:
        source: 0 (int) untuk kamera default, atau path string/Path
            ke file video yang sudah ada.

    Returns:
        (cap, is_live, fps) — cap adalah cv2.VideoCapture, is_live
        menandakan apakah sumbernya kamera real-time (mempengaruhi
        cara timestamp dihitung), fps dipakai untuk file video.
    """
    is_live = source == 0
    if not is_live:
        source = str(source)
        if not Path(source).exists():
            raise RuntimeError(f"File video tidak ditemukan: {source}")

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        if is_live:
            raise RuntimeError("Tidak bisa membuka kamera. Periksa apakah kamera dipakai aplikasi lain.")
        raise RuntimeError(f"Tidak bisa membuka file video: {source}. Periksa format/codec-nya.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not is_live and (fps is None or fps <= 0):
        print("[Peringatan] FPS file video tidak terbaca, memakai asumsi 30 FPS.")
        fps = 30.0

    return cap, is_live, fps


def run_capture_session(participant_id: str, session_id: str, source=0,
                         default_label=None, allow_manual_relabel=True,
                         show_window=True):
    """
    Menjalankan akuisisi data dari kamera live ATAU file video yang sudah ada.

    Args:
        source: 0 untuk kamera default, atau path ke file video (str/Path).
        default_label: label stres awal. Jika None, pakai settings.DEFAULT_LABEL.
            Untuk video testing tanpa label pasti, kirim "Unlabeled" di sini.
        allow_manual_relabel: jika True, tombol 1/2/3 tetap bisa dipakai
            untuk mengganti label di tengah pemutaran (berguna untuk video
            yang ingin ditandai manual per segmen waktu).
        show_window: jika False, video diproses tanpa menampilkan jendela
            (lebih cepat untuk batch processing banyak file).
    """
    cap, is_live, fps = open_source(source)

    if show_window:
        cv2.namedWindow(settings.WINDOW_TITLE, cv2.WINDOW_NORMAL)

    session = SessionState(participant_id=participant_id, session_id=session_id)
    if default_label is not None:
        session.current_label = default_label
    if allow_manual_relabel:
        print_label_legend(show_window=show_window, current_label=session.current_label)

    ms_per_frame = 1000.0 / fps if not is_live else None

    try:
        with build_face_landmarker() as face_landmarker, DatasetWriter(
            participant_id=participant_id, session_id=session_id
        ) as writer:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    reason = "Stream kamera berhenti" if is_live else "Video selesai diproses"
                    print(f"{reason} (total {session.frame_count} frame).")
                    break

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

                if is_live:
                    timestamp_ms = int(time.time() * 1000)
                else:
                    # Timestamp diturunkan dari posisi frame, bukan jam sistem
                    # (lihat catatan di docstring modul ini).
                    timestamp_ms = int(session.frame_count * ms_per_frame)

                result = face_landmarker.detect_for_video(mp_image, timestamp_ms)

                h, w = frame.shape[:2]

                if result.face_landmarks:
                    landmarks = result.face_landmarks[0]
                    draw_overlay(frame, landmarks, w, h, settings.DRAW_ALL_LANDMARKS)

                    try:
                        norm_result = normalize_landmarks(landmarks)
                        writer.write_row(
                            frame_idx=session.frame_count,
                            stress_label=session.current_label,
                            feature_values=norm_result.values,
                        )
                    except (IndexError, ZeroDivisionError) as e:
                        print(f"[Peringatan] Gagal menormalisasi frame {session.frame_count}: {e}")
                else:
                    writer.write_missing_row(
                        frame_idx=session.frame_count,
                        stress_label=session.current_label,
                    )
                
                if show_window:
                    cv2.imshow(settings.WINDOW_TITLE, frame)
                

                session.tick()

                if show_window:
                    # waitKey(1) dipakai juga untuk video file: tetap memberi
                    # waktu event loop OpenCV merender jendela. Untuk batch
                    # processing tanpa jendela, langkah ini dilewati sama
                    # sekali (show_window=False) supaya prosesnya secepat CPU.
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:  # ESC
                        break
                    if allow_manual_relabel:
                        key_char = chr(key) if 0 <= key < 128 else ""
                        if key_char in settings.STRESS_LABELS:
                            session.set_label(key_char)
                            print(f"[Frame {session.frame_count}] Label diganti -> {session.current_label}")

    finally:
        cap.release()
        if show_window:
            cv2.destroyAllWindows()

    print(f"\nSesi selesai. Total frame terekam: {session.frame_count}")
    print(f"Riwayat pergantian label: {session.label_changes}")
    return session
