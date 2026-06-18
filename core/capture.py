"""
Loop akuisisi data utama: kamera → MediaPipe → normalisasi → CSV.

Modul ini HANYA orkestrasi (memanggil modul lain secara berurutan).
Logika murni (normalisasi, klasifikasi ekspresi, penulisan CSV) sudah
dipisah ke modul masing-masing supaya:
  1. Bisa diuji unit test tanpa kamera fisik.
  2. Mudah dibaca: loop ini jadi pendek dan jelas alurnya.
  3. Bisa dipakai ulang untuk mode lain (mis. baca dari file video,
     bukan webcam) tanpa duplikasi logika.
"""

import time

import cv2
import mediapipe as mp

from config import settings
from core.dataset_writer import DatasetWriter
from core.session import SessionState, print_label_legend
from utils.expression import classify_expression
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
        output_face_blendshapes=True,
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


def run_capture_session(participant_id: str, session_id: str):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Tidak bisa membuka kamera. Periksa apakah kamera dipakai aplikasi lain.")

    cv2.namedWindow(settings.WINDOW_TITLE, cv2.WINDOW_NORMAL)
    session = SessionState(participant_id=participant_id, session_id=session_id)
    print_label_legend()

    p_time = 0.0

    try:
        with build_face_landmarker() as face_landmarker, DatasetWriter(
            participant_id=participant_id, session_id=session_id
        ) as writer:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("Stream kamera berhenti (tidak ada frame baru).")
                    break

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                timestamp_ms = int(time.time() * 1000)
                result = face_landmarker.detect_for_video(mp_image, timestamp_ms)

                h, w = frame.shape[:2]
                expression = "Netral"
                expr_color = (0, 0, 255)

                if result.face_landmarks:
                    landmarks = result.face_landmarks[0]
                    draw_overlay(frame, landmarks, w, h, settings.DRAW_ALL_LANDMARKS)

                    if result.face_blendshapes:
                        expression, expr_color = classify_expression(result.face_blendshapes[0])

                    try:
                        norm_result = normalize_landmarks(landmarks)
                        writer.write_row(
                            frame_idx=session.frame_count,
                            stress_label=session.current_label,
                            expression=expression,
                            feature_values=norm_result.values,
                        )
                    except (IndexError, ZeroDivisionError) as e:
                        print(f"[Peringatan] Gagal menormalisasi frame {session.frame_count}: {e}")
                else:
                    expression = "Occlusion"
                    expr_color = (100, 100, 100)
                    writer.write_missing_row(
                        frame_idx=session.frame_count,
                        stress_label=session.current_label,
                    )

                # --- Overlay info ke frame ---
                cv2.putText(frame, f"Label Stres: {session.current_label}", (10, 45),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                cv2.putText(frame, f"Ekspresi: {expression}", (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, expr_color, 2)

                c_time = time.time()
                fps = 1 / (c_time - p_time) if p_time else 0
                p_time = c_time
                cv2.putText(frame, f"FPS: {int(fps)}", (10, 115),
                            cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)

                cv2.imshow(settings.WINDOW_TITLE, frame)
                session.tick()

                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
                key_char = chr(key) if 0 <= key < 128 else ""
                if key_char in settings.STRESS_LABELS:
                    session.set_label(key_char)
                    print(f"[Frame {session.frame_count}] Label diganti -> {session.current_label}")

    finally:
        cap.release()
        cv2.destroyAllWindows()

    print(f"\nSesi selesai. Total frame terekam: {session.frame_count}")
    print(f"Riwayat pergantian label: {session.label_changes}")
    return session
