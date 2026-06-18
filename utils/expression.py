"""
Klasifikasi ekspresi berbasis blendshape — UNTUK OVERLAY VISUAL SAJA.

PENTING (lihat code review): hasil fungsi ini TIDAK dipakai sebagai
label "class" pada dataset training. Memakai ekspresi otomatis sebagai
label stres bersifat sirkular — model akan belajar meniru aturan
if-else di bawah ini, bukan pola stres yang sesungguhnya.

Label stres yang valid berasal dari protokol sesi yang ditentukan
operator (lihat core/session.py). Kolom ekspresi di sini hanya
disimpan sebagai metadata tambahan untuk eksplorasi data, dan untuk
feedback visual real-time ke operator saat merekam.
"""

from config import settings


def classify_expression(face_blendshapes) -> tuple[str, tuple[int, int, int]]:
    """
    Args:
        face_blendshapes: list kategori blendshape dari satu wajah hasil
            face_landmarker.detect_for_video(...).face_blendshapes[i]

    Returns:
        (label, warna_bgr) — label string dan warna teks untuk overlay OpenCV.
    """
    scores = {c.category_name: c.score for c in face_blendshapes}
    t = settings.EXPR_THRESHOLDS

    if scores.get("jawOpen", 0) > t["jaw_open"] and scores.get("browInnerUp", 0) > t["brow_inner_up"]:
        return "Kaget", (0, 165, 255)
    if scores.get("eyeBlinkLeft", 0) > t["eye_blink"] and scores.get("eyeBlinkRight", 0) > t["eye_blink"]:
        return "Tutup_Mata", (255, 255, 0)
    if scores.get("mouthSmileLeft", 0) > t["mouth_smile"] or scores.get("mouthSmileRight", 0) > t["mouth_smile"]:
        return "Senyum", (0, 255, 0)
    if scores.get("mouthFrownLeft", 0) > t["mouth_frown"] or scores.get("mouthFrownRight", 0) > t["mouth_frown"]:
        return "Sedih", (255, 0, 0)
    return "Netral", (0, 0, 255)
