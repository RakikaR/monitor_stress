"""
Post-processing dataset: imputasi nilai hilang + pembentukan sekuens
sliding window untuk training TCN.

Catatan penting soal sliding window (lihat README untuk detail):
window dibentuk PER (participant_id, session_id), bukan lintas batas
sesi/partisipan. Kode asli membentuk window dari seluruh CSV secara
linear — kalau ada >1 sesi dalam satu file, window di perbatasan dua
sesi akan mencampur frame dari dua orang/kondisi berbeda yang tidak
ada hubungan temporalnya. Ini bug halus yang mudah terlewat karena
tidak melempar error, hanya mencemari kualitas data training.
"""

import numpy as np
import pandas as pd

from config import settings


def impute_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Interpolasi linear untuk kolom numerik, dilakukan PER sesi agar
    tidak menginterpolasi lintas batas sesi yang tidak berkaitan."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [c for c in numeric_cols if c != "frame_idx"]

    missing_before = df[numeric_cols].isna().sum().sum()
    if missing_before == 0:
        print("Tidak ada data yang hilang.")
        return df

    print(f"Menginterpolasi {missing_before} nilai kosong (NaN) per sesi...")

    # Pakai transform per kolom (bukan groupby().apply() pada seluruh
    # sub-dataframe) supaya perilakunya stabil lintas versi pandas dan
    # tidak berisiko men-drop kolom grouping (participant_id/session_id).
    group_keys = df.groupby(["participant_id", "session_id"]).ngroup()
    for col in numeric_cols:
        df[col] = (
            df.groupby(group_keys)[col]
            .transform(lambda s: s.interpolate(method="linear", limit_direction="both"))
        )

    remaining = df[numeric_cols].isna().sum().sum()
    if remaining > 0:
        print(f"[Peringatan] {remaining} nilai masih NaN (sesi dengan SEMUA frame occluded).")
    else:
        print("Imputasi selesai.")
    return df


def build_sliding_windows(df: pd.DataFrame, window_size=None, step_size=None):
    """
    Membentuk tensor 3D (Batch, Timesteps, Features) untuk TCN.

    Window dibentuk per (participant_id, session_id) secara terpisah
    sehingga tidak ada window yang mencampur dua sesi berbeda.
    """
    window_size = window_size or settings.WINDOW_SIZE
    step_size = step_size or settings.STEP_SIZE

    feature_cols = [
        c for c in df.columns
        if c not in ("participant_id", "session_id", "frame_idx", "class")
    ]

    X_sequences, y_sequences = [], []
    skipped_groups = []

    for (pid, sid), group in df.groupby(["participant_id", "session_id"]):
        group = group.sort_values("frame_idx")
        features = group[feature_cols].values
        labels = group["class"].values
        num_frames = len(features)

        if num_frames < window_size:
            skipped_groups.append((pid, sid, num_frames))
            continue

        for i in range(0, num_frames - window_size + 1, step_size):
            window_features = features[i: i + window_size]
            window_labels = labels[i: i + window_size]

            unique_labels, counts = np.unique(window_labels, return_counts=True)
            majority_label = unique_labels[np.argmax(counts)]

            X_sequences.append(window_features)
            y_sequences.append(majority_label)

    if skipped_groups:
        print(f"[Peringatan] {len(skipped_groups)} sesi dilewati karena terlalu pendek (< {window_size} frame):")
        for pid, sid, n in skipped_groups:
            print(f"  - {pid}/{sid}: {n} frame")

    if not X_sequences:
        raise ValueError(
            f"Tidak ada window yang terbentuk. Semua sesi lebih pendek dari "
            f"WINDOW_SIZE={window_size}. Rekam sesi lebih panjang atau kecilkan WINDOW_SIZE."
        )

    X_tensor = np.array(X_sequences, dtype=np.float32)
    y_tensor = np.array(y_sequences)
    return X_tensor, y_tensor


def run_post_processing(csv_path=None):
    csv_path = csv_path or settings.DATASET_CSV

    print("\n--- Tahap 1: Imputasi Data (Linear Interpolation per Sesi) ---")
    df = pd.read_csv(csv_path, sep=";")
    df = impute_missing(df)
    df.to_csv(csv_path, sep=";", index=False)

    print("\n--- Tahap 2: Pembentukan Sekuens Temporal (Sliding Window) ---")
    X_tensor, y_tensor = build_sliding_windows(df)

    x_path = settings.DATA_DIR / "X_tcn_data.npy"
    y_path = settings.DATA_DIR / "y_tcn_labels.npy"
    np.save(x_path, X_tensor)
    np.save(y_path, y_tensor)

    print(f"\n[SUKSES] Data tensor 3D berhasil dibentuk!")
    print(f"X (Fitur) : {X_tensor.shape} -> (Batch_Size, Timesteps, Features)")
    print(f"y (Label) : {y_tensor.shape} -> (Batch_Size,)")
    print(f"Distribusi label: {dict(zip(*np.unique(y_tensor, return_counts=True)))}")
    print(f"Disimpan ke: {x_path} dan {y_path}")

    return X_tensor, y_tensor
