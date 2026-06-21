"""
Penulis CSV untuk dataset landmark.

PERBAIKAN PERFORMA (lihat code review poin #2): kode asli membuka dan
menutup file pada SETIAP frame (puluhan kali per detik). Membuka file
melibatkan system call yang relatif mahal dibanding operasi tulis itu
sendiri; pada storage lambat (SD card, HDD eksternal) ini bisa jadi
bottleneck FPS yang nyata.

Solusi di sini: buka file SEKALI di awal sesi, simpan file handle,
tutup secara eksplisit di akhir (lewat context manager). csv.writer
tetap menulis baris demi baris secara streaming — bukan menumpuk
semua di memori — sehingga aman untuk sesi rekaman yang panjang.
"""

import csv

from config import settings


class DatasetWriter:
    """Context manager untuk menulis dataset landmark wajah ke CSV."""

    def __init__(self, csv_path=None, selected_indices=None, participant_id="", session_id=""):
        self.csv_path = csv_path or settings.DATASET_CSV
        self.selected_indices = selected_indices or settings.SELECTED_INDICES
        self.participant_id = participant_id
        self.session_id = session_id
        self._file = None
        self._writer = None

    def _build_header(self):
        header = ["participant_id", "session_id", "frame_idx", "class"]
        for idx in self.selected_indices:
            header += [f"x{idx}", f"y{idx}", f"z{idx}"]
        return header

    def __enter__(self):
        is_new_file = not self.csv_path.exists()
        self._file = open(self.csv_path, mode="a", newline="", encoding="utf-8")
        self._writer = csv.writer(
            self._file, delimiter=";", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        if is_new_file:
            self._writer.writerow(self._build_header())
        return self

    def write_row(self, frame_idx: int, stress_label: str, feature_values):
        """Menulis satu baris data (satu frame)."""
        row = [self.participant_id, self.session_id, frame_idx, stress_label]
        row.extend(feature_values)
        self._writer.writerow(row)

    def write_missing_row(self, frame_idx: int, stress_label: str, reason: str = "Occlusion"):
        """Menulis baris NaN saat wajah tidak terdeteksi."""
        n_values = len(self.selected_indices) * 3
        # PERBAIKAN: Hapus variabel 'reason' dari array di bawah ini
        row = [self.participant_id, self.session_id, frame_idx, stress_label] 
        row.extend([float("nan")] * n_values)
        self._writer.writerow(row)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._file:
            self._file.close()
        return False
