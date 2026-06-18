"""
Manajemen sesi & label stres.

Label stres ("class" pada dataset) ditentukan OPERATOR secara sadar
melalui keyboard, bukan diturunkan otomatis dari ekspresi wajah.
Alasan lengkap ada di code review / README — intinya menghindari
label yang sirkular (lihat utils/expression.py).

Operator menekan tombol 1/2/3 untuk mengganti label stres aktif
KAPAN SAJA selama sesi (misalnya saat transisi protokol dari fase
istirahat ke fase tugas kognitif). Semua frame yang ditangkap
sebelum pergantian akan tetap memakai label sebelumnya.
"""

from dataclasses import dataclass, field

from config import settings


@dataclass
class SessionState:
    participant_id: str
    session_id: str
    current_label: str = settings.DEFAULT_LABEL
    frame_count: int = 0
    label_changes: list = field(default_factory=list)  # log (frame_idx, label_baru)

    def set_label(self, key: str) -> bool:
        """
        Mengubah label aktif berdasarkan tombol yang ditekan.

        Returns:
            True jika key valid dan label berubah, False jika key tidak dikenal.
        """
        new_label = settings.STRESS_LABELS.get(key)
        if new_label is None:
            return False
        if new_label != self.current_label:
            self.current_label = new_label
            self.label_changes.append((self.frame_count, new_label))
        return True

    def tick(self):
        self.frame_count += 1


def prompt_session_info() -> tuple[str, str]:
    """Meminta ID partisipan & sesi dari operator sebelum kamera dibuka."""
    participant_id = input("Masukkan ID partisipan (contoh: P01): ").strip() or "P00"
    session_id = input("Masukkan ID sesi (contoh: S01): ").strip() or "S00"
    return participant_id, session_id


def print_label_legend():
    print("\n=== Kontrol Label Stres ===")
    for key, label in settings.STRESS_LABELS.items():
        print(f"  Tekan [{key}] -> {label}")
    print(f"  Label awal: {settings.DEFAULT_LABEL}")
    print("  Tekan [ESC] untuk berhenti merekam.\n")
