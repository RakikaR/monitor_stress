# Stress Monitor — Modul Akuisisi Data (TCN-ready)

Refactor dari `cv & mp.py` menjadi package modular untuk pengumpulan
dataset landmark wajah yang akan dipakai melatih model Temporal
Convolutional Network (TCN) untuk deteksi tingkat stres.

## Ringkasan Code Review (kode asli → versi ini)

| # | Temuan di kode asli | Risiko | Perbaikan |
|---|---|---|---|
| 1 | `class` di CSV diisi label **ekspresi** (Senyum/Kaget/dst), bukan label **stres** | Model akan belajar memprediksi hal yang salah — tidak ada baris kode yang memetakan ekspresi → stres | Label stres kini ditentukan **operator** via keyboard (1/2/3), disimpan terpisah dari kolom ekspresi |
| 2 | File CSV dibuka & ditutup di **setiap frame** (`open(...)` di dalam loop) | Overhead syscall berulang puluhan kali/detik — bottleneck FPS di storage lambat | File dibuka sekali per sesi (`DatasetWriter` sebagai context manager) |
| 3 | Tidak ada ID partisipan/sesi pada baris data | Tidak bisa memisahkan data per orang untuk validasi, dan window sliding bisa mencampur frame dari sesi berbeda | Kolom `participant_id`, `session_id`, `frame_idx` ditambahkan; semua diminta di awal program |
| 4 | Sliding window dibentuk dari **seluruh CSV sekaligus** secara linear | Window di perbatasan dua sesi/partisipan mencampur frame yang tidak berkaitan temporal — bug halus, tidak error tapi mencemari data training | Window dibentuk **per (participant_id, session_id)** secara terpisah (lihat `core/postprocessing.py`) |
| 5 | Variabel `c` di `h, w, c = frame.shape` tidak terpakai | Linter warning, kode kurang bersih | Diganti `h, w = frame.shape[:2]` |
| 6 | `import pandas`/`import numpy` diulang di tengah file | Tidak rapi | Semua import dikumpulkan di atas tiap modul |
| 7 | Tidak ada `min_tracking_confidence` | Tracking bisa jitter pada hardware lemah | Ditambahkan, dikonfigurasi di `config/settings.py` |
| 8 | Drawing 478 titik selalu aktif | Komputasi tambahan yang tidak perlu saat mode capture (bukan visualisasi) di perangkat low-resource | `DRAW_ALL_LANDMARKS = False` secara default — hanya gambar titik terpilih |
| 9 | Kegagalan unduh model / kamera tidak ditangani dengan jelas | Traceback mentah membingungkan pengguna non-teknis | Pesan error yang jelas + saran tindakan di `utils/model_downloader.py` dan `core/capture.py` |
| 10 | `chr(keycode)` pada keycode di luar ASCII bisa crash | Potensi `ValueError` saat tombol non-standar ditekan | Guard range `0 <= key < 128` sebelum `chr()` |

### Keputusan desain penting: kenapa label stres tidak otomatis?

Kode asli punya logika `if scores.get('jawOpen') > 0.3 ...: status = "Kaget"`
yang sebelumnya langsung jadi label di CSV. Ini **sirkular** untuk tujuan
deteksi stres: kamu memakai aturan blendshape buatan sendiri untuk
membuat label, lalu fitur blendshape yang sama untuk memprediksi label
itu. Model TCN akhirnya hanya akan belajar meniru if-else yang sudah
kamu tulis, bukan pola stres yang sesungguhnya.

Solusi pada versi ini: label stres (`Rileks` / `Stres_Ringan` / `Stres_Berat`)
dikontrol **operator** lewat tombol 1/2/3 saat sesi rekam — biasanya
sesuai protokol stres terinduksi (baseline santai → tugas kognitif
sedang → tugas tekanan waktu). Kolom ekspresi (Senyum/Kaget/dst) tetap
disimpan sebagai metadata tambahan, bukan sebagai label.

## Struktur Project

```
stress_monitor/
├── main.py                  # entry point — jalankan ini
├── config/settings.py       # semua parameter & "magic numbers"
├── core/
│   ├── capture.py           # loop kamera (orkestrasi)
│   ├── session.py           # state sesi & kontrol label oleh operator
│   ├── dataset_writer.py    # penulis CSV (file dibuka sekali)
│   └── postprocessing.py    # imputasi + sliding window per sesi
├── utils/
│   ├── model_downloader.py  # unduh model MediaPipe
│   ├── normalization.py     # normalisasi koordinat (pure function)
│   └── expression.py        # klasifikasi ekspresi (HANYA overlay visual)
├── data/                     # CSV & .npy hasil akan tersimpan di sini
└── models/                   # file .task MediaPipe akan tersimpan di sini
```

## Cara Menjalankan

```bash
pip install -r requirements.txt
python main.py
```

Saat program berjalan:
1. Masukkan ID partisipan dan ID sesi saat diminta.
2. Pilih sumber input: **kamera live** atau **file video yang sudah ada**.
   - Jika memilih file video, program akan tanya apakah kamu sudah
     tahu label stres yang pasti untuk video itu. Kalau belum (sekadar
     menguji pipeline), semua frame akan diberi label `Unlabeled` —
     **data ini sebaiknya tidak dicampur ke dataset training final**,
     karena tidak mencerminkan kondisi stres yang sesungguhnya.
3. Jendela tampilan akan terbuka (kecuali dipakai lewat kode dengan
   `show_window=False`, untuk batch processing tanpa GUI). Tekan
   `1`/`2`/`3` kapan saja untuk mengubah label stres aktif.
4. Tekan `ESC` untuk berhenti merekam (kamera), atau biarkan video
   selesai diproses sampai akhir secara otomatis.
5. Program otomatis menjalankan imputasi dan membentuk tensor TCN
   (`data/X_tcn_data.npy`, `data/y_tcn_labels.npy`).

### Catatan teknis: kamera vs file video

Kode (`core/capture.py`) menangani kamera dan file video lewat fungsi
yang sama, tapi dengan satu perbedaan penting: cara menghitung
**timestamp** yang dikirim ke MediaPipe.
- **Kamera live**: timestamp diambil dari jam sistem (`time.time()`),
  karena tidak ada timeline intrinsik — frame diambil "sekarang".
- **File video**: timestamp dihitung dari posisi frame
  (`frame_idx * (1000 / fps)`), BUKAN dari jam sistem. Ini penting
  karena saat memproses file, loop berjalan secepat CPU bisa (tidak
  menunggu seperti webcam real-time) — kalau dipaksa pakai jam
  sistem, timestamp antar-frame bisa terlalu rapat atau bahkan tidak
  monoton, dan MediaPipe VIDEO mode akan menolaknya.

Bagian ini sudah diverifikasi dengan video sintetis (60 frame, 30 FPS):
timestamp yang dihasilkan monoton naik dan akurat merepresentasikan
durasi video asli.

## Yang Masih Perlu Kamu Putuskan (di luar scope kode ini)

- **Protokol stres terinduksi**: tugas/stimulus konkret apa yang akan
  memicu tiap level stres saat rekaman (lihat saran di percakapan).
- **Validasi label** (opsional tapi disarankan): kuesioner standar
  seperti PSS setelah sesi, untuk mengonfirmasi label operator cocok
  dengan kondisi subjektif partisipan.
- **Jumlah partisipan & durasi rekaman** per kelas — agar dataset
  cukup besar dan seimbang (`y_tensor` jangan didominasi satu label).
- Arsitektur TCN itu sendiri belum dibangun di sini — modul ini
  berhenti di titik menghasilkan `X_tcn_data.npy` / `y_tcn_labels.npy`
  yang siap dipakai sebagai input training.
