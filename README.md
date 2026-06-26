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
  memicu tiap level stres saat rekaman .
- **Validasi label** (opsional tapi disarankan): kuesioner standar
  seperti PSS setelah sesi, untuk mengonfirmasi label operator cocok
  dengan kondisi subjektif partisipan.
- **Jumlah partisipan & durasi rekaman** per kelas — agar dataset
  cukup besar dan seimbang (`y_tensor` jangan didominasi satu label).
- Arsitektur TCN itu sendiri belum dibangun di sini — modul ini
  berhenti di titik menghasilkan `X_tcn_data.npy` / `y_tcn_labels.npy`
  yang siap dipakai sebagai input training.
