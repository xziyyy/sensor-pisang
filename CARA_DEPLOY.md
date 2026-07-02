# Cara Deploy — SmartFruit (bisa diakses siapa saja, termasuk HP)

## Yang perlu kamu siapkan dulu
Kumpulkan file-file ini jadi satu folder/repo:
- `app_web.py`          (yang baru dibuat, ganti kamera desktop -> kamera browser)
- `ekstraksi_ciri.py`   (punya kamu, tidak diubah)
- `model_pisang.pkl`    (hasil `latih_model.py` yang sudah kamu jalankan — WAJIB ikut diupload, jangan cuma script latih_model.py-nya)
- `requirements.txt`
- `database_pisang.csv` (opsional, kalau mau tampilkan saran per kategori)

`deteksi_kamera.py` dan `latih_model.py` yang lama **tidak perlu ikut dideploy** —
`deteksi_kamera.py` cuma untuk kamu coba lokal (buka kamera laptop sendiri),
`latih_model.py` cuma dijalankan sekali di komputer kamu untuk menghasilkan
`model_pisang.pkl`.

## Langkah 1 — Coba dulu di laptop sendiri
```
pip install -r requirements.txt
streamlit run app_web.py
```
Browser otomatis kebuka. Coba ambil foto pisang, pastikan hasilnya masuk akal
sebelum lanjut deploy.

## Langkah 2 — Upload ke GitHub
1. Buat repo baru di github.com (boleh public/private).
2. Upload semua file di atas ke repo itu (termasuk `model_pisang.pkl`).
   - Kalau `model_pisang.pkl` ukurannya lebih dari 100MB, GitHub biasa akan
     menolak — kalau kejadian, kabari saya, ada cara lain (Git LFS atau
     compress model KNN-nya).

## Langkah 3 — Deploy gratis pakai Streamlit Community Cloud
1. Buka **share.streamlit.io**, login pakai akun GitHub.
2. Klik "New app", pilih repo yang tadi diupload.
3. Isi "Main file path" = `app_web.py`.
4. Klik Deploy.
5. Setelah selesai build (biasanya 1-3 menit), kamu dapat link publik seperti:
   `https://nama-app-kamu.streamlit.app`

Link ini bisa dibuka **dari HP siapa saja** lewat browser (Chrome/Safari) —
tidak perlu install apa-apa. Saat mereka buka `st.camera_input`, browser HP
mereka akan minta izin akses kamera, dan itu kamera HP mereka sendiri yang
kepakai (bukan kamera server).

## Alternatif lain (kalau perlu)
| Platform | Gratis? | Cocok untuk |
|---|---|---|
| Streamlit Community Cloud | Ya | Cara paling gampang, direkomendasikan |
| Hugging Face Spaces (pilih SDK: Streamlit) | Ya | Alternatif kalau Streamlit Cloud lagi penuh/limit |
| Render.com | Ada free tier terbatas | Kalau nanti mau upgrade jadi Flask/FastAPI |

## Kalau nanti errornya soal opencv
Di `requirements.txt` sengaja saya pakai `opencv-python-headless`, BUKAN
`opencv-python` biasa. Versi biasa butuh library GUI (untuk `cv2.imshow`)
yang tidak ada di server cloud, dan akan bikin proses deploy gagal.
Fungsi `ambil_ciri()` di `ekstraksi_ciri.py` tidak pakai `cv2.imshow` sama
sekali, jadi aman pakai versi headless.
