Deteksi Kematangan Pisang - Mobile Pipeline

Ringkasan:
- `train_cnn.py`: latih model CNN ringan dari folder `train/` dan keluarannya `model.h5`.
- `convert_to_tflite.py`: convert `model.h5` ke `model.tflite`.
- `app/`: contoh Kivy app yang menampilkan kamera dan memuat `model.tflite` (jika ada) untuk inferensi.

Langkah cepat (desktop):
1. Install dependencies (disarankan virtualenv):

```bash
pip install tensorflow opencv-python numpy kivy kivymd
```

2. Latih model:

```bash
python train_cnn.py --data_dir train --output model.h5 --epochs 10
```

3. Convert ke TFLite:

```bash
python convert_to_tflite.py --input model.h5 --output model.tflite
```

4. Jalankan contoh Kivy (desktop):

```bash
python app/main.py
```

Membangun APK (ringkasan):
- Gunakan Buildozer pada Linux/WSL; di Google Colab bisa di-setup tetapi perlu banyak download (SDK/NDK) dan bisa memakan waktu/ruang.
- Perbarui `buildozer.spec` sesuai kebutuhan dan tambahkan `tflite-runtime` atau `tensorflow` di `requirements`.

Catatan teknis & tradeoffs:
- Mengonversi model Keras ke TFLite memberi performa on-device yang baik.
- Jika Anda ingin memakai ekstraksi fitur klasik (`ekstraksi_ciri.py` + scikit-learn), porting ke Android memerlukan bundling OpenCV native dan scikit-learn yang lebih sulit.
