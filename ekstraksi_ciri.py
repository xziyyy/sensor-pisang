"""
Modul ekstraksi ciri (feature extraction) untuk deteksi kematangan pisang.

Strategi peningkatan akurasi dibanding versi awal:
1. Segmentasi: pisahkan objek pisang dari background sebelum hitung ciri,
   supaya warna meja/tangan/dinding di belakang tidak ikut "dianggap" warna pisang.
   Tiga tahap penyaringan dipakai supaya tidak asal ambil objek berwarna:
     a. Saturasi tinggi (Otsu) - objek lebih jenuh warnanya dari background.
     b. Rentang warna khas pisang (hijau-kuning) + bercak gelap/coklat -
        kulit tangan/wajah manusia umumnya di luar rentang ini.
     c. Bentuk memanjang (lonjong) - pisang berbentuk panjang, bukan
        bulat/kompak seperti tangan mengepal atau wajah.
2. Ciri lebih kaya: bukan cuma rata-rata & std HSV, tapi juga histogram Hue
   (menangkap campuran warna kuning+coklat) dan rasio bercak gelap/coklat
   (indikator kuat tingkat kematangan & pembusukan).
3. Ciri tekstur (GLCM) untuk membedakan kulit licin (mentah) vs kulit
   berbintik (mateng/busuk).

PENTING: kalau file ini diubah, model_pisang.pkl HARUS dilatih ulang
(jalankan latih_model.py lagi). Model lama dilatih dari ciri hasil
segmentasi versi sebelumnya, jadi tidak lagi cocok/akurat dipakai
bersama versi segmentasi yang baru.
"""

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

# Rentang warna khas pisang dalam HSV (H: 0-179 di OpenCV). Mencakup
# hijau (mentah) sampai kuning (matang). Nilai ini heuristik/perkiraan -
# kalau di uji coba nyata pisang asli malah sering ditolak, coba lebarkan
# rentangnya; kalau tangan/benda lain masih sering lolos, coba persempit.
BATAS_WARNA_PISANG_BAWAH = (15, 35, 30)
BATAS_WARNA_PISANG_ATAS = (95, 255, 255)

# Bercak coklat/hitam pada pisang sangat matang/busuk seringkali value-nya
# rendah (gelap) dan tidak selalu masuk rentang warna di atas, jadi
# ditambahkan sebagai jalur terpisah supaya tetap terhitung sebagai bagian
# dari pisang, bukan dibuang sebagai "background".
BATAS_BERCAK_GELAP_BAWAH = (0, 0, 0)
BATAS_BERCAK_GELAP_ATAS = (179, 130, 95)

# Rasio panjang:lebar minimum (dari minAreaRect) supaya kontur dianggap
# berbentuk pisang (lonjong), bukan objek bulat/kompak seperti tangan atau
# wajah. Naikkan kalau masih banyak salah deteksi objek bulat; turunkan
# kalau pisang asli malah sering ditolak (misal pisang difoto dari ujung,
# atau tumpukan pisang yang bentuknya lebih kompak).
AMBANG_RASIO_MEMANJANG = 1.4


def segmentasi_pisang(gambar_bgr):
    hsv = cv2.cvtColor(gambar_bgr, cv2.COLOR_BGR2HSV)
    s_channel = hsv[:, :, 1]

    _, mask_saturasi = cv2.threshold(s_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    mask_warna_pisang = cv2.inRange(hsv, BATAS_WARNA_PISANG_BAWAH, BATAS_WARNA_PISANG_ATAS)
    mask_bercak_gelap = cv2.inRange(hsv, BATAS_BERCAK_GELAP_BAWAH, BATAS_BERCAK_GELAP_ATAS)
    mask_warna = cv2.bitwise_or(mask_warna_pisang, mask_bercak_gelap)

    mask = cv2.bitwise_and(mask_saturasi, mask_warna)

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    kontur, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not kontur:
        return np.full(s_channel.shape, 255, dtype=np.uint8)

    kontur_terbesar = max(kontur, key=cv2.contourArea)
    luas_relatif = cv2.contourArea(kontur_terbesar) / (mask.shape[0] * mask.shape[1])

    if luas_relatif < 0.05:
        return np.full(s_channel.shape, 255, dtype=np.uint8)

    (_, (lebar_kotak, tinggi_kotak), _) = cv2.minAreaRect(kontur_terbesar)
    sisi_pendek = min(lebar_kotak, tinggi_kotak)
    sisi_panjang = max(lebar_kotak, tinggi_kotak)
    rasio_memanjang = sisi_panjang / max(sisi_pendek, 1)

    if rasio_memanjang < AMBANG_RASIO_MEMANJANG:
        return np.full(s_channel.shape, 255, dtype=np.uint8)

    mask_final = np.zeros_like(mask)
    cv2.drawContours(mask_final, [kontur_terbesar], -1, 255, -1)
    return mask_final


def _ciri_warna(hsv, mask):
    mean, std = cv2.meanStdDev(hsv, mask=mask)
    mean = mean.flatten()
    std = std.flatten()

    hist_h = cv2.calcHist([hsv], [0], mask, [8], [0, 180])
    hist_h = cv2.normalize(hist_h, hist_h).flatten()

    return np.concatenate([mean, std, hist_h])


def _ciri_bercak(hsv, mask):
    """
    Rasio piksel 'gelap/coklat' (bercak kematangan/busuk) terhadap
    total piksel objek. Pisang yang makin matang/busuk biasanya makin
    banyak bercak coklat-kehitaman pada kulitnya.
    """
    v_channel = hsv[:, :, 2]
    area_objek = max(cv2.countNonZero(mask), 1)

    gelap = (v_channel < 90).astype(np.uint8) * 255
    bercak_mask = cv2.bitwise_and(gelap, mask)
    rasio_bercak = cv2.countNonZero(bercak_mask) / area_objek

    return np.array([rasio_bercak])


def _ciri_tekstur(gambar_bgr, mask):
    """Ciri tekstur GLCM pada area objek saja (kulit licin vs berbintik)."""
    abu = cv2.cvtColor(gambar_bgr, cv2.COLOR_BGR2GRAY)
    abu_masked = cv2.bitwise_and(abu, abu, mask=mask)
    abu_kecil = cv2.resize(abu_masked, (64, 64))

    glcm = graycomatrix(abu_kecil, distances=[1], angles=[0],
                         levels=256, symmetric=True, normed=True)
    kontras = graycoprops(glcm, "contrast")[0, 0]
    homogenitas = graycoprops(glcm, "homogeneity")[0, 0]
    energi = graycoprops(glcm, "energy")[0, 0]

    return np.array([kontras, homogenitas, energi])


def ambil_ciri(gambar_bgr):
    """Ekstrak satu vektor ciri lengkap dari satu gambar pisang."""
    gambar_bgr = cv2.resize(gambar_bgr, (200, 200))
    mask = segmentasi_pisang(gambar_bgr)
    hsv = cv2.cvtColor(gambar_bgr, cv2.COLOR_BGR2HSV)

    ciri_warna = _ciri_warna(hsv, mask)
    ciri_bercak = _ciri_bercak(hsv, mask)
    ciri_tekstur = _ciri_tekstur(gambar_bgr, mask)

    return np.concatenate([ciri_warna, ciri_bercak, ciri_tekstur])
