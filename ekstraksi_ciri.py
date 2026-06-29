"""
Modul ekstraksi ciri (feature extraction) untuk deteksi kematangan pisang.

Strategi peningkatan akurasi dibanding versi awal:
1. Segmentasi: pisahkan objek pisang dari background sebelum hitung ciri,
   supaya warna meja/tangan/dinding di belakang tidak ikut "dianggap" warna pisang.
2. Ciri lebih kaya: bukan cuma rata-rata & std HSV, tapi juga histogram Hue
   (menangkap campuran warna kuning+coklat) dan rasio bercak gelap/coklat
   (indikator kuat tingkat kematangan & pembusukan).
3. Ciri tekstur (GLCM) untuk membedakan kulit licin (mentah) vs kulit
   berbintik (mateng/busuk).
"""

import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops


def segmentasi_pisang(gambar_bgr):

    hsv = cv2.cvtColor(gambar_bgr, cv2.COLOR_BGR2HSV)
    s_channel = hsv[:, :, 1]

    _, mask = cv2.threshold(s_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

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
