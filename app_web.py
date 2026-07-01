"""
Versi WEB dari deteksi_kamera.py — supaya bisa diakses orang lain lewat
link, termasuk dari HP.

Perbedaan utama dari deteksi_kamera.py:
- Kamera TIDAK diambil lewat cv2.VideoCapture(0) (itu kamera server/laptop
  kamu sendiri, orang lain gak akan bisa akses kamera kamu dari HP mereka).
  Di sini kamera diambil lewat st.camera_input(), yaitu kamera BROWSER
  milik orang yang buka link ini (kamera laptop/HP mereka sendiri).
- Tidak pakai cv2.imshow()/while-loop/waitKey, karena itu butuh window
  desktop yang gak ada di server. Streamlit yang urus tampilan di browser.
- Model (model_pisang.pkl) dan ekstraksi_ciri.py TIDAK diubah sama sekali,
  dipakai persis seperti aslinya.

Cara coba di komputer sendiri dulu sebelum deploy:
    pip install -r requirements.txt
    streamlit run app_web.py
Nanti otomatis buka di browser http://localhost:8501
"""

import cv2
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from ekstraksi_ciri import ambil_ciri

FILE_MODEL = "model_pisang.pkl"
FILE_DATABASE = "database_pisang.csv"

LABEL_TAMPILAN = {
    "busuk": "Terlalu Matang / Busuk",
    "mentah": "Mentah",
    "mateng": "Matang",
    "kematengan": "Sangat Matang",
}

WARNA_TAMPILAN = {
    "busuk": "red",
    "mentah": "green",
    "mateng": "orange",
    "kematengan": "orange",
}


@st.cache_resource
def muat_model():
    return joblib.load(FILE_MODEL)


@st.cache_data
def muat_database():
    try:
        return pd.read_csv(FILE_DATABASE)
    except FileNotFoundError:
        return None


def foto_ke_gambar_bgr(file_foto):
    """Ubah hasil st.camera_input (format bytes) jadi array BGR untuk OpenCV."""
    bytes_data = file_foto.getvalue()
    array_np = np.frombuffer(bytes_data, dtype=np.uint8)
    return cv2.imdecode(array_np, cv2.IMREAD_COLOR)


def main():
    st.set_page_config(page_title="Deteksi Kematangan Pisang", page_icon="🍌")
    st.title("🍌 Deteksi Kematangan Pisang")
    st.write(
        "Ambil foto pisang pakai kamera (bisa kamera HP kalau dibuka dari HP), "
        "lalu sistem akan memprediksi tingkat kematangannya."
    )

    model = muat_model()
    database = muat_database()

    foto = st.camera_input("Arahkan kamera ke pisang, lalu ambil foto")

    if foto is not None:
        gambar_bgr = foto_ke_gambar_bgr(foto)

        with st.spinner("Menganalisis..."):
            ciri = ambil_ciri(gambar_bgr).reshape(1, -1)
            probabilitas = model.predict_proba(ciri)[0]
            idx_terbaik = probabilitas.argmax()
            kategori = model.classes_[idx_terbaik]
            keyakinan = probabilitas[idx_terbaik]

        label = LABEL_TAMPILAN.get(kategori, kategori)
        warna = WARNA_TAMPILAN.get(kategori, "gray")

        st.markdown(f"### Status: :{warna}[{label}]")
        st.write(f"Keyakinan model: **{keyakinan:.0%}**")

        with st.expander("Lihat detail probabilitas tiap kelas"):
            for kelas, prob in zip(model.classes_, probabilitas):
                st.write(f"- {LABEL_TAMPILAN.get(kelas, kelas)}: {prob:.1%}")

        if database is not None:
            baris = database[
                database["nama_indonesia"].str.contains(
                    label.split(" / ")[0], case=False, na=False
                )
            ]
            if not baris.empty:
                st.info(f"💡 Saran: {baris.iloc[0]['saran']}")


if __name__ == "__main__":
    main()
