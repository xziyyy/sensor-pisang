"""
Deteksi kematangan pisang, realtime lewat browser (bukan jepret foto satu-satu).

Perbedaan dari versi camera_input sebelumnya:
- Pakai streamlit-webrtc supaya kamera menyala terus dan tiap frame video
  langsung diproses & dikasih label, mirip versi desktop (deteksi_kamera.py),
  tapi kameranya tetap kamera milik orang yang buka halaman ini (termasuk HP),
  bukan kamera server.
- Ada smoothing (voting mayoritas beberapa frame terakhir) supaya label
  tidak berkedip-kedip ganti tiap frame hanya karena noise sesaat.
- Ada panel referensi kematangan, diambil dari database_pisang.csv (hasil
  rangkuman dari struktur dataset di folder train/: mentah, mateng,
  kematengan, busuk), supaya orang yang belum familiar bisa paham arti
  tiap label yang muncul di video.

Model (model_pisang.pkl) dan ekstraksi_ciri.py tidak diubah sama sekali.

Cara coba lokal:
    pip install -r requirements.txt
    streamlit run app_web.py
"""

import collections
import queue
import time

import av
import cv2
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

from ekstraksi_ciri import ambil_ciri

FILE_MODEL = "model_pisang.pkl"
FILE_DATABASE = "database_pisang.csv"
JUMLAH_FRAME_SMOOTHING = 10

NAMA_KELAS = {
    "mentah": "Mentah",
    "mateng": "Matang",
    "kematengan": "Sangat Matang",
    "busuk": "Terlalu Matang",
}

WARNA_BGR = {
    "mentah": (86, 156, 74),
    "mateng": (66, 173, 244),
    "kematengan": (43, 125, 236),
    "busuk": (60, 60, 200),
}

RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            # TURN server gratis (Open Relay Project). Diperlukan karena
            # jaringan Streamlit Community Cloud sering memblokir koneksi
            # WebRTC langsung (STUN saja tidak cukup di sana). Layanan
            # gratis ini kadang kurang stabil; kalau masih sering gagal,
            # pertimbangkan pindah ke TURN berbayar seperti Twilio.
            {
                "urls": ["turn:openrelay.metered.ca:80"],
                "username": "openrelayproject",
                "credential": "openrelayproject",
            },
            {
                "urls": ["turn:openrelay.metered.ca:443"],
                "username": "openrelayproject",
                "credential": "openrelayproject",
            },
            {
                "urls": ["turn:openrelay.metered.ca:443?transport=tcp"],
                "username": "openrelayproject",
                "credential": "openrelayproject",
            },
        ]
    }
)


def suntik_gaya():
    st.markdown(
        """
        <style>
        .block-container { padding-top: 2rem; max-width: 980px; }
        h1 { font-weight: 600; letter-spacing: -0.5px; margin-bottom: 0.2rem; }
        .subjudul { color: #6b6b6b; font-size: 0.95rem; margin-bottom: 1.6rem; }
        .kartu {
            border: 1px solid #e4e4e4;
            border-radius: 10px;
            padding: 1rem 1.2rem;
            background: #fafafa;
        }
        .kartu h4 { margin-top: 0; margin-bottom: 0.4rem; font-size: 0.95rem; }
        .baris-referensi {
            display: flex;
            justify-content: space-between;
            padding: 0.35rem 0;
            border-bottom: 1px solid #ececec;
            font-size: 0.9rem;
        }
        .baris-referensi:last-child { border-bottom: none; }
        .titik {
            display: inline-block;
            width: 9px; height: 9px;
            border-radius: 50%;
            margin-right: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def muat_model():
    return joblib.load(FILE_MODEL)


@st.cache_data
def muat_database():
    try:
        return pd.read_csv(FILE_DATABASE)
    except FileNotFoundError:
        return None


class ProsesorVideo:
    """
    Jalan di thread terpisah milik streamlit-webrtc. Tiap frame video masuk
    lewat recv(), diproses pakai model yang sudah dilatih, lalu label
    digambar langsung di atas frame sebelum dikirim balik ke browser.

    Hasil tiap frame juga dikirim ke antrian (queue) supaya panel di luar
    video (statistik singkat) bisa ikut update tanpa perlu jepret ulang.
    """

    def __init__(self, model, antrian_hasil):
        self.model = model
        self.antrian_hasil = antrian_hasil
        self.riwayat = collections.deque(maxlen=JUMLAH_FRAME_SMOOTHING)

    def recv(self, frame):
        gambar = frame.to_ndarray(format="bgr24")

        try:
            ciri = ambil_ciri(gambar).reshape(1, -1)
            probabilitas = self.model.predict_proba(ciri)[0]
            idx_terbaik = probabilitas.argmax()
            kategori_frame = self.model.classes_[idx_terbaik]
            keyakinan = probabilitas[idx_terbaik]
        except Exception:
            return av.VideoFrame.from_ndarray(gambar, format="bgr24")

        self.riwayat.append(kategori_frame)
        kategori_stabil = collections.Counter(self.riwayat).most_common(1)[0][0]

        label = NAMA_KELAS.get(kategori_stabil, kategori_stabil)
        warna = WARNA_BGR.get(kategori_stabil, (200, 200, 200))

        tinggi = gambar.shape[0]
        cv2.rectangle(gambar, (0, tinggi - 70), (420, tinggi), (25, 25, 25), -1)
        cv2.putText(gambar, label, (16, tinggi - 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.95, warna, 2, cv2.LINE_AA)
        cv2.putText(gambar, f"keyakinan {keyakinan:.0%}", (16, tinggi - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

        try:
            self.antrian_hasil.put_nowait((kategori_stabil, float(keyakinan)))
        except queue.Full:
            pass

        return av.VideoFrame.from_ndarray(gambar, format="bgr24")


def tampilkan_panel_referensi(database):
    st.markdown("<div class='kartu'>", unsafe_allow_html=True)
    st.markdown("<h4>Referensi tingkat kematangan</h4>", unsafe_allow_html=True)

    if database is None:
        st.write("File database_pisang.csv tidak ditemukan.")
    else:
        peta_warna_hex = {
            "Mentah": "#569c4a",
            "Mengkal": "#a3b845",
            "Matang": "#f4ad44",
            "Sangat Matang": "#ec7d2c",
            "Terlalu Matang/Busuk": "#c83c3c",
        }
        for _, baris in database.iterrows():
            warna_hex = peta_warna_hex.get(baris["nama_indonesia"], "#999")
            st.markdown(
                f"<div class='baris-referensi'>"
                f"<span><span class='titik' style='background:{warna_hex}'></span>"
                f"{baris['nama_indonesia']}</span>"
                f"<span style='color:#8a8a8a'>{baris['warna_dominan']}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)


def tampilkan_panel_status(antrian_hasil, sedang_jalan):
    st.markdown("<div class='kartu'>", unsafe_allow_html=True)
    st.markdown("<h4>Status saat ini</h4>", unsafe_allow_html=True)
    tempat_status = st.empty()

    if not sedang_jalan:
        tempat_status.write("Kamera belum aktif.")
        st.markdown("</div>", unsafe_allow_html=True)
        return

    hasil_terakhir = None
    batas_waktu = time.time() + 0.3
    while time.time() < batas_waktu:
        try:
            hasil_terakhir = antrian_hasil.get(timeout=0.1)
        except queue.Empty:
            break

    if hasil_terakhir is None:
        tempat_status.write("Menunggu data dari kamera...")
    else:
        kategori, keyakinan = hasil_terakhir
        label = NAMA_KELAS.get(kategori, kategori)
        tempat_status.markdown(f"**{label}**  \nkeyakinan {keyakinan:.0%}")

    st.markdown("</div>", unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="Deteksi Kematangan Pisang", layout="centered")
    suntik_gaya()

    st.title("Deteksi Kematangan Pisang")
    st.markdown(
        "<div class='subjudul'>Arahkan kamera ke pisang. Label dan tingkat "
        "keyakinan model tampil langsung di video, tanpa perlu jepret foto.</div>",
        unsafe_allow_html=True,
    )

    model = muat_model()
    database = muat_database()

    if "antrian_hasil" not in st.session_state:
        st.session_state.antrian_hasil = queue.Queue(maxsize=1)

    # Ambil ke variabel biasa dulu. video_processor_factory dipanggil dari
    # thread lain milik streamlit-webrtc (bukan thread utama Streamlit),
    # dan st.session_state tidak bisa diakses dari thread itu -> harus
    # sudah berupa objek Python biasa sebelum masuk ke closure/lambda.
    antrian_hasil = st.session_state.antrian_hasil

    kolom_video, kolom_panel = st.columns([3, 2])

    with kolom_video:
        ctx = webrtc_streamer(
            key="deteksi-pisang",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=lambda: ProsesorVideo(
                model, antrian_hasil
            ),
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

    with kolom_panel:
        tampilkan_panel_status(antrian_hasil, ctx.state.playing)
        tampilkan_panel_referensi(database)

    st.markdown(
        "<div class='subjudul' style='margin-top:1.4rem'>"
        "Model dilatih dari kumpulan foto pisang pada berbagai tahap kematangan "
        "(folder train: mentah, mateng, kematengan, busuk). Akurasi bergantung "
        "pada pencahayaan dan seberapa jelas pisang terlihat di kamera."
        "</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
