"""
SmartFruit - deteksi kematangan pisang lewat browser, dua mode:
- Realtime : kamera menyala terus, label update tiap frame.
- Ambil Foto: jepret satu kali, hasil dianalisis dari foto itu saja.

Mode dipilih lewat menu (tombol "Menu" di kanan atas, membuka panel kecil
berisi pilihan mode) -- bukan selector besar yang selalu tampil di layar.

Fitur lain yang sudah ada sebelumnya, dipakai di kedua mode:
- Video/foto dibalik (cv2.flip) supaya tidak mirror/kaca.
- Kotak pembatas digambar di sekeliling pisang yang terdeteksi.
- Klasifikasi hanya jalan kalau ada objek yang tersegmentasi dari
  background (lihat deteksi_area_pisang()).
- Panel referensi tingkat kematangan (skala Von Loesecke) selalu tampil
  di bawah, di kedua mode.

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
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

from ekstraksi_ciri import ambil_ciri, segmentasi_pisang

FILE_MODEL = "model_pisang.pkl"
JUMLAH_FRAME_SMOOTHING = 10
UKURAN_ANALISIS = (200, 200)  # harus sama dengan resize di dalam ambil_ciri()

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

# Referensi tingkat kematangan, dipetakan dari skala Von Loesecke - skala
# 7 tahap warna kulit pisang yang jadi rujukan umum di riset & industri
# pisang sejak 1949, dan masih dipakai sampai sekarang untuk penyortiran
# pisang secara visual. Dipetakan ke 4 kategori yang dipakai model ini.
REFERENSI_KEMATANGAN = [
    {
        "kelas": "mentah",
        "label": "Mentah",
        "deskripsi": "Kulit hijau penuh, atau hijau dengan sedikit semburat kuning",
        "warna_hex": "#569c4a",
    },
    {
        "kelas": "mateng",
        "label": "Matang",
        "deskripsi": "Kulit kuning dengan ujung sedikit hijau, sampai kuning penuh",
        "warna_hex": "#f4ad44",
    },
    {
        "kelas": "kematengan",
        "label": "Sangat Matang",
        "deskripsi": "Kuning penuh dengan bintik-bintik coklat mulai muncul",
        "warna_hex": "#ec7d2c",
    },
    {
        "kelas": "busuk",
        "label": "Terlalu Matang",
        "deskripsi": "Bintik coklat meluas atau kulit sudah menghitam",
        "warna_hex": "#c83c3c",
    },
]

RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            # TURN server gratis (Open Relay Project). Diperlukan karena
            # jaringan Streamlit Community Cloud sering memblokir koneksi
            # WebRTC langsung (STUN saja tidak cukup di sana).
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
        .block-container { padding-top: 1.6rem; max-width: 980px; }
        h1 { font-weight: 600; letter-spacing: -0.5px; margin: 0; font-size: 1.6rem; }
        .subjudul { color: #6b6b6b; font-size: 0.95rem; margin: 0.3rem 0 1.4rem 0; }
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
            gap: 0.8rem;
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
            flex-shrink: 0;
        }
        div[data-testid="stPopover"] button {
            border-radius: 8px;
            font-size: 1.1rem;
            padding: 0.25rem 0.7rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def muat_model():
    return joblib.load(FILE_MODEL)


def deteksi_area_pisang(gambar_bgr):
    """
    Pakai ulang segmentasi_pisang() dari ekstraksi_ciri.py untuk mengecek
    apakah ada objek yang cukup besar tersegmentasi dari background, dan
    sekaligus menghitung kotak pembatas (bounding box) di sekelilingnya.

    Mengembalikan (ada_objek, kotak) dengan kotak berupa (x, y, w, h)
    dalam koordinat frame ASLI, atau None kalau tidak ada objek.
    """
    tinggi_asli, lebar_asli = gambar_bgr.shape[:2]
    gambar_kecil = cv2.resize(gambar_bgr, UKURAN_ANALISIS)
    mask = segmentasi_pisang(gambar_kecil)

    if cv2.countNonZero(mask) >= mask.size:
        return False, None

    x, y, w, h = cv2.boundingRect(mask)
    skala_x = lebar_asli / UKURAN_ANALISIS[0]
    skala_y = tinggi_asli / UKURAN_ANALISIS[1]
    kotak = (int(x * skala_x), int(y * skala_y), int(w * skala_x), int(h * skala_y))
    return True, kotak


def proses_frame(model, gambar_bgr):
    """
    Satu langkah klasifikasi untuk satu gambar (dipakai di kedua mode).
    Mengembalikan (ada_objek, kotak, kategori, keyakinan).
    Tidak ada smoothing di sini -- smoothing khusus dipakai di mode
    realtime saja (lihat ProsesorVideo).
    """
    ada_objek, kotak = deteksi_area_pisang(gambar_bgr)
    if not ada_objek:
        return False, None, None, 0.0

    try:
        ciri = ambil_ciri(gambar_bgr).reshape(1, -1)
        probabilitas = model.predict_proba(ciri)[0]
        idx_terbaik = probabilitas.argmax()
        kategori = model.classes_[idx_terbaik]
        keyakinan = float(probabilitas[idx_terbaik])
    except Exception:
        return False, None, None, 0.0

    return True, kotak, kategori, keyakinan


def gambar_kotak_dan_label(gambar_bgr, kotak, kategori, keyakinan):
    x, y, w, h = kotak
    label = NAMA_KELAS.get(kategori, kategori)
    warna = WARNA_BGR.get(kategori, (200, 200, 200))
    tinggi = gambar_bgr.shape[0]

    cv2.rectangle(gambar_bgr, (x, y), (x + w, y + h), warna, 2)
    cv2.rectangle(gambar_bgr, (0, tinggi - 70), (420, tinggi), (25, 25, 25), -1)
    cv2.putText(gambar_bgr, label, (16, tinggi - 38),
                cv2.FONT_HERSHEY_SIMPLEX, 0.95, warna, 2, cv2.LINE_AA)
    cv2.putText(gambar_bgr, f"keyakinan {keyakinan:.0%}", (16, tinggi - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    return gambar_bgr


def gambar_status_kosong(gambar_bgr, teks="Tidak ada pisang terdeteksi"):
    tinggi = gambar_bgr.shape[0]
    cv2.rectangle(gambar_bgr, (0, tinggi - 70), (420, tinggi), (25, 25, 25), -1)
    cv2.putText(gambar_bgr, teks, (16, tinggi - 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (190, 190, 190), 2, cv2.LINE_AA)
    return gambar_bgr


class ProsesorVideo:
    """
    Jalan di thread terpisah milik streamlit-webrtc. Tiap frame video masuk
    lewat recv(), diproses pakai model yang sudah dilatih, lalu label +
    kotak pembatas digambar langsung di atas frame sebelum dikirim balik
    ke browser. Dipakai khusus untuk mode Realtime.
    """

    def __init__(self, model, antrian_hasil, balik_gambar):
        self.model = model
        self.antrian_hasil = antrian_hasil
        self.balik_gambar = balik_gambar
        self.riwayat = collections.deque(maxlen=JUMLAH_FRAME_SMOOTHING)

    def recv(self, frame):
        gambar = frame.to_ndarray(format="bgr24")
        if self.balik_gambar:
            # Kamera depan biasanya dikirim browser dalam kondisi "kaca
            # cermin" (kanan-kiri terbalik) untuk preview normal, jadi
            # dibalik lagi di sini. Kamera belakang TIDAK di-mirror oleh
            # browser, jadi tidak perlu (dan tidak boleh) dibalik.
            gambar = cv2.flip(gambar, 1)

        ada_objek, kotak, kategori_frame, keyakinan = proses_frame(self.model, gambar)

        if not ada_objek:
            self.riwayat.clear()
            gambar = gambar_status_kosong(gambar)
            try:
                self.antrian_hasil.put_nowait((None, 0.0))
            except queue.Full:
                pass
            return av.VideoFrame.from_ndarray(gambar, format="bgr24")

        self.riwayat.append(kategori_frame)
        kategori_stabil = collections.Counter(self.riwayat).most_common(1)[0][0]

        gambar = gambar_kotak_dan_label(gambar, kotak, kategori_stabil, keyakinan)

        try:
            self.antrian_hasil.put_nowait((kategori_stabil, keyakinan))
        except queue.Full:
            pass

        return av.VideoFrame.from_ndarray(gambar, format="bgr24")


def tampilkan_panel_referensi():
    st.markdown("<div class='kartu'>", unsafe_allow_html=True)
    st.markdown("<h4>Referensi tingkat kematangan</h4>", unsafe_allow_html=True)
    for item in REFERENSI_KEMATANGAN:
        st.markdown(
            f"<div class='baris-referensi'>"
            f"<span><span class='titik' style='background:{item['warna_hex']}'></span>"
            f"{item['label']}</span>"
            f"<span style='color:#8a8a8a; text-align:right'>{item['deskripsi']}</span>"
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
        if kategori is None:
            tempat_status.write("Tidak ada pisang terdeteksi di kamera.")
        else:
            label = NAMA_KELAS.get(kategori, kategori)
            tempat_status.markdown(f"**{label}**  \nkeyakinan {keyakinan:.0%}")

    st.markdown("</div>", unsafe_allow_html=True)


def mode_realtime(model, arah_kamera):
    if "antrian_hasil" not in st.session_state:
        st.session_state.antrian_hasil = queue.Queue(maxsize=1)
    antrian_hasil = st.session_state.antrian_hasil

    balik_gambar = arah_kamera == "Depan"
    facing_mode = "user" if arah_kamera == "Depan" else "environment"

    kolom_video, kolom_panel = st.columns([3, 2])

    with kolom_video:
        ctx = webrtc_streamer(
            key="deteksi-pisang-realtime",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=lambda: ProsesorVideo(
                model, antrian_hasil, balik_gambar
            ),
            media_stream_constraints={
                "video": {"facingMode": facing_mode},
                "audio": False,
            },
            async_processing=True,
        )

    with kolom_panel:
        tampilkan_panel_status(antrian_hasil, ctx.state.playing)


def mode_ambil_foto(model, arah_kamera):
    balik_gambar = arah_kamera == "Depan"
    kolom_foto, kolom_hasil = st.columns([3, 2])

    with kolom_foto:
        foto = st.camera_input("Ambil foto pisang", label_visibility="collapsed")

    with kolom_hasil:
        st.markdown("<div class='kartu'>", unsafe_allow_html=True)
        st.markdown("<h4>Hasil</h4>", unsafe_allow_html=True)

        if foto is None:
            st.write("Belum ada foto. Klik ikon kamera di sebelah kiri.")
        else:
            bytes_data = foto.getvalue()
            array_np = np.frombuffer(bytes_data, dtype=np.uint8)
            gambar = cv2.imdecode(array_np, cv2.IMREAD_COLOR)
            if balik_gambar:
                gambar = cv2.flip(gambar, 1)

            with st.spinner("Menganalisis..."):
                ada_objek, kotak, kategori, keyakinan = proses_frame(model, gambar)

            if not ada_objek:
                st.write("Tidak ada pisang terdeteksi pada foto ini.")
            else:
                label = NAMA_KELAS.get(kategori, kategori)
                st.markdown(f"**{label}**  \nkeyakinan {keyakinan:.0%}")

        st.markdown("</div>", unsafe_allow_html=True)

    if foto is not None and ada_objek:
        gambar_ditandai = gambar_kotak_dan_label(gambar.copy(), kotak, kategori, keyakinan)
        gambar_rgb = cv2.cvtColor(gambar_ditandai, cv2.COLOR_BGR2RGB)
        with kolom_foto:
            st.image(gambar_rgb, use_container_width=True)


def panel_menu():
    """
    Menu tersembunyi di kanan atas (bentuk tombol kecil "Menu"), isinya
    pilihan mode dan arah kamera. Default-nya tertutup, tidak memenuhi
    layar seperti selector besar yang selalu tampil.
    """
    if "mode_terpilih" not in st.session_state:
        st.session_state.mode_terpilih = "Realtime"
    if "arah_kamera" not in st.session_state:
        st.session_state.arah_kamera = "Depan"

    kolom_judul, kolom_menu = st.columns([5, 1])
    with kolom_judul:
        st.title("SmartFruit")
    with kolom_menu:
        with st.popover("Menu", use_container_width=True):
            st.radio(
                "Mode deteksi",
                options=["Realtime", "Ambil Foto"],
                key="mode_terpilih",
            )
            st.radio(
                "Kamera",
                options=["Depan", "Belakang"],
                key="arah_kamera",
                help="Kamera depan biasanya perlu dibalik (anti-mirror). "
                     "Kamera belakang tidak perlu dibalik.",
            )


def main():
    st.set_page_config(page_title="SmartFruit", layout="centered")
    suntik_gaya()

    panel_menu()

    model = muat_model()
    mode = st.session_state.mode_terpilih
    arah_kamera = st.session_state.arah_kamera

    if mode == "Realtime":
        st.markdown(
            "<div class='subjudul'>Arahkan kamera ke pisang. Label dan kotak "
            "pembatas tampil langsung di video.</div>",
            unsafe_allow_html=True,
        )
        mode_realtime(model, arah_kamera)
    else:
        st.markdown(
            "<div class='subjudul'>Ambil satu foto pisang, hasil analisis "
            "muncul setelah foto diambil.</div>",
            unsafe_allow_html=True,
        )
        mode_ambil_foto(model, arah_kamera)

    st.markdown("<div style='margin-top:1.2rem'></div>", unsafe_allow_html=True)
    tampilkan_panel_referensi()


if __name__ == "__main__":
    main()
