import collections
import cv2
import joblib

from ekstraksi_ciri import ambil_ciri

LISTFOLDER = {
    "busuk": "Busuk",
    "mentah": "Mentah",
    "mateng": "Matang",
    "kematengan": "Terlalu Matang",
}

WARNA_TEKS = {
    "busuk": (0, 0, 255),
    "mentah": (0, 255, 0),
    "mateng": (0, 255, 255),
    "kematengan": (0, 165, 255),
}

JUMLAH_FRAME_SMOOTHING = 10
FILE_MODEL = "model_pisang.pkl"


def main():
    model = joblib.load(FILE_MODEL)
    print("Model dimuat. Tekan Q untuk keluar.")

    riwayat_prediksi = collections.deque(maxlen=JUMLAH_FRAME_SMOOTHING)
    kamera = cv2.VideoCapture(0)

    while True:
        ret, frame = kamera.read()
        if not ret:
            print("Kamera tidak terdeteksi!")
            break

        ciri = ambil_ciri(frame).reshape(1, -1)
        probabilitas = model.predict_proba(ciri)[0]
        idx_terbaik = probabilitas.argmax()
        kategori_frame = model.classes_[idx_terbaik]
        keyakinan = probabilitas[idx_terbaik]

        riwayat_prediksi.append(kategori_frame)
        kategori_stabil = collections.Counter(riwayat_prediksi).most_common(1)[0][0]

        label = LISTFOLDER.get(kategori_stabil, kategori_stabil)
        warna = WARNA_TEKS.get(kategori_stabil, (255, 255, 255))

        cv2.putText(frame, f"Status: {label}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, warna, 3)
        cv2.putText(frame, f"Keyakinan: {keyakinan:.0%}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Deteksi Kematangan Pisang", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    kamera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
