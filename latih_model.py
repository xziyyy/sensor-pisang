"""
Latih & evaluasi model klasifikasi kematangan pisang.

Fitur dinormalisasi (StandardScaler) sebelum masuk KNN, nilai n_neighbors
dicari otomatis lewat cross-validation (GridSearchCV), dan data dibagi
train/test (80/20) supaya akurasi yang dilaporkan dihitung dari data yang
tidak dipakai untuk belajar. Model + scaler disimpan ke model_pisang.pkl.

Cara pakai:
    python latih_model.py

Struktur folder dataset yang diharapkan:
    train/
        busuk/      *.jpg
        mentah/     *.jpg
        mateng/     *.jpg
        kematengan/ *.jpg
"""

import os
import numpy as np
import cv2
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

from ekstraksi_ciri import ambil_ciri

FOLDER_DATASET = "train"
FILE_MODEL = "model_pisang.pkl"


def muat_dataset(folder=FOLDER_DATASET):
    list_ciri, list_label = [], []
    for kategori in sorted(os.listdir(folder)):
        path_kategori = os.path.join(folder, kategori)
        if not os.path.isdir(path_kategori):
            continue

        jumlah = 0
        for nama_file in os.listdir(path_kategori):
            path_gambar = os.path.join(path_kategori, nama_file)
            gambar = cv2.imread(path_gambar)
            if gambar is None:
                continue
            list_ciri.append(ambil_ciri(gambar))
            list_label.append(kategori)
            jumlah += 1

        print(f"  - {kategori}: {jumlah} gambar")

    return np.array(list_ciri), np.array(list_label)


def main():
    print("Memuat dataset...")
    X, y = muat_dataset()
    print(f"Total data: {len(y)}")

    if len(set(y)) < 2:
        print("Dataset perlu minimal 2 kategori dengan gambar valid. "
              "Cek nama folder & path FOLDER_DATASET.")
        return

    nilai, jumlah_tiap = np.unique(y, return_counts=True)
    for k, j in zip(nilai, jumlah_tiap):
        if j < 15:
            print(f"⚠️  Kategori '{k}' hanya punya {j} gambar. "
                  f"Idealnya minimal 30-50 gambar per kategori, beragam "
                  f"pencahayaan & background, supaya model tidak overfit.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsClassifier()),
    ])

    grid_parameter = {
        "knn__n_neighbors": [3, 5, 7, 9, 11],
        "knn__weights": ["uniform", "distance"],
    }

    pencarian = GridSearchCV(pipeline, grid_parameter, cv=5, scoring="accuracy")
    pencarian.fit(X_train, y_train)

    print(f"\nParameter terbaik: {pencarian.best_params_}")
    print(f"Akurasi cross-validation (data latih): {pencarian.best_score_:.2%}")

    model_terbaik = pencarian.best_estimator_
    prediksi = model_terbaik.predict(X_test)

    print("\n Laporan evaluasi pada data uji (belum pernah dilihat model):")
    print(classification_report(y_test, prediksi))
    print("Confusion matrix (baris=label asli, kolom=label prediksi):")
    print(model_terbaik.classes_)
    print(confusion_matrix(y_test, prediksi, labels=model_terbaik.classes_))

    joblib.dump(model_terbaik, FILE_MODEL)
    print(f"\nModel disimpan ke {FILE_MODEL}")


if __name__ == "__main__":
    main()