"""
Latih model CNN sederhana dari folder `train/` dan simpan `model.h5`.
Expectasi: struktur folder `train/<label>/*.jpg` seperti saat ini.
"""

import os
import argparse
import tensorflow as tf
from tensorflow.keras import layers, models


def build_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dropout(0.4),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer="adam",
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def main(args):
    data_dir = args.data_dir
    img_size = (128, 128)
    batch_size = 32

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        labels='inferred',
        label_mode='int',
        image_size=img_size,
        batch_size=batch_size,
        validation_split=0.2,
        subset='training',
        seed=123
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        labels='inferred',
        label_mode='int',
        image_size=img_size,
        batch_size=batch_size,
        validation_split=0.2,
        subset='validation',
        seed=123
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)
    print("Found classes:", class_names)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    model = build_model(input_shape=(img_size[0], img_size[1], 3), num_classes=num_classes)
    model.summary()

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(patience=3, factor=0.5, verbose=1),
        tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True)
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=args.epochs, callbacks=callbacks)

    out_path = args.output
    model.save(out_path)
    print(f"Saved Keras model to {out_path}")

    # save class names
    with open("labels.txt", "w", encoding="utf-8") as f:
        for name in class_names:
            f.write(name + "\n")
    print("Saved labels to labels.txt")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='train', help='Path to train dataset')
    parser.add_argument('--output', default='model.h5', help='Keras model output file')
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()
    main(args)
