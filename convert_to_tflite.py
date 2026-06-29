"""
Convert `model.h5` (Keras) to `model.tflite` with optional quantization.
Usage:
    python convert_to_tflite.py --input model.h5 --output model.tflite
"""

import argparse
import tensorflow as tf


def convert(input_path, output_path, quantize=False):
    model = tf.keras.models.load_model(input_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    print(f'Wrote TFLite model to {output_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='model.h5')
    parser.add_argument('--output', default='model.tflite')
    parser.add_argument('--quantize', action='store_true')
    args = parser.parse_args()
    convert(args.input, args.output, args.quantize)
