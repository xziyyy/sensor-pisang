"""
Contoh Kivy app yang menampilkan kamera (OpenCV) dan menjalankan inference
menggunakan model TFLite (`model.tflite`) jika tersedia.

Cara jalankan (desktop):
    pip install kivy opencv-python numpy tensorflow
    python app/main.py

Catatan: di Android gunakan Buildozer dan sertakan `tflite-runtime` atau
`tensorflow` di requirements (lihat buildozer.spec).
"""
from pathlib import Path
import threading

import cv2
import numpy as np
from kivy.app import App
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.lang import Builder
from kivy.properties import StringProperty
from kivy.uix.boxlayout import BoxLayout

try:
    import tflite_runtime.interpreter as tflite_rt
    TFLITE_RUNTIME = True
except Exception:
    try:
        import tensorflow as tf
        TFLITE_RUNTIME = False
    except Exception:
        tf = None
        tflite_rt = None
        TFLITE_RUNTIME = False

KV = Path(__file__).parent.joinpath('main.kv').read_text()

LABELS_FILE = Path(__file__).parent.parent.joinpath('labels.txt')
MODEL_FILE = Path(__file__).parent.parent.joinpath('model.tflite')

LISTFOLDER = {
    "busuk": "Busuk",
    "mentah": "Mentah",
    "mateng": "Matang",
    "kematengan": "Terlalu Matang",
}

WARNA_HEX = {
    "busuk": "#ff0000",
    "mentah": "#00ff00",
    "mateng": "#ffff00",
    "kematengan": "#ffa500",
}


class CameraWidget(BoxLayout):
    status_text = StringProperty("--")

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.capture = cv2.VideoCapture(0)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.interpreter = None
        self.labels = []
        self.load_model()
        Clock.schedule_interval(self.update, 1.0 / 30.0)

    def load_model(self):
        if MODEL_FILE.exists() and LABELS_FILE.exists():
            try:
                if TFLITE_RUNTIME:
                    self.interpreter = tflite_rt.Interpreter(model_path=str(MODEL_FILE))
                else:
                    self.interpreter = tf.lite.Interpreter(model_path=str(MODEL_FILE))
                self.interpreter.allocate_tensors()
                with open(LABELS_FILE, 'r', encoding='utf-8') as f:
                    self.labels = [l.strip() for l in f.readlines()]
                print('Loaded TFLite model and labels:', self.labels)
            except Exception as e:
                print('Failed to load TFLite model:', e)
                self.interpreter = None
        else:
            print('No model.tflite or labels.txt found; running in placeholder mode')

    def update(self, dt):
        ret, frame = self.capture.read()
        if not ret:
            return

        # flip for selfie-like view
        frame = cv2.flip(frame, 1)
        buf = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, _ = buf.shape
        texture = Texture.create(size=(w, h))
        texture.blit_buffer(buf.tobytes(), colorfmt='rgb', bufferfmt='ubyte')
        texture.flip_vertical()
        self.ids.cam_preview.texture = texture

        # run inference in background thread to avoid UI lag
        if self.interpreter is not None:
            threading.Thread(target=self.run_inference, args=(frame,)).start()

    def run_inference(self, frame):
        # Preprocess to model input (128x128)
        img = cv2.resize(frame, (128, 128))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        input_data = np.expand_dims(img, axis=0)

        try:
            input_details = self.interpreter.get_input_details()
            output_details = self.interpreter.get_output_details()
            # handle different input types
            if input_details[0]['dtype'] == np.uint8:
                input_scale, input_zero_point = input_details[0]['quantization']
                if input_scale:
                    input_data = (input_data / input_scale + input_zero_point).astype(np.uint8)
            self.interpreter.set_tensor(input_details[0]['index'], input_data)
            self.interpreter.invoke()
            output_data = self.interpreter.get_tensor(output_details[0]['index'])[0]

            idx = int(np.argmax(output_data))
            label = self.labels[idx] if idx < len(self.labels) else str(idx)
            conf = float(output_data[idx])
            display = LISTFOLDER.get(label, label)
            self.status_text = f"{display} ({conf:.0%})"
        except Exception as e:
            print('Inference error:', e)


class MainApp(App):
    def build(self):
        return Builder.load_string(KV)


if __name__ == '__main__':
    MainApp().run()
