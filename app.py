#import torch
#import cv2

#model = torch.hub.load('ultralytics/yolov5','custom',patch)

from flask import Flask, Response
import torch
import cv2
import numpy as np
import requests
from io import BytesIO
from pydantic import BaseModel

import pathlib

temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Leemos el modelo
model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path='D:/Todo/model/estesiosi.pt')

app = Flask(__name__)


def generate_frames():
    # Inicia la captura de video desde la cámara (puede ser 0, 1, etc. dependiendo de tu configuración)
    cap = cv2.VideoCapture(0)

    while True:
        # Captura un frame de la cámara
        ret, frame = cap.read()

        if not ret:
            break

        # Convierte el frame a un formato adecuado para la transmisión
        ret, buffer = cv2.imencode('.jpg', frame)

        # Convierte el buffer a bytes
        frame_bytes = buffer.tobytes()

        # Genera un frame para enviar al cliente
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    # Libera la cámara al terminar
    cap.release()


def detectCamera():
    # Inicia la captura de video desde la cámara (puede ser 0, 1, etc. dependiendo de tu configuración)
    cap = cv2.VideoCapture(0)

    while True:
        # Captura un frame de la cámara
        ret, frame = cap.read()

        if not ret:
            break

        # realizamos detecciones
        detect = model(frame)

        info = detect.pandas().xyxy[0]  # im1 predictions
        print(info)

        # Mostramos los FPS
        cv2.imshow('Detector de Casco', np.squeeze(detect.render()))

        # Convierte el frame a un formato adecuado para la transmisión
        ret, buffer = cv2.imencode('.jpg', frame)

        # Convierte el buffer a bytes
        frame_bytes = buffer.tobytes()

        # Genera un frame para enviar al cliente
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    # Libera la cámara al terminar
    cap.release()


def detectImageLink(image_url):
    # Descargar la imagen desde la URL
    response = requests.get(image_url)
    if response.status_code != 200:
        raise ValueError("No se pudo descargar la imagen desde la URL proporcionada.")

    # Convertir la respuesta a una matriz de bytes
    image_bytes = BytesIO(response.content).read()

    # Convertir los bytes a una matriz de imagen
    frame = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)

    # Realizar detecciones
    detect = model(frame)

    info = detect.pandas().xyxy[0]  # im1 predictions

    print(info, "Resultado de las detecciones")

    # Mostrar el resultado de las detecciones
    cv2.imshow('Detector de Casco', np.squeeze(detect.render()))

    # Convertir el frame a un formato adecuado para la transmisión
    ret, buffer = cv2.imencode('.jpg', frame)

    # Convertir el buffer a bytes
    frame_bytes = buffer.tobytes()

    # Generar un frame para enviar al cliente
    yield (b'--frame\r\n'
           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


class Image(BaseModel):
    imagen: str


def detectImage(image: Image):
    # Descargar la imagen desde la URL
    response = requests.get(image.imagen)
    if response.status_code != 200:
        raise ValueError("No se pudo descargar la imagen desde la URL proporcionada.")

    # Convertir la respuesta a una matriz de bytes
    image_bytes = BytesIO(response.content).read()

    # Convertir los bytes a una matriz de imagen
    frame = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), -1)

    # Realizar detecciones
    detect = model(frame)

    info = detect.pandas().xyxy[0]  # im1 predictions

    print(info, "Resultado de las detecciones")

    return Response(detect.render(), mimetype='multipart/x-mixed-replace; boundary=frame')

    # Mostrar el resultado de las detecciones
    cv2.imshow('Detector de Casco', np.squeeze(detect.render()))

    # Convertir el frame a un formato adecuado para la transmisión
    ret, buffer = cv2.imencode('.jpg', frame)

    # Convertir el buffer a bytes
    frame_bytes = buffer.tobytes()

    # Generar un frame para enviar al cliente
    yield (b'--frame\r\n'
           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(
        detectImageLink("https://upload.wikimedia.org/wikipedia/commons/thumb/f/f1/Pink_eye.jpg/320px-Pink_eye.jpg"),
        mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def home():
    return Response(detectCamera(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/detect', methods=['POST'])
def detect(image: Image):
    return Response(detectImage(image), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
