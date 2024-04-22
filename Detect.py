import torch
import cv2
import numpy as np

import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Leemos el modelo
model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path='D:/Todo/model/estesi.pt')

# realizamos videocaptura
cap = cv2.VideoCapture(0)  # camara

# empezamos
while True:
    # realzar videocaptura
    ret, frame = cap.read()

    # realizamos detecciones
    detect = model(frame)

    info = detect.pandas().xyxy[0]  # im1 predictions
    print(info)

    # Mostramos los FPS
    cv2.imshow('Detector de Casco', np.squeeze(detect.render()))

    # Leer por teclado
    t = cv2.waitKey(5)
    if t == 27:
        break
cap.release()
cv2.destroyAllWindows()