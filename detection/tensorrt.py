import torch
import torchvision.models as models
from facenet_pytorch import MTCNN, InceptionResnetV1

import cv2
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from torchvision import transforms
import torchvision.models as models

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
capture.set(cv2.CAP_PROP_FPS, 60)
plt.ion()

mtcnn = MTCNN()

while True:
    ret, frame = capture.read()
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)

    boxes, _ = mtcnn.detect(pil_image)
    gca = plt.gca()
    for box in boxes:
        plt.imshow(rgb_image)
        coords = (box[0], box[1]), box[2]-box[0], box[3]-box[1]
        gca.add_patch(plt.Rectangle(*coords, fill=False))
    plt.draw()
    plt.pause(0.01)
    plt.cla()


capture.release()
plt.close()
