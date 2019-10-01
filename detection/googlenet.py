import torch
import torchvision.models as models

import cv2
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import torchvision.models as models

capture = cv2.VideoCapture(0)
plt.ion()

transform = transforms.Compose([            #[1]
    transforms.Resize(256),                    #[2]
    transforms.CenterCrop(224),                #[3]
    transforms.ToTensor(),                     #[4]
    transforms.Normalize(                      #[5]
        mean=[0.485, 0.456, 0.406],                #[6]
        std=[0.229, 0.224, 0.225]                  #[7]
)])

model = models.googlenet(pretrained=True).eval()

while True:
    ret, frame = capture.read()
    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)
    img_t = transform(pil_image)
    batch_t = torch.unsqueeze(img_t, 0)
    out = model(batch_t)

    with open('imagenet_classes.txt') as f:
        labels = [line.strip() for line in f.readlines()]

    _, index = torch.max(out, 1)
 
    percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
 
    print(labels[index[0]], percentage[index[0]].item())

    #  plt.imshow(rgb_image)
    #  plt.draw()
    #  plt.pause(0.1)
