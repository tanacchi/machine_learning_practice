import torch
import torchvision.models as models
from facenet_pytorch import MTCNN, InceptionResnetV1


mtcnn = MTCNN()

dummy_input = torch.randn(1, 3, 640, 480)
input_names = [ "actual_input_1" ]
output_names = [ "boxes", "ahi" ]

torch.onnx.export(
    mtcnn, dummy_input, "mtcnn.onnx", verbose=True,
    input_names=input_names, output_names=output_names
)
