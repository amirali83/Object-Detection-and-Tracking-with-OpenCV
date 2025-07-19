import torch
import torchvision
from torchvision.transforms import functional as F
import cv2
import time
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.to(device)
model.eval()
