from pyexpat import model
import torch
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from dart.models.autoencoder import (
    DartHybridQuantizer,
    DARTAutoEncoder,
    DARTAutoEncoderWithDisc
)
from skimage.metrics import structural_similarity as ssim
import numpy as np
from dart.models.transformer import DARTForT2I

llm_path = "/Users/kairoliu/Documents/Dart/hart/llm"

model = DARTForT2I.from_pretrained(llm_path, ignore_mismatched_sizes=True)
print(model)