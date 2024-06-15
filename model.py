import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
from PIL import Image
import numpy as np


class DenseNet201(nn.Module):
    def __init__(self):
        super().__init__()
        classifier = nn.Sequential(nn.Linear(1000, 512, bias=True),
                                   nn.LeakyReLU(inplace=True),
                                   nn.Linear(512, 101, bias=True),
                                   nn.LeakyReLU(inplace=True))
        self.model = torchvision.models.densenet201(pretrained=False)
        self.model.fc = classifier

    def forward(self, x):
        out = self.model(x)
        return out


def model_pipeline(image):
    model = DenseNet201()
    model.load_state_dict(torch.load('model_weights.ckpt', map_location=torch.device('cpu')))
    model.eval()
    if image.mode != 'RGB':
        image = image.convert('RGB')
    img_transform = transforms.Compose([transforms.Resize(256),
                        transforms.CenterCrop(224),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])])
    image = img_transform(image)
    image = image.view(1, *image.size())
    with torch.no_grad():
        out = model(image)
    out = torch.nn.functional.softmax(out, dim=1)
    result = torch.argmax(out.detach().cpu(), -1).numpy()[0]
    label = open('labels.txt').read().split('\n')[result]
    return label
