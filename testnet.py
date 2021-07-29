import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torch.nn as nn
from torchvision.utils import save_image


class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.gen = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(True),
            nn.Linear(256, 256), nn.ReLU(True), nn.Linear(256, 784), nn.Tanh())

    def forward(self, x):
        x = self.gen(x)
        return x.squeeze(-1) 

model = generator()
model.load_state_dict(torch.load("generator.pth",map_location='cpu'))
z = torch.randn(100)
model.eval()
out = model(z)

def to_img(x):
    out = 0.5 * (x + 1)
    out = out.clamp(0, 1)
    out = out.view(-1, 1, 28, 28)
    return out

fake_images = to_img(out.data)
save_image(fake_images, 'predict.png')