from torchvision import datasets, models, transforms
import os
import torch
from torch.autograd import Variable
from skimage import io
import numpy as np
from torch import nn
import datetime
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn import metrics
import argparse

def roll_n(X, axis, n):
    f_idx = tuple(slice(None, None, None) if i != axis else slice(0, n, None)
                  for i in range(X.dim()))
    b_idx = tuple(slice(None, None, None) if i != axis else slice(n, None, None)
                  for i in range(X.dim()))
    front = X[f_idx]
    back = X[b_idx]
    return torch.cat([back, front], axis)

def fftshift(tensor):
    real, imag = tensor[..., 0], tensor[..., 1]
    for dim in range(2, len(tensor.shape)):
        real, imag = roll_n(real, axis=dim, n=real.size(dim) // 2), roll_n(imag, axis=dim, n=imag.size(dim) // 2)
    return torch.stack([real, imag], dim=-1)

class DnCNN(nn.Module):
    def __init__(self, num_layers=17, num_features=64):
        super(DnCNN, self).__init__()
        layers = [nn.Sequential(nn.Conv2d(3, num_features, kernel_size=3, stride=1, padding=1),
                                nn.ReLU(inplace=True))]
        for i in range(num_layers - 2):
            layers.append(nn.Sequential(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
                                        nn.BatchNorm2d(num_features),
                                        nn.ReLU(inplace=True)))
        layers.append(nn.Conv2d(num_features, 3, kernel_size=3, padding=1))
        self.layers = nn.Sequential(*layers)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, inputs):
        y = inputs
        residual = self.layers(y)
        residual_1 = residual.clone()

        residual_gray = 0.299 * residual_1[:, 0, :, :] + 0.587 * residual_1[:, 1, :, :] + 0.114 * residual_1[:, 2, :, :]

        # Compute 2D FFT
        thirdPart_fft_1 = torch.fft.fft2(residual_gray, dim=(-2, -1))
        thirdPart_fft_1_orig = thirdPart_fft_1.clone()

        thirdPart_fft_1 = fftshift(thirdPart_fft_1)
        magnitude = torch.sqrt(thirdPart_fft_1[..., 0] ** 2 + thirdPart_fft_1[..., 1] ** 2)

        n = 25
        _, w, h = magnitude.shape
        half_w, half_h = int(w / 2), int(h / 2)
        thirdPart_fft_2 = magnitude[:, half_w - n:half_w + n + 1, half_h - n:half_h + n + 1].clone()
        thirdPart_fft_3 = magnitude.clone()
        thirdPart_fft_3[:, half_w - n:half_w + n + 1, half_h - n:half_h + n + 1] = 0
        max_value = torch.max(thirdPart_fft_3)
        thirdPart_fft_4 = magnitude.clone()
        thirdPart_fft_4 = torch.transpose(thirdPart_fft_4, 1, 2)

        return magnitude, thirdPart_fft_2
