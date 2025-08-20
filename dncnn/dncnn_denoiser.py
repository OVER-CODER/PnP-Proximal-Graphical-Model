import torch
import torch.nn as nn
import numpy as np
import cv2

class DnCNN(nn.Module):
    def __init__(self, channels=1, num_of_layers=20):  # <= FIXED to 20
        super(DnCNN, self).__init__()
        layers = []

        layers.append(nn.Conv2d(in_channels=channels, out_channels=64, kernel_size=3, padding=1, bias=False))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(num_of_layers - 2):
            layers.append(nn.Conv2d(64, 64, 3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(64))
            layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv2d(in_channels=64, out_channels=channels, kernel_size=3, padding=1, bias=False))

        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        return x - self.dncnn(x)  # residual learning


def load_dncnn_model(model_path='dncnn_gray.pth'):
    model = DnCNN(channels=1, num_of_layers=20)  # <= FIXED to 20 layers
    state_dict = torch.load(model_path, map_location='cpu')

    # Strip 'module.' if present
    new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    model.load_state_dict(new_state_dict, strict=True)
    model.eval()
    return model

def dncnn_luma_denoise(img, model):
    """Convert RGB to YCrCb, denoise Y channel with DnCNN, and convert back to RGB."""
    img = np.clip(img, 0, 1)
    img_ycrcb = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2YCrCb)
    y, cr, cb = cv2.split(img_ycrcb)

    y = y.astype(np.float32) / 255.0
    inp = torch.from_numpy(y).float().unsqueeze(0).unsqueeze(0)
    
    with torch.no_grad():
        denoised_y = model(inp).squeeze().numpy()

    denoised_y = np.clip(denoised_y * 255.0, 0, 255).astype(np.uint8)
    img_ycrcb = cv2.merge([denoised_y, cr, cb])
    img_rgb = cv2.cvtColor(img_ycrcb, cv2.COLOR_YCrCb2RGB)

    return img_rgb.astype(np.float32) / 255.0