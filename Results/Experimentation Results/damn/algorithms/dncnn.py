import numpy as np
from bm3d import bm3d_rgb  # still used for baseline
from sporco.pgm.ppp import PPP
from sporco.interp import bilinear_demosaic
from sporco import metric, util, plot
from skimage.metrics import structural_similarity as ssim
from skimage import img_as_float
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
plot.config_notebook_plotting()

# ---- CFA subsampling / back-projection ----
def A(x):
    """Map an RGB image to a single channel image with each pixel
    representing a single colour according to the colour filter array.
    """
    y = np.zeros(x.shape[0:2])
    y[1::2, 1::2] = x[1::2, 1::2, 0]  # Red
    y[0::2, 1::2] = x[0::2, 1::2, 1]  # Green
    y[1::2, 0::2] = x[1::2, 0::2, 1]  # Green
    y[0::2, 0::2] = x[0::2, 0::2, 2]  # Blue
    return y


def AT(x):
    """Back project a single channel raw image to an RGB image with zeros
    at the locations of undefined samples.
    """
    y = np.zeros(x.shape + (3,))
    y[1::2, 1::2, 0] = x[1::2, 1::2]  # Red
    y[0::2, 1::2, 1] = x[0::2, 1::2]  # Green
    y[1::2, 0::2, 1] = x[1::2, 0::2]  # Green
    y[0::2, 0::2, 2] = x[0::2, 0::2]  # Blue
    return y


# ---- Image ----
# img = util.ExampleImages().image('4.png', anti_aliasing=True, idxexp=np.s_[0:767,0:767])
# img = img_as_float(img)


from skimage.io import imread
from skimage.transform import resize
from skimage.util import img_as_float

# Load your own image instead of using 'util.ExampleImages'
img = imread("5.png")  # Replace with actual image path
img = resize(img, (767, 767), anti_aliasing=True)  # Resize to expected size
img = img_as_float(img)

# Check if grayscale, convert to RGB
if img.ndim == 2:
    img = np.stack([img]*3, axis=-1)
elif img.shape[2] == 4:
    img = img[:, :, :3]  # Drop alpha channel

print("Image shape before CFA sampling:", img.shape)
np.random.seed(12345)
s = A(img)
nsigma = 3e-2
sn = s + nsigma * np.random.randn(*s.shape)

def f(x):
    return 0.5 * np.linalg.norm((A(x) - sn).ravel())**2

def gradf(x):
    return AT(A(x) - sn)

# ---- DnCNN Model ----
# ---- Define DnCNN model ----
class DnCNN(nn.Module):
    def __init__(self, channels=3, num_of_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = [nn.Conv2d(channels, features, kernel_size, padding=padding), nn.ReLU(inplace=True)]
        for _ in range(num_of_layers - 2):
            layers += [nn.Conv2d(features, features, kernel_size, padding=padding),
                       nn.BatchNorm2d(features), nn.ReLU(inplace=True)]
        layers += [nn.Conv2d(features, channels, kernel_size, padding=padding)]
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        return x - self.dncnn(x)  # residual learning

# ---- Instantiate and use untrained model ----
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dncnn = DnCNN(channels=3).to(device)
dncnn.eval()  # No training, but needed to disable dropout/batchnorm in eval mode


# ---- PPP Proxg using DnCNN ----
def proxg(x, L):
    x_t = torch.from_numpy(x.transpose(2, 0, 1)).float().unsqueeze(0).to(device)
    with torch.no_grad():
        out = dncnn(x_t).squeeze(0).cpu().numpy().transpose(1, 2, 0)
    return np.clip(out, 0, 1)

# ---- Baseline using BM3D on demosaiced image ----
imgb = bm3d_rgb(bilinear_demosaic(sn), 3 * nsigma)

# ---- PPP Solve ----
opt = PPP.Options({'Verbose': True, 'RelStopTol': 1e-3,
                   'MaxMainIter': 30, 'L': 0.5, 'X0': imgb})
b = PPP(img.shape, f, gradf, proxg, opt=opt)
imgp = b.solve()

# ---- Evaluation ----
ssim_imgb = ssim(img, imgb, channel_axis=2, data_range=1.0)
ssim_imgp = ssim(img, imgp, channel_axis=2, data_range=1.0)

print("PPP PGM solve time:         %5.2f s" % b.timer.elapsed('solve'))
print("Baseline demosaicing PSNR:  %5.2f dB" % metric.psnr(img, imgb))
print("Baseline demosaicing SSIM:  %5.4f"    % ssim_imgb)
print("PPP demosaicing PSNR:       %5.2f dB" % metric.psnr(img, imgp))
print("PPP demosaicing SSIM:       %5.4f"    % ssim_imgp)

# ---- Visualization ----
fig, ax = plot.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(21, 7))
plot.imview(img, title='Reference', fig=fig, ax=ax[0])
plot.imview(imgb, title='BM3D baseline: %.2f dB' % metric.psnr(img, imgb), fig=fig, ax=ax[1])
plot.imview(imgp, title='PPP + DnCNN: %.2f dB' % metric.psnr(img, imgp), fig=fig, ax=ax[2])
fig.savefig("demosaic_dncnn_og4_again.jpg", dpi=300)
fig.show()
