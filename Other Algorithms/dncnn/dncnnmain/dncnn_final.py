from __future__ import print_function
from builtins import input, range

import numpy as np

# Removed bm3d import as it's being replaced
# from bm3d import bm3d_rgb

try:
    from colour_demosaicing import demosaicing_CFA_Bayer_Menon2007
except ImportError:
    have_demosaic = False
else:
    have_demosaic = True

from sporco.pgm.ppp import PPP
from sporco.interp import bilinear_demosaic
from sporco import metric
from sporco import util
from sporco import plot
from skimage.metrics import structural_similarity as ssim
plot.config_notebook_plotting()

# --- DnCNN Integration ---
import torch
import torch.nn as nn

# Define a basic DnCNN model (simplified for demonstration purposes)
# In a practical scenario, you would typically use a pre-trained model
# from a well-established repository (e.g., from cszn/DnCNN on GitHub)
# and load its weights.
class DnCNN(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, depth=17, num_filters=64):
        super(DnCNN, self).__init__()
        layers = []
        # First layer: Conv + ReLU
        layers.append(nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1, bias=False))
        layers.append(nn.ReLU(inplace=True))

        # Middle layers: Conv + BatchNorm + ReLU
        for _ in range(depth - 2):
            layers.append(nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(num_filters))
            layers.append(nn.ReLU(inplace=True))

        # Last layer: Conv (output is the residual/noise)
        layers.append(nn.Conv2d(num_filters, out_channels, kernel_size=3, padding=1, bias=False))

        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        # DnCNN typically learns the residual (noise)
        # Input `x` is noisy image. Output `self.dncnn(x)` is predicted noise.
        # Denoised image = Noisy image - Predicted noise
        return x - self.dncnn(x)

class DnCNNDenoiser:
    def __init__(self, noise_level, model_path=None):
        # Initialize DnCNN model
        self.model = DnCNN(in_channels=3, out_channels=3, depth=17, num_filters=64)
        if model_path:
            # Load pre-trained weights if available.
            # Make sure the model architecture matches the loaded weights.
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
                print(f"Loaded DnCNN model from {model_path}")
            except Exception as e:
                print(f"Warning: Could not load DnCNN model weights from {model_path}. "
                      f"Using randomly initialized weights. Error: {e}")
        self.model.eval() # Set to evaluation mode

        self.noise_level = noise_level # Sigma for the denoiser, passed to DnCNN

        # Check for CUDA availability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        print(f"DnCNN denoiser running on: {self.device}")

    def __call__(self, img_noisy_np):
        # img_noisy_np is a NumPy array in HxWxC format, values typically [0, 1]
        # Convert to PyTorch tensor: HxWxC -> CxHxW, add batch dimension, and move to device
        img_noisy_tensor = torch.from_numpy(img_noisy_np).permute(2, 0, 1).unsqueeze(0).float().to(self.device)

        with torch.no_grad(): # No need to calculate gradients for inference
            # DnCNN might not explicitly take noise_level as an input to its forward pass,
            # but rather is trained for a range of noise levels or specific ones.
            # If your DnCNN implementation *does* take noise_level as an input,
            # modify this line accordingly:
            # denoised_tensor = self.model(img_noisy_tensor, self.noise_level)
            
            # For a standard DnCNN model, the model is trained to handle a certain noise distribution,
            # and the noise_level is typically used during training or implied by the chosen model.
            # Here, it's just used for consistency with BM3D's bsigma parameter.
            denoised_tensor = self.model(img_noisy_tensor)

        # Convert back to NumPy array: remove batch dimension, CxHxW -> HxWxC
        denoised_np = denoised_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()

        # Ensure output is within the valid image range [0, 1]
        denoised_np = np.clip(denoised_np, 0, 1)
        return denoised_np

# --- Original Code (modified for DnCNN) ---

def A(x):
    """Map an RGB image to a single channel image with each pixel
    representing a single colour according to the colour filter array.
    """

    y = np.zeros(x.shape[0:2])
    y[1::2, 1::2] = x[1::2, 1::2, 0]
    y[0::2, 1::2] = x[0::2, 1::2, 1]
    y[1::2, 0::2] = x[1::2, 0::2, 1]
    y[0::2, 0::2] = x[0::2, 0::2, 2]
    return y


def AT(x):
    """Back project a single channel raw image to an RGB image with zeros
    at the locations of undefined samples.
    """

    y = np.zeros(x.shape + (3,))
    y[1::2, 1::2, 0] = x[1::2, 1::2]
    y[0::2, 1::2, 1] = x[0::2, 1::2]
    y[1::2, 0::2, 1] = x[1::2, 0::2]
    y[0::2, 0::2, 2] = x[0::2, 0::2]
    return y


if have_demosaic:
    def demosaic(cfaimg):
        return demosaicing_CFA_Bayer_Menon2007(cfaimg, pattern='BGGR')
else:
    def demosaic(cfaimg):
        return bilinear_demosaic(cfaimg)
    

img = util.ExampleImages().image('1.png', scaled=True,
                                 idxexp=np.s_[0:767,0:767])

np.random.seed(12345)
s = A(img)
rgbshp = s.shape + (3,)   # Shape of reconstructed RGB image
rgbsz = s.size * 3        # Size of reconstructed RGB image
nsigma = 3e-2             # Noise standard deviation
sn = s + nsigma * np.random.randn(*s.shape)

def f(x):
    return 0.5 * np.linalg.norm((A(x) - sn).ravel())**2

def gradf(x):
    return AT(A(x) - sn)

# Denoiser parameter (used as sigma for DnCNN)
bsigma = 4.5e-2

# Initialize DnCNN denoiser
# If you have pre-trained weights, specify the path here:
# dncnn_denoiser = DnCNNDenoiser(bsigma, model_path='path/to/your/dncnn_weights.pth')
dncnn_denoiser = DnCNNDenoiser(bsigma) # Using randomly initialized weights if no path provided

def proxg(x, L):
    # This is where DnCNN is used instead of BM3D
    return dncnn_denoiser(x)

# Baseline demosaicing using the simple demosaic function and then DnCNN for denoising
# Note: bm3d_rgb takes 'bsigma' (noise_level) as an argument.
# If your DnCNN model takes noise_level directly, adjust the call below.
# Here, we pass the `3 * nsigma` as an example, but a pre-trained DnCNN's effectiveness
# might depend on the specific noise levels it was trained on.
# We'll use the dncnn_denoiser for the baseline denoising as well for consistency.
imgb = dncnn_denoiser(demosaic(sn)) # Demosaic and then denoise with DnCNN

opt = PPP.Options({'Verbose': True, 'RelStopTol': 1e-5,
                   'MaxMainIter': 30, 'L': 0.5, 'X0': imgb})

b = PPP(img.shape, f, gradf, proxg, opt=opt)
imgp = b.solve()

# Compute SSIM
ssim_imgb = ssim(img, imgb, channel_axis=2, data_range=img.max() - img.min())
ssim_imgp = ssim(img, imgp, channel_axis=2, data_range=img.max() - img.min())

# Print metrics
print("PPP PGM solve time:          %5.2f s" % b.timer.elapsed('solve'))
print("Baseline demosaicing PSNR:   %5.2f dB" % metric.psnr(img, imgb))
print("Baseline demosaicing SSIM:   %5.4f"      % ssim_imgb)
print("PPP demosaicing PSNR:        %5.2f dB" % metric.psnr(img, imgp))
print("PPP demosaicing SSIM:        %5.4f"      % ssim_imgp)


fig, ax = plot.subplots(nrows=1, ncols=3, sharex=True, sharey=True,
                         figsize=(21, 7))
plot.imview(img, title='Reference', fig=fig, ax=ax[0])
plot.imview(imgb, title='Baseline demosaiced + DnCNN denoised: %.2f (dB)' %
             metric.psnr(img, imgb), fig=fig, ax=ax[1])
plot.imview(imgp, title='PPP demosaicing with DnCNN: %.2f (dB)' %
             metric.psnr(img, imgp), fig=fig, ax=ax[2])

fig.savefig("demosaic_dncnn_Final_1.jpg", dpi=300)

# fig.show() # This is typically not needed when saving to file in a non-interactive environment