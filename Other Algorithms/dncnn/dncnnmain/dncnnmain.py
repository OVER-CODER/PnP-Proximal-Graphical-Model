from __future__ import print_function
from builtins import input, range

import numpy as np
import cv2 # Added for grayscale conversion
import torch
import torch.nn as nn
from torchvision.transforms import ToTensor, ToPILImage

# from bm3d import bm3d_rgb # No longer needed if using DnCNN directly on grayscale

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


# Define the DnCNN model
class DnCNN(nn.Module):
    def __init__(self, channels=1, num_of_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers - 2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.dncnn(x)
        return x - out # Residual learning

# Load the pre-trained grayscale DnCNN model
# You need to download 'model.pth' from the provided GitHub link and place it in the same directory
dncnn_model_path = 'model.pth'
dncnn_model = DnCNN(channels=1, num_of_layers=17) # Assuming 17 layers for this common DnCNN
try:
    # Load the entire model directly
    dncnn_model = torch.load(dncnn_model_path, map_location=torch.device('cpu'), weights_only=False)
    dncnn_model.eval() # Set to evaluation mode
    print(f"DnCNN model loaded from {dncnn_model_path}")
except FileNotFoundError:
    print(f"Error: DnCNN model file '{dncnn_model_path}' not found. Please download it from the GitHub link and place it in the same directory.")
    exit()
except Exception as e: # Catch other potential loading errors
    print(f"An error occurred during model loading: {e}")
    # If the model.pth actually contains only a state_dict, you might revert to the previous approach
    # and adjust the class definition if it differs from the saved one.
    print("Attempting to load as state_dict in case the file format is different...")
    try:
        dncnn_model = DnCNN(channels=1, num_of_layers=17) # Re-initialize
        dncnn_model.load_state_dict(torch.load(dncnn_model_path, map_location=torch.device('cpu'), weights_only=False))
        dncnn_model.eval()
        print(f"DnCNN model loaded successfully as state_dict from {dncnn_model_path}")
    except Exception as se:
        print(f"Failed to load as state_dict either: {se}")
        print("Please ensure 'model.pth' is the correct file and matches the expected DnCNN architecture.")
        exit()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dncnn_model.to(device)


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
rgbshp = s.shape + (3,)
rgbsz = s.size * 3
nsigma = 1e-1
sn = s + nsigma * np.random.randn(*s.shape)

def f(x):
    return 0.5 * np.linalg.norm((A(x) - sn).ravel())**2

def gradf(x):
    return AT(A(x) - sn)

# bsigma parameter might still be useful for other aspects, but DnCNN
# is generally trained for specific noise levels or a range.
# For simplicity, we're assuming the loaded DnCNN handles the noise level.
# If your DnCNN model allows it, you might pass a noise level parameter.
# For this specific DnCNN model, it's typically trained for a fixed sigma (e.g., sigma=25 in the model name).
# The "bsigma" parameter was for BM3D's internal noise estimation.
# For DnCNN, it often takes a fixed noise level, or you might need to select a model
# trained on a similar noise level to your 'nsigma'.
# For now, we'll ignore bsigma in proxg_dncnn and assume the model is suitable.



def proxg_dncnn(x_rgb, L):
    print("\n--- Inside proxg_dncnn ---")

    # --- NEW: Explicitly clip the input x_rgb to [0, 1] range ---
    # Convert to float32 first if it's float64, for consistency and potential slight speedup with torch/cv2
    x_rgb = x_rgb.astype(np.float32)
    x_rgb_clipped_input = np.clip(x_rgb, 0.0, 1.0)
    print(f"Clipped Input x_rgb shape: {x_rgb_clipped_input.shape}, dtype: {x_rgb_clipped_input.dtype}, min: {x_rgb_clipped_input.min():.4f}, max: {x_rgb_clipped_input.max():.4f}")

    # Now use x_rgb_clipped_input for all subsequent operations
    x_rgb_uint8 = (x_rgb_clipped_input * 255).astype(np.uint8)
    print(f"x_rgb_uint8 shape: {x_rgb_uint8.shape}, dtype: {x_rgb_uint8.dtype}, min: {x_rgb_uint8.min()}, max: {x_rgb_uint8.max()}")

    ycbcr_img_uint8 = cv2.cvtColor(x_rgb_uint8, cv2.COLOR_RGB2YCrCb)
    print(f"ycbcr_img_uint8 shape: {ycbcr_img_uint8.shape}, dtype: {ycbcr_img_uint8.dtype}, min_Y: {ycbcr_img_uint8[:,:,0].min()}, max_Y: {ycbcr_img_uint8[:,:,0].max()}")

    y_channel = ycbcr_img_uint8[:, :, 0].astype(np.float32) / 255.0
    print(f"y_channel shape: {y_channel.shape}, dtype: {y_channel.dtype}, min: {y_channel.min():.4f}, max: {y_channel.max():.4f}")

    # Prepare Y channel for DnCNN (PyTorch expects NCHW format)
    y_tensor = torch.from_numpy(y_channel).unsqueeze(0).unsqueeze(0).to(device)
    print(f"y_tensor shape (before DnCNN): {y_tensor.shape}, min: {y_tensor.min():.4f}, max: {y_tensor.max():.4f}")

    # Apply DnCNN to the Y channel
    with torch.no_grad():
        denoised_y_tensor = dncnn_model(y_tensor)

    print(f"denoised_y_tensor shape (after DnCNN): {denoised_y_tensor.shape}, min: {denoised_y_tensor.min():.4f}, max: {denoised_y_tensor.max():.4f}")

    # Convert denoised Y back to NumPy array and original range (0-255)
    denoised_y = (denoised_y_tensor.squeeze(0).squeeze(0).cpu().numpy() * 255).astype(np.uint8)
    print(f"denoised_y shape: {denoised_y.shape}, dtype: {denoised_y.dtype}, min: {denoised_y.min()}, max: {denoised_y.max()}")

    # Replace original Y channel with denoised Y
    ycbcr_denoised_uint8 = ycbcr_img_uint8.copy()
    ycbcr_denoised_uint8[:, :, 0] = denoised_y

    # Convert back to RGB (0-255 uint8), then to 0-1 float for sporco
    denoised_rgb_uint8 = cv2.cvtColor(ycbcr_denoised_uint8, cv2.COLOR_YCrCb2RGB)
    denoised_rgb = denoised_rgb_uint8.astype(np.float32) / 255.0

    # Ensure the final output is clipped to [0, 1] as well
    denoised_rgb = np.clip(denoised_rgb, 0, 1)
    print(f"Output denoised_rgb shape: {denoised_rgb.shape}, dtype: {denoised_rgb.dtype}, min: {denoised_rgb.min():.4f}, max: {denoised_rgb.max():.4f}")
    print("--- Exiting proxg_dncnn ---\n")

    return denoised_rgb


# Baseline denoising with demosaicing + DnCNN
# First demosaic the noisy CFA image, then denoise the resulting RGB image using DnCNN
# Note: For the baseline, we're applying DnCNN on the already-demosaiced RGB image.
# We'll need a way to apply DnCNN to an RGB image for this baseline.
# The simplest approach is to convert the demosaiced RGB image to grayscale, denoise, and convert back.
demosaiced_noisy_rgb = demosaic(sn)

# Apply proxg_dncnn (which converts to grayscale, denoises, and converts back to RGB)
imgb = proxg_dncnn(demosaiced_noisy_rgb, None) # L is not used in proxg_dncnn


opt = PPP.Options({'Verbose': True, 'RelStopTol': 1e-3,
                   'MaxMainIter': 50, 'L': 0.05, 'X0': imgb}) # X0 now uses the DnCNN-denoised baseline

# Use the DnCNN proximal operator
b = PPP(img.shape, f, gradf, proxg_dncnn, opt=opt) # Changed proxg to proxg_dncnn
imgp = b.solve()

# Compute SSIM
ssim_imgb = ssim(img, imgb, channel_axis=2, data_range=img.max() - img.min())
ssim_imgp = ssim(img, imgp, channel_axis=2, data_range=img.max() - img.min())

# Print metrics
print("PPP PGM solve time:     %5.2f s" % b.timer.elapsed('solve'))
print("Baseline demosaicing PSNR: %5.2f dB" % metric.psnr(img, imgb))
print("Baseline demosaicing SSIM: %5.4f"  % ssim_imgb)
print("PPP demosaicing PSNR:    %5.2f dB" % metric.psnr(img, imgp))
print("PPP demosaicing SSIM:    %5.4f"  % ssim_imgp)


fig, ax = plot.subplots(nrows=1, ncols=3, sharex=True, sharey=True,
                         figsize=(21, 7))
plot.imview(img, title='Reference', fig=fig, ax=ax[0])
plot.imview(imgb, title='Baseline demosaic (DnCNN): %.2f (dB)' %
             metric.psnr(img, imgb), fig=fig, ax=ax[1])
plot.imview(imgp, title='PPP demosaic (DnCNN): %.2f (dB)' %
             metric.psnr(img, imgp), fig=fig, ax=ax[2])

fig.savefig("demosaic_dncnn_test1.jpg", dpi=300)

fig.show()