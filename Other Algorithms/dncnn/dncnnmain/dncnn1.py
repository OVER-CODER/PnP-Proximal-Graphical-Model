from __future__ import print_function
from builtins import input, range

import numpy as np

# from bm3d import bm3d_rgb  # Comment out or remove BM3D import
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

# Import TensorFlow and Keras
import tensorflow as tf
from tensorflow import keras

# Import the DnCNN class from your model.py
# Make sure 'model.py' is in the same directory as this script.
from model import DnCNN as DnCNN_Architecture # Renamed to avoid confusion with loaded instance

# --- DnCNN Model Loading ---
dncnn_model = None # Initialize to None

# Define the path to your pre-trained weights file
WEIGHTS_FILE_PATH = 'DnCNN_Default_SIDD_20211113-160850.h5'

try:
    # 1. Instantiate the DnCNN architecture from model.py
    # This assumes your DnCNN_Architecture class automatically sets up for 3 channels
    # via `Input(shape=(None, None, 3))` as seen in your provided model.py
    dncnn_builder = DnCNN_Architecture()
    dncnn_model = dncnn_builder.get_model()

    # 2. Load the pre-trained weights onto the *built architecture*
    dncnn_model.load_weights(WEIGHTS_FILE_PATH)

    print(f"DnCNN model architecture built and weights loaded successfully from {WEIGHTS_FILE_PATH}!")
    # Optionally print model summary to verify
    # print(dncnn_model.summary())
except Exception as e:
    print(f"Error building or loading weights for DnCNN model from {WEIGHTS_FILE_PATH}: {e}")
    print("Please ensure 'model.py' is present and that the weights file matches its architecture.")
    print("Proceeding without DnCNN, results will be a placeholder.")
    dncnn_model = None # Set to None if loading fails

def dncnn_denoise(image, sigma):
    """
    DnCNN denoiser implementation using the loaded Keras model.
    'image' is a NumPy array (H, W, C) with values typically in [0, 1].
    'sigma' is the noise standard deviation (Note: This specific DnCNN model
    is likely fixed-sigma or blind, so 'sigma' might not be directly used by the model itself,
    but it's kept in the function signature for consistency).
    """
    if dncnn_model is None:
        print("DnCNN model not loaded, returning original image as placeholder.")
        return image # Return original image if model failed to load

    # Convert image to float32
    image = image.astype(np.float32)

    # Normalize to [0, 255] range for the DnCNN model
    input_image_normalized = image * 255.0

    # Add batch dimension for Keras (1, H, W, C)
    # Your model.py defines `Input(shape=(None, None, 3))`, so it expects 3 channels directly.
    input_tensor = np.expand_dims(input_image_normalized, axis=0)

    # Predict with the DnCNN model
    # The model definition `outputs = inputs - dncnn_network` means
    # the model directly outputs the denoised image.
    denoised_output_batch = dncnn_model.predict(input_tensor, verbose=0)

    # Remove batch dimension
    denoised_image = np.squeeze(denoised_output_batch)

    # De-normalize back to [0, 1] range (assuming model outputs in [0, 255])
    denoised_image_denormalized = denoised_image / 255.0

    # Clip to ensure values are within [0, 1]
    denoised_image_clipped = np.clip(denoised_image_denormalized, 0, 1)

    return denoised_image_clipped


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
                                 idxexp=np.s_[0:255,0:255])


np.random.seed(12345)
s = A(img)
rgbshp = s.shape + (3,)  # Shape of reconstructed RGB image
rgbsz = s.size * 3       # Size of reconstructed RGB image
nsigma = 3e-2            # Noise standard deviation
sn = s + nsigma * np.random.randn(*s.shape)

def f(x):
    return 0.5 * np.linalg.norm((A(x) - sn).ravel())**2

def gradf(x):
    return AT(A(x) - sn)

bsigma = 4.5e-2  # Denoiser parameter for the Proximal operator

def proxg(x, L):
    # Here, x is an RGB image, and we want to denoise it using DnCNN
    # The 'L' parameter is from the PGM algorithm, not directly used by DnCNN
    return dncnn_denoise(x, bsigma)


# Baseline using the demosaic function directly with DnCNN
imgb = dncnn_denoise(demosaic(sn), 3 * nsigma)

opt = PPP.Options({'Verbose': True, 'RelStopTol': 1e-3,
                   'MaxMainIter': 30, 'L': 0.5, 'X0': imgb})


b = PPP(img.shape, f, gradf, proxg, opt=opt)
imgp = b.solve()

# Compute SSIM
ssim_imgb = ssim(img, imgb, channel_axis=2, data_range=img.max() - img.min())
ssim_imgp = ssim(img, imgp, channel_axis=2, data_range=img.max() - img.min())

# Print metrics
print("PPP PGM solve time:        %5.2f s" % b.timer.elapsed('solve'))
print("Baseline demosaicing PSNR: %5.2f dB" % metric.psnr(img, imgb))
print("Baseline demosaicing SSIM: %5.4f"    % ssim_imgb)
print("PPP demosaicing PSNR:      %5.2f dB" % metric.psnr(img, imgp))
print("PPP demosaicing SSIM:      %5.4f"    % ssim_imgp)


fig, ax = plot.subplots(nrows=1, ncols=3, sharex=True, sharey=True,
                         figsize=(21, 7))
plot.imview(img, title='Reference', fig=fig, ax=ax[0])
plot.imview(imgb, title='Baseline demosaic (DnCNN): %.2f (dB)' %
             metric.psnr(img, imgb), fig=fig, ax=ax[1])
plot.imview(imgp, title='PPP demosaic (DnCNN): %.2f (dB)' %
             metric.psnr(img, imgp), fig=fig, ax=ax[2])

fig.savefig("demosaic_1test1_dncnn_modelpy.jpg", dpi=300)

fig.show()