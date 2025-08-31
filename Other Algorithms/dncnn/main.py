from __future__ import print_function
from builtins import input, range

import numpy as np
from sporco.pgm.ppp import PPP
from sporco.interp import bilinear_demosaic
from sporco import metric, util, plot
from skimage.metrics import structural_similarity as ssim

from dncnn_denoiser import load_dncnn_model, dncnn_luma_denoise

try:
    from colour_demosaicing import demosaicing_CFA_Bayer_Menon2007
    have_demosaic = True
except ImportError:
    have_demosaic = False

plot.config_notebook_plotting()

def A(x):
    y = np.zeros(x.shape[0:2])
    y[1::2, 1::2] = x[1::2, 1::2, 0]
    y[0::2, 1::2] = x[0::2, 1::2, 1]
    y[1::2, 0::2] = x[1::2, 0::2, 1]
    y[0::2, 0::2] = x[0::2, 0::2, 2]
    return y

def AT(x):
    y = np.zeros(x.shape + (3,))
    y[1::2, 1::2, 0] = x[1::2, 1::2]
    y[0::2, 1::2, 1] = x[0::2, 1::2]
    y[1::2, 0::2, 1] = x[1::2, 0::2]
    y[0::2, 0::2, 2] = x[0::2, 0::2]
    return y

def demosaic(cfaimg):
    if have_demosaic:
        return demosaicing_CFA_Bayer_Menon2007(cfaimg, pattern='BGGR')
    else:
        return bilinear_demosaic(cfaimg)

# Load image
img = util.ExampleImages().image('1.png', scaled=True, idxexp=np.s_[0:767, 0:767])
np.random.seed(12345)

s = A(img)
nsigma = 3e-2  # Noise std
sn = s + nsigma * np.random.randn(*s.shape)

def f(x): return 0.5 * np.linalg.norm((A(x) - sn).ravel())**2
def gradf(x): return AT(A(x) - sn)

# Load DnCNN model
dncnn_model = load_dncnn_model('dncnn_gray.pth')  # <-- Download this model and place here

def proxg(x, L):
    return dncnn_luma_denoise(x, dncnn_model)

imgb = dncnn_luma_denoise(demosaic(sn), dncnn_model)

# Run PPP optimization
opt = PPP.Options({'Verbose': True, 'RelStopTol': 1e-3,
                   'MaxMainIter': 10, 'L': 0.5, 'X0': imgb})

b = PPP(img.shape, f, gradf, proxg, opt=opt)
imgp = b.solve()

# Metrics
ssim_imgb = ssim(img, imgb, channel_axis=2, data_range=img.max() - img.min())
ssim_imgp = ssim(img, imgp, channel_axis=2, data_range=img.max() - img.min())

print("PPP PGM solve time:         %5.2f s" % b.timer.elapsed('solve'))
print("Baseline demosaicing PSNR:  %5.2f dB" % metric.psnr(img, imgb))
print("Baseline demosaicing SSIM:  %5.4f"    % ssim_imgb)
print("PPP demosaicing PSNR:       %5.2f dB" % metric.psnr(img, imgp))
print("PPP demosaicing SSIM:       %5.4f"    % ssim_imgp)

# Visualization
fig, ax = plot.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(21, 7))
plot.imview(img, title='Reference', fig=fig, ax=ax[0])
plot.imview(imgb, title='Baseline DnCNN: %.2f (dB)' % metric.psnr(img, imgb), fig=fig, ax=ax[1])
plot.imview(imgp, title='PPP DnCNN: %.2f (dB)' % metric.psnr(img, imgp), fig=fig, ax=ax[2])
fig.savefig("demosaic_dncnn_result.jpg", dpi=300)
fig.show()
