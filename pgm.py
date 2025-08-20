from __future__ import print_function
from builtins import input, range

import numpy as np

from bm3d import bm3d_rgb
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
from skimage.restoration import denoise_tv_chambolle


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
    
img = util.ExampleImages().image('4.png', scaled=True,
                                 idxexp=np.s_[0:767,0:767])


np.random.seed(12345)
s = A(img)
rgbshp = s.shape + (3,)  # Shape of reconstructed RGB image
rgbsz = s.size * 3       # Size of reconstructed RGB image
nsigma = 2e-2            # Noise standard deviation
sn = s + nsigma * np.random.randn(*s.shape)


def f(x):
    return 0.5 * np.linalg.norm((A(x) - sn).ravel())**2

def gradf(x):
    return AT(A(x) - sn)


bsigma = 0.033  # Denoiser parameter

def proxg(x, L):
    return bm3d_rgb(x, bsigma)

sn_rgb = AT(sn)
sn_denoised = bm3d_rgb(sn_rgb, 3 * nsigma)
imgb = demosaic(A(sn_denoised))



opt = PPP.Options({'Verbose': True, 'RelStopTol': 1e-3,
                   'MaxMainIter': 20, 'L': 0.70, 'X0': imgb})


b = PPP(img.shape, f, gradf, proxg, opt=opt)
imgp = b.solve()

demosaiced_sn = demosaic(sn)
img_tv = denoise_tv_chambolle(demosaiced_sn, weight=0.1, channel_axis=2)
img_tv = np.clip(img_tv, 0, 1)

psnr_tv = metric.psnr(img, img_tv)
ssim_tv = ssim(img, img_tv, channel_axis=2, data_range=img.max() - img.min())

ssim_imgb = ssim(img, imgb, channel_axis=2, data_range=img.max() - img.min())
ssim_imgp = ssim(img, imgp, channel_axis=2, data_range=img.max() - img.min())


print("PPP PGM solve time:        %5.2f s" % b.timer.elapsed('solve'))
print("Baseline demosaicing PSNR:  %5.2f dB" % metric.psnr(img, imgb))
print("Baseline demosaicing SSIM:  %5.4f"    % ssim_imgb)
print("PPP demosaicing PSNR:       %5.2f dB" % metric.psnr(img, imgp))
print("PPP demosaicing SSIM:       %5.4f"    % ssim_imgp)
print("TV Demosaicing PSNR:        %5.2f dB" % psnr_tv)
print("TV Demosaicing SSIM:        %5.4f"     % ssim_tv)


fig, ax = plot.subplots(nrows=1, ncols=3, sharex=True, sharey=True,
                        figsize=(21, 7))
plot.imview(img, title='Reference', fig=fig, ax=ax[0])
plot.imview(imgb, title='Baseline demoisac: %.2f (dB)' %
            metric.psnr(img, imgb), fig=fig, ax=ax[1])
plot.imview(imgp, title='PPP demoisac: %.2f (dB)' %
            metric.psnr(img, imgp), fig=fig, ax=ax[2])

# fig.savefig("demosaic_pgm_1.jpg", dpi=300)

fig.show()