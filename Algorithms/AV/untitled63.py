# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 16:51:07 2025

@author: DELL
"""

from __future__ import print_function
from builtins import input, range

import numpy as np
from scipy.sparse.linalg import LinearOperator

from bm3d import bm3d_rgb
try:
    from colour_demosaicing import demosaicing_CFA_Bayer_Menon2007
except ImportError:
    have_demosaic = False
else:
    have_demosaic = True

from sporco.linalg import _cg_wrapper
from sporco.admm.ppp import PPP
from sporco.interp import bilinear_demosaic
from sporco import metric
from sporco import util
from sporco import plot
plot.config_notebook_plotting()


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
    

img = util.ExampleImages().image('kodim23.png', scaled=True,
                                 idxexp=np.s_[160:416,60:316])

np.random.seed(12345)
s = A(img)
rgbshp = s.shape + (3,)  # Shape of reconstructed RGB image
rgbsz = s.size * 3       # Size of reconstructed RGB image
nsigma = 2e-2            # Noise standard deviation
sn = s + nsigma * np.random.randn(*s.shape)


def f(x):
    return 0.5 * np.linalg.norm((A(x) - sn).ravel())**2

def proxf(x, rho, tol=1e-3, maxit=100):
    ATA = lambda z: AT(A(z))
    ATAI = lambda z: ATA(z.reshape(rgbshp)).ravel() + rho * z.ravel()
    lop = LinearOperator((rgbsz, rgbsz), matvec=ATAI, dtype=s.dtype)
    b = AT(sn) + rho * x
    vx, cgit = _cg_wrapper(lop, b.ravel(), None, tol, maxit)
    return vx.reshape(rgbshp)



bsigma = 6.1e-2  # Denoiser parameter

def proxg(x, rho):
    return bm3d_rgb(x, bsigma)


imgb = bm3d_rgb(demosaic(sn), 3 * nsigma)


opt = PPP.Options({'Verbose': True, 'RelStopTol': 1e-3,
                   'MaxMainIter': 12, 'rho': 1.8e-1, 'Y0': imgb})


b = PPP(img.shape, f, proxf, proxg, opt=opt)
imgp = b.solve()

print("PPP ADMM solve time:        %5.2f s" % b.timer.elapsed('solve'))
print("Baseline demosaicing PSNR:  %5.2f dB" % metric.psnr(img, imgb))
print("PPP demosaicing PSNR:       %5.2f dB" % metric.psnr(img, imgp))

fig, ax = plot.subplots(nrows=1, ncols=3, sharex=True, sharey=True,
                        figsize=(21, 7))
plot.imview(img, title='Reference', fig=fig, ax=ax[0])
plot.imview(imgb, title='Baseline demoisac: %.2f (dB)' %
            metric.psnr(img, imgb), fig=fig, ax=ax[1])
plot.imview(imgp, title='PPP demoisac: %.2f (dB)' %
            metric.psnr(img, imgp), fig=fig, ax=ax[2])
fig.show()





















































