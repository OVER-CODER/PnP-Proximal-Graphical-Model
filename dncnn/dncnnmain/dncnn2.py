import os
import numpy as np
import torch
import torch.nn as nn
import h5py
from sporco.pgm.ppp import PPP
from sporco.interp import bilinear_demosaic
from sporco import metric, util, plot
from skimage.metrics import structural_similarity as ssim

# Try importing Menon2007 demosaicing
try:
    from colour_demosaicing import demosaicing_CFA_Bayer_Menon2007
    have_demosaic = True
except ImportError:
    have_demosaic = False

plot.config_notebook_plotting()

# CFA functions
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
    return bilinear_demosaic(cfaimg)

# ----------------------
# DnCNN (PyTorch version)
# ----------------------
class DnCNN(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, depth=17, num_filters=64):
        super(DnCNN, self).__init__()
        layers = [nn.Conv2d(in_channels, num_filters, kernel_size=3, padding=1, bias=False), nn.ReLU(inplace=True)]
        for _ in range(depth - 2):
            layers += [nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False),
                       nn.BatchNorm2d(num_filters), nn.ReLU(inplace=True)]
        layers.append(nn.Conv2d(num_filters, out_channels, kernel_size=3, padding=1, bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        return x - self.dncnn(x)

class DnCNNDenoiser:
    def __init__(self, noise_level, model_path=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.noise_level = noise_level

        if model_path and model_path.endswith(".npz"):
            weights_npz = np.load(model_path)
            self.model = DnCNN(in_channels=1, out_channels=1)  # Assuming grayscale
            self.model.to(self.device)
            self.transfer_weights_from_npz(weights_npz)
            print("✅ DnCNN model loaded from .npz weights.")
        else:
            print("❌ No valid .npz model provided. Using random init 3-channel model.")
            self.model = DnCNN(in_channels=3, out_channels=3)
            self.model.to(self.device)

        self.model.eval()

    def transfer_weights_from_npz(self, weights_npz):
        print("Keys in NPZ file:")
        for key in weights_npz:
            print(key)

        conv_keys = sorted(
            [k for k in weights_npz if k.startswith("conv2d")],
            key=lambda x: int(x.split('_')[-1]) if '_' in x and x.split('_')[-1].isdigit() else -1
        )

        bn_keys = sorted(
            [k for k in weights_npz if k.startswith("batch_normalization")],
            key=lambda x: int(x.split('_')[-1]) if '_' in x and x.split('_')[-1].isdigit() else -1
        )

        pytorch_layers = list(self.model.dncnn.children())

        conv_layer_idx = 0
        for conv_key in conv_keys:
            conv_weights = weights_npz[conv_key]
            
            if conv_weights.ndim == 4:  # kernel
                kernel = np.transpose(conv_weights, (3, 2, 0, 1))  # Keras (H, W, inC, outC) to PyTorch (outC, inC, H, W)
                pytorch_layers[conv_layer_idx].weight.data.copy_(torch.from_numpy(kernel))
            elif conv_weights.ndim == 1 and np.issubdtype(conv_weights.dtype, np.number):  # bias
                if pytorch_layers[conv_layer_idx].bias is not None:
                    pytorch_layers[conv_layer_idx].bias.data.copy_(torch.from_numpy(conv_weights))



            # Move to next conv block every 3 layers (Conv → BN → ReLU)
            if conv_layer_idx + 3 < len(pytorch_layers):
                conv_layer_idx += 3

        bn_layer_idx = 1
        for bn_key in bn_keys:
            bn_weights = weights_npz[bn_key]  # shape: (4, C) or similar
            gamma, beta, mean, var = bn_weights

            bn_layer = pytorch_layers[bn_layer_idx]
            bn_layer.weight.data.copy_(torch.from_numpy(gamma))
            bn_layer.bias.data.copy_(torch.from_numpy(beta))
            bn_layer.running_mean.copy_(torch.from_numpy(mean))
            bn_layer.running_var.copy_(torch.from_numpy(var))

            bn_layer_idx += 3



    def __call__(self, img_np):
        if img_np.ndim == 2:
            img_np = np.expand_dims(img_np, axis=-1)

        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            denoised = self.model(img_tensor)
        return np.clip(denoised.squeeze().permute(1, 2, 0).cpu().numpy(), 0, 1)

# ----------------------
# Main Image Processing
# ----------------------
def main():
    # Load example image
    try:
        img = util.ExampleImages().image('1.png', scaled=True, idxexp=np.s_[0:767, 0:767])
    except Exception as e:
        print("❌ Could not load example image: '1.png'. Place it in correct directory.")
        return

    np.random.seed(12345)
    s = A(img)
    nsigma = 3e-2
    sn = s + nsigma * np.random.randn(*s.shape)

    def f(x): return 0.5 * np.linalg.norm((A(x) - sn).ravel())**2
    def gradf(x): return AT(A(x) - sn)

    # Denoiser
    bsigma = 4.5e-2
    model_path = 'dncnn_weights_extracted.npz'
    if not os.path.exists(model_path):
        print(f"❌ Model file {model_path} not found.")
        return

    denoiser = DnCNNDenoiser(bsigma, model_path=model_path)

    imgb = denoiser(demosaic(sn))  # Baseline demosaic + denoise

    def proxg(x, L): return denoiser(x)

    opt = PPP.Options({'Verbose': True, 'RelStopTol': 1e-5, 'MaxMainIter': 20, 'L': 0.5, 'X0': imgb})
    solver = PPP(img.shape, f, gradf, proxg, opt)
    imgp = solver.solve()

    ssim_imgb = ssim(img, imgb, channel_axis=2, data_range=img.max() - img.min())
    ssim_imgp = ssim(img, imgp, channel_axis=2, data_range=img.max() - img.min())

    print(f"✅ PPP PGM solve time:        {solver.timer.elapsed('solve'):.2f} s")
    print(f"✅ Baseline PSNR:             {metric.psnr(img, imgb):.2f} dB, SSIM: {ssim_imgb:.4f}")
    print(f"✅ PPP Restored PSNR:         {metric.psnr(img, imgp):.2f} dB, SSIM: {ssim_imgp:.4f}")

    fig, ax = plot.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=(21, 7))
    plot.imview(img, title='Reference', fig=fig, ax=ax[0])
    plot.imview(imgb, title='Baseline Demosaic + DnCNN', fig=fig, ax=ax[1])
    plot.imview(imgp, title='PPP + DnCNN', fig=fig, ax=ax[2])
    fig.savefig("results_demosaic_dncnn.jpg", dpi=300)
    print("🖼️ Saved comparison figure as 'results_demosaic_dncnn.jpg'")

if __name__ == "__main__":
    main()
