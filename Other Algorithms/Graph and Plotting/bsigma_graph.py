import matplotlib.pyplot as plt

# Your sorted data
bsigma_sorted = [0.02, 0.024, 0.028, 0.033, 0.036]
psnr_sorted = [37.81, 37.89, 37.89, 37.86, 37.84]
ssim_sorted = [0.8688, 0.8743, 0.8765, 0.8778, 0.8781]

# PSNR plot
fig_psnr, ax_psnr = plt.subplots(figsize=(6, 4), dpi=600)
ax_psnr.plot(bsigma_sorted, psnr_sorted, marker='o', linestyle='-', color='blue')
ax_psnr.set_title('PSNR vs BM3D σ (bsigma)')
ax_psnr.set_xlabel('bsigma')
ax_psnr.set_ylabel('PSNR (dB)')
ax_psnr.grid(True)
fig_psnr.tight_layout()
fig_psnr.savefig("psnr_vs_bsigma.png", dpi=600)

# SSIM plot
fig_ssim, ax_ssim = plt.subplots(figsize=(6, 4), dpi=600)
ax_ssim.plot(bsigma_sorted, ssim_sorted, marker='o', linestyle='-', color='green')
ax_ssim.set_title('SSIM vs BM3D σ (bsigma)')
ax_ssim.set_xlabel('bsigma')
ax_ssim.set_ylabel('SSIM')
ax_ssim.grid(True)
fig_ssim.tight_layout()
fig_ssim.savefig("ssim_vs_bsigma.png", dpi=600)
