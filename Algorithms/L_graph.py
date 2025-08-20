import matplotlib.pyplot as plt

# Sorted data for α (previously L)
alpha_sorted = [0.6, 0.68, 0.71, 0.74, 0.8]
psnr_sorted = [19.35, 37.89, 37.89, 37.88, 37.87]
ssim_sorted = [0.8726, 0.8765, 0.8767, 0.8769, 0.8773]

# PSNR plot
fig_psnr, ax_psnr = plt.subplots(figsize=(6, 4), dpi=600)
ax_psnr.plot(alpha_sorted, psnr_sorted, marker='o', linestyle='-', color='blue')
ax_psnr.set_title(r'PSNR vs PnP Step Size ($\alpha$)')
ax_psnr.set_xlabel(r'$\alpha$')
ax_psnr.set_ylabel('PSNR (dB)')
ax_psnr.grid(True)
fig_psnr.tight_layout()
fig_psnr.savefig("psnr_vs_alpha.png", dpi=600)

# SSIM plot
fig_ssim, ax_ssim = plt.subplots(figsize=(6, 4), dpi=600)
ax_ssim.plot(alpha_sorted, ssim_sorted, marker='o', linestyle='-', color='green')
ax_ssim.set_title(r'SSIM vs PnP Step Size ($\alpha$)')
ax_ssim.set_xlabel(r'$\alpha$')
ax_ssim.set_ylabel('SSIM')
ax_ssim.grid(True)
fig_ssim.tight_layout()
fig_ssim.savefig("ssim_vs_alpha.png", dpi=600)
