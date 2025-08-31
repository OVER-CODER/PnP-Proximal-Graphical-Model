import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Data provided by the user
methods = ["PnP ADMM", "PnP ADMM Consensus", "PnP PGM (DnCNN)", "PnP PGM (Bm3D)"]
average_ssim = [0.90528, 0.90318, 0.90682, 0.92974]
average_psnr = [40.91, 40.84, 40.90, 42.28]

# Create a DataFrame
df = pd.DataFrame({
    'Method': methods,
    'Average SSIM': average_ssim,
    'Average PSNR': average_psnr
})

# SSIM Bar Chart
fig_ssim, ax_ssim = plt.subplots(figsize=(8, 5), dpi=300)
ax_ssim.bar(df['Method'], df['Average SSIM'], color='blue')
ax_ssim.set_title("Average SSIM", fontsize=14)
ax_ssim.set_ylabel("SSIM", fontsize=12)
ax_ssim.set_xlabel("Method", fontsize=12)
ax_ssim.set_ylim(min(df['Average SSIM']) * 0.98, max(df['Average SSIM']) * 1.02) # Adjust y-lim for better visualization
ax_ssim.grid(axis='y', linestyle='--', alpha=0.7)
# plt.xticks(rotation=15, ha='right') # Rotate x-axis labels for readability
plt.tight_layout()
plt.savefig("average_ssim_plot.png", bbox_inches='tight')


# PSNR Bar Chart
fig_psnr, ax_psnr = plt.subplots(figsize=(8, 5), dpi=300)
ax_psnr.bar(df['Method'], df['Average PSNR'], color='blue')
ax_psnr.set_title("Average PSNR (dB)", fontsize=14)
ax_psnr.set_ylabel("PSNR (dB)", fontsize=12)
ax_psnr.set_xlabel("Method", fontsize=12)
ax_psnr.set_ylim(min(df['Average PSNR']) * 0.98, max(df['Average PSNR']) * 1.02) # Adjust y-lim
ax_psnr.grid(axis='y', linestyle='--', alpha=0.7)
# plt.xticks(rotation=15, ha='right') # Rotate x-axis labels for readability
plt.tight_layout()
plt.savefig("average_psnr_plot.png", bbox_inches='tight')


# You can also display the data in a table if needed
fig_table, ax_table = plt.subplots(figsize=(7, 2), dpi=300)
ax_table.axis('tight')
ax_table.axis('off')
table = ax_table.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.auto_set_column_width(col=list(range(len(df.columns))))
plt.tight_layout()
plt.savefig("data_table.png", bbox_inches='tight')
