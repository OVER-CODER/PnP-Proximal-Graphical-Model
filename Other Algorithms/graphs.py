# import matplotlib.pyplot as plt
# import pandas as pd
# import numpy as np

# data = {
#     "Method": ["PPP ADMM", "PPP ADMM", "PPP ADMM", "PPP ADMM", "PPP ADMM", "PPP ADMM", "PPP ADMM Consensus", "PPP ADMM Consensus", "PPP ADMM Consensus", "PPP ADMM Consensus", "PPP ADMM Consensus", "PPP ADMM Consensus", "PPP PGM", "PPP PGM", "PPP PGM", "PPP PGM", "PPP PGM", "PPP PGM"],
#     "Label": ["b", "f", "j", "n", "r", "Average", "c", "g", "k", "o", "s", "Average", "d", "h", "l", "p", "t", "Average"],
#     "Solve Time (s)": [215, 210.37, 208.81, 209.26, 209.37, 210.562, 172.97, 174.79, 177.03, 175.74, 175.81, 175.268, 363.19, 357.63, 362.27, 347.78, 359.91, 358.156],
#     "Baseline PSNR (dB)": [37.81, 38.91, 38.47, 48.79, 39.52, 40.7, 37.81, 38.91, 38.47, 48.79, 39.52, 40.7, 37.81, 38.91, 38.47, 48.79, 39.52, 40.7],
#     "PPP PSNR (dB)": [37.97, 38.98, 38.5, 48.6, 39.75, 40.76, 37.83, 38.92, 38.46, 49.33, 39.64, 40.836, 37.86, 38.98, 38.57, 50.23, 39.68, 41.064],
#     "SSIM": [0.999995177033425, 0.999996858376645, 0.999997666119113, 0.999999444484303, 0.999997093834063, 0.999997247969510, 0.999995097769153, 0.999996854375502, 0.999997653506629, 0.999999434478341, 0.999997054090096, 0.999997218843944, 0.999995220823246, 0.999996979043802, 0.999997796913434, 0.999999553518867, 0.999997128235025, 0.999997335706875]
# }

# df = pd.DataFrame(data)


# df_avg = df[df['Label'] == 'Average']

# fig_ssim, ax_ssim = plt.subplots(figsize=(6, 4), dpi=300)
# ax_ssim.bar(df_avg['Method'], df_avg['SSIM'], color=['blue'])
# ax_ssim.set_title("Average SSIM")
# ax_ssim.set_ylabel("SSIM")
# ax_ssim.set_xlabel("Method")
# ax_ssim.set_ylim(0.999997, 0.999998)
# ax_ssim.grid(True, linestyle='--', linewidth=0.7, alpha=0.6)
# fig_ssim.savefig("ssim_graph.png", bbox_inches='tight')

# fig_psnr, ax_psnr = plt.subplots(figsize=(6, 4), dpi=300)
# ax_psnr.bar(df_avg['Method'], df_avg['PPP PSNR (dB)'], color=['blue'])
# ax_psnr.set_title("Average PPP PSNR (dB)")
# ax_psnr.set_ylabel("PSNR (dB)")
# ax_psnr.set_xlabel("Method")
# ax_psnr.set_ylim(40.6, 41.2)
# ax_psnr.grid(True, linestyle='--', linewidth=0.7, alpha=0.6)
# fig_psnr.savefig("psnr_graph.png", bbox_inches='tight')

# fig_table, ax_table = plt.subplots(figsize=(10, 4), dpi=300)
# ax_table.axis('tight')
# ax_table.axis('off')
# table = ax_table.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
# table.auto_set_font_size(False)
# table.set_fontsize(8)
# table.auto_set_column_width([0, 1, 2, 3, 4, 5])
# fig_table.savefig("table.png", bbox_inches='tight')

# plt.show()



import matplotlib.pyplot as plt
import pandas as pd

# Updated method names
method_names = {
    "M1": "PnP ADMM",
    "M2": "PnP ADMM Consensus",
    "M3": "PnP PGM (DnCNN)",
    "M4": "PnP PGM (Bm3D)"
}

# Recomputed data
data = {
    "Method": ["M1", "M1", "M1", "M1", "M1", "M1", "M2", "M2", "M2", "M2", "M2", "M2", "M3", "M3", "M3", "M3", "M3", "M3", "M4", "M4", "M4", "M4", "M4", "M4"],
    "Label": ["b", "f", "j", "n", "r", "Average", "c", "g", "k", "o", "s", "Average", "d", "h", "l", "p", "t", "Average", "e", "i", "m", "q", "u", "Average"],
    "Solve Time (s)": [215, 210.37, 208.81, 209.26, 209.37, 210.562, 172.97, 174.79, 177.03, 175.74, 175.81, 175.268, 363.19, 357.63, 362.27, 347.78, 359.91, 358.156],
    "Recomputed SSIM": [0.8748, 0.8899, 0.8823, 0.9760, 0.9034, 0.90528, 0.8737, 0.8897, 0.8806, 0.9699, 0.9020, 0.90318, 0.8778, 0.8958, 0.8884, 0.9651, 0.907, 0.90682, 0.9073, 0.921, 0.9351, 0.9783, 0.907, 0.92974],
    "Re PSNR": [37.97, 38.94, 38.50, 49.40, 39.74, 40.91, 37.83, 38.92, 38.46, 49.33, 39.64, 40.84, 37.85, 38.97, 38.57, 49.43, 39.67, 40.90, 38.68, 39.85, 41.56, 50.24, 41.06, 42.28],
    "Re Base PSNR": [37.81, 38.91, 38.47, 48.79, 39.52, 40.70, 37.81, 38.91, 38.47, 48.79, 39.52, 40.70, 37.81, 36.70, 34.27, 38.56, 36.82, 36.83]
}

df = pd.DataFrame(data)

# Replace method codes with full names
df["Method"] = df["Method"].map(method_names)
df_avg = df[df['Label'] == 'Average']

# Colors for each method
colors = ['blue']

# SSIM bar chart
fig_ssim, ax_ssim = plt.subplots(figsize=(6, 4), dpi=300)
ax_ssim.bar(df_avg['Method'], df_avg['Recomputed SSIM'], color=colors)
ax_ssim.set_title("Average SSIM")
ax_ssim.set_ylabel("SSIM")
ax_ssim.set_xlabel("Method")
ax_ssim.set_ylim(0.9, 0.92)
ax_ssim.grid(True, linestyle='--', linewidth=0.7, alpha=0.6)
fig_ssim.savefig("ssim_graph.png", bbox_inches='tight')

# PSNR bar chart
fig_psnr, ax_psnr = plt.subplots(figsize=(6, 4), dpi=300)
ax_psnr.bar(df_avg['Method'], df_avg['Re PSNR'], color=colors)
ax_psnr.set_title("Average PSNR (dB)")
ax_psnr.set_ylabel("PSNR (dB)")
ax_psnr.set_xlabel("Method")
ax_psnr.set_ylim(40.6, 41.2)
ax_psnr.grid(True, linestyle='--', linewidth=0.7, alpha=0.6)
fig_psnr.savefig("psnr_graph.png", bbox_inches='tight')

# Table with full names
fig_table, ax_table = plt.subplots(figsize=(11, 4), dpi=300)
ax_table.axis('tight')
ax_table.axis('off')
table = ax_table.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
table.auto_set_font_size(False)
table.set_fontsize(8)
table.auto_set_column_width([0, 1, 2, 3, 4, 5])
fig_table.savefig("recomputed_table.png", bbox_inches='tight')

plt.show()
