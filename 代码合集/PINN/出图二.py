from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

out_dir = Path("figure2_outputs")
out_dir.mkdir(exist_ok=True)

systems = ["M1","M2","M3","M4","M5","M6"]
cases = [r"$\nu=0.8$", r"$\nu=1.0$", r"$\nu=1.2$"]
markers = ["o","s","^"]
x = np.arange(6, dtype=float)
offsets = [-0.17, 0.0, 0.17]

data = {
    "pde": np.array([
        [4.658e-4,1.223e-4,7.842e-4],
        [9.140,8.800,8.930],
        [7.925e-7,3.310e-7,7.671e-7],
        [6.329e-7,4.240e-7,5.999e-7],
        [3.807e-7,5.549e-4,1.714e-6],
        [4.420e-7,3.233e-5,7.032e-7],
    ]),
    "ephi": np.array([
        [1,1,1],[2,2,2],
        [5.011e-3,5.004e-3,5.005e-3],
        [5.009e-3,5.006e-3,5.009e-3],
        [5.005e-3,6.280e-3,5.008e-3],
        [5.004e-3,5.050e-3,5.005e-3],
    ]),
    "etraj": np.array([
        [1.020e-1,1.550e-1,4.080e-2],
        [2.650e-1,2.370e-1,2.250e-1],
        [2.357e-6,8.415e-7,1.221e-6],
        [2.334e-6,9.469e-7,1.380e-6],
        [2.358e-6,7.044e-6,1.648e-6],
        [2.058e-6,2.256e-6,1.051e-6],
    ]),
    "time": np.array([
        [213.9,214.1,214.7],
        [552.1,540.8,630.3],
        [3982.8,2497.3,3937.8],
        [588.4,730.0,533.2],
        [1493.3,3644.2,2760.8],
        [570.9,1539.7,665.8],
    ]),
}

specs = [
    ("pde", r"$\mathrm{MSE}^{\mathrm{all}}_{\mathrm{PDE}}$", "(a) Independent Full-Domain PDE Residual", True),
    ("ephi", r"$E_{\Phi,\max}$", "(b) Maximum Relative Complex-Field Error", True),
    ("etraj", r"$E_{\mathrm{traj}}$", "(c) Normalized Centroid-Trajectory Error", True),
    ("time", "Optimization time (s)", "(d) Wall-Clock Optimization Time", False),
]

panel_pngs = []

for idx, (key, ylabel, title, logy) in enumerate(specs):
    fig, ax = plt.subplots(figsize=(6.6,4.8))
    vals = data[key]
    for j, (case, marker, dx) in enumerate(zip(cases, markers, offsets)):
        ax.scatter(x+dx, vals[:,j], s=58, marker=marker, label=case, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(systems)
    ax.set_xlabel("PINN System")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    if logy:
        ax.set_yscale("log")
    # ========== 修改网格：只画水平y主网格，去掉竖网格、去掉log次要细碎网格 ==========
    ax.grid(True, axis="y", which="major", alpha=0.22)
    # 删除原来minor网格绘制
    # if logy:
    #     ax.grid(True, which="minor", alpha=0.10)
    # ========= 删除阴影竖条 ax.axvspan =========
    # ax.axvspan(2.72, 3.28, alpha=0.07)
    # 竖向网格线：对应 M1, M2, M3, M4, M5, M6
    ax.axvspan(2.72, 3.28, alpha=0.07)
    ax.grid(True, axis="x", which="major", alpha=0.22)
    if idx == 0:
        ax.legend(frameon=True, fontsize=10)
    fig.tight_layout()

    p_png = out_dir / f"figure2_{key}.png"
    p_pdf = out_dir / f"figure2_{key}.pdf"
    fig.savefig(p_png, dpi=320, bbox_inches="tight")
    fig.savefig(p_pdf, bbox_inches="tight")
    plt.close(fig)
    panel_pngs.append(p_png)

imgs = [Image.open(p).convert("RGB") for p in panel_pngs]
w = max(im.width for im in imgs)
h = max(im.height for im in imgs)
canvas = Image.new("RGB", (2*w, 2*h), "white")
for i, im in enumerate(imgs):
    col, row = i % 2, i // 2
    canvas.paste(im, (col*w + (w-im.width)//2, row*h + (h-im.height)//2))

canvas.save(out_dir / "Figure2_cross_regime_comparison.png", dpi=(320,320))
canvas.save(out_dir / "Figure2_cross_regime_comparison.pdf", "PDF", resolution=320)