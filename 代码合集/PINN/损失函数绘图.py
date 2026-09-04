from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =====================================================
# 1. 改这里：六种方法的 history.csv 路径
# =====================================================

METHODS = [
    {
        "name": "Classic Soft-IC PINN",
        "file": Path(r"C:\Users\50971\Desktop\方程文件\代码\classic_soft_ic_pinn_nf10000_v1.0_d3.23\history.csv"),
    },
    {
        "name": "Classic Hard-IC PINN",
        "file": Path(r"C:\Users\50971\Desktop\方程文件\代码\classic_hard_ic_pinn_v1.0_d3.23\history.csv"),
    },
    {
        "name": "Analytic Hard-IC Stage-II only",
        "file": Path(r"C:\Users\50971\Desktop\方程文件\代码\hard_ic_final_solov1.0_d3.23\history.csv"),
    },
    {
        "name": "Analytic Hard-IC Two-stage",
        "file": Path(r"C:\Users\50971\Desktop\方程文件\代码\hard_ic_final_v1.0_d3.23\history.csv"),
    },
    {
        "name": "Analytic Soft-IC Stage-II Only",
        "file": Path(r"C:\Users\50971\Desktop\方程文件\代码\soft_ic_final_solo_v1.0_d3.23\history.csv"),
    },
    {
        "name": "Analytic Soft-IC Two-stage",
        "file": Path(r"C:\Users\50971\Desktop\方程文件\代码\soft_ic_final_v1.0_d3.23\history.csv"),
    },
]


# 图片保存位置
OUT_DIR = Path(r"C:\Users\50971\Desktop\方程文件\代码\loss_compare_v1")
OUT_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================
# 2. 读取 loss 文件
# =====================================================

def read_history(method):
    csv_path = method["file"]

    if not csv_path.exists():
        raise FileNotFoundError(f"找不到文件: {csv_path}")

    df = pd.read_csv(csv_path)

    # 兼容 global_step / step 两种写法
    if "global_step" in df.columns:
        steps = df["global_step"].to_numpy(dtype=float)
    elif "step" in df.columns:
        steps = df["step"].to_numpy(dtype=float)
    else:
        raise KeyError(f"{csv_path} 中没有 global_step 或 step 列")

    if "total" not in df.columns:
        raise KeyError(f"{csv_path} 中没有 total 列")

    if "pde" not in df.columns:
        raise KeyError(f"{csv_path} 中没有 pde 列")

    total = df["total"].to_numpy(dtype=float)
    pde = df["pde"].to_numpy(dtype=float)

    # total 不能小于等于 0，否则 log 坐标会出错
    total = np.where(total > 0.0, total, np.nan)

    # 对完整二阶段，pretrain 阶段没有真正计算 PDE
    # 所以 stage == pretrain 的 pde 不画
    if "stage" in df.columns:
        stage = df["stage"].astype(str).str.lower().to_numpy()
        pretrain_mask = np.array(["pretrain" in s for s in stage])
        pde = np.where(pretrain_mask, np.nan, pde)

    # pde <= 0 的点也不画，避免 log 坐标显示假平台
    pde = np.where(pde > 0.0, pde, np.nan)

    return steps, total, pde


histories = []

for method in METHODS:
    steps, total, pde = read_history(method)
    histories.append(
        {
            "name": method["name"],
            "steps": steps,
            "total": total,
            "pde": pde,
        }
    )


# =====================================================
# 3. 通用画图函数
# =====================================================

def beautify_axis(ax):
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    # 只保留稀疏主网格，删掉细碎次网格
    ax.grid(True, which="major", alpha=0.2, color="#cccccc", linewidth=0.8)
    ax.set_yscale("log")
    # 网格放在曲线下方，不会盖住线条
    ax.set_axisbelow(True)

    # 完整二阶段分界线：7000 步后进入 PDE fine-tuning
    ax.axvline(
        7000,
        linestyle="--",
        linewidth=1.0,
        color="0.4",
        alpha=0.6,
        label="PDE Fine-Tuning Starts",
        zorder=1,
    )

    ax.legend(
        fontsize=8,
        loc="best",
        frameon=True,
    )

# =====================================================
# 4. 图一：六种方法 total loss
# =====================================================

fig, ax = plt.subplots(figsize=(8.2, 5.2))

for item in histories:
    ax.plot(
        item["steps"],
        item["total"],
        linewidth=2.0,
        # marker="o",
        # markersize=3.5,
        label=item["name"],
    )

ax.set_title(r"Total Loss Comparison, $\nu=1.0,\ d=3.23$")
beautify_axis(ax)

fig.tight_layout()
fig.savefig(OUT_DIR / "all_methods_total_loss_v1.png", dpi=300)
plt.close(fig)


# =====================================================
# 5. 图二：六种方法 PDE loss
# =====================================================

fig, ax = plt.subplots(figsize=(8.2, 5.2))

for item in histories:
    ax.plot(
        item["steps"],
        item["pde"],
        linewidth=2.0,
        # marker="o",
        # markersize=3.5,
        label=item["name"],
    )

ax.set_title(r"PDE Loss Comparison, $\nu=1.0,\ d=3.23$")
beautify_axis(ax)

fig.tight_layout()
fig.savefig(OUT_DIR / "all_methods_pde_loss_v1.png", dpi=300)
plt.close(fig)


# =====================================================
# 6. 图三：total 和 PDE 放在一张图里
# =====================================================

fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.2), sharex=True)

ax = axes[0]
for item in histories:
    ax.plot(
        item["steps"],
        item["total"],
        linewidth=2.0,
        # marker="o",
        # markersize=3.5,
        label=item["name"],
    )
ax.set_title("Total Loss")
beautify_axis(ax)

ax = axes[1]
for item in histories:
    ax.plot(
        item["steps"],
        item["pde"],
        linewidth=2.0,
        marker="o",
        markersize=3.5,
        label=item["name"],
    )
ax.set_title("PDE Loss")
beautify_axis(ax)

fig.suptitle(r"Loss Comparison Of Six PINN Variants, $\nu=1.0,\ d=3.23$", fontsize=14)
fig.tight_layout()
fig.savefig(OUT_DIR / "all_methods_total_and_pde_loss_v1.png", dpi=300)
plt.close(fig)


# =====================================================
# 7. 打印保存路径
# =====================================================

print("Saved figures:")
print("  ", OUT_DIR / "all_methods_total_loss_v1.png")
print("  ", OUT_DIR / "all_methods_pde_loss_v1.png")
print("  ", OUT_DIR / "all_methods_total_and_pde_loss_v1.png")