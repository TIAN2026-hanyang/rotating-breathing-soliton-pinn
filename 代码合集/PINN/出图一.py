from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterMathtext, NullLocator, NullFormatter
from matplotlib.ticker import FuncFormatter
from matplotlib.colors import LogNorm

# =========================================================
# 1. 改这里：三组结果文件夹
# =========================================================
CASES = [
    {
        "tag": "v0.8_d3.60",
        "label": r"$\nu=0.8$",
        "nu": 0.8,
        "d": 3.60,
        "result_dir": Path(r"C:\Users\50971\Desktop\代码\hard_ic_final_v0.8_d3.60"),
    },
    {
        "tag": "v1.0_d3.23",
        "label": r"$\nu=1.0$",
        "nu": 1.0,
        "d": 3.23,
        "result_dir": Path(r"C:\Users\50971\Desktop\代码\hard_ic_final_v1.0_d3.23"),
    },
    {
        "tag": "v1.2_d2.94",
        "label": r"$\nu=1.2$",
        "nu": 1.2,
        "d": 2.94,
        "result_dir": Path(r"C:\Users\50971\Desktop\代码\hard_ic_final_v1.2_d2.94"),
    },
]

# 单独新建一个文件夹保存三组合图
OUTPUT_DIR = Path(r"C:\Users\50971\Desktop\代码\three_velocity_combined_results")

# 功率雷达图：
# False = 画 P(z)
# True  = 画 |P(z)-P_target|
USE_POWER_ERROR_FOR_RADAR = False

# 三种柔和配色（不要太冲）
COLORS = [
    "#4C78A8",  # 柔和蓝
    "#72B7B2",  # 柔和青绿
    "#E6B36A",  # 柔和橙棕
]


# =========================================================
# 2. 工具函数
# =========================================================
def set_clean_log_y_axis(ax, positive_values):
    positive_values = np.asarray(positive_values, dtype=float)
    positive_values = positive_values[positive_values > 0.0]

    if positive_values.size == 0:
        return

    y_min = float(np.min(positive_values))
    y_max = float(np.max(positive_values))

    ax.set_yscale("log")
    ax.set_ylim(y_min * 0.55, y_max * 2.2)

    ax.yaxis.set_major_locator(LogLocator(base=10.0))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))

    # 去掉次刻度，避免太密太乱
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_formatter(NullFormatter())

    ax.grid(True, axis="y", which="major", alpha=0.25)


# def add_bar_labels_slanted(ax, bars, values, fmt="{:.2e}", rotation=28, fontsize=9):
#     for bar, value in zip(bars, values):
#         x = bar.get_x() + bar.get_width() / 2.0
#         y = bar.get_height()
#
#         ax.text(
#             x,
#             y * 1.06,
#             fmt.format(value),
#             ha="left",
#             va="bottom",
#             rotation=rotation,
#             fontsize=fontsize,
#         )
def add_bar_labels_slanted(
    ax,
    bars,
    values,
    fmt="{:.2e}",
    y_factor=1.08,
    rotation=45,
    fontsize=8,
):
    """
    给柱子顶部加斜着的数字。
    y_factor 用来控制数字高度。
    三组柱子分别用不同 y_factor，可以避免数字重叠。
    """
    for bar, value in zip(bars, values):
        if value <= 0.0:
            continue

        x = bar.get_x() + bar.get_width() / 2.0
        y = bar.get_height()

        ax.text(
            x,
            y * y_factor,
            fmt.format(value),
            ha="center",
            va="bottom",
            rotation=rotation,
            fontsize=fontsize,
            color="black",
            zorder=10,
        )

# =========================================================
# 3. 读取单个 case
# =========================================================
def load_case_data(case):
    npz_path = case["result_dir"] / "prediction_results.npz"
    summary_path = case["result_dir"] / "summary.json"

    if not npz_path.exists():
        raise FileNotFoundError(f"找不到文件: {npz_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"找不到文件: {summary_path}")

    data = np.load(npz_path, allow_pickle=False)

    z_values = data["z"]
    powers = data["powers"]

    centroids_pinn = data["centroids_pinn"]
    centroids_reference = data["centroids_reference"]

    radius_pinn = data["radius_pinn"]
    radius_reference = data["radius_reference"]

    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    d_value = float(summary.get("d", case["d"]))
    nu_value = float(summary.get("nu", case["nu"]))
    target_power = float(summary["config"]["target_power"])

    z_plot = z_values / np.pi

    # ---------- 轨迹误差 ----------
    if centroids_reference.size == 0:
        raise ValueError(f"{case['tag']} 中 centroids_reference 为空，无法计算轨迹误差。")

    centroid_diff = centroids_pinn - centroids_reference
    trajectory_error_z = (
        np.sqrt(np.mean(np.sum(centroid_diff ** 2, axis=2), axis=1)) / d_value
    )
    trajectory_error_global = float(
        np.sqrt(np.mean(np.sum(centroid_diff ** 2, axis=2))) / d_value
    )

    # ---------- 半径误差 ----------
    if radius_reference.size == 0:
        raise ValueError(f"{case['tag']} 中 radius_reference 为空，无法计算半径误差。")

    radius_abs_error = np.abs(radius_pinn - radius_reference)

    return {
        "tag": case["tag"],
        "label": case["label"],
        "nu": nu_value,
        "d": d_value,
        "target_power": target_power,
        "z_plot": z_plot,
        "powers": powers,
        "trajectory_error_z": trajectory_error_z,
        "trajectory_error_global": trajectory_error_global,
        "radius_abs_error": radius_abs_error,
    }


# =========================================================
# 4. 三种速度：轨迹误差柱状图
# =========================================================
def plot_three_case_trajectory_error(cases_data, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    z_ref = cases_data[0]["z_plot"]
    for item in cases_data[1:]:
        if len(item["z_plot"]) != len(z_ref) or not np.allclose(item["z_plot"], z_ref):
            raise ValueError("三个 case 的 z 切片不一致，不能直接画在同一张图上。")

    x_labels = [f"{z:.1f}" for z in z_ref]
    x = np.arange(len(z_ref))

    fig, ax = plt.subplots(figsize=(10, 6.5))

    width = 0.22
    offsets = [-width, 0.0, width]

    all_positive_values = []

    for i, item in enumerate(cases_data):
        values = item["trajectory_error_z"]
        all_positive_values.extend(values[values > 0.0])

        bars = ax.bar(
            x + offsets[i],
            values,
            width=width,
            color=COLORS[i],
            label=item["label"],
            edgecolor="none",
            alpha=0.95,
        )

        # add_bar_labels_slanted(ax, bars, values, fmt="{:.2e}", rotation=28, fontsize=9)
        label_y_factors = [1.00, 1.45, 0.90]

        add_bar_labels_slanted(
            ax,
            bars,
            values,
            fmt="{:.2e}",
            y_factor=label_y_factors[i],
            rotation=45,
            fontsize=7,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=12)
    ax.set_xlabel(r"$z/\pi$", fontsize=15)
    set_clean_log_y_axis(ax, np.asarray(all_positive_values))

    ax.legend(loc="upper left", fontsize=12, frameon=True, framealpha=0.95)

    fig.tight_layout()
    save_path = output_dir / "centroid_trajectory_error_3cases_bar.png"
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

    print(f"[已保存] {save_path}")

def plot_three_case_trajectory_error_violin(cases_data, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.2, 6.2))

    data_list = [np.asarray(item["trajectory_error_z"], dtype=float) for item in cases_data]
    labels = [item["label"] for item in cases_data]

    vp = ax.violinplot(
        data_list,
        positions=np.arange(1, len(data_list) + 1),
        widths=0.75,
        showmeans=False,
        showmedians=True,
        showextrema=False,
    )

    # 小提琴上色
    for body, color in zip(vp["bodies"], COLORS):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.28)

    if "cmedians" in vp:
        vp["cmedians"].set_color("black")
        vp["cmedians"].set_linewidth(1.8)

    # 叠加真实散点
    for i, vals in enumerate(data_list, start=1):
        jitter = np.linspace(-0.07, 0.07, len(vals))
        ax.scatter(
            i + jitter,
            vals,
            s=28,
            color=COLORS[i - 1],
            alpha=0.9,
            zorder=3,
        )

    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=12)

    all_positive_values = np.concatenate([vals[vals > 0] for vals in data_list])
    set_clean_log_y_axis(ax, all_positive_values)

    ax.set_ylabel(r"$e_{\mathrm{traj}}$", fontsize=16)
    ax.set_title("Centroid trajectory error distribution", fontsize=18)
    ax.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    save_path = output_dir / "centroid_trajectory_error_3cases_violin.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[已保存] {save_path}")

# =========================================================
# 5. 三种速度：ΔR 柱状图
# =========================================================
def plot_three_case_radius_error(cases_data, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    z_ref = cases_data[0]["z_plot"]
    for item in cases_data[1:]:
        if len(item["z_plot"]) != len(z_ref) or not np.allclose(item["z_plot"], z_ref):
            raise ValueError("三个 case 的 z 切片不一致，不能直接画在同一张图上。")

    x_labels = [f"{z:.1f}" for z in z_ref]
    x = np.arange(len(z_ref))

    fig, ax = plt.subplots(figsize=(10, 6.5))

    width = 0.22
    offsets = [-width, 0.0, width]

    all_positive_values = []

    for i, item in enumerate(cases_data):
        values = item["radius_abs_error"]
        all_positive_values.extend(values[values > 0.0])

        bars = ax.bar(
            x + offsets[i],
            values,
            width=width,
            color=COLORS[i],
            label=item["label"],
            edgecolor="none",
            alpha=0.95,
        )

        # add_bar_labels_slanted(ax, bars, values, fmt="{:.2e}", rotation=28, fontsize=9)
        label_y_factors = [1.00, 1.05, 1.00]

        add_bar_labels_slanted(
            ax,
            bars,
            values,
            fmt="{:.2e}",
            y_factor=label_y_factors[i],
            rotation=45,
            fontsize=7,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=12)
    ax.set_xlabel(r"$z/\pi$", fontsize=15)
    ax.set_ylabel(r"$\Delta R(z)$", fontsize=16)

    set_clean_log_y_axis(ax, np.asarray(all_positive_values))

    ax.legend(loc="upper right", fontsize=12, frameon=True, framealpha=0.95)

    fig.tight_layout()
    save_path = output_dir / "molecular_radius_error_3cases_bar.png"
    fig.savefig(save_path, dpi=300)
    plt.close(fig)

    print(f"[已保存] {save_path}")
def plot_three_case_radius_error_box(cases_data, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.2, 6.2))

    data_list = [np.asarray(item["radius_abs_error"], dtype=float) for item in cases_data]
    labels = [item["label"] for item in cases_data]

    bp = ax.boxplot(
        data_list,
        patch_artist=True,
        tick_labels=labels,
        widths=0.55,
        showfliers=True,
        medianprops=dict(color="black", linewidth=1.8),
        boxprops=dict(linewidth=1.5),
        whiskerprops=dict(linewidth=1.3),
        capprops=dict(linewidth=1.3),
    )

    # 上色
    for patch, color in zip(bp["boxes"], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.35)
        patch.set_edgecolor(color)

    # 叠加原始散点，更直观
    for i, vals in enumerate(data_list, start=1):
        jitter = np.linspace(-0.06, 0.06, len(vals))
        ax.scatter(
            i + jitter,
            vals,
            s=28,
            color=COLORS[i - 1],
            alpha=0.9,
            zorder=3,
        )

    all_positive_values = np.concatenate([vals[vals > 0] for vals in data_list])
    set_clean_log_y_axis(ax, all_positive_values)

    ax.set_ylabel(r"$\Delta R$", fontsize=16)
    ax.set_title("Molecular radius error distribution", fontsize=18)
    ax.grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    save_path = output_dir / "molecular_radius_error_3cases_box.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[已保存] {save_path}")


# =========================================================
# 5.5 三种速度：误差热力图
# =========================================================
def plot_error_heatmap(
    data_matrix,
    row_labels,
    col_labels,
    title,
    cbar_label,
    save_path,
    cmap="YlGnBu",
    use_log=True,
    annotate=True,
):
    """
    通用误差热力图。
    data_matrix shape = (3, 6):
        行：nu = 0.8, 1.0, 1.2
        列：z/pi = 0.0, 0.1, ..., 0.5
    """
    data = np.asarray(data_matrix, dtype=float)

    if data.ndim != 2:
        raise ValueError(f"data_matrix 必须是二维矩阵，但现在 shape={data.shape}")

    positive_values = data[data > 0.0]
    if positive_values.size == 0:
        raise ValueError("热力图数据全为 0 或非正数，无法使用 log 颜色标尺。")

    vmin = float(np.min(positive_values))
    vmax = float(np.max(positive_values))

    fig, ax = plt.subplots(figsize=(9.2, 4.8))

    if use_log:
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        norm = None

    im = ax.imshow(
        data,
        cmap=cmap,
        norm=norm,
        aspect="auto",
        origin="upper",
    )

    # 坐标轴标签
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=12)

    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=12)

    ax.set_xlabel(r"$z/\pi$", fontsize=15)
    ax.set_ylabel(r"$\nu$", fontsize=15)
    ax.set_title(title, fontsize=18, pad=12)

    # 白色格线
    ax.set_xticks(np.arange(-0.5, data.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, data.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.3)
    ax.tick_params(which="minor", bottom=False, left=False)

    # 每个格子写数值
    if annotate:
        if use_log:
            threshold = np.sqrt(vmin * vmax)
        else:
            threshold = 0.5 * (vmin + vmax)

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                value = float(data[i, j])
                text_color = "white" if value >= threshold else "black"

                ax.text(
                    j,
                    i,
                    f"{value:.2e}",
                    ha="center",
                    va="center",
                    fontsize=10,
                    color=text_color,
                )

    # colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.035)
    cbar.ax.set_ylabel(cbar_label, rotation=90, fontsize=13)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[已保存] {save_path}")


def plot_three_case_trajectory_error_heatmap(cases_data, output_dir):
    """
    三种速度的运动轨迹误差热力图：
    行：nu
    列：z/pi
    值：e_traj(z)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    z_ref = cases_data[0]["z_plot"]
    for item in cases_data[1:]:
        if len(item["z_plot"]) != len(z_ref) or not np.allclose(item["z_plot"], z_ref):
            raise ValueError("三个 case 的 z 切片不一致，不能直接画热力图。")

    row_labels = [item["label"] for item in cases_data]
    col_labels = [f"{z:.1f}" for z in z_ref]

    traj_matrix = np.vstack([
        np.asarray(item["trajectory_error_z"], dtype=float)
        for item in cases_data
    ])

    save_path = output_dir / "centroid_trajectory_error_3cases_heatmap.png"

    plot_error_heatmap(
        data_matrix=traj_matrix,
        row_labels=row_labels,
        col_labels=col_labels,
        title="Centroid Trajectory Error Heatmap",
        cbar_label=r"$e_{\mathrm{traj}}(z)$",
        save_path=save_path,
        cmap="YlGnBu",
        use_log=True,
        annotate=True,
    )


def plot_three_case_radius_error_heatmap(cases_data, output_dir):
    """
    三种速度的分子半径误差热力图：
    行：nu
    列：z/pi
    值：Delta R(z)=|R_PINN(z)-R_SSFM(z)|
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    z_ref = cases_data[0]["z_plot"]
    for item in cases_data[1:]:
        if len(item["z_plot"]) != len(z_ref) or not np.allclose(item["z_plot"], z_ref):
            raise ValueError("三个 case 的 z 切片不一致，不能直接画热力图。")

    row_labels = [item["label"] for item in cases_data]
    col_labels = [f"{z:.1f}" for z in z_ref]

    radius_matrix = np.vstack([
        np.asarray(item["radius_abs_error"], dtype=float)
        for item in cases_data
    ])

    save_path = output_dir / "molecular_radius_error_3cases_heatmap.png"

    plot_error_heatmap(
        data_matrix=radius_matrix,
        row_labels=row_labels,
        col_labels=col_labels,
        title=r"Molecular Radius Error Heatmap",
        cbar_label=r"$\Delta R(z)$",
        save_path=save_path,
        cmap="YlOrBr",
        use_log=True,
        annotate=True,
    )


# =========================================================
# 6. 三种速度：功率雷达图
# =========================================================
def plot_three_case_power_radar(cases_data, output_dir, use_error=False):
    output_dir.mkdir(parents=True, exist_ok=True)

    z_ref = cases_data[0]["z_plot"]
    for item in cases_data[1:]:
        if len(item["z_plot"]) != len(z_ref) or not np.allclose(item["z_plot"], z_ref):
            raise ValueError("三个 case 的 z 切片不一致，不能直接画在同一张图上。")

    # =====================================================
    # 角度标签
    # =====================================================
    xtick_labels = []
    for z in z_ref:
        if np.isclose(z, 0.0):
            xtick_labels.append(r"$0$")
        else:
            xtick_labels.append(rf"${z:.1f}\pi$")

    n = len(z_ref)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    angles_closed = np.r_[angles, angles[0]]

    fig, ax = plt.subplots(figsize=(8.2, 8.2), subplot_kw=dict(polar=True))

    # 从正上方开始，顺时针
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)

    # =====================================================
    # 数据准备
    # =====================================================
    all_values = []
    for item in cases_data:
        if use_error:
            vals = np.abs(item["powers"] - item["target_power"])
        else:
            vals = item["powers"]
        all_values.extend(vals.tolist())

    all_values = np.asarray(all_values, dtype=float)

    # 标准功率固定为 4
    target_power = 4.0

    # =====================================================
    # 径向范围与等间距刻度
    # 目标：
    # 1. 等间距刻度；
    # 2. target power = 4.0 正好是一条灰色网格线；
    # 3. 外圈不要离数据太远；
    # 4. 刻度类似 3.999997, 3.999998, ..., 4.000000, ...
    # =====================================================
    if use_error:
        values_min = float(np.min(all_values))
        values_max = float(np.max(all_values))
        span = max(values_max - values_min, 1.0e-12)

        rmin = max(0.0, values_min - 0.08 * span)
        rmax = values_max + 0.08 * span

        rticks = np.linspace(rmin, rmax, 5)

    else:
        # 固定等间距步长
        step = 1.0e-6

        # 真实数据相对 P=4 的最大偏差
        dev_min = float(np.min(all_values - target_power))
        dev_max = float(np.max(all_values - target_power))

        # 找到能包住所有数据的整数刻度层级
        low_level = int(np.floor(dev_min / step))
        high_level = int(np.ceil(dev_max / step))

        # 至少保证上下各有两格，不然图太扁
        low_level = min(low_level, -2)
        high_level = max(high_level, 2)

        # 生成等间距刻度，并保证 4.000000 在其中
        levels = np.arange(low_level, high_level + 1)
        rticks = target_power + levels * step

        rmin = float(rticks[0])
        rmax = float(rticks[-1])

    ax.set_ylim(rmin, rmax)
    ax.set_yticks(rticks)

    # =====================================================
    # 画三组功率曲线
    # =====================================================
    for i, item in enumerate(cases_data):
        if use_error:
            values = np.abs(item["powers"] - item["target_power"])
        else:
            values = item["powers"]

        plot_values = np.r_[values, values[0]]

        ax.plot(
            angles_closed,
            plot_values,
            linewidth=2.2,
            color=COLORS[i],
            label=item["label"],
        )

        ax.fill(
            angles_closed,
            plot_values,
            color=COLORS[i],
            alpha=0.10,
        )

    # =====================================================
    # 角度坐标
    # =====================================================
    ax.set_xticks(angles)
    ax.set_xticklabels(xtick_labels, fontsize=12)

    # =====================================================
    # 径向刻度格式
    # =====================================================
    if use_error:
        ax.set_yticklabels([f"{t:.1e}" for t in rticks], fontsize=11)
    else:
        ax.set_yticklabels([f"{t:.6f}" for t in rticks], fontsize=11)

    # 径向数字放在右上角附近
    ax.set_rlabel_position(32)
    ax.tick_params(axis="y", labelsize=11)

    # =====================================================
    # 网格线
    # =====================================================
    ax.grid(True, color="0.70", linewidth=1.0, alpha=0.85)

    # 最外层黑色边框保留
    ax.spines["polar"].set_color("black")
    ax.spines["polar"].set_linewidth(1.2)

    theta_dense = np.linspace(0, 2 * np.pi, 720)

    # 额外描一圈最外层黑色圆，使外圈更清楚
    ax.plot(
        theta_dense,
        np.full_like(theta_dense, rmax),
        color="black",
        linewidth=1.0,
        zorder=5,
    )

    # =====================================================
    # 标准功率 P=4 的参考线
    # 它会和 4.000000 这条灰色网格线重合
    # =====================================================
    if not use_error:
        ax.plot(
            theta_dense,
            np.full_like(theta_dense, target_power),
            linestyle="--",
            color="black",
            linewidth=1.5,
            dashes=(6, 3),
            label=rf"Target Power $P={target_power:.1f}$",
            zorder=6,
        )

    # 不要标题
    ax.set_title("", fontsize=18, pad=28)

    # 图例
    ax.legend(
        loc="upper right",
        bbox_to_anchor=(1.18, 1.12),
        fontsize=12,
        frameon=True,
        framealpha=0.95,
    )

    fig.tight_layout()

    if use_error:
        save_path = output_dir / "power_error_3cases_radar.png"
    else:
        save_path = output_dir / "power_3cases_radar.png"

    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[已保存] {save_path}")
# =========================================================
# 6. 三种速度：功率柱状图
# =========================================================
def plot_three_case_power_bar(cases_data, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)

    z_ref = cases_data[0]["z_plot"]
    for item in cases_data[1:]:
        if len(item["z_plot"]) != len(z_ref) or not np.allclose(item["z_plot"], z_ref):
            raise ValueError("三个 case 的 z 切片不一致，不能直接画在同一张图上。")

    x_labels = [f"{z:.1f}" for z in z_ref]
    x = np.arange(len(z_ref))

    fig, ax = plt.subplots(figsize=(10, 6.5))

    width = 0.22
    offsets = [-width, 0.0, width]

    # 标准功率固定为 4
    target_power = 4.0

    all_power_values = []

    for i, item in enumerate(cases_data):
        values = np.asarray(item["powers"], dtype=float)
        all_power_values.extend(values.tolist())

        bars = ax.bar(
            x + offsets[i],
            values,
            width=width,
            color=COLORS[i],
            label=item["label"],
            edgecolor="none",
            alpha=0.95,
        )

        # 柱子上方标数字，斜着写
        label_offsets = [
            (-8, 5),    # nu=0.8
            (0, 12),    # nu=1.0
            (8, 5),     # nu=1.2
        ]

        # for bar, value in zip(bars, values):
        #     ax.annotate(
        #         f"{value:.6f}",
        #         xy=(bar.get_x() + bar.get_width() / 2.0, value),
        #         xytext=label_offsets[i],
        #         textcoords="offset points",
        #         ha="center",
        #         va="bottom",
        #         rotation=35,
        #         fontsize=7,
        #         color="black",
        #         clip_on=False,
        #         zorder=10,
        #     )

    # 标准功率参考线 P=4
    ax.axhline(
        target_power,
        color="black",
        linestyle="--",
        linewidth=1.6,
        label=r"Target Power $P=4$",
        zorder=3,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=12)

    ax.set_xlabel(r"$z/\pi$", fontsize=15)
    ax.set_ylabel(r"$P(z)$", fontsize=16)

    # 不用标题
    ax.set_title("")

    # 纵坐标只截取 4 附近一小段
    all_power_values = np.asarray(all_power_values, dtype=float)
    max_dev = float(np.max(np.abs(all_power_values - target_power)))

    # 防止所有值太接近 4 导致范围太窄
    max_dev = max(max_dev, 1.0e-6)

    pad = 0.35 * max_dev

    ax.set_ylim(
        target_power - max_dev - pad,
        target_power + max_dev + pad,
    )

    # 不显示 offset，不出现 1e-6 + 4 这种形式
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: f"{y:.6f}"))

    ax.grid(True, axis="y", alpha=0.25)

    ax.legend(
        loc="upper right",
        fontsize=12,
        frameon=True,
        framealpha=0.95,
    )

    fig.tight_layout()

    save_path = output_dir / "power_3cases_bar.png"
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"[已保存] {save_path}")

# =========================================================
# 7. main
# =========================================================
def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cases_data = []
    for case in CASES:
        print(f"[读取] {case['tag']}")
        item = load_case_data(case)
        cases_data.append(item)

        print(
            f"  -> nu={item['nu']:.1f}, d={item['d']:.2f}, "
            f"global E_traj={item['trajectory_error_global']:.2e}"
        )

    # # 1) 轨迹误差柱状图
    # plot_three_case_trajectory_error(cases_data, OUTPUT_DIR)
    #
    # # 2) 分子半径误差 ΔR 柱状图
    # plot_three_case_radius_error(cases_data, OUTPUT_DIR)
    #
    # # 3) 功率雷达图
    # plot_three_case_power_radar(cases_data, OUTPUT_DIR, use_error=USE_POWER_ERROR_FOR_RADAR)
    # plot_three_case_power_bar(cases_data, OUTPUT_DIR)

    # 1) 运动轨迹误差热力图
    plot_three_case_trajectory_error_heatmap(cases_data, OUTPUT_DIR)

    # 2) 分子半径误差 ΔR 热力图
    plot_three_case_radius_error_heatmap(cases_data, OUTPUT_DIR)

    # 3) 功率雷达图：保持原来的函数不动
    plot_three_case_power_radar(cases_data, OUTPUT_DIR, use_error=False)
    print("\n全部完成。输出文件夹：")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()