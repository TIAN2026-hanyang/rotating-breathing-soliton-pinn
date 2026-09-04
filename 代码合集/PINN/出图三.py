from pathlib import Path
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterMathtext, NullFormatter, NullLocator
from matplotlib.patches import Wedge, FancyArrowPatch
from matplotlib.lines import Line2D

# =========================================================
# 只改这里：三组结果文件夹路径
# =========================================================
CASES = [
    {
        "tag": "v0.8_d3.60",
        "nu": 0.8,
        "d": 3.60,
        "result_dir": Path(r"C:\Users\50971\Desktop\代码\hard_ic_final_v0.8_d3.60"),
    },
    {
        "tag": "v1.0_d3.23",
        "nu": 1.0,
        "d": 3.23,
        "result_dir": Path(r"C:\Users\50971\Desktop\代码\hard_ic_final_v1.0_d3.23"),
    },
    {
        "tag": "v1.2_d2.94",
        "nu": 1.2,
        "d": 2.94,
        "result_dir": Path(r"C:\Users\50971\Desktop\代码\hard_ic_final_v1.2_d2.94"),
    },
]


# =========================================================
# 工具函数
# =========================================================
def analytic_orbit_xy(beam_index, alpha, nu_value, d_value):
    """
    解析质心轨道：
        x_n(z) = d [ cos(phi_n) cos(alpha) - nu sin(phi_n) sin(alpha) ]
        y_n(z) = d [ sin(phi_n) cos(alpha) + nu cos(phi_n) sin(alpha) ]
    其中 phi_n = n*pi/2
    """
    phi = beam_index * np.pi / 2.0

    x = d_value * (
        np.cos(phi) * np.cos(alpha)
        - nu_value * np.sin(phi) * np.sin(alpha)
    )
    y = d_value * (
        np.sin(phi) * np.cos(alpha)
        + nu_value * np.cos(phi) * np.sin(alpha)
    )
    return x, y


def add_curved_arrow_on_orbit(ax, beam_index, alpha_start_deg, alpha_end_deg, nu_value, d_value, color="0.25"):
    """
    在解析轨道上画弯曲箭头。
    alpha 递增方向就是传播方向。
    """
    alpha_deg = np.linspace(alpha_start_deg, alpha_end_deg, 80)
    alpha = np.deg2rad(alpha_deg)

    x, y = analytic_orbit_xy(beam_index, alpha, nu_value, d_value)

    ax.plot(
        x, y,
        color=color,
        linewidth=3.0,
        solid_capstyle="round",
        zorder=2,
    )

    ax.annotate(
        "",
        xy=(x[-1], y[-1]),
        xytext=(x[-8], y[-8]),
        arrowprops=dict(
            arrowstyle="-|>",
            color=color,
            linewidth=3.0,
            mutation_scale=18,
            shrinkA=0,
            shrinkB=0,
        ),
        zorder=3,
    )

def set_dynamic_radius_ylim(ax, radius_pinn, radius_reference, d_value, min_window=0.010):
    """
    动态设置 R(z) 的纵坐标范围。

    如果半径几乎围绕 d 小幅波动，例如 v=1.0,d=3.23，
    会自动给出类似 3.2250 到 3.2350 的范围。

    如果半径变化较大，例如 v=0.8 或 v=1.2 的椭圆轨道，
    会自动覆盖全部数据范围，不会截断。
    """
    values = [np.asarray(radius_pinn, dtype=float), np.asarray([d_value], dtype=float)]

    if radius_reference is not None and radius_reference.size > 0:
        values.append(np.asarray(radius_reference, dtype=float))

    all_values = np.concatenate(values)
    y_min = float(np.min(all_values))
    y_max = float(np.max(all_values))

    data_span = y_max - y_min

    if data_span < min_window:
        center = float(d_value)
        half_window = min_window / 2.0
        ax.set_ylim(center - half_window, center + half_window)
    else:
        pad = 0.08 * data_span
        ax.set_ylim(y_min - pad, y_max + pad)


def set_clean_log_y_axis(ax, values, lower_factor=0.5, upper_factor=2.0):
    """
    自动设置干净的 log y 轴。
    适合 1e-6、1e-5、1e-3、1e-2 等任意数量级误差。
    """
    positive_values = np.asarray(values, dtype=float)
    positive_values = positive_values[positive_values > 0.0]

    if positive_values.size == 0:
        return

    y_min = float(np.min(positive_values))
    y_max = float(np.max(positive_values))

    ax.set_yscale("log")
    ax.set_ylim(y_min * lower_factor, y_max * upper_factor)

    ax.yaxis.set_major_locator(LogLocator(base=10.0))
    ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))

    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_formatter(NullFormatter())

    ax.grid(True, axis="y", which="major", alpha=0.25)

def add_split_marker(ax, x, y, color_a, color_b, radius=0.11, mode="lr", zorder=6):
    """
    双色点
    mode="lr": 左右分半，color_a=左半，color_b=右半
    mode="ud": 上下分半，color_a=上半，color_b=下半
    """
    if mode == "lr":
        patch_a = Wedge(
            center=(x, y),
            r=radius,
            theta1=90,
            theta2=270,
            facecolor=color_a,
            edgecolor="white",
            linewidth=0.6,
            zorder=zorder,
        )
        patch_b = Wedge(
            center=(x, y),
            r=radius,
            theta1=-90,
            theta2=90,
            facecolor=color_b,
            edgecolor="white",
            linewidth=0.6,
            zorder=zorder,
        )

    elif mode == "ud":
        patch_a = Wedge(
            center=(x, y),
            r=radius,
            theta1=0,
            theta2=180,
            facecolor=color_a,
            edgecolor="white",
            linewidth=0.6,
            zorder=zorder,
        )
        patch_b = Wedge(
            center=(x, y),
            r=radius,
            theta1=180,
            theta2=360,
            facecolor=color_b,
            edgecolor="white",
            linewidth=0.6,
            zorder=zorder,
        )
    else:
        raise ValueError("mode must be 'lr' or 'ud'")

    ax.add_patch(patch_a)
    ax.add_patch(patch_b)


def add_black_cardinal_arrows_n4(ax, nu_value, d_value):
    """
    N=4 时，在正右、正上、正左、正下四个位置加逆时针黑色箭头。

    关键修正：
    右、左箭头走 beam 0/2 对应轨道；
    上、下箭头走 beam 1/3 对应轨道。
    这样 v=0.8 和 v=1.2 时，上下箭头也会从分子点附近出发。
    """

    # 每个元素是：
    # (beam_index, alpha_start)
    #
    # 右侧点：beam 0 轨道，alpha = 0
    # 上侧点：beam 1 轨道，alpha = 0
    # 左侧点：beam 0 轨道，alpha = pi
    # 下侧点：beam 1 轨道，alpha = pi
    arrow_specs = [
        (0, 0.0),          # right
        (1, 0.0),          # top
        (0, np.pi),        # left
        (1, np.pi),        # bottom
    ]

    eps = 0.035      # 从分子点边缘后一点开始；越小越靠近点
    seg = 0.20       # 箭头沿轨道长度；越大越长
    n_curve = 40
    lw = 2.2

    for beam_index, alpha0 in arrow_specs:
        alpha_curve = np.linspace(alpha0 + eps, alpha0 + seg, n_curve)

        x_curve, y_curve = analytic_orbit_xy(
            beam_index,
            alpha_curve,
            nu_value,
            d_value,
        )

        # 先画一小段黑色弧线，保证贴合对应灰色轨道
        ax.plot(
            x_curve[:-3],
            y_curve[:-3],
            color="black",
            linewidth=lw,
            solid_capstyle="round",
            zorder=3,
        )

        # 末端加箭头头部
        arrow = FancyArrowPatch(
            (float(x_curve[-4]), float(y_curve[-4])),
            (float(x_curve[-1]), float(y_curve[-1])),
            arrowstyle="-|>",
            mutation_scale=22,
            linewidth=lw,
            color="black",
            shrinkA=0,
            shrinkB=0,
            zorder=4,
        )
        ax.add_patch(arrow)


def add_radius_d_line(ax, d_value):
    """
    从圆心到右侧边点画半径 d，并在圆心加黑点。
    """
    ax.plot(
        [0.0, d_value],
        [0.0, 0.0],
        color="black",
        linewidth=2.0,
        linestyle="-",
        zorder=2,
    )

    # 圆心黑点
    ax.scatter(
        [0.0], [0.0],
        s=28,
        color="black",
        zorder=5,
    )

    ax.text(
        0.50 * d_value,
        0.03 * d_value,
        r"$d$",
        fontsize=15,
        ha="center",
        va="bottom",
        color="black",
        zorder=8,
    )

# =========================================================
# 单个 case 画图
# =========================================================
def plot_one_case(case):
    RESULT_DIR = case["result_dir"]

    npz_path = RESULT_DIR / "prediction_results.npz"
    summary_path = RESULT_DIR / "summary.json"

    if not npz_path.exists():
        print(f"[跳过] 找不到文件: {npz_path}")
        return

    if not summary_path.exists():
        print(f"[跳过] 找不到文件: {summary_path}")
        return

    data = np.load(npz_path, allow_pickle=False)

    z_values = data["z"]
    powers = data["powers"]

    centroids_pinn = data["centroids_pinn"]
    centroids_reference = data["centroids_reference"]

    radius_pinn = data["radius_pinn"]
    radius_reference = data["radius_reference"]

    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)

    tag = summary.get("tag", case["tag"])
    target_power = float(summary["config"]["target_power"])

    # 用 summary 里的值；如果没有就退回 case 里的值
    d_value = float(summary.get("d", case["d"]))
    nu_value = float(summary.get("nu", case["nu"]))
    title_suffix = f"v={nu_value:.1f}, d={d_value:.2f}"

    n_z, n_components, _ = centroids_pinn.shape
    z_plot = z_values / np.pi
    # =====================================================
    # 运动轨迹误差 e_traj(z)
    # =====================================================
    if centroids_reference.size > 0:
        centroid_diff = centroids_pinn - centroids_reference

        # 每个 z 切片的中心轨迹误差
        # shape: (n_z,)
        trajectory_error_z = (
                np.sqrt(np.mean(np.sum(centroid_diff ** 2, axis=2), axis=1))
                / d_value
        )

        # 整体公式 (36) 指标
        trajectory_error_global = float(
            np.sqrt(np.mean(np.sum(centroid_diff ** 2, axis=2)))
            / d_value
        )
    else:
        trajectory_error_z = np.array([])
        trajectory_error_global = None
    # 颜色
    colors = [f"C{i}" for i in range(n_components)]

    # =====================================================
    # 1. 解析轨道背景 + PINN 质心点图（带数字）
    # =====================================================
    fig, ax = plt.subplots(figsize=(7, 7))

    theta = np.linspace(0.0, 2.0 * np.pi, 1000)

    # nu=1 时四条轨道重合成一个圆
    if abs(nu_value - 1.0) < 1.0e-12:
        orbit_beams = [0]
    else:
        # nu!=1 时，两条不同椭圆轨道：beam 0/2 一条，beam 1/3 一条
        orbit_beams = [0, 1]

    # 画灰色解析轨道
    for j, beam_index in enumerate(orbit_beams):
        orbit_x, orbit_y = analytic_orbit_xy(
            beam_index,
            theta,
            nu_value,
            d_value,
        )

        ax.plot(
            orbit_x,
            orbit_y,
            color="0.65",
            linewidth=2.2,
            linestyle="-",
            zorder=1,
        )

    # 半径 d：从圆心到右侧边点
    add_radius_d_line(ax, d_value)

    # N=4 时，在正右、正上、正左、正下附近加纯黑方向箭头
    if n_components == 4:
        add_black_cardinal_arrows_n4(ax, nu_value, d_value)

    # =====================================================
    # 画 PINN 质心点
    # 普通点正常 scatter，四个 0/5 重合点单独画双色半圆
    # =====================================================
    split_marker_radius = 0.035 * d_value

    for n in range(n_components):
        xp = centroids_pinn[:, n, 0]
        yp = centroids_pinn[:, n, 1]

        # 中间切片正常画点，首尾重合点后面用双色半圆点画
        if n_components == 4 and n_z >= 2:
            mid_indices = np.arange(1, n_z - 1)
        else:
            mid_indices = np.arange(n_z)

        ax.scatter(
            xp[mid_indices],
            yp[mid_indices],
            s=80,
            color=colors[n],
            label=f"PINN beam {n}",
            zorder=4,
        )

    # =====================================================
    # 四个 0/5 重合点画成双色半圆
    # =====================================================
    if n_components == 4 and n_z >= 2:
        # 正右：上半蓝(beam0)，下半绿(beam2)
        p_right = 0.5 * (centroids_pinn[0, 0, :] + centroids_pinn[-1, 2, :])
        add_split_marker(
            ax,
            p_right[0],
            p_right[1],
            color_a=colors[0],  # 上半蓝
            color_b=colors[2],  # 下半绿
            radius=split_marker_radius,
            mode="ud",
        )

        # 正上：左半橙(beam1)，右半红(beam3)
        p_top = 0.5 * (centroids_pinn[0, 1, :] + centroids_pinn[-1, 3, :])
        add_split_marker(
            ax,
            p_top[0],
            p_top[1],
            color_a=colors[1],  # 左半橙
            color_b=colors[3],  # 右半红
            radius=split_marker_radius,
            mode="lr",
        )

        # 正左：上半蓝(beam0)，下半绿(beam2)
        p_left = 0.5 * (centroids_pinn[0, 2, :] + centroids_pinn[-1, 0, :])
        add_split_marker(
            ax,
            p_left[0],
            p_left[1],
            color_a=colors[0],  # 上半蓝
            color_b=colors[2],  # 下半绿
            radius=split_marker_radius,
            mode="ud",
        )

        # 正下：左半橙(beam1)，右半红(beam3)
        p_bottom = 0.5 * (centroids_pinn[0, 3, :] + centroids_pinn[-1, 1, :])
        add_split_marker(
            ax,
            p_bottom[0],
            p_bottom[1],
            color_a=colors[1],  # 左半橙
            color_b=colors[3],  # 右半红
            radius=split_marker_radius,
            mode="lr",
        )

    # =====================================================
    # 四个重合点的 0/5 数字手动标注
    # 只改数字，不改球的颜色
    # =====================================================
    label_fs = 10

    # 正右：上半蓝 beam0 是 0，下半绿 beam2 是 5
    ax.text(
        p_right[0] + 0.12,
        p_right[1] + 0.08,
        "0",
        color=colors[0],
        fontsize=label_fs,
        ha="left",
        va="bottom",
        zorder=9,
    )
    ax.text(
        p_right[0] + 0.12,
        p_right[1] - 0.08,
        "5",
        color=colors[2],
        fontsize=label_fs,
        ha="left",
        va="top",
        zorder=9,
    )

    # 正左：上半蓝 beam0 是 5，下半绿 beam2 是 0
    ax.text(
        p_left[0] - 0.12,
        p_left[1] + 0.08,
        "5",
        color=colors[0],
        fontsize=label_fs,
        ha="right",
        va="bottom",
        zorder=9,
    )
    ax.text(
        p_left[0] - 0.12,
        p_left[1] - 0.08,
        "0",
        color=colors[2],
        fontsize=label_fs,
        ha="right",
        va="top",
        zorder=9,
    )

    # 正上：左半橙 beam1 是 0，右半红 beam3 是 5
    ax.text(
        p_top[0] - 0.08,
        p_top[1] + 0.12,
        "0",
        color=colors[1],
        fontsize=label_fs,
        ha="right",
        va="bottom",
        zorder=9,
    )
    ax.text(
        p_top[0] + 0.08,
        p_top[1] + 0.12,
        "5",
        color=colors[3],
        fontsize=label_fs,
        ha="left",
        va="bottom",
        zorder=9,
    )

    # 正下：左半橙 beam1 是 5，右半红 beam3 是 0
    ax.text(
        p_bottom[0] - 0.08,
        p_bottom[1] - 0.12,
        "5",
        color=colors[1],
        fontsize=label_fs,
        ha="right",
        va="top",
        zorder=9,
    )
    ax.text(
        p_bottom[0] + 0.08,
        p_bottom[1] - 0.12,
        "0",
        color=colors[3],
        fontsize=label_fs,
        ha="left",
        va="top",
        zorder=9,
    )

    # =====================================================
    # 给每个切片点标 0,1,2,3,4,5
    # =====================================================
    for n in range(n_components):
        xp = centroids_pinn[:, n, 0]
        yp = centroids_pinn[:, n, 1]

        for k in range(n_z):

            # N=4 时，四个首尾重合点的 0/5 数字后面手动画
            # 这里先跳过，避免重复和错位
            if n_components == 4 and k in (0, n_z - 1):
                continue

            xk = xp[k]
            yk = yp[k]
            rk = np.hypot(xk, yk)

            radial_offset = 0.16
            tangent_offset = 0.14

            if rk > 1.0e-12:
                rx = xk / rk
                ry = yk / rk

                tx = -ry
                ty = rx

                if n in (0, 1):
                    tangent_sign = 1.0
                else:
                    tangent_sign = -1.0

                xt = xk + radial_offset * rx + tangent_sign * tangent_offset * tx
                yt = yk + radial_offset * ry + tangent_sign * tangent_offset * ty
            else:
                xt = xk + 0.05
                yt = yk + 0.05

            ax.text(
                xt,
                yt,
                f"{k}",
                fontsize=10,
                ha="center",
                va="center",
                color=colors[n],
                zorder=8,
                bbox=dict(
                    boxstyle="round,pad=0.15",
                    facecolor="white",
                    edgecolor="none",
                    alpha=0.80,
                ),
            )

    # 坐标轴 LaTeX 形式
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")

    # ax.set_title(f"PINN centroid points on analytic orbit: {title_suffix}")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)

    # =====================================================
    # 图例：PINN beam 0,1,2,3 横着一排，并整体往上放
    # =====================================================
    orbit_handle = Line2D(
        [],
        [],
        color="0.65",
        linewidth=2.2,
        linestyle="-",
        label="analytic center orbit",
    )

    radius_handle = Line2D(
        [],
        [],
        color="black",
        linewidth=2.0,
        linestyle="-",
        label=r"radius $d$",
    )

    beam_handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=8,
            markerfacecolor=colors[i],
            markeredgecolor=colors[i],
            label=f"PINN beam {i}",
        )
        for i in range(n_components)
    ]

    # =====================================================
    # 图例：三行两列
    # 第1行：analytic center orbit | radius d
    # 第2行：PINN beam 0          | PINN beam 1
    # 第3行：PINN beam 2          | PINN beam 3
    #
    # 注意：Matplotlib 的 ncol=2 会按“列”填充，
    # 所以这里 handles 顺序必须写成：
    # 第1列：orbit, beam0, beam2
    # 第2列：radius, beam1, beam3
    # =====================================================

    handles = [
        orbit_handle,
        beam_handles[0],
        beam_handles[2],
        radius_handle,
        beam_handles[1],
        beam_handles[3],
    ]

    labels = [
        "analytic center orbit",
        "PINN beam 0",
        "PINN beam 2",
        r"radius $d$",
        "PINN beam 1",
        "PINN beam 3",
    ]

    ax.legend(
        handles,
        labels,
        fontsize=9,
        ncol=2,
        loc="center",
        bbox_to_anchor=(0.50, 0.65),
        frameon=True,
        framealpha=0.95,
        borderpad=0.35,
        labelspacing=0.38,
        columnspacing=1.20,
        handletextpad=0.65,
        handlelength=1.9,
    )
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "centroid_points_on_analytic_orbit_numbered.png", dpi=300)
    plt.close(fig)

    # =====================================================
    # 1.5 运动轨迹误差柱状图 e_traj(z)
    # =====================================================
    if trajectory_error_z.size > 0:
        fig, ax = plt.subplots(figsize=(7, 5))

        x_labels = [f"{v:.1f}" for v in z_plot]
        x_pos = np.arange(len(z_plot))

        ax.bar(
            x_pos,
            trajectory_error_z,
            width=0.58,
            label=r"$e_{\mathrm{traj}}(z)$",
        )

        ax.axhline(
            trajectory_error_global,
            linestyle="--",
            linewidth=1.8,
            label=rf"global $E_{{traj}}$ = {trajectory_error_global:.2e}",
        )

        # 在每个柱子上标数值
        for i, value in enumerate(trajectory_error_z):
            ax.text(
                x_pos[i],
                value * 1.08,
                f"{value:.2e}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels)

        ax.set_xlabel(r"$z/\pi$")
        # ax.set_ylabel(r"normalized centroid trajectory error")
        # ax.set_title(f"Centroid trajectory error: {title_suffix}")

        # y 轴改成 10^{-8}, 10^{-7}, 10^{-6} 这种形式
        ax.set_yscale("log")

        positive_values = trajectory_error_z[trajectory_error_z > 0.0]
        y_min = float(np.min(positive_values))
        y_max = float(np.max(positive_values))

        ax.set_ylim(y_min * 0.5, y_max * 2.0)

        ax.yaxis.set_major_locator(LogLocator(base=10.0))
        ax.yaxis.set_major_formatter(LogFormatterMathtext(base=10.0))

        # 不显示 log 坐标的次刻度，避免很多密集小线
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_minor_formatter(NullFormatter())

        # 只画主网格线，也就是 10^{-8}, 10^{-7}, 10^{-6} 对应的线
        ax.grid(True, axis="y", which="major", alpha=0.25)

        ax.legend(loc="best")

        fig.tight_layout()
        fig.savefig(RESULT_DIR / "centroid_trajectory_error_bar.png", dpi=300)
        plt.close(fig)

    # =====================================================
    # 2. 分子半径柱状对比图 R(z)
    # =====================================================
    fig, ax = plt.subplots(figsize=(7, 5))

    x_labels = [f"{v:.1f}" for v in z_plot]
    x_pos = np.arange(len(z_plot))
    bar_width = 0.36

    bars_pinn = ax.bar(
        x_pos - bar_width / 2.0,
        radius_pinn,
        width=bar_width,
        label="PINN",
    )

    bars_ssfm = ax.bar(
        x_pos + bar_width / 2.0,
        radius_reference,
        width=bar_width,
        label="SSFM",
    )

    ax.axhline(
        d_value,
        linestyle=":",
        linewidth=2.0,
        label=rf"$R=d={d_value:.2f}$",
    )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel(r"$z/\pi$")
    ax.set_ylabel(r"$R(z)$")
    ax.set_title(f"Molecular radius: {title_suffix}")

    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper right")
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)

    set_dynamic_radius_ylim(ax, radius_pinn, radius_reference, d_value)

    def add_bar_labels_top_right(ax, bars, fmt="{:.4f}", dx=-4, dy=4, rotation=35):
        for bar in bars:
            h = bar.get_height()
            x = bar.get_x() + bar.get_width() / 2.0

            ax.annotate(
                fmt.format(h),
                xy=(x, h),
                xytext=(dx, dy),
                textcoords="offset points",
                ha="left",
                va="bottom",
                rotation=rotation,
                fontsize=8,
                color="black",
            )

    add_bar_labels_top_right(ax, bars_pinn, fmt="{:.6f}")
    add_bar_labels_top_right(ax, bars_ssfm, fmt="{:.6f}")

    fig.tight_layout()
    fig.savefig(RESULT_DIR / "molecular_radius_bar.png", dpi=300)
    plt.close(fig)
    # =====================================================
    # 3. 半径绝对误差柱状图
    # =====================================================
    if radius_reference.size > 0:
        radius_abs_error = np.abs(radius_pinn - radius_reference)

        fig, ax = plt.subplots(figsize=(7, 5))

        x_labels = [f"{v:.1f}" for v in z_plot]
        x_pos = np.arange(len(z_plot))

        bars_radius_err = ax.bar(
            x_pos,
            radius_abs_error,
            width=0.58,
            label="absolute radius error",
        )

        # 在误差柱子上方标数字
        for bar, value in zip(bars_radius_err, radius_abs_error):
            if value <= 0.0:
                continue

            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                value * 1.08,
                f"{value:.2e}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels)

        ax.set_xlabel(r"$z/\pi$")
        ax.set_ylabel(r"$|R_{\mathrm{PINN}}(z)-R_{\mathrm{SSFM}}(z)|$")
        ax.set_title(f"Molecular radius error: {title_suffix}")

        # y 轴用数量级显示，例如 10^{-7}, 10^{-6}, 10^{-5}
        set_clean_log_y_axis(ax, radius_abs_error)

        ax.legend(loc="upper right")

        fig.tight_layout()
        fig.savefig(RESULT_DIR / "molecular_radius_error_clear.png", dpi=300)
        plt.close(fig)

    # =====================================================
    # 4. 功率图 P(z)
    # =====================================================
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(z_plot, powers, marker="o", label="PINN power")
    ax.axhline(target_power, linestyle="--", label=f"target power = {target_power}")

    ax.set_xlabel(r"$z/\pi$")
    ax.set_ylabel(r"$P(z)$")
    # ax.set_title(f"{tag}: power conservation")
    ax.set_title(f"Power conservation: {title_suffix}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.ticklabel_format(axis="y", style="plain", useOffset=False)

    # 不夸大小偏差
    ax.set_ylim(target_power - 1.0e-4, target_power + 1.0e-4)

    max_power_error = float(np.max(np.abs(powers - target_power)))
    ax.text(
        0.02,
        0.92,
        rf"max $|P-{target_power:g}|$ = {max_power_error:.2e}",
        transform=ax.transAxes,
        fontsize=11,
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
    )

    fig.tight_layout()
    fig.savefig(RESULT_DIR / "power_curve_clear.png", dpi=300)
    plt.close(fig)

    print(f"[完成] {tag}")
    print("  ", RESULT_DIR / "centroid_points_on_analytic_orbit_numbered.png")
    print("  ", RESULT_DIR / "centroid_trajectory_error_clear.png")
    print("  ", RESULT_DIR / "molecular_radius_clear.png")
    print("  ", RESULT_DIR / "molecular_radius_error_clear.png")
    print("  ", RESULT_DIR / "power_curve_clear.png")


# =========================================================
# 主程序：依次跑三组
# =========================================================
def main():
    for case in CASES:
        plot_one_case(case)


if __name__ == "__main__":
    main()