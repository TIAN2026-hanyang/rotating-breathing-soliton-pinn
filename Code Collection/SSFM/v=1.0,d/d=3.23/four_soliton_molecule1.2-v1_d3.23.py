
"""
四孤子分子解析场与绘图
模型（精确解析式对应 w=0）：
    i psi_z + (psi_xx + psi_yy) + g2*(x^2+y^2)*psi = 0,  g2 < 0

新增功能：
1. 分别保存各传播位置的光强图；
2. 保存质心轨迹图；
3. 将全部光强图和质心轨迹放入一张总体图 four_soliton_overview.png。
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import matplotlib.pyplot as plt
import numpy as np


# 所有结果都保存到本 Python 文件所在的文件夹
OUTPUT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class FourSolitonParams:
    g2: float = -1.0
    d: float = 3.23
    nu: float = 1.0
    a0: float = 1.0
    A0: float = 4.0
    Pr: float = 1.0
    target_power: float = 4.0
    relative_phases: Tuple[float, float, float, float] = (
        0.0,
        np.pi,
        0.0,
        np.pi,
    )

    def __post_init__(self) -> None:
        if self.g2 >= 0:
            raise ValueError("自聚焦强非局域抛物势要求 g2 < 0。")
        if self.d <= 0 or self.a0 <= 0 or self.Pr <= 0:
            raise ValueError("d、a0、Pr 必须为正数。")
        if len(self.relative_phases) != 4:
            raise ValueError("relative_phases 必须包含四个相位。")

    @property
    def beta0(self) -> float:
        return 2.0 * np.sqrt(-self.g2)


def resolve_output_path(output_path: Path | str) -> Path:
    """把相对路径统一解析到 Python 文件所在目录。"""
    path = Path(output_path)
    if not path.is_absolute():
        path = OUTPUT_DIR / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def beam_shape_parameters(z: float, p: FourSolitonParams):
    """返回 alpha、D、a(z)、b(z)、theta(z)。"""
    beta0 = p.beta0
    alpha = beta0 * z
    D = np.cos(alpha) ** 2 + p.Pr * np.sin(alpha) ** 2

    a = p.a0 * np.sqrt(D)

    # 当前 PDE 的横向 Laplacian 系数为 1
    b = beta0 * (p.Pr - 1.0) * np.sin(2.0 * alpha) / (8.0 * D)

    # atan2 可避免普通 arctan 的分支跳变
    theta = -np.arctan2(
        np.sqrt(p.Pr) * np.sin(alpha),
        np.cos(alpha),
    )
    return alpha, D, a, b, theta


def centers_and_phase_gradients(z: float, p: FourSolitonParams):
    """
    返回：
        centers.shape = (4, 2)，每行为 [x_n(z), y_n(z)]
        momenta.shape = (4, 2)，每行为 [p_xn(z), p_yn(z)]

    其中 p_n = R_n'(z)/2 为线性相位梯度。
    """
    beta0 = p.beta0
    alpha = beta0 * z

    varphi = np.arange(4, dtype=float) * np.pi / 2.0
    cphi = np.cos(varphi)
    sphi = np.sin(varphi)
    ca = np.cos(alpha)
    sa = np.sin(alpha)

    x_n = p.d * (cphi * ca - p.nu * sphi * sa)
    y_n = p.d * (sphi * ca + p.nu * cphi * sa)

    px_n = -0.5 * beta0 * p.d * (
        cphi * sa + p.nu * sphi * ca
    )
    py_n = 0.5 * beta0 * p.d * (
        -sphi * sa + p.nu * cphi * ca
    )

    centers = np.column_stack((x_n, y_n))
    momenta = np.column_stack((px_n, py_n))
    return centers, momenta


def common_translation_phase(z: float, p: FourSolitonParams) -> float:
    """
    对称切向入射时四个组分具有相同平移附加相位：
        phi(z) = beta0*d^2*(1-nu^2)/8 * sin(2*beta0*z)
    """
    alpha = p.beta0 * z
    return (
        p.beta0
        * p.d**2
        * (1.0 - p.nu**2)
        * np.sin(2.0 * alpha)
        / 8.0
    )


def raw_four_soliton_field(
    x: np.ndarray,
    y: np.ndarray,
    z: float,
    p: FourSolitonParams,
) -> np.ndarray:
    """未乘总功率归一化系数 G0 的四孤子复场。"""
    _, _, a, b, theta = beam_shape_parameters(z, p)
    centers, momenta = centers_and_phase_gradients(z, p)
    phi = common_translation_phase(z, p)

    amplitude = p.A0 * p.a0 / a
    psi = np.zeros_like(x, dtype=np.complex128)

    for n in range(4):
        xn, yn = centers[n]
        pxn, pyn = momenta[n]
        qn = (x - xn) ** 2 + (y - yn) ** 2

        phase = (
            b * qn
            + pxn * x
            + pyn * y
            + phi
            + theta
            + p.relative_phases[n]
        )

        psi += amplitude * np.exp(
            -qn / (2.0 * a**2) + 1j * phase
        )

    return psi


def integration_power(
    psi: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
) -> float:
    """二维梯形积分计算 P = ∫∫ |psi|^2 dxdy。"""
    intensity = np.abs(psi) ** 2
    int_x = np.trapezoid(intensity, x=x_axis, axis=1)
    return float(np.trapezoid(int_x, x=y_axis, axis=0))


def normalization_factor(
    x: np.ndarray,
    y: np.ndarray,
    x_axis: np.ndarray,
    y_axis: np.ndarray,
    p: FourSolitonParams,
) -> float:
    """在 z=0 计算固定的 G0，使总输入功率等于 target_power。"""
    psi0 = raw_four_soliton_field(x, y, 0.0, p)
    power0 = integration_power(psi0, x_axis, y_axis)
    if power0 <= 0:
        raise RuntimeError("计算得到的初始总功率非正，无法归一化。")
    return np.sqrt(p.target_power / power0)


def four_soliton_field(
    x: np.ndarray,
    y: np.ndarray,
    z: float,
    p: FourSolitonParams,
    G0: float = 1.0,
) -> np.ndarray:
    return G0 * raw_four_soliton_field(x, y, z, p)


def plot_intensity(
    z: float,
    p: FourSolitonParams,
    xlim=(-8.0, 8.0),
    ylim=(-8.0, 8.0),
    points=501,
    output_path=OUTPUT_DIR / "four_soliton_intensityv1.png",
):
    """绘制并保存某一传播位置 z 的二维光强图。"""
    x_axis = np.linspace(xlim[0], xlim[1], points)
    y_axis = np.linspace(ylim[0], ylim[1], points)
    x, y = np.meshgrid(x_axis, y_axis)

    G0 = normalization_factor(x, y, x_axis, y_axis, p)
    psi = four_soliton_field(x, y, z, p, G0=G0)
    intensity = np.abs(psi) ** 2

    fig, ax = plt.subplots(figsize=(6.6, 5.5))
    image = ax.imshow(
        intensity,
        extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
        origin="lower",
        aspect="equal",
    )

    centers, _ = centers_and_phase_gradients(z, p)
    ax.scatter(centers[:, 0], centers[:, 1], marker="x", s=50)

    alpha = p.beta0 * z
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(
        rf"$|\Psi_4|^2$, $z={alpha / np.pi:.2f}\pi$, "
    )
    fig.colorbar(image, ax=ax, label="Intensity")
    fig.tight_layout()

    save_path = resolve_output_path(output_path)
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return save_path


def trajectory_data(
    p: FourSolitonParams,
    samples: int = 1000,
):
    """返回一个完整质心运动周期内的轨迹数据。"""
    z_period = 2.0 * np.pi / p.beta0
    z_values = np.linspace(0.0, z_period, samples)

    all_centers = np.empty((samples, 4, 2), dtype=float)
    for j, z in enumerate(z_values):
        all_centers[j], _ = centers_and_phase_gradients(z, p)

    return z_values, all_centers


def draw_trajectories_on_axis(
    ax: plt.Axes,
    p: FourSolitonParams,
    samples: int = 1000,
) -> None:
    """在给定坐标轴上绘制四孤子质心轨迹。"""
    _, all_centers = trajectory_data(p, samples=samples)

    for n in range(4):
        ax.plot(all_centers[:, n, 0], all_centers[:, n, 1])
        ax.scatter(
            all_centers[0, n, 0],
            all_centers[0, n, 1],
            marker="o",
            s=36,
        )

    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(
        rf"Center trajectories, $\nu={p.nu:.3g}$"
    )
    ax.grid(True)


def plot_trajectories(
    p: FourSolitonParams,
    output_path=OUTPUT_DIR / "four_soliton_trajectoriesv1.png",
    samples=1000,
):
    """单独绘制并保存一个完整周期内的四条质心轨迹。"""
    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    draw_trajectories_on_axis(ax, p, samples=samples)
    fig.tight_layout()

    save_path = resolve_output_path(output_path)
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return save_path


def generate_snapshots(
    p: FourSolitonParams,
    alpha_values: Iterable[float] = (
        0.0,
        np.pi / 5.0,
        2.0 * np.pi / 5.0,
        3.0 * np.pi / 5.0,
        4.0 * np.pi / 5.0,
        np.pi,
    ),
):
    """分别保存给定 alpha = beta0*z 位置处的光强图。"""
    paths = []

    for index, alpha in enumerate(alpha_values):
        z = alpha / p.beta0
        path = plot_intensity(
            z=z,
            p=p,
            output_path=(
                OUTPUT_DIR
                / f"intensity_{index:02d}_alpha_{alpha / np.pi:.2f}pi.png"
            ),
        )
        paths.append(path)

    return paths



def plot_overview(
    p: FourSolitonParams,
    alpha_values: Iterable[float] = (
        0.0,
        np.pi / 5.0,
        2.0 * np.pi / 5.0,
        3.0 * np.pi / 5.0,
        4.0 * np.pi / 5.0,
        np.pi,
    ),
    xlim=(-8.0, 8.0),
    ylim=(-8.0, 8.0),
    points=401,
    columns=3,
    output_path=OUTPUT_DIR / "four_soliton_overview.png",
):
    """
    将全部传播位置的光强图绘制在同一张总体图中。

    注意：
    1. 总体图中不包含质心轨迹图；
    2. 所有子图使用统一的光强颜色范围，便于横向比较；
    3. 子图总数等于 len(alpha_values)。
    """
    alpha_values = tuple(alpha_values)

    if not alpha_values:
        raise ValueError("alpha_values 不能为空。")
    if columns < 1:
        raise ValueError("columns 必须大于等于 1。")

    x_axis = np.linspace(xlim[0], xlim[1], points)
    y_axis = np.linspace(ylim[0], ylim[1], points)
    x, y = np.meshgrid(x_axis, y_axis)

    # 归一化系数只需在 z=0 计算一次
    G0 = normalization_factor(x, y, x_axis, y_axis, p)

    intensities = []
    centers_list = []

    for alpha in alpha_values:
        z = alpha / p.beta0

        psi = four_soliton_field(
            x,
            y,
            z,
            p,
            G0=G0,
        )
        intensities.append(np.abs(psi) ** 2)

        centers, _ = centers_and_phase_gradients(z, p)
        centers_list.append(centers)

    # 所有传播截面使用统一颜色范围，便于比较强度变化
    global_vmax = max(
        float(np.max(intensity))
        for intensity in intensities
    )

    panel_count = len(alpha_values)
    rows = int(np.ceil(panel_count / columns))

    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(5.2 * columns, 4.6 * rows),
        squeeze=False,
        constrained_layout=True,
    )
    flat_axes = axes.ravel()

    last_image = None
    used_axes = []

    for index, (alpha, intensity, centers) in enumerate(
        zip(alpha_values, intensities, centers_list)
    ):
        ax = flat_axes[index]

        last_image = ax.imshow(
            intensity,
            extent=[xlim[0], xlim[1], ylim[0], ylim[1]],
            origin="lower",
            aspect="equal",
            vmin=0.0,
            vmax=global_vmax,
        )

        # 标出四个组分孤子的质心位置
        ax.scatter(
            centers[:, 0],
            centers[:, 1],
            marker="x",
            s=38,
        )

        z = alpha / p.beta0
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(
            rf" $z={z/ np.pi:.2f}*\pi$, "
        )

        used_axes.append(ax)

    # 当子图总数不能整除 columns 时，关闭多余坐标轴
    for ax in flat_axes[panel_count:]:
        ax.axis("off")

    # 所有光强图共用一个色标
    if last_image is not None:
        fig.colorbar(
            last_image,
            ax=used_axes,
            label="Intensity",
            shrink=0.88,
            pad=0.02,
        )

    fig.suptitle(
        rf"Four-soliton molecule overview: "
        rf"$d={p.d:.3g}$, "
        rf"$\nu={p.nu:.3g}$, "
        rf"$P_r={p.Pr:.3g}$, "
        rf"$\beta_0={p.beta0:.3g}$",
        fontsize=15,
    )

    save_path = resolve_output_path(output_path)
    fig.savefig(
        save_path,
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(fig)

    return save_path


if __name__ == "__main__":
    params = FourSolitonParams(
        g2=-1.0,
        d=3.23,
        nu=1.0,
        a0=1.0,
        A0=4.0,
        Pr=1.0,
        target_power=4.0,
        relative_phases=(0.0, np.pi, 0.0, np.pi),
    )

    alpha_values = (
        0.0,
        np.pi / 5.0,
        2.0 * np.pi / 5.0,
        3.0 * np.pi / 5.0,
        4.0 * np.pi / 5.0,
        np.pi,
    )

    # 1. 分别保存六张传播光强图
    generated = generate_snapshots(
        params,
        alpha_values=alpha_values,
    )

    # 2. 单独生成四个质点（质心）轨迹图
    trajectory_path = plot_trajectories(
        params,
        output_path=OUTPUT_DIR / "four_soliton_trajectories-v1.png",
        samples=1000,
    )

    # 3. 生成只包含六张光强图的总体图
    overview_path = plot_overview(
        params,
        alpha_values=alpha_values,
        columns=6,
        output_path=OUTPUT_DIR / "four_soliton_overview-v1.png",
    )

    print("已生成单独光强图：")
    for path in generated:
        print(path.resolve())

    print("已生成质点轨迹图：")
    print(trajectory_path.resolve())

    print("已生成总体图：")
    print(overview_path.resolve())
