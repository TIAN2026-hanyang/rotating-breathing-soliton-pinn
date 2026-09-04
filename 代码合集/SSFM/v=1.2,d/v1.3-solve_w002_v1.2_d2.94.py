
"""
修正版：按解析解脚本的坐标与数据设置，用二维分步傅里叶法求解

    i psi_z + psi_xx + psi_yy
    + w |psi|^2 psi + g2 (x^2+y^2) psi = 0.

默认参数：
    w = 0.02, g2 = -1,
    d = 2.94, nu = 1.2,
    a0 = 1, A0 = 4, Pr = 1,
    target_power = 4.

重要坐标关系：
    alpha = beta0*z,
    beta0 = 2*sqrt(-g2).

解析脚本的六个切片是
    alpha = 0, pi/5, 2pi/5, 3pi/5, 4pi/5, pi.

当 g2=-1 时 beta0=2，因此真实传播位置是
    z = 0, pi/10, pi/5, 3pi/10, 2pi/5, pi/2.

所有输出均直接保存到本 Python 文件所在目录，不再建立子目录。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import json

import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Params:
    g2: float = -1.0
    w: float = 0.02

    d: float = 2.94
    nu: float = 1.2
    a0: float = 1.0
    A0: float = 4.0
    Pr: float = 1.0
    target_power: float = 4.0

    relative_phases: tuple[float, float, float, float] = (
        0.0,
        np.pi,
        0.0,
        np.pi,
    )

    @property
    def beta0(self) -> float:
        return 2.0 * np.sqrt(-self.g2)

    def validate(self) -> None:
        if self.g2 >= 0:
            raise ValueError("束缚型抛物势要求 g2 < 0。")
        if self.w < 0:
            raise ValueError("当前模型设定要求 w >= 0。")
        if min(self.d, self.a0, self.A0, self.Pr, self.target_power) <= 0:
            raise ValueError("d、a0、A0、Pr 和 target_power 必须为正。")
        if len(self.relative_phases) != 4:
            raise ValueError("relative_phases 必须包含 4 个相位。")

        # w=0 高斯解析解的一致性条件
        exact_condition = self.beta0**2 * self.a0**4 * self.Pr
        if not np.isclose(exact_condition, 4.0, rtol=0.0, atol=1e-12):
            expected_pr = 4.0 / (self.beta0**2 * self.a0**4)
            raise ValueError(
                "当前参数不满足 w=0 精确高斯解条件 "
                "beta0^2*a0^4*Pr=4。"
                f"当前值为 {exact_condition:.16g}，"
                f"应取 Pr={expected_pr:.16g}。"
            )


@dataclass(frozen=True)
class Numerics:
    L: float = 8.0
    Nx: int = 256
    Ny: int = 256
    n_steps: int = 2500

    # 这里存的是 alpha，不是 z
    alpha_values: tuple[float, ...] = (
        0.0,
        np.pi / 5.0,
        2.0 * np.pi / 5.0,
        3.0 * np.pi / 5.0,
        4.0 * np.pi / 5.0,
        np.pi,
    )

    def validate(self) -> None:
        if self.L <= 0:
            raise ValueError("L 必须为正。")
        if self.Nx < 32 or self.Ny < 32:
            raise ValueError("Nx 和 Ny 过小。")
        if self.n_steps <= 0:
            raise ValueError("n_steps 必须为正整数。")

        alpha = np.asarray(self.alpha_values, dtype=float)
        if alpha.ndim != 1 or alpha.size < 2:
            raise ValueError("alpha_values 至少需要两个切片。")
        if not np.isclose(alpha[0], 0.0):
            raise ValueError("alpha_values 的第一个值必须是 0。")
        if np.any(np.diff(alpha) <= 0):
            raise ValueError("alpha_values 必须严格递增。")


def raw_initial_field(
    X: np.ndarray,
    Y: np.ndarray,
    p: Params,
) -> np.ndarray:
    """w=0 四孤子解析场在 z=0 的未归一化初值。"""
    beta0 = p.beta0
    psi0 = np.zeros_like(X, dtype=np.complex128)

    for n in range(4):
        varphi = n * np.pi / 2.0
        x0 = p.d * np.cos(varphi)
        y0 = p.d * np.sin(varphi)

        # 线性相位梯度 p_n(0)=R'_n(0)/2
        px0 = -0.5 * beta0 * p.nu * p.d * np.sin(varphi)
        py0 = 0.5 * beta0 * p.nu * p.d * np.cos(varphi)

        q2 = (X - x0) ** 2 + (Y - y0) ** 2
        phase = px0 * X + py0 * Y + p.relative_phases[n]

        psi0 += p.A0 * np.exp(
            -q2 / (2.0 * p.a0**2) + 1j * phase
        )

    return psi0


def discrete_power(
    psi: np.ndarray,
    dx: float,
    dy: float,
) -> float:
    """FFT 周期网格上的二维矩形求积。"""
    return float(np.sum(np.abs(psi) ** 2) * dx * dy)


def normalized_initial_field(
    X: np.ndarray,
    Y: np.ndarray,
    dx: float,
    dy: float,
    p: Params,
):
    raw = raw_initial_field(X, Y, p)
    raw_power = discrete_power(raw, dx, dy)

    if raw_power <= 0:
        raise RuntimeError("初始功率非正，不能归一化。")

    G0 = np.sqrt(p.target_power / raw_power)
    psi0 = G0 * raw
    return psi0, G0, raw_power


def analytic_w0_field(
    X: np.ndarray,
    Y: np.ndarray,
    z: float,
    p: Params,
    G0: float,
) -> np.ndarray:
    """w=0 的四孤子精确解析场。"""
    beta0 = p.beta0
    alpha = beta0 * z

    ca = np.cos(alpha)
    sa = np.sin(alpha)

    D = ca**2 + p.Pr * sa**2
    a = p.a0 * np.sqrt(D)
    amplitude = G0 * p.A0 * p.a0 / a

    # 对 PDE 中 Laplacian 系数为 1 的情形，分母是 8D
    b = (
        beta0
        * (p.Pr - 1.0)
        * np.sin(2.0 * alpha)
        / (8.0 * D)
    )

    # 整体宽度相位，atan2 用于正确处理象限
    theta = -np.arctan2(
        np.sqrt(p.Pr) * sa,
        ca,
    )

    # 全局坐标线性相位下的公共平移相位
    common_phase = (
        beta0
        * p.d**2
        * (1.0 - p.nu**2)
        * np.sin(2.0 * alpha)
        / 8.0
    )

    psi = np.zeros_like(X, dtype=np.complex128)

    for n in range(4):
        varphi = n * np.pi / 2.0
        cphi = np.cos(varphi)
        sphi = np.sin(varphi)

        xn = p.d * (cphi * ca - p.nu * sphi * sa)
        yn = p.d * (sphi * ca + p.nu * cphi * sa)

        pxn = -0.5 * beta0 * p.d * (
            cphi * sa + p.nu * sphi * ca
        )
        pyn = 0.5 * beta0 * p.d * (
            -sphi * sa + p.nu * cphi * ca
        )

        q2 = (X - xn) ** 2 + (Y - yn) ** 2

        phase = (
            b * q2
            + pxn * X
            + pyn * Y
            + common_phase
            + theta
            + p.relative_phases[n]
        )

        psi += amplitude * np.exp(
            -q2 / (2.0 * a**2) + 1j * phase
        )

    return psi


def relative_l2(a: np.ndarray, b: np.ndarray) -> float:
    denominator = np.linalg.norm(b.ravel())
    if denominator == 0:
        return float(np.linalg.norm((a - b).ravel()))
    return float(np.linalg.norm((a - b).ravel()) / denominator)


def boundary_max_intensity(psi: np.ndarray) -> float:
    boundary = np.concatenate(
        [
            np.abs(psi[0, :]) ** 2,
            np.abs(psi[-1, :]) ** 2,
            np.abs(psi[:, 0]) ** 2,
            np.abs(psi[:, -1]) ** 2,
        ]
    )
    return float(np.max(boundary))


def solve_ssfm(
    p: Params,
    n: Numerics,
):
    p.validate()
    n.validate()

    # 所有输出直接位于脚本所在目录
    out_dir = SCRIPT_DIR

    # FFT 周期网格必须使用 endpoint=False
    x = np.linspace(-n.L, n.L, n.Nx, endpoint=False)
    y = np.linspace(-n.L, n.L, n.Ny, endpoint=False)
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    X, Y = np.meshgrid(x, y, indexing="xy")
    r2 = X**2 + Y**2

    kx = 2.0 * np.pi * np.fft.fftfreq(n.Nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(n.Ny, d=dy)
    KX, KY = np.meshgrid(kx, ky, indexing="xy")
    k2 = KX**2 + KY**2

    alpha_values = np.asarray(n.alpha_values, dtype=float)
    z_targets = alpha_values / p.beta0
    z_max = float(z_targets[-1])
    dz = z_max / n.n_steps

    diffraction_factor = np.exp(-1j * k2 * dz)

    psi, G0, raw_power = normalized_initial_field(
        X, Y, dx, dy, p
    )

    # 验证数值初值与解析初值严格一致
    psi0_analytic = analytic_w0_field(X, Y, 0.0, p, G0)
    initial_field_mismatch = relative_l2(psi, psi0_analytic)

    target_steps = np.rint(z_targets / dz).astype(int)
    target_steps[0] = 0
    target_steps[-1] = n.n_steps

    # 对当前六个等间隔 alpha，2500 步恰好可精确命中所有切片
    actual_z_targets = target_steps * dz
    actual_alpha_targets = p.beta0 * actual_z_targets

    numerical_slices = [psi.copy()]
    analytic_slices = [psi0_analytic.copy()]
    actual_z = [0.0]
    next_slice = 1

    power_z = [0.0]
    power_values = [discrete_power(psi, dx, dy)]

    for step in range(1, n.n_steps + 1):
        # Strang 前半步：局域非线性 + 抛物势
        local_1 = p.w * np.abs(psi) ** 2 + p.g2 * r2
        psi *= np.exp(0.5j * dz * local_1)

        # 整步：衍射
        psi_hat = np.fft.fft2(psi)
        psi_hat *= diffraction_factor
        psi = np.fft.ifft2(psi_hat)

        # Strang 后半步：局域非线性 + 抛物势
        local_2 = p.w * np.abs(psi) ** 2 + p.g2 * r2
        psi *= np.exp(0.5j * dz * local_2)

        z_now = step * dz

        sample_stride = max(1, n.n_steps // 200)
        if step % sample_stride == 0 or step == n.n_steps:
            power_z.append(z_now)
            power_values.append(discrete_power(psi, dx, dy))

        if (
            next_slice < len(target_steps)
            and step == target_steps[next_slice]
        ):
            numerical_slices.append(psi.copy())
            analytic_slices.append(
                analytic_w0_field(X, Y, z_now, p, G0)
            )
            actual_z.append(z_now)
            next_slice += 1

    if next_slice != len(target_steps):
        raise RuntimeError("未能记录全部目标切片，请检查 n_steps。")

    numerical_slices = np.stack(numerical_slices)
    analytic_slices = np.stack(analytic_slices)
    actual_z = np.asarray(actual_z)
    power_z = np.asarray(power_z)
    power_values = np.asarray(power_values)

    intensity_num = np.abs(numerical_slices) ** 2
    intensity_ana = np.abs(analytic_slices) ** 2
    intensity_diff = np.abs(intensity_num - intensity_ana)

    field_errors = np.array(
        [
            relative_l2(numerical_slices[j], analytic_slices[j])
            for j in range(len(actual_z))
        ]
    )
    intensity_errors = np.array(
        [
            relative_l2(intensity_num[j], intensity_ana[j])
            for j in range(len(actual_z))
        ]
    )
    boundary_max = np.array(
        [boundary_max_intensity(s) for s in numerical_slices]
    )

    data_path = out_dir / "v1.2_d2.94_matched_slices_data.npz"
    np.savez_compressed(
        data_path,
        x=x,
        y=y,
        alpha_requested=alpha_values,
        alpha_actual=actual_alpha_targets,
        z_requested=z_targets,
        z_actual=actual_z,
        psi_numerical=numerical_slices,
        psi_analytic_w0=analytic_slices,
        intensity_numerical=intensity_num,
        intensity_analytic_w0=intensity_ana,
        intensity_absolute_difference=intensity_diff,
        relative_field_error=field_errors,
        relative_intensity_error=intensity_errors,
        boundary_max_intensity=boundary_max,
        power_z=power_z,
        power=power_values,
        G0=G0,
        raw_power=raw_power,
        initial_field_mismatch=initial_field_mismatch,
    )

    config_path = out_dir / "v1.2_d2.94_matched_config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "physics": asdict(p),
                "numerics": asdict(n),
                "coordinate_relation": "alpha = beta0*z",
                "beta0": p.beta0,
                "z_max": z_max,
                "dz": dz,
                "G0": G0,
                "raw_power": raw_power,
                "normalized_power": power_values[0],
                "initial_field_relative_mismatch": initial_field_mismatch,
                "alpha_requested": alpha_values.tolist(),
                "alpha_actual": actual_alpha_targets.tolist(),
                "z_requested": z_targets.tolist(),
                "z_actual": actual_z.tolist(),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    error_csv_path = out_dir / "v1.2_d2.94_slice_errors.csv"
    with error_csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "index",
                "alpha_over_pi",
                "z_over_pi",
                "z",
                "relative_field_error",
                "relative_intensity_error",
                "boundary_max_intensity",
            ]
        )
        for j in range(len(actual_z)):
            writer.writerow(
                [
                    j,
                    actual_alpha_targets[j] / np.pi,
                    actual_z[j] / np.pi,
                    actual_z[j],
                    field_errors[j],
                    intensity_errors[j],
                    boundary_max[j],
                ]
            )

    return {
        "x": x,
        "y": y,
        "alpha": actual_alpha_targets,
        "z": actual_z,
        "psi_num": numerical_slices,
        "psi_ana": analytic_slices,
        "I_num": intensity_num,
        "I_ana": intensity_ana,
        "I_diff": intensity_diff,
        "field_errors": field_errors,
        "intensity_errors": intensity_errors,
        "boundary_max": boundary_max,
        "power_z": power_z,
        "power": power_values,
        "G0": G0,
        "raw_power": raw_power,
        "initial_field_mismatch": initial_field_mismatch,
        "out_dir": out_dir,
        "data_path": data_path,
        "config_path": config_path,
        "error_csv_path": error_csv_path,
    }


def draw_intensity(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    intensity: np.ndarray,
    vmin: float,
    vmax: float,
):
    """
    使用 pcolormesh，避免 endpoint=False FFT 网格配合 imshow extent
    时产生半个网格的坐标偏移。
    """
    image = ax.pcolormesh(
        x,
        y,
        intensity,
        shading="auto",
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
    )
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(y[0], y[-1])
    ax.set_aspect("equal")
    return image


def plot_numerical_slices(result, p: Params) -> Path:
    x = result["x"]
    y = result["y"]
    alpha_values = result["alpha"]
    z_values = result["z"]
    intensity = result["I_num"]
    out_dir = result["out_dir"]

    vmax = float(np.max(intensity))

    fig, axes = plt.subplots(
        1,
        len(z_values),
        figsize=(26, 4.6),
        constrained_layout=True,
    )

    last = None
    for j, ax in enumerate(np.atleast_1d(axes)):
        last = draw_intensity(
            ax, x, y, intensity[j], 0.0, vmax
        )
        ax.set_title(
            rf"$z={z_values[j]/np.pi:.1f}\pi$"
        )
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")

    fig.colorbar(
        last,
        ax=np.atleast_1d(axes).tolist(),
        shrink=0.84,
        label=r"$|\psi_{\rm num}|^2$",
    )

    fig.suptitle(
        rf"Numerical propagation: $w={p.w}$, $g_2={p.g2}$, "
        rf"$\nu={p.nu}$, $d={p.d}$, $P(0)={p.target_power}$",
        fontsize=15,
    )

    path = out_dir / "v1.2_d2.94_numerical_slices.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_comparison(result, p: Params) -> Path:
    x = result["x"]
    y = result["y"]
    alpha_values = result["alpha"]
    z_values = result["z"]
    I_num = result["I_num"]
    I_ana = result["I_ana"]
    diff = result["I_diff"]
    out_dir = result["out_dir"]

    vmax = max(float(np.max(I_num)), float(np.max(I_ana)))
    diff_vmax = max(float(np.max(diff)), np.finfo(float).eps)

    fig, axes = plt.subplots(
        3,
        len(z_values),
        figsize=(25, 11),
        constrained_layout=True,
    )

    im_main = None
    im_diff = None

    for j in range(len(z_values)):
        im_main = draw_intensity(
            axes[0, j], x, y, I_ana[j], 0.0, vmax
        )
        draw_intensity(
            axes[1, j], x, y, I_num[j], 0.0, vmax
        )
        im_diff = draw_intensity(
            axes[2, j], x, y, diff[j], 0.0, diff_vmax
        )

        axes[0, j].set_title(
            rf"$z={z_values[j]/np.pi:.1f}\pi$"
        )

        for row in range(3):
            axes[row, j].set_xlabel(r"$x$")
            axes[row, j].set_ylabel(r"$y$")

    axes[0, 0].set_ylabel("Analytic $w=0$\n" + r"$y$")
    axes[1, 0].set_ylabel("Numerical $w=0.02$\n" + r"$y$")
    axes[2, 0].set_ylabel(
        "Absolute intensity difference\n" + r"$y$"
    )

    fig.colorbar(
        im_main,
        ax=axes[:2, :].ravel().tolist(),
        shrink=0.75,
        label="Intensity",
    )
    fig.colorbar(
        im_diff,
        ax=axes[2, :].ravel().tolist(),
        shrink=0.75,
        label=r"$|I_{\rm num}-I_{\rm ana}|$",
    )

    fig.suptitle(
        rf"Matched comparison: $\nu={p.nu}$, $d={p.d}$, "
        rf"$\alpha=\beta_0 z$, $\beta_0={p.beta0:.3g}$",
        fontsize=15,
    )

    path = out_dir / "v1.2_d2.94_analytic_vs_numerical.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_power(result) -> Path:
    z = result["power_z"]
    p_values = result["power"]
    out_dir = result["out_dir"]

    relative = (p_values - p_values[0]) / p_values[0]

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(z, relative)
    ax.set_xlabel(r"$z$")
    ax.set_ylabel(r"$[P(z)-P(0)]/P(0)$")
    ax.set_title("Relative power drift")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = out_dir / "v1.2_d2.94_power_drift.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    p = Params()

    n = Numerics(
        L=8.0,
        Nx=256,
        Ny=256,
        n_steps=2500,
        alpha_values=(
            0.0,
            np.pi / 5.0,
            2.0 * np.pi / 5.0,
            3.0 * np.pi / 5.0,
            4.0 * np.pi / 5.0,
            np.pi,
        ),
    )

    result = solve_ssfm(p, n)

    numerical_path = plot_numerical_slices(result, p)
    comparison_path = plot_comparison(result, p)
    power_path = plot_power(result)

    final_power_drift = (
        result["power"][-1] - result["power"][0]
    ) / result["power"][0]

    print("=" * 80)
    print("修正版数值传播完成")
    print(f"脚本目录                    = {SCRIPT_DIR}")
    print(f"nu                         = {p.nu:.12f}")
    print(f"d                          = {p.d:.12f}")
    print(f"beta0                      = {p.beta0:.12f}")
    print("坐标关系                    = alpha = beta0*z")
    print(f"真实 z_max                 = {result['z'][-1]:.12f}")
    print(f"raw initial power          = {result['raw_power']:.12e}")
    print(f"normalization G0           = {result['G0']:.12e}")
    print(f"normalized P(0)            = {result['power'][0]:.12e}")
    print(
        "initial field mismatch     = "
        f"{result['initial_field_mismatch']:.6e}"
    )
    print(f"final relative power drift = {final_power_drift:.6e}")
    print("alpha/pi slices            =", result["alpha"] / np.pi)
    print("z/pi slices                =", result["z"] / np.pi)
    print(
        "relative intensity errors =",
        result["intensity_errors"],
    )
    print(f"数值切片图                  = {numerical_path}")
    print(f"解析-数值对比图             = {comparison_path}")
    print(f"功率漂移图                  = {power_path}")
    print(f"切片数据                    = {result['data_path']}")
    print(f"配置文件                    = {result['config_path']}")
    print(f"误差表                      = {result['error_csv_path']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
