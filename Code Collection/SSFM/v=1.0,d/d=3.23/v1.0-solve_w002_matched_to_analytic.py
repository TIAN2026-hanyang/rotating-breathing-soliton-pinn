
"""
按照解析解脚本的数据设置，使用二维分步傅里叶法求解

    i psi_z + psi_xx + psi_yy
    + w |psi|^2 psi + g2 (x^2+y^2) psi = 0,

其中
    w = 0.02, g2 = -1.

初值严格采用 w=0 四孤子解析场在 z=0 的表达式，并按解析代码
将总输入功率归一化为 target_power = 4。

切片位置与解析代码一致：
    alpha = beta0*z = 0, pi/5, 2pi/5, 3pi/5, 4pi/5, pi
即在 g2=-1、beta0=2 时：
    z = 0, pi/10, pi/5, 3pi/10, 2pi/5, pi/2.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class Params:
    g2: float = -1.0
    w: float = 0.02

    d: float = 3.23
    nu: float = 1.0
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


@dataclass(frozen=True)
class Numerics:
    L: float = 8.0
    Nx: int = 256
    Ny: int = 256

    # alpha_max = pi，对应 z_max = pi/beta0 = pi/2
    alpha_max: float = np.pi
    n_steps: int = 2500

    output_dir: str = "matched_numerical_output"


def raw_initial_field(
    X: np.ndarray,
    Y: np.ndarray,
    p: Params,
) -> np.ndarray:
    """
    w=0 四孤子解析解在 z=0 的未归一化初值。
    与解析脚本 raw_four_soliton_field(..., z=0) 一致。
    """
    beta0 = p.beta0
    psi0 = np.zeros_like(X, dtype=np.complex128)

    for n in range(4):
        varphi = n * np.pi / 2.0
        x0 = p.d * np.cos(varphi)
        y0 = p.d * np.sin(varphi)

        px0 = -0.5 * beta0 * p.nu * p.d * np.sin(varphi)
        py0 =  0.5 * beta0 * p.nu * p.d * np.cos(varphi)

        q2 = (X - x0) ** 2 + (Y - y0) ** 2
        phase = (
            px0 * X
            + py0 * Y
            + p.relative_phases[n]
        )

        psi0 += p.A0 * np.exp(
            -q2 / (2.0 * p.a0**2)
            + 1j * phase
        )

    return psi0


def discrete_power(
    psi: np.ndarray,
    dx: float,
    dy: float,
) -> float:
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
    """
    w=0 时的四孤子解析场，用于与 w=0.02 数值解对照。
    """
    beta0 = p.beta0
    alpha = beta0 * z

    D = np.cos(alpha) ** 2 + p.Pr * np.sin(alpha) ** 2
    a = p.a0 * np.sqrt(D)
    amplitude = G0 * p.A0 * p.a0 / a

    b = (
        beta0
        * (p.Pr - 1.0)
        * np.sin(2.0 * alpha)
        / (8.0 * D)
    )

    theta = -np.arctan2(
        np.sqrt(p.Pr) * np.sin(alpha),
        np.cos(alpha),
    )

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

        xn = p.d * (
            cphi * np.cos(alpha)
            - p.nu * sphi * np.sin(alpha)
        )
        yn = p.d * (
            sphi * np.cos(alpha)
            + p.nu * cphi * np.sin(alpha)
        )

        pxn = -0.5 * beta0 * p.d * (
            cphi * np.sin(alpha)
            + p.nu * sphi * np.cos(alpha)
        )
        pyn = 0.5 * beta0 * p.d * (
            -sphi * np.sin(alpha)
            + p.nu * cphi * np.cos(alpha)
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
            -q2 / (2.0 * a**2)
            + 1j * phase
        )

    return psi


def solve_ssfm(
    p: Params,
    n: Numerics,
):
    out_dir = Path(n.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

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

    z_max = n.alpha_max / p.beta0
    dz = z_max / n.n_steps

    diffraction_factor = np.exp(-1j * k2 * dz)

    psi, G0, raw_power = normalized_initial_field(
        X, Y, dx, dy, p
    )

    # 与解析脚本完全一致的六个 alpha 切片
    alpha_values = np.array(
        [
            0.0,
            np.pi / 5.0,
            2.0 * np.pi / 5.0,
            3.0 * np.pi / 5.0,
            4.0 * np.pi / 5.0,
            np.pi,
        ]
    )
    z_targets = alpha_values / p.beta0

    target_steps = np.rint(z_targets / dz).astype(int)
    target_steps[-1] = n.n_steps

    numerical_slices = []
    analytic_slices = []
    actual_z = []

    numerical_slices.append(psi.copy())
    analytic_slices.append(
        analytic_w0_field(X, Y, 0.0, p, G0)
    )
    actual_z.append(0.0)

    next_slice = 1

    power_z = [0.0]
    power_values = [discrete_power(psi, dx, dy)]

    for step in range(1, n.n_steps + 1):
        # 前半步：抛物势 + 局域非线性
        local_1 = p.w * np.abs(psi) ** 2 + p.g2 * r2
        psi *= np.exp(0.5j * dz * local_1)

        # 整步：衍射
        psi_hat = np.fft.fft2(psi)
        psi_hat *= diffraction_factor
        psi = np.fft.ifft2(psi_hat)

        # 后半步：抛物势 + 局域非线性
        local_2 = p.w * np.abs(psi) ** 2 + p.g2 * r2
        psi *= np.exp(0.5j * dz * local_2)

        z_now = step * dz

        if step % max(1, n.n_steps // 200) == 0:
            power_z.append(z_now)
            power_values.append(discrete_power(psi, dx, dy))

        while (
            next_slice < len(target_steps)
            and step >= target_steps[next_slice]
        ):
            numerical_slices.append(psi.copy())
            analytic_slices.append(
                analytic_w0_field(X, Y, z_now, p, G0)
            )
            actual_z.append(z_now)
            next_slice += 1

    numerical_slices = np.stack(numerical_slices)
    analytic_slices = np.stack(analytic_slices)
    actual_z = np.asarray(actual_z)

    power_z = np.asarray(power_z)
    power_values = np.asarray(power_values)

    intensity_num = np.abs(numerical_slices) ** 2
    intensity_ana = np.abs(analytic_slices) ** 2

    np.savez_compressed(
        out_dir / "matched_slices_data.npz",
        x=x,
        y=y,
        alpha=alpha_values,
        z=actual_z,
        psi_numerical=numerical_slices,
        psi_analytic_w0=analytic_slices,
        intensity_numerical=intensity_num,
        intensity_analytic_w0=intensity_ana,
        power_z=power_z,
        power=power_values,
        G0=G0,
        raw_power=raw_power,
    )

    with open(
        out_dir / "matched_config.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            {
                "physics": p.__dict__,
                "numerics": n.__dict__,
                "beta0": p.beta0,
                "z_max": z_max,
                "dz": dz,
                "G0": G0,
                "raw_power": raw_power,
                "normalized_power": power_values[0],
                "alpha_values": alpha_values.tolist(),
                "z_values": actual_z.tolist(),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    return {
        "x": x,
        "y": y,
        "X": X,
        "Y": Y,
        "alpha": alpha_values,
        "z": actual_z,
        "psi_num": numerical_slices,
        "psi_ana": analytic_slices,
        "I_num": intensity_num,
        "I_ana": intensity_ana,
        "power_z": power_z,
        "power": power_values,
        "G0": G0,
        "raw_power": raw_power,
        "out_dir": out_dir,
    }


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
        6,
        figsize=(26, 4.5),
        constrained_layout=True,
    )

    last = None
    for j, ax in enumerate(axes):
        last = ax.imshow(
            intensity[j],
            extent=[x[0], x[-1], y[0], y[-1]],
            origin="lower",
            aspect="equal",
            vmin=0.0,
            vmax=vmax,
        )
        ax.set_title(
            rf"$\alpha={alpha_values[j]/np.pi:.1f}\pi$"
            + "\n"
            + rf"$z={z_values[j]:.4f}$"
        )
        ax.set_xlabel(r"$x$")
        ax.set_ylabel(r"$y$")

    fig.colorbar(
        last,
        ax=axes.tolist(),
        shrink=0.84,
        label=r"$|\psi_{\rm num}|^2$",
    )

    fig.suptitle(
        rf"Numerical propagation matched to analytic data: "
        rf"$w={p.w}$, $g_2={p.g2}$, $P(0)=4$",
        fontsize=15,
    )

    path = out_dir / "numerical_slices_matched_to_analytic.png"
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
    out_dir = result["out_dir"]

    vmax = max(float(np.max(I_num)), float(np.max(I_ana)))
    diff = np.abs(I_num - I_ana)
    diff_vmax = float(np.max(diff))

    fig, axes = plt.subplots(
        3,
        6,
        figsize=(25, 11),
        constrained_layout=True,
    )

    im_main = None
    im_diff = None

    for j in range(6):
        im_main = axes[0, j].imshow(
            I_ana[j],
            extent=[x[0], x[-1], y[0], y[-1]],
            origin="lower",
            aspect="equal",
            vmin=0.0,
            vmax=vmax,
        )
        axes[0, j].set_title(
            rf"$z={z_values[j]/np.pi:.1f}\pi$, "
        )

        axes[1, j].imshow(
            I_num[j],
            extent=[x[0], x[-1], y[0], y[-1]],
            origin="lower",
            aspect="equal",
            vmin=0.0,
            vmax=vmax,
        )

        im_diff = axes[2, j].imshow(
            diff[j],
            extent=[x[0], x[-1], y[0], y[-1]],
            origin="lower",
            aspect="equal",
            vmin=0.0,
            vmax=diff_vmax,
        )

        for row in range(3):
            axes[row, j].set_xlabel(r"$x$")
            axes[row, j].set_ylabel(r"$y$")

    axes[0, 0].set_ylabel("Analytic $w=0$\n" + r"$y$")
    axes[1, 0].set_ylabel("Numerical $w=0.02$\n" + r"$y$")
    axes[2, 0].set_ylabel("Absolute intensity difference\n" + r"$y$")

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
        rf"Matched comparison, $P(0)=4$, "
        rf"$w_{{num}}={p.w}$, $w_{{ana}}=0$",
        fontsize=15,
    )

    path = out_dir / "analytic_vs_numerical_matched_comparison.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_power(result) -> Path:
    z = result["power_z"]
    power_values = result["power"]
    out_dir = result["out_dir"]

    relative = (
        power_values - power_values[0]
    ) / power_values[0]

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ax.plot(z, relative)
    ax.set_xlabel(r"$z$")
    ax.set_ylabel(r"$[P(z)-P(0)]/P(0)$")
    ax.set_title("Relative power drift")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    path = out_dir / "matched_power_drift.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    p = Params(
        g2=-1.0,
        w=0.02,
        d=3.23,
        nu=1.0,
        a0=1.0,
        A0=4.0,
        Pr=1.0,
        target_power=4.0,
        relative_phases=(0.0, np.pi, 0.0, np.pi),
    )

    n = Numerics(
        L=8.0,
        Nx=256,
        Ny=256,
        alpha_max=np.pi,
        n_steps=2500,
        output_dir=str(
            Path(__file__).resolve().parent
            / "matched_numerical_output"
        ),
    )

    result = solve_ssfm(p, n)
    numerical_path = plot_numerical_slices(result, p)
    comparison_path = plot_comparison(result, p)
    power_path = plot_power(result)

    final_power_drift = (
        result["power"][-1] - result["power"][0]
    ) / result["power"][0]

    print("=" * 76)
    print("按照解析解数据设置的数值传播已完成")
    print(f"beta0                 = {p.beta0:.12f}")
    print(f"raw initial power     = {result['raw_power']:.12e}")
    print(f"normalization G0      = {result['G0']:.12e}")
    print(f"normalized P(0)       = {result['power'][0]:.12e}")
    print(f"final relative drift  = {final_power_drift:.6e}")
    print("alpha slices          =", result["alpha"])
    print("z slices              =", result["z"])
    print(f"numerical slices      = {numerical_path}")
    print(f"comparison figure     = {comparison_path}")
    print(f"power figure          = {power_path}")
    print(f"data file             = {result['out_dir'] / 'matched_slices_data.npz'}")
    print("=" * 76)


if __name__ == "__main__":
    main()
