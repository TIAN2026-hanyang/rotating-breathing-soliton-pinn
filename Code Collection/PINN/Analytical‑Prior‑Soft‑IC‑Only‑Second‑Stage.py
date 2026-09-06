
"""
hard_ic_two_stage_forward.py

解析主导 soft-IC 两阶段 PINN 正问题：
1. 软初值约束，通过 L_IC 约束 psi(x,y,0)=psi0(x,y)；
2. 解析先验使用相对 L2 损失；
3. 解析先验点与 PDE 配点均采用“全域 LHS + 光束区域增强 LHS”混合抽样；
4. 第一阶段学习 w=0 解析先验；
5. 第二阶段使用 w=0.02 PDE，逐渐降低解析先验权重，并保留软初值约束；
6. w=0.02 SSFM 数值解只用于训练后验证，绝不进入训练损失；
7. 损失曲线横坐标使用真实累计训练迭代次数；
8. 训练后在固定独立 LHS 测试点上输出五类 PDE 残差指标。

方程：
    i psi_z + psi_xx + psi_yy
    + 0.02 |psi|^2 psi - (x^2+y^2) psi = 0.

默认研究配置仍为 6x80 网络、2500 解析点、20000 PDE 点。
命令行可给出较小配置，用于 CPU 环境的实际运行。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import norm, qmc


SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_DIR = SCRIPT_DIR
CASE_TAGS = ("v0.8_d3.60", "v1.0_d3.23", "v1.2_d2.94")
CASE_PARAMETERS = {
    "v0.8_d3.60": (0.8, 3.60),
    "v1.0_d3.23": (1.0, 3.23),
    "v1.2_d2.94": (1.2, 2.94),
}


@dataclass
class Config:

    hidden_layers: int = 6
    width: int = 80

    n_analytic: int = 500
    analytic_beam_fraction: float = 0.25
    n_f: int = 10000
    pde_beam_fraction: float = 0.25
    beam_sigma_data: float = 1.20
    beam_sigma_pde: float = 1.50

    pretrain_steps: int = 0
    pretrain_lr: float = 1.0e-3
    finetune_steps: int = 10000
    finetune_lr: float = 5.0e-4
    lbfgs_steps: int = 0

    lambda_analytic_pre: float = 1.0
    lambda_analytic_start: float = 1.0
    lambda_analytic_end: float = 0.05
    lambda_analytic_lbfgs: float = 0.0
    lambda_ic: float = 1.0
    lambda_bc: float = 1.0
    lambda_pde: float = 1.0

    analytic_batch: int = 256
    boundary_batch: int = 256
    pde_batch: int = 256

    # Fixed, independent post-training PDE evaluation set.
    pde_eval_points: int = 10000
    pde_eval_batch: int = 512
    beam_region_radius: float = 2.0
    boundary_region_width: float = 1.0

    # Diagnostics for Eq. (36) and Eq. (37).
    # The centroid of each constituent is computed in a local region
    # around the corresponding analytical beam centre.
    centroid_region_radius: float = 2.0
    centroid_eps: float = 1.0e-12

    L: float = 8.0
    z_max: float = math.pi / 2.0
    w: float = 0.02
    g2: float = -1.0
    a0: float = 1.0
    A0: float = 4.0
    Pr: float = 1.0
    target_power: float = 4.0

    boundary_mode: str = "zero"
    prediction_grid: int = 128
    seed: int = 20260626

    eval_seed: int = 20269626
    dtype: str = "float32"


def dtype_from_name(name: str) -> torch.dtype:
    return torch.float64 if name == "float64" else torch.float32


def beta0(cfg: Config) -> float:
    return 2.0 * math.sqrt(-cfg.g2)


def relative_phases_numpy() -> np.ndarray:
    return np.asarray([0.0, np.pi, 0.0, np.pi], dtype=np.float64)


def normalization_factor(
    nu: float,
    d: float,
    cfg: Config,
    points: int = 256,
) -> tuple[float, float]:
    x = np.linspace(-cfg.L, cfg.L, points, endpoint=False)
    y = np.linspace(-cfg.L, cfg.L, points, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="xy")
    psi = np.zeros_like(X, dtype=np.complex128)
    b0 = beta0(cfg)

    for n, phase0 in enumerate(relative_phases_numpy()):
        phi = n * np.pi / 2.0
        x0 = d * np.cos(phi)
        y0 = d * np.sin(phi)
        px0 = -0.5 * b0 * nu * d * np.sin(phi)
        py0 = 0.5 * b0 * nu * d * np.cos(phi)
        q2 = (X - x0) ** 2 + (Y - y0) ** 2
        psi += cfg.A0 * np.exp(
            -q2 / (2.0 * cfg.a0**2)
            + 1j * (px0 * X + py0 * Y + phase0)
        )

    dx = x[1] - x[0]
    dy = y[1] - y[0]
    raw_power = float(np.sum(np.abs(psi) ** 2) * dx * dy)
    G0 = math.sqrt(cfg.target_power / raw_power)
    return G0, raw_power


def analytic_field_numpy(
    xyz: np.ndarray,
    nu: float,
    d: float,
    G0: float,
    cfg: Config,
) -> np.ndarray:
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    b0 = beta0(cfg)
    alpha = b0 * z
    ca = np.cos(alpha)
    sa = np.sin(alpha)
    D = ca**2 + cfg.Pr * sa**2
    a = cfg.a0 * np.sqrt(D)
    amplitude = G0 * cfg.A0 * cfg.a0 / a
    chirp = b0 * (cfg.Pr - 1.0) * np.sin(2.0 * alpha) / (8.0 * D)
    theta = -np.arctan2(np.sqrt(cfg.Pr) * sa, ca)
    common = b0 * d**2 * (1.0 - nu**2) * np.sin(2.0 * alpha) / 8.0

    psi = np.zeros(len(xyz), dtype=np.complex128)
    for n, phase0 in enumerate(relative_phases_numpy()):
        phi = n * np.pi / 2.0
        cp = np.cos(phi)
        sp = np.sin(phi)

        xn = d * (cp * ca - nu * sp * sa)
        yn = d * (sp * ca + nu * cp * sa)
        px = -0.5 * b0 * d * (cp * sa + nu * sp * ca)
        py = 0.5 * b0 * d * (-sp * sa + nu * cp * ca)

        q2 = (x - xn) ** 2 + (y - yn) ** 2
        phase = chirp * q2 + px * x + py * y + common + theta + phase0
        psi += amplitude * np.exp(-q2 / (2.0 * a**2) + 1j * phase)

    return psi


def beam_centres_numpy(
    z: np.ndarray,
    beam_index: np.ndarray,
    nu: float,
    d: float,
    cfg: Config,
) -> tuple[np.ndarray, np.ndarray]:
    alpha = beta0(cfg) * z
    phi = beam_index * np.pi / 2.0
    x = d * (
        np.cos(phi) * np.cos(alpha)
        - nu * np.sin(phi) * np.sin(alpha)
    )
    y = d * (
        np.sin(phi) * np.cos(alpha)
        + nu * np.cos(phi) * np.sin(alpha)
    )
    return x, y


def mixed_lhs_points(
    count: int,
    beam_fraction: float,
    sigma: float,
    nu: float,
    d: float,
    cfg: Config,
    seed: int,
) -> np.ndarray:
    n_beam = int(round(count * beam_fraction))
    n_global = count - n_beam

    global_sampler = qmc.LatinHypercube(d=3, seed=seed)
    global_xyz = qmc.scale(
        global_sampler.random(n_global),
        np.array([-cfg.L, -cfg.L, 0.0]),
        np.array([cfg.L, cfg.L, cfg.z_max]),
    )

    if n_beam == 0:
        return global_xyz

    # LHS variables: z, discrete beam selector, Gaussian dx, Gaussian dy.
    unit = qmc.LatinHypercube(d=4, seed=seed + 1).random(n_beam)
    z = cfg.z_max * unit[:, 0]
    beam_index = np.minimum((4.0 * unit[:, 1]).astype(int), 3)
    centre_x, centre_y = beam_centres_numpy(
        z, beam_index, nu, d, cfg
    )
    eps = 1.0e-6
    dx = sigma * norm.ppf(np.clip(unit[:, 2], eps, 1.0 - eps))
    dy = sigma * norm.ppf(np.clip(unit[:, 3], eps, 1.0 - eps))
    beam_xyz = np.column_stack(
        (
            np.clip(centre_x + dx, -cfg.L, cfg.L),
            np.clip(centre_y + dy, -cfg.L, cfg.L),
            z,
        )
    )

    xyz = np.vstack((global_xyz, beam_xyz))
    rng = np.random.default_rng(seed + 2)
    rng.shuffle(xyz)
    return xyz


def initial_condition_points(
    count: int,
    beam_fraction: float,
    sigma: float,
    d: float,
    cfg: Config,
    seed: int,
) -> np.ndarray:
    """
    Generate soft-IC points on z=0.

    The samples consist of global 2D LHS points and beam-enhanced
    points around the four initial beam centres. This is stricter than
    projecting 3D trajectory samples back to z=0.
    """
    n_beam = int(round(count * beam_fraction))
    n_global = count - n_beam

    global_sampler = qmc.LatinHypercube(d=2, seed=seed)
    xy_global = qmc.scale(
        global_sampler.random(n_global),
        np.array([-cfg.L, -cfg.L]),
        np.array([cfg.L, cfg.L]),
    )

    if n_beam == 0:
        return np.column_stack(
            (
                xy_global[:, 0],
                xy_global[:, 1],
                np.zeros(n_global),
            )
        )

    unit = qmc.LatinHypercube(d=3, seed=seed + 1).random(n_beam)
    beam_index = np.minimum((4.0 * unit[:, 0]).astype(int), 3)
    phi = beam_index * np.pi / 2.0

    centre_x = d * np.cos(phi)
    centre_y = d * np.sin(phi)

    eps = 1.0e-6
    dx = sigma * norm.ppf(np.clip(unit[:, 1], eps, 1.0 - eps))
    dy = sigma * norm.ppf(np.clip(unit[:, 2], eps, 1.0 - eps))

    xy_beam = np.column_stack(
        (
            np.clip(centre_x + dx, -cfg.L, cfg.L),
            np.clip(centre_y + dy, -cfg.L, cfg.L),
        )
    )

    xy = np.vstack((xy_global, xy_beam))
    rng = np.random.default_rng(seed + 2)
    rng.shuffle(xy)

    return np.column_stack(
        (
            xy[:, 0],
            xy[:, 1],
            np.zeros(len(xy)),
        )
    )


def boundary_points(
    count: int,
    cfg: Config,
    seed: int,
) -> list[np.ndarray]:
    faces = []
    yz = qmc.scale(
        qmc.LatinHypercube(d=2, seed=seed).random(count),
        np.array([-cfg.L, 0.0]),
        np.array([cfg.L, cfg.z_max]),
    )
    faces.append(np.column_stack((np.full(count, -cfg.L), yz[:, 0], yz[:, 1])))
    faces.append(np.column_stack((np.full(count, cfg.L), yz[:, 0], yz[:, 1])))

    xz = qmc.scale(
        qmc.LatinHypercube(d=2, seed=seed + 1).random(count),
        np.array([-cfg.L, 0.0]),
        np.array([cfg.L, cfg.z_max]),
    )
    faces.append(np.column_stack((xz[:, 0], np.full(count, -cfg.L), xz[:, 1])))
    faces.append(np.column_stack((xz[:, 0], np.full(count, cfg.L), xz[:, 1])))
    return faces


class CorrectionNet(torch.nn.Module):
    def __init__(
        self,
        lower: np.ndarray,
        upper: np.ndarray,
        hidden_layers: int,
        width: int,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "lower", torch.as_tensor(lower, dtype=dtype).reshape(1, 3)
        )
        self.register_buffer(
            "upper", torch.as_tensor(upper, dtype=dtype).reshape(1, 3)
        )

        modules: list[torch.nn.Module] = []
        in_dim = 3
        for _ in range(hidden_layers):
            layer = torch.nn.Linear(in_dim, width, dtype=dtype)
            torch.nn.init.xavier_normal_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
            modules.extend((layer, torch.nn.Tanh()))
            in_dim = width
        output = torch.nn.Linear(in_dim, 2, dtype=dtype)
        torch.nn.init.xavier_normal_(output.weight)
        torch.nn.init.zeros_(output.bias)
        modules.append(output)
        self.net = torch.nn.Sequential(*modules)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        scaled = 2.0 * (xyz - self.lower) / (self.upper - self.lower) - 1.0
        return self.net(scaled)


class HardICPINN(torch.nn.Module):
    """
    Analytical-prior soft-IC PINN.

    The w=0 analytic field is embedded as the base solution:
        psi_theta = psi_w0 + N_theta.

    The initial condition is not hard-enforced by the lifting factor.
    Instead, it is imposed through a soft initial-condition loss L_IC.
    """

    def __init__(
        self,
        nu: float,
        d: float,
        G0: float,
        cfg: Config,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.nu = float(nu)
        self.d = float(d)
        self.G0 = float(G0)
        self.cfg = cfg
        self.correction = CorrectionNet(
            np.array([-cfg.L, -cfg.L, 0.0]),
            np.array([cfg.L, cfg.L, cfg.z_max]),
            cfg.hidden_layers,
            cfg.width,
            dtype,
        )
        self.register_buffer(
            "phases",
            torch.as_tensor(
                [0.0, math.pi, 0.0, math.pi],
                dtype=dtype,
            ),
        )

    def analytic_uv(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        b0 = beta0(self.cfg)
        alpha = b0 * z
        ca = torch.cos(alpha)
        sa = torch.sin(alpha)

        D = ca.square() + self.cfg.Pr * sa.square()
        width = self.cfg.a0 * torch.sqrt(D)
        amplitude = (
            self.G0
            * self.cfg.A0
            * self.cfg.a0
            / width
        )

        chirp = (
            b0
            * (self.cfg.Pr - 1.0)
            * torch.sin(2.0 * alpha)
            / (8.0 * D)
        )
        theta = -torch.atan2(
            math.sqrt(self.cfg.Pr) * sa,
            ca,
        )
        common = (
            b0
            * self.d**2
            * (1.0 - self.nu**2)
            * torch.sin(2.0 * alpha)
            / 8.0
        )

        u = torch.zeros_like(x)
        v = torch.zeros_like(x)

        for n in range(4):
            phi = n * math.pi / 2.0
            cp = math.cos(phi)
            sp = math.sin(phi)

            xn = self.d * (
                cp * ca - self.nu * sp * sa
            )
            yn = self.d * (
                sp * ca + self.nu * cp * sa
            )
            px = -0.5 * b0 * self.d * (
                cp * sa + self.nu * sp * ca
            )
            py = 0.5 * b0 * self.d * (
                -sp * sa + self.nu * cp * ca
            )

            q2 = (x - xn).square() + (y - yn).square()
            phase = (
                chirp * q2
                + px * x
                + py * y
                + common
                + theta
                + self.phases[n]
            )
            amp = amplitude * torch.exp(
                -q2 / (2.0 * width.square())
            )
            u = u + amp * torch.cos(phase)
            v = v + amp * torch.sin(phase)

        return torch.cat((u, v), dim=1)

    def initial_uv(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.analytic_uv(x, y, torch.zeros_like(x))

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        x = xyz[:, 0:1]
        y = xyz[:, 1:2]
        z = xyz[:, 2:3]

        analytic_base = self.analytic_uv(x, y, z)

        # Soft-IC analytical-prior version:
        # no z/zmax lifting, so the initial condition is enforced by L_IC.
        return analytic_base + self.correction(xyz)


def derivative(
    output: torch.Tensor,
    input_: torch.Tensor,
) -> torch.Tensor:
    return torch.autograd.grad(
        output,
        input_,
        grad_outputs=torch.ones_like(output),
        create_graph=True,
        retain_graph=True,
    )[0]


def residual(
    model: HardICPINN,
    xyz: torch.Tensor,
    cfg: Config,
) -> tuple[torch.Tensor, torch.Tensor]:
    xyz = xyz.clone().detach().requires_grad_(True)
    uv = model(xyz)
    u = uv[:, 0:1]
    v = uv[:, 1:2]

    gu = derivative(u, xyz)
    gv = derivative(v, xyz)
    ux, uy, uz = gu[:, 0:1], gu[:, 1:2], gu[:, 2:3]
    vx, vy, vz = gv[:, 0:1], gv[:, 1:2], gv[:, 2:3]
    uxx = derivative(ux, xyz)[:, 0:1]
    uyy = derivative(uy, xyz)[:, 1:2]
    vxx = derivative(vx, xyz)[:, 0:1]
    vyy = derivative(vy, xyz)[:, 1:2]

    x = xyz[:, 0:1]
    y = xyz[:, 1:2]
    intensity = u.square() + v.square()
    coefficient = cfg.w * intensity + cfg.g2 * (x.square() + y.square())
    fu = -vz + uxx + uyy + coefficient * u
    fv = uz + vxx + vyy + coefficient * v
    return fu, fv


def relative_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1.0e-12,
) -> torch.Tensor:
    return (
        torch.sum((prediction - target).square())
        / (torch.sum(target.square()) + eps)
    )


def normalized_boundary_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    amplitude_scale: float,
) -> torch.Tensor:
    return torch.mean((prediction - target).square()) / (
        amplitude_scale**2 + 1.0e-12
    )


# def normalized_pde_loss(
#     fu: torch.Tensor,
#     fv: torch.Tensor,
#     xyz: torch.Tensor,
# ) -> torch.Tensor:
#     # Suppress artificial domination by the large harmonic-potential scale
#     # near the outer computational boundary.
#     x = xyz[:, 0:1]
#     y = xyz[:, 1:2]
#     scale = 1.0 + x.square() + y.square()
#     return torch.mean((fu.square() + fv.square()) / scale.square())

def normalized_pde_loss(
    fu: torch.Tensor,
    fv: torch.Tensor,
    xyz: torch.Tensor,
) -> torch.Tensor:
    # Unnormalized raw PDE loss:
    # L_PDE = mean(fu^2 + fv^2)
    return torch.mean(fu.square() + fv.square())


def independent_pde_test_points(
    count: int,
    cfg: Config,
    seed: int,
) -> np.ndarray:
    """Generate one fixed full-domain LHS set that is never used in training."""
    if count <= 0:
        raise ValueError("pde_eval_points must be positive.")
    sampler = qmc.LatinHypercube(d=3, seed=seed)
    return qmc.scale(
        sampler.random(count),
        np.array([-cfg.L, -cfg.L, 0.0], dtype=np.float64),
        np.array([cfg.L, cfg.L, cfg.z_max], dtype=np.float64),
    )


def pde_evaluation_region_masks(
    xyz: np.ndarray,
    nu: float,
    d: float,
    cfg: Config,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Define two geometry-based subsets of the same independent test set.

    beam_mask:
        Points whose transverse distance to the nearest analytic beam centre
        is no larger than cfg.beam_region_radius.
    boundary_mask:
        Points lying in the side-boundary shell of width
        cfg.boundary_region_width.
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("xyz must have shape (N, 3).")
    if cfg.beam_region_radius <= 0.0:
        raise ValueError("beam_region_radius must be positive.")
    if not 0.0 < cfg.boundary_region_width <= cfg.L:
        raise ValueError("boundary_region_width must lie in (0, L].")

    z = xyz[:, 2]
    min_distance = np.full(len(xyz), np.inf, dtype=np.float64)
    for beam_index in range(4):
        indices = np.full(len(xyz), beam_index, dtype=np.int64)
        centre_x, centre_y = beam_centres_numpy(z, indices, nu, d, cfg)
        distance = np.sqrt(
            (xyz[:, 0] - centre_x) ** 2
            + (xyz[:, 1] - centre_y) ** 2
        )
        min_distance = np.minimum(min_distance, distance)

    beam_mask = min_distance <= cfg.beam_region_radius
    distance_to_side_boundary = np.minimum(
        cfg.L - np.abs(xyz[:, 0]),
        cfg.L - np.abs(xyz[:, 1]),
    )
    boundary_mask = distance_to_side_boundary <= cfg.boundary_region_width
    return beam_mask, boundary_mask


def evaluate_pde_metrics(
    model: HardICPINN,
    xyz_np: np.ndarray,
    nu: float,
    d: float,
    cfg: Config,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[dict[str, float | int], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate PDE residuals after training on a fixed, independent test set.

    The five requested metrics are:
        weighted_pde_mse
        raw_pde_mse
        raw_pde_rmse
        beam_region_pde_mse
        boundary_region_pde_mse

    The regional metrics use the unweighted residual f_R^2 + f_I^2.
    """
    xyz_np = np.asarray(xyz_np, dtype=np.float64)
    beam_mask, boundary_mask = pde_evaluation_region_masks(
        xyz_np, nu, d, cfg
    )
    if not np.any(beam_mask):
        raise RuntimeError(
            "No independent PDE test point entered the beam region; "
            "increase pde_eval_points or beam_region_radius."
        )
    if not np.any(boundary_mask):
        raise RuntimeError(
            "No independent PDE test point entered the boundary region; "
            "increase pde_eval_points or boundary_region_width."
        )

    residual_squared = np.empty(len(xyz_np), dtype=np.float64)
    weighted_residual_squared = np.empty(len(xyz_np), dtype=np.float64)

    was_training = model.training
    model.eval()
    batch_size = max(1, int(cfg.pde_eval_batch))

    for start in range(0, len(xyz_np), batch_size):
        stop = min(start + batch_size, len(xyz_np))
        xyz = torch.as_tensor(
            xyz_np[start:stop],
            dtype=dtype,
            device=device,
        )
        # Automatic differentiation is required here, even though no model
        # parameter update is performed.
        with torch.enable_grad():
            fu, fv = residual(model, xyz, cfg)
            raw = fu.square() + fv.square()
            x = xyz[:, 0:1]
            y = xyz[:, 1:2]
            scale = 1.0 + x.square() + y.square()
            weighted = raw / scale.square()

        residual_squared[start:stop] = (
            raw.detach().cpu().numpy().reshape(-1)
        )
        weighted_residual_squared[start:stop] = (
            weighted.detach().cpu().numpy().reshape(-1)
        )

    if was_training:
        model.train()

    raw_mse = float(np.mean(residual_squared))
    metrics: dict[str, float | int] = {
        "weighted_pde_mse": float(np.mean(weighted_residual_squared)),
        "raw_pde_mse": raw_mse,
        "raw_pde_rmse": float(np.sqrt(raw_mse)),
        "beam_region_pde_mse": float(np.mean(residual_squared[beam_mask])),
        "boundary_region_pde_mse": float(
            np.mean(residual_squared[boundary_mask])
        ),
        "evaluation_points": int(len(xyz_np)),
        "beam_region_points": int(np.count_nonzero(beam_mask)),
        "boundary_region_points": int(np.count_nonzero(boundary_mask)),
        "evaluation_seed": int(cfg.eval_seed),
    }
    return (
        metrics,
        residual_squared,
        weighted_residual_squared,
        beam_mask,
        boundary_mask,
    )


def batch_pair(
    xyz: torch.Tensor,
    uv: torch.Tensor,
    size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if size >= len(xyz):
        return xyz, uv
    index = torch.randint(0, len(xyz), (size,), device=xyz.device)
    return xyz[index], uv[index]


def batch_tensor(x: torch.Tensor, size: int) -> torch.Tensor:
    if size >= len(x):
        return x
    index = torch.randint(0, len(x), (size,), device=x.device)
    return x[index]


def analytic_weight(
    step: int,
    total: int,
    start: float,
    end: float,
) -> float:
    if total <= 1:
        return end
    s = step / (total - 1)
    return end + (start - end) * 0.5 * (1.0 + math.cos(math.pi * s))


@torch.no_grad()
def predict_chunks(
    model: HardICPINN,
    xyz: np.ndarray,
    device: torch.device,
    dtype: torch.dtype,
    chunk: int = 32768,
) -> np.ndarray:
    output = []
    for start in range(0, len(xyz), chunk):
        tensor = torch.as_tensor(
            xyz[start:start + chunk],
            dtype=dtype,
            device=device,
        )
        output.append(model(tensor).cpu().numpy())
    return np.vstack(output)


def component_count() -> int:
    """Number of constituents in the current original N=4 code."""
    return int(len(relative_phases_numpy()))


def constituent_centroids_and_radius(
    x: np.ndarray,
    y: np.ndarray,
    z_values: np.ndarray,
    intensity_slices: np.ndarray,
    nu: float,
    d: float,
    cfg: Config,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the constituent centroids c_n^X(z) and molecular radius R_X(z).

    This implements the quantities used in Eq. (36) and Eq. (37).
    For each z slice and each constituent, the centroid is computed from
    the intensity in a local region around the analytical centre of that
    constituent.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    z_values = np.asarray(z_values, dtype=np.float64)
    intensity_slices = np.asarray(intensity_slices, dtype=np.float64)

    X, Y = np.meshgrid(x, y, indexing="xy")
    n_components = component_count()
    centroids = np.empty((len(z_values), n_components, 2), dtype=np.float64)
    radius = np.empty(len(z_values), dtype=np.float64)

    local_radius2 = float(cfg.centroid_region_radius) ** 2
    eps = float(cfg.centroid_eps)

    for k, z_value in enumerate(z_values):
        I = intensity_slices[k]
        beam_indices = np.arange(n_components, dtype=np.int64)
        z_for_centres = np.full(n_components, z_value, dtype=np.float64)
        centre_x, centre_y = beam_centres_numpy(
            z_for_centres, beam_indices, nu, d, cfg
        )

        distance2 = np.empty((n_components,) + X.shape, dtype=np.float64)
        for n in range(n_components):
            distance2[n] = (X - centre_x[n]) ** 2 + (Y - centre_y[n]) ** 2

        nearest = np.argmin(distance2, axis=0)

        for n in range(n_components):
            mask = (nearest == n) & (distance2[n] <= local_radius2)
            if not np.any(mask):
                mask = nearest == n

            weights = I[mask]
            denom = float(np.sum(weights))

            if denom <= eps:
                # Fallback to analytical centre if the local intensity is too small.
                centroids[k, n, 0] = centre_x[n]
                centroids[k, n, 1] = centre_y[n]
            else:
                centroids[k, n, 0] = float(np.sum(X[mask] * weights) / denom)
                centroids[k, n, 1] = float(np.sum(Y[mask] * weights) / denom)

        radius[k] = float(np.mean(np.linalg.norm(centroids[k], axis=1)))

    return centroids, radius


def collective_motion_diagnostics(
    x: np.ndarray,
    y: np.ndarray,
    z_values: np.ndarray,
    intensity_prediction: np.ndarray,
    intensity_reference: np.ndarray | None,
    nu: float,
    d: float,
    cfg: Config,
) -> dict:
    """
    Diagnostics for Eq. (36) and Eq. (37).

    Eq. (36): normalized centroid-trajectory error E_traj.
    Eq. (37): molecular radius R_X(z).
    """
    centroids_pinn, radius_pinn = constituent_centroids_and_radius(
        x, y, z_values, intensity_prediction, nu, d, cfg
    )

    result = {
        "centroids_pinn": centroids_pinn,
        "radius_pinn": radius_pinn,
        "centroids_reference": None,
        "radius_reference": None,
        "trajectory_relative_error": None,
        "radius_absolute_errors": None,
        "radius_relative_errors": None,
        "radius_error_normalized_by_d": None,
    }

    if intensity_reference is None:
        return result

    centroids_ref, radius_ref = constituent_centroids_and_radius(
        x, y, z_values, intensity_reference, nu, d, cfg
    )

    centroid_diff = centroids_pinn - centroids_ref
    trajectory_error = float(
        np.sqrt(np.mean(np.sum(centroid_diff**2, axis=2)))
        / max(abs(float(d)), float(cfg.centroid_eps))
    )

    radius_abs = np.abs(radius_pinn - radius_ref)
    radius_rel = radius_abs / np.maximum(
        np.abs(radius_ref),
        float(cfg.centroid_eps),
    )
    radius_error_by_d = float(
        np.sqrt(np.mean((radius_pinn - radius_ref) ** 2))
        / max(abs(float(d)), float(cfg.centroid_eps))
    )

    result.update(
        {
            "centroids_reference": centroids_ref,
            "radius_reference": radius_ref,
            "trajectory_relative_error": trajectory_error,
            "radius_absolute_errors": radius_abs,
            "radius_relative_errors": radius_rel,
            "radius_error_normalized_by_d": radius_error_by_d,
        }
    )
    return result


def load_numerical_reference(tag: str):
    """Load an optional SSFM reference using portable, script-relative paths."""
    candidates = [
        REFERENCE_DIR / f"lhs_{tag}.npz",
        SCRIPT_DIR / "pinn_two_stage_forward_inverse" / f"lhs_{tag}.npz",
    ]
    if tag == "v1.0_d3.23":
        candidates.insert(0, SCRIPT_DIR / "v1.1_matched_slices_data.npz")

    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        print(f"[{tag}] no SSFM reference found; prediction will still be saved.")
        return None

    raw = np.load(path, allow_pickle=False)

    def first_key(*keys):
        for key in keys:
            if key in raw.files:
                return np.asarray(raw[key])
        raise KeyError(
            f"{path}: none of {keys} found. Available keys: {raw.files}"
        )

    reference = {
        "x": first_key("grid_x", "x"),
        "y": first_key("grid_y", "y"),
        "z": first_key("grid_z", "z_actual", "z", "z_values"),
        "psi": first_key("psi_slices", "psi_numerical", "psi_reference", "psi"),
    }
    print(f"[{tag}] loaded SSFM reference: {path}")
    return reference


def train_case(
    tag: str,
    cfg: Config,
    device: torch.device,
) -> dict:
    nu, d = CASE_PARAMETERS[tag]
    dtype = dtype_from_name(cfg.dtype)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    G0, raw_power = normalization_factor(nu, d, cfg)

    ana_xyz_np = mixed_lhs_points(
        cfg.n_analytic,
        cfg.analytic_beam_fraction,
        cfg.beam_sigma_data,
        nu,
        d,
        cfg,
        cfg.seed + 10,
    )
    ana_psi = analytic_field_numpy(ana_xyz_np, nu, d, G0, cfg)
    ana_uv_np = np.column_stack((ana_psi.real, ana_psi.imag))

    pde_xyz_np = mixed_lhs_points(
        cfg.n_f,
        cfg.pde_beam_fraction,
        cfg.beam_sigma_pde,
        nu,
        d,
        cfg,
        cfg.seed + 20,
    )

    # Initial-condition points for soft-IC training.
    # These points lie exactly on z=0 and include global + beam-enhanced samples.
    ic_xyz_np = initial_condition_points(
        cfg.n_analytic,
        cfg.analytic_beam_fraction,
        cfg.beam_sigma_data,
        d,
        cfg,
        cfg.seed + 40,
    )
    ic_psi = analytic_field_numpy(ic_xyz_np, nu, d, G0, cfg)
    ic_uv_np = np.column_stack((ic_psi.real, ic_psi.imag))

    n_bc = max(256, min(1250, cfg.n_analytic // 2))
    bc_faces_np = boundary_points(n_bc, cfg, cfg.seed + 30)
    if cfg.boundary_mode == "analytic":
        bc_uv_np = []
        for face in bc_faces_np:
            psi = analytic_field_numpy(face, nu, d, G0, cfg)
            bc_uv_np.append(np.column_stack((psi.real, psi.imag)))
    else:
        bc_uv_np = [np.zeros((n_bc, 2), dtype=np.float64) for _ in range(4)]

    ana_xyz = torch.as_tensor(ana_xyz_np, dtype=dtype, device=device)
    ana_uv = torch.as_tensor(ana_uv_np, dtype=dtype, device=device)
    pde_xyz = torch.as_tensor(pde_xyz_np, dtype=dtype, device=device)
    ic_xyz = torch.as_tensor(ic_xyz_np, dtype=dtype, device=device)
    ic_uv = torch.as_tensor(ic_uv_np, dtype=dtype, device=device)
    bc_faces = [
        torch.as_tensor(face, dtype=dtype, device=device)
        for face in bc_faces_np
    ]
    bc_targets = [
        torch.as_tensor(values, dtype=dtype, device=device)
        for values in bc_uv_np
    ]

    model = HardICPINN(nu, d, G0, cfg, dtype).to(device)
    amplitude_scale = float(np.sqrt(np.max(np.abs(ana_psi) ** 2)))

    output_dir = SCRIPT_DIR / f"soft_ic_final_solo_{tag}"
    # output_dir = SCRIPT_DIR / f"hard_ic_{tag}_seed{cfg.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict] = []
    start_time = time.time()

    def boundary_loss(training: bool) -> torch.Tensor:
        total = torch.zeros((), dtype=dtype, device=device)
        for xyz_all, target_all in zip(bc_faces, bc_targets):
            if training:
                xyz_use, target_use = batch_pair(
                    xyz_all, target_all, cfg.boundary_batch
                )
            else:
                xyz_use, target_use = xyz_all, target_all
            total = total + normalized_boundary_mse(
                model(xyz_use), target_use, amplitude_scale
            )
        return total / 4.0

    # Stage 1: relative analytic-prior pretraining.
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.pretrain_lr)
    for step in range(1, cfg.pretrain_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        xyz_b, uv_b = batch_pair(
            ana_xyz, ana_uv, cfg.analytic_batch
        )
        l_ana = relative_mse(model(xyz_b), uv_b)

        xyz_i, uv_i = batch_pair(
            ic_xyz, ic_uv, cfg.analytic_batch
        )
        l_ic = relative_mse(model(xyz_i), uv_i)

        l_bc = boundary_loss(training=True)
        loss = (
            cfg.lambda_analytic_pre * l_ana
            + cfg.lambda_ic * l_ic
            + cfg.lambda_bc * l_bc
        )
        loss.backward()
        optimizer.step()

        if step == 1 or step % max(1, cfg.pretrain_steps // 10) == 0:
            history.append(
                {
                    "stage": "pretrain",
                    "step": step,
                    "global_step": step,
                    "total": float(loss.detach().cpu()),
                    "analytic": float(l_ana.detach().cpu()),
                    "ic": float(l_ic.detach().cpu()),
                    "bc": float(l_bc.detach().cpu()),
                    "pde": 0.0,
                    "lambda_analytic": cfg.lambda_analytic_pre,
                }
            )
            print(
                f"[{tag}] pre {step}/{cfg.pretrain_steps} "
                f"L={float(loss.detach().cpu()):.3e} "
                f"Lana={float(l_ana.detach().cpu()):.3e} "
                f"Lic={float(l_ic.detach().cpu()):.3e}"
            )

    torch.save(
        {"state": model.state_dict(), "config": asdict(cfg), "tag": tag},
        output_dir / "pretrained_model.pt",
    )

    # Stage 2: target PDE and decaying analytic prior.
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.finetune_lr)
    for step in range(1, cfg.finetune_steps + 1):
        lam_ana = analytic_weight(
            step - 1,
            cfg.finetune_steps,
            cfg.lambda_analytic_start,
            cfg.lambda_analytic_end,
        )
        optimizer.zero_grad(set_to_none=True)

        xyz_a, uv_a = batch_pair(
            ana_xyz, ana_uv, cfg.analytic_batch
        )
        xyz_f = batch_tensor(pde_xyz, cfg.pde_batch)

        l_ana = relative_mse(model(xyz_a), uv_a)

        xyz_i, uv_i = batch_pair(
            ic_xyz, ic_uv, cfg.analytic_batch
        )
        l_ic = relative_mse(model(xyz_i), uv_i)

        l_bc = boundary_loss(training=True)
        fu, fv = residual(model, xyz_f, cfg)
        l_pde = normalized_pde_loss(fu, fv, xyz_f)

        loss = (
            lam_ana * l_ana
            + cfg.lambda_ic * l_ic
            + cfg.lambda_bc * l_bc
            + cfg.lambda_pde * l_pde
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        if step == 1 or step % max(1, cfg.finetune_steps // 10) == 0:
            history.append(
                {
                    "stage": "finetune",
                    "step": step,
                    "global_step": cfg.pretrain_steps + step,
                    "total": float(loss.detach().cpu()),
                    "analytic": float(l_ana.detach().cpu()),
                    "ic": float(l_ic.detach().cpu()),
                    "bc": float(l_bc.detach().cpu()),
                    "pde": float(l_pde.detach().cpu()),
                    "lambda_analytic": lam_ana,
                }
            )
            print(
                f"[{tag}] fine {step}/{cfg.finetune_steps} "
                f"L={float(loss.detach().cpu()):.3e} lambda={lam_ana:.3f} "
                f"Lpde={float(l_pde.detach().cpu()):.3e} "
                f"Lic={float(l_ic.detach().cpu()):.3e}"
            )

    # Optional small final L-BFGS on a fixed moderate subset.
    if cfg.lbfgs_steps > 0:
        ana_small = ana_xyz[: min(len(ana_xyz), 512)]
        uv_small = ana_uv[: len(ana_small)]
        ic_small = ic_xyz[: min(len(ic_xyz), 512)]
        ic_uv_small = ic_uv[: len(ic_small)]
        pde_small = pde_xyz[: min(len(pde_xyz), 512)]
        lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=0.5,
            max_iter=cfg.lbfgs_steps,
            max_eval=cfg.lbfgs_steps,
            history_size=20,
            tolerance_grad=1.0e-9,
            tolerance_change=1.0e-12,
            line_search_fn=None,
        )

        count = {"n": 0}

        def closure():
            lbfgs.zero_grad(set_to_none=True)
            l_ana = relative_mse(model(ana_small), uv_small)
            l_ic = relative_mse(model(ic_small), ic_uv_small)
            l_bc = boundary_loss(training=False)
            fu, fv = residual(model, pde_small, cfg)
            l_pde = normalized_pde_loss(fu, fv, pde_small)
            loss = (
                cfg.lambda_analytic_lbfgs * l_ana
                + cfg.lambda_ic * l_ic
                + cfg.lambda_bc * l_bc
                + cfg.lambda_pde * l_pde
            )
            loss.backward()
            count["n"] += 1
            return loss

        lbfgs.step(closure)

    elapsed = time.time() - start_time

    # Fixed independent post-training PDE evaluation. These points are
    # generated with a seed not used by analytic, PDE, or boundary training
    # samples and never enter any optimization loss.
    pde_eval_xyz_np = independent_pde_test_points(
        cfg.pde_eval_points,
        cfg,
        cfg.eval_seed,
    )
    (
        pde_metrics,
        pde_residual_squared,
        pde_weighted_residual_squared,
        pde_beam_mask,
        pde_boundary_mask,
    ) = evaluate_pde_metrics(
        model,
        pde_eval_xyz_np,
        nu,
        d,
        cfg,
        device,
        dtype,
    )

    pde_eval_npz_path = output_dir / "pde_evaluation_points.npz"
    np.savez_compressed(
        pde_eval_npz_path,
        xyz=pde_eval_xyz_np,
        residual_squared=pde_residual_squared,
        weighted_residual_squared=pde_weighted_residual_squared,
        beam_region_mask=pde_beam_mask,
        boundary_region_mask=pde_boundary_mask,
        beam_region_radius=np.array(cfg.beam_region_radius),
        boundary_region_width=np.array(cfg.boundary_region_width),
        evaluation_seed=np.array(cfg.eval_seed),
    )

    pde_eval_csv_path = output_dir / "pde_evaluation_metrics.csv"
    with pde_eval_csv_path.open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(pde_metrics.keys()))
        writer.writeheader()
        writer.writerow(pde_metrics)

    pde_eval_json_path = output_dir / "pde_evaluation_metrics.json"
    with pde_eval_json_path.open("w", encoding="utf-8") as file:
        json.dump(pde_metrics, file, ensure_ascii=False, indent=2)

    print(f"[{tag}] independent PDE evaluation:")
    for metric_name in (
        "weighted_pde_mse",
        "raw_pde_mse",
        "raw_pde_rmse",
        "beam_region_pde_mse",
        "boundary_region_pde_mse",
    ):
        print(f"  {metric_name} = {pde_metrics[metric_name]:.6e}")

    # Prediction and validation against w=0.02 numerical reference.
    reference = load_numerical_reference(tag)
    if reference is None:
        x = np.linspace(-cfg.L, cfg.L, cfg.prediction_grid, endpoint=False)
        y = x.copy()
        z_values = np.array(
            [0, np.pi/10, np.pi/5, 3*np.pi/10, 2*np.pi/5, np.pi/2]
        )
        psi_reference = None
    else:
        full_x = reference["x"]
        full_y = reference["y"]
        stride = max(1, len(full_x) // cfg.prediction_grid)
        x = full_x[::stride]
        y = full_y[::stride]
        z_values = reference["z"]
        psi_reference = reference["psi"][:, ::stride, ::stride]

    X, Y = np.meshgrid(x, y, indexing="xy")
    psi_prediction = []
    for z_value in z_values:
        xyz = np.column_stack(
            (X.ravel(), Y.ravel(), np.full(X.size, z_value))
        )
        uv = predict_chunks(model, xyz, device, dtype)
        psi_prediction.append(
            (uv[:, 0] + 1j * uv[:, 1]).reshape(X.shape)
        )
    psi_prediction = np.stack(psi_prediction)
    intensity_prediction = np.abs(psi_prediction) ** 2

    if psi_reference is not None:
        intensity_reference = np.abs(psi_reference) ** 2
        field_errors = np.asarray(
            [
                np.linalg.norm(psi_prediction[i] - psi_reference[i])
                / np.linalg.norm(psi_reference[i])
                for i in range(len(z_values))
            ]
        )
        intensity_errors = np.asarray(
            [
                np.linalg.norm(
                    intensity_prediction[i] - intensity_reference[i]
                )
                / np.linalg.norm(intensity_reference[i])
                for i in range(len(z_values))
            ]
        )
    else:
        intensity_reference = None
        field_errors = None
        intensity_errors = None

    dx = x[1] - x[0]
    dy = y[1] - y[0]
    powers = np.sum(intensity_prediction, axis=(1, 2)) * dx * dy
    maxima = np.max(intensity_prediction, axis=(1, 2))

    collective = collective_motion_diagnostics(
        x=x,
        y=y,
        z_values=z_values,
        intensity_prediction=intensity_prediction,
        intensity_reference=intensity_reference,
        nu=nu,
        d=d,
        cfg=cfg,
    )

    np.savez_compressed(
        output_dir / "prediction_results.npz",
        x=x,
        y=y,
        z=z_values,
        psi_prediction=psi_prediction,
        intensity_prediction=intensity_prediction,
        psi_reference=(
            np.array([]) if psi_reference is None else psi_reference
        ),
        field_relative_errors=(
            np.array([]) if field_errors is None else field_errors
        ),
        intensity_relative_errors=(
            np.array([]) if intensity_errors is None else intensity_errors
        ),
        powers=powers,
        maxima=maxima,
        centroids_pinn=collective["centroids_pinn"],
        radius_pinn=collective["radius_pinn"],
        centroids_reference=(
            np.array([])
            if collective["centroids_reference"] is None
            else collective["centroids_reference"]
        ),
        radius_reference=(
            np.array([])
            if collective["radius_reference"] is None
            else collective["radius_reference"]
        ),
        radius_absolute_errors=(
            np.array([])
            if collective["radius_absolute_errors"] is None
            else collective["radius_absolute_errors"]
        ),
        radius_relative_errors=(
            np.array([])
            if collective["radius_relative_errors"] is None
            else collective["radius_relative_errors"]
        ),
    )

    # Separate color scale for PINN and reference to avoid false blank panels.
    rows = 3 if intensity_reference is not None else 1
    fig, axes = plt.subplots(
        rows,
        len(z_values),
        figsize=(4.1 * len(z_values), 4.0 * rows),
        constrained_layout=True,
        squeeze=False,
    )
    pred_vmax = max(float(np.max(intensity_prediction)), 1.0e-12)
    ref_vmax = (
        max(float(np.max(intensity_reference)), 1.0e-12)
        if intensity_reference is not None
        else 1.0
    )

    for i, z_value in enumerate(z_values):
        axes[0, i].pcolormesh(
            x, y, intensity_prediction[i],
            shading="auto", vmin=0.0, vmax=pred_vmax,
        )
        axes[0, i].set_title(
            rf"PINN $z/\pi={z_value/np.pi:.1f}$"
        )
        if intensity_reference is not None:
            axes[1, i].pcolormesh(
                x, y, intensity_reference[i],
                shading="auto", vmin=0.0, vmax=ref_vmax,
            )
            axes[1, i].set_title(
                rf"SSFM $z/\pi={z_value/np.pi:.1f}$"
            )
            difference = np.abs(
                intensity_prediction[i] - intensity_reference[i]
            )
            axes[2, i].pcolormesh(
                x, y, difference,
                shading="auto",
            )
            axes[2, i].set_title(
                rf"abs diff, err={intensity_errors[i]:.2e}"
            )
        for row in range(rows):
            axes[row, i].set_aspect("equal")
            axes[row, i].set_xlabel("x")
            axes[row, i].set_ylabel("y")

    fig.suptitle(
        rf"Soft-IC analytical-prior PINN: $\nu={nu}$, $d={d}$, $w=0.02$"
    )
    figure_path = output_dir / "prediction_comparison.png"
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Training curves.
    with (output_dir / "history.csv").open(
        "w", encoding="utf-8", newline=""
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "stage", "step", "global_step", "total", "analytic",
                "ic", "bc", "pde", "lambda_analytic",
            ],
        )
        writer.writeheader()
        writer.writerows(history)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    global_steps = np.asarray(
        [h["global_step"] for h in history], dtype=np.int64
    )
    total_history = np.maximum(
        [h["total"] for h in history], 1.0e-30
    )
    analytic_history = np.maximum(
        [h["analytic"] for h in history], 1.0e-30
    )
    ic_history = np.maximum(
        [h["ic"] for h in history], 1.0e-30
    )
    # PDE is not evaluated during pretraining. NaN avoids displaying a
    # misleading artificial 1e-30 plateau before fine-tuning starts.
    pde_history = np.asarray(
        [
            np.nan if h["stage"] == "pretrain"
            else max(h["pde"], 1.0e-30)
            for h in history
        ],
        dtype=np.float64,
    )
    # Plot component losses first.
    ax.semilogy(
        global_steps,
        analytic_history,
        linestyle="--",
        linewidth=1.8,
        label=r"$L_{ana}$",
    )

    ax.semilogy(
        global_steps,
        ic_history,
        linestyle=":",
        linewidth=2.0,
        label=r"$L_{IC}$",
    )

    ax.semilogy(
        global_steps,
        pde_history,
        linestyle="-.",
        linewidth=1.8,
        label=r"$L_{PDE}$",
    )

    # Plot total loss last, so it will not be covered by PDE.
    ax.semilogy(
        global_steps,
        total_history,
        linestyle="-",
        linewidth=2.6,
        label="total",
        zorder=10,
    )

    if cfg.pretrain_steps > 0 and cfg.finetune_steps > 0:
        ax.axvline(
            cfg.pretrain_steps,
            linestyle="--",
            linewidth=1.2,
            label="PDE fine-tuning starts",
        )
    ax.set_xlabel("training iteration")
    ax.set_ylabel("loss")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "loss_history.png", dpi=180)
    plt.close(fig)

    # Soft-IC check on dense grid.
    xyz0 = np.column_stack(
        (X.ravel(), Y.ravel(), np.zeros(X.size))
    )
    pred0 = predict_chunks(model, xyz0, device, dtype)
    exact0 = analytic_field_numpy(xyz0, nu, d, G0, cfg)
    initial_condition_error = np.linalg.norm(
        pred0[:, 0] + 1j * pred0[:, 1] - exact0
    ) / np.linalg.norm(exact0)

    summary = {
        "tag": tag,
        "nu": nu,
        "d": d,
        "G0": G0,
        "raw_power": raw_power,
        "hard_initial_embedding": False,
        "soft_initial_constraint": True,
        "analytic_base_embedding": True,
        "initial_condition_relative_error": float(initial_condition_error),
        "soft_ic_relative_error": float(initial_condition_error),
        "hard_ic_relative_error": None,
        "training_loss": "Stage I: L_ana + L_IC + L_BC; Stage II: lambda_a(k)*L_ana + L_IC + L_BC + L_PDE",
        "network_parameterization": "psi_theta = psi_w0(x,y,z) + N_theta(x,y,z)",
        "training_uses_numerical_solution": False,
        "numerical_solution_used_only_for_validation": True,
        "sampling": {
            "analytic_total": cfg.n_analytic,
            "analytic_beam_fraction": cfg.analytic_beam_fraction,
            "pde_total": cfg.n_f,
            "pde_beam_fraction": cfg.pde_beam_fraction,
        },
        "powers": powers.tolist(),
        "max_intensities": maxima.tolist(),
        "collective_motion": {
            "centroid_region_radius": cfg.centroid_region_radius,
            "trajectory_relative_error": (
                collective["trajectory_relative_error"]
            ),
            "radius_pinn": collective["radius_pinn"].tolist(),
            "radius_reference": (
                None
                if collective["radius_reference"] is None
                else collective["radius_reference"].tolist()
            ),
            "radius_absolute_errors": (
                None
                if collective["radius_absolute_errors"] is None
                else collective["radius_absolute_errors"].tolist()
            ),
            "radius_relative_errors": (
                None
                if collective["radius_relative_errors"] is None
                else collective["radius_relative_errors"].tolist()
            ),
            "radius_error_normalized_by_d": (
                collective["radius_error_normalized_by_d"]
            ),
        },
        "field_relative_errors": (
            None if field_errors is None else field_errors.tolist()
        ),
        "intensity_relative_errors": (
            None if intensity_errors is None else intensity_errors.tolist()
        ),
        "training_seconds": elapsed,
        "pde_evaluation": {
            **pde_metrics,
            "beam_region_radius": cfg.beam_region_radius,
            "boundary_region_width": cfg.boundary_region_width,
            "metrics_csv": str(pde_eval_csv_path),
            "metrics_json": str(pde_eval_json_path),
            "pointwise_data": str(pde_eval_npz_path),
        },
        "config": asdict(cfg),
        "model": str(output_dir / "final_model.pt"),
        "figure": str(figure_path),
    }
    torch.save(
        {
            "model_state": model.state_dict(),
            "tag": tag,
            "nu": nu,
            "d": d,
            "G0": G0,
            "config": asdict(cfg),
        },
        output_dir / "final_model.pt",
    )
    with (output_dir / "summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print(
        f"[{tag}] done: soft IC err={initial_condition_error:.3e}, "
        f"P(z)={powers}, maxI={maxima}, time={elapsed:.1f}s"
    )
    return summary


def main():
    # 以后所有训练参数都只改 Config，不再用命令行 parser 覆盖
    cfg = Config()
    # 只跑 v=1.0, d=3.23 这一组
    tags = ("v1.2_d2.94",)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    summaries = []

    print(f"device={device}; soft IC + analytic prior + beam-enhanced LHS")
    print(f"running cases = {tags}")
    print(f"pretrain_steps = {cfg.pretrain_steps}")
    print(f"finetune_steps = {cfg.finetune_steps}")
    print(f"n_analytic = {cfg.n_analytic}")
    print(f"n_f = {cfg.n_f}")
    print(f"prediction_grid = {cfg.prediction_grid}")

    for tag in tags:
        summaries.append(train_case(tag, cfg, device))

    with (SCRIPT_DIR / "run_summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(summaries, file, ensure_ascii=False, indent=2)

# def main():
#     # 五组随机种子
#     seeds = (
#         20260626,
#         20260627,
#         20260628,
#         20260629,
#         20260630,
#     )
#
#     # 先只测试主 case
#     tags = ("v1.0_d3.23",)
#
#     device = torch.device(
#         "cuda" if torch.cuda.is_available() else "cpu"
#     )
#
#     all_summaries = []
#
#     print(f"device={device}; soft IC + analytic prior + beam-enhanced LHS")
#     print(f"running cases = {tags}")
#     print(f"running seeds = {seeds}")
#
#     for seed in seeds:
#         cfg = Config()
#         cfg.seed = int(seed)
#
#         print("\n" + "=" * 80)
#         print(f"Starting seed = {cfg.seed}")
#         print(f"pretrain_steps = {cfg.pretrain_steps}")
#         print(f"finetune_steps = {cfg.finetune_steps}")
#         print(f"n_analytic = {cfg.n_analytic}")
#         print(f"n_f = {cfg.n_f}")
#         print(f"prediction_grid = {cfg.prediction_grid}")
#         print("=" * 80)
#
#         for tag in tags:
#             summary = train_case(tag, cfg, device)
#             all_summaries.append(summary)
#
#     with (SCRIPT_DIR / "run_summary_all_seeds.json").open(
#         "w", encoding="utf-8"
#     ) as file:
#         json.dump(all_summaries, file, ensure_ascii=False, indent=2)
#

if __name__ == "__main__":
    main()
