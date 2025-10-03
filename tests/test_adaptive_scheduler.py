"""AdaptiveScheduler 单元测试

验证核心行为：
1. ratio 持续低于 low 阈值 => distance_weight / entropy_beta 分别向 max 方向单调逼近
2. ratio 持续高于 high 阈值 => 分别向 min 方向单调逼近
3. ratio 在区间内震荡 => 不再发生显著更新
4. EMA 与 AVG 的方向性一致
"""
from __future__ import annotations

import math

from src.utils.adaptive import AdaptiveConfig, AdaptiveScheduler


def _approx(a: float, b: float, eps: float = 1e-6) -> bool:
    return abs(a - b) < eps


def test_adaptive_low_phase():
    cfg = AdaptiveConfig(
        window=10,
        dw_max=0.05,
        dw_min=0.005,
        dw_low=0.2,
        dw_high=0.4,
        dw_lr=0.5,
        ent_max=0.02,
        ent_min=0.002,
        ent_low=0.2,
        ent_high=0.4,
        ent_lr=0.5,
    )
    sched = AdaptiveScheduler(
        cfg, base_distance_weight=0.01, base_entropy_beta=0.005, base_lr_scale=1.0
    )

    updates_dw = []
    updates_ent = []
    for _ in range(12):
        ndw, ne, _, metrics = sched.step(0.0)  # 极低 ratio
        if ndw is not None:
            updates_dw.append(ndw)
        if ne is not None:
            updates_ent.append(ne)
        # 单调性断言
        if len(updates_dw) >= 2:
            assert updates_dw[-1] >= updates_dw[-2]
        if len(updates_ent) >= 2:
            assert updates_ent[-1] >= updates_ent[-2]
        assert metrics["adaptive_ratio_avg"] <= 0.01

    # 接近上限
    assert updates_dw[-1] <= cfg.dw_max + 1e-9
    assert updates_ent[-1] <= cfg.ent_max + 1e-9


def test_adaptive_high_phase():
    cfg = AdaptiveConfig(
        window=5,
        dw_max=0.05,
        dw_min=0.005,
        dw_low=0.2,
        dw_high=0.4,
        dw_lr=0.5,
        ent_max=0.02,
        ent_min=0.002,
        ent_low=0.2,
        ent_high=0.4,
        ent_lr=0.5,
    )
    sched = AdaptiveScheduler(
        cfg, base_distance_weight=0.04, base_entropy_beta=0.015, base_lr_scale=1.0
    )

    for _ in range(10):
        sched.step(0.9)  # 极高 ratio

    assert math.isclose(sched.distance_weight, cfg.dw_min, rel_tol=0.01, abs_tol=1e-6)
    assert math.isclose(sched.entropy_beta, cfg.ent_min, rel_tol=0.01, abs_tol=1e-6)


def test_adaptive_stable_band():
    cfg = AdaptiveConfig(
        window=8,
        dw_max=0.05,
        dw_min=0.005,
        dw_low=0.2,
        dw_high=0.4,
        dw_lr=0.5,
        ent_max=0.02,
        ent_min=0.002,
        ent_low=0.2,
        ent_high=0.4,
        ent_lr=0.5,
    )
    base_dw = 0.02
    base_ent = 0.01
    sched = AdaptiveScheduler(
        cfg,
        base_distance_weight=base_dw,
        base_entropy_beta=base_ent,
        base_lr_scale=1.0,
    )

    # 先几步进入稳态（中间带）
    for _ in range(20):
        sched.step(0.3)  # 中间区间
    # 不应远离初始值太多
    assert abs(sched.distance_weight - base_dw) < 0.01
    assert abs(sched.entropy_beta - base_ent) < 0.01


def test_adaptive_ema_direction_consistency():
    cfg = AdaptiveConfig(
        window=4,
        dw_max=0.05,
        dw_min=0.005,
        dw_low=0.2,
        dw_high=0.4,
        dw_lr=0.5,
        ent_max=0.02,
        ent_min=0.002,
        ent_low=0.2,
        ent_high=0.4,
        ent_lr=0.5,
    )
    sched = AdaptiveScheduler(cfg, base_distance_weight=0.01, base_entropy_beta=0.005)

    ema_values = []
    avg_values = []
    for _ in range(15):
        _, _, _, metrics = sched.step(0.0)  # 推向上限
        ema_values.append(metrics["adaptive_ratio_ema"])
        avg_values.append(metrics["adaptive_ratio_avg"])

    # 两者都应保持非增（因为 ratio=0 固定）且接近 0
    assert all(ema_values[i] >= ema_values[i + 1] - 1e-9 for i in range(len(ema_values) - 1))
    assert all(avg_values[i] >= avg_values[i + 1] - 1e-9 for i in range(len(avg_values) - 1))
    assert ema_values[-1] < 0.05
    assert avg_values[-1] < 0.05


def test_adaptive_lr_scale_low_ratio_increase():
    cfg = AdaptiveConfig(
        window=4,
        dw_max=0.05,
        dw_min=0.005,
        dw_low=0.2,
        dw_high=0.4,
        dw_lr=0.5,
        lr_scale_max=1.5,
        lr_scale_min=0.5,
        lr_lr=0.5,
    )
    sched = AdaptiveScheduler(
        cfg, base_distance_weight=0.02, base_entropy_beta=0.01, base_lr_scale=1.0
    )

    scales = []
    for _ in range(6):
        _, _, new_scale, metrics = sched.step(0.0)
        if new_scale is not None:
            scales.append(new_scale)
        if "adaptive_lr_scale" in metrics:
            scales.append(metrics["adaptive_lr_scale"])

    assert scales  # should have updates
    assert all(sc >= 1.0 for sc in scales)
    assert sched.lr_scale <= cfg.lr_scale_max + 1e-6
