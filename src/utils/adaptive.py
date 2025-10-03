"""自适应调度模块 | Adaptive scheduling utilities.

该模块封装基于 `env_positive_dx_ratio` 的距离奖励权重与策略熵系数( entropy_beta )双通道自适应逻辑，
支持窗口平均 (SMA) 与指数滑动平均 (EMA) 去抖。独立成类便于单元测试覆盖。

使用流程：
1. 在训练初始化阶段创建 AdaptiveScheduler，传入初始 distance_weight / entropy_beta 与阈值上限下限。
2. 每次得到最新的正向位移占比 ratio 后调用 `step(ratio)`，返回 (maybe_new_distance_weight, maybe_new_entropy_beta, metrics_dict)。
3. 训练脚本据返回值选择是否写回 wrapper 的 RewardConfig 与 optimizer.beta_entropy。

调度策略：
    ratio < low_threshold  : 推向上限 (max)
    ratio > high_threshold : 推向下限 (min)
    其余区间              : 维持（仅轻微数值漂移，由 lr 控制）

EMA 系数：若未显式提供 alpha，则按 2/(window+1) 计算（经典滑动平均对齐周期）。

线程安全：该类内部无锁，不适用于多线程同时写。训练主循环单线程调用即可。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class AdaptiveConfig:
    window: int = 50
    # Distance weight bounds & thresholds
    dw_max: Optional[float] = None
    dw_min: Optional[float] = None
    dw_low: float = 0.05
    dw_high: float = 0.30
    dw_lr: float = 0.05  # interpolation step
    # Entropy beta bounds & thresholds
    ent_max: Optional[float] = None
    ent_min: Optional[float] = None
    ent_low: float = 0.05
    ent_high: float = 0.30
    ent_lr: float = 0.10
    # EMA smoothing
    ema_alpha: Optional[float] = None  # if None => 2/(window+1)


class AdaptiveScheduler:
    """调度器主类。

    注意：只有在 *同时* 设置 max / min 时某通道才启用。
    """

    def __init__(
        self, cfg: AdaptiveConfig, base_distance_weight: float, base_entropy_beta: float
    ) -> None:
        self.cfg = cfg
        self._ratios: deque[float] = deque(maxlen=max(5, cfg.window))
        self._ema: Optional[float] = None
        self._distance_weight = base_distance_weight
        self._entropy_beta = base_entropy_beta
        if self.cfg.ema_alpha is None:
            self._alpha = 2.0 / (cfg.window + 1.0)
        else:
            self._alpha = self.cfg.ema_alpha

    def enabled(self) -> bool:
        return (
            self.cfg.dw_max is not None
            and self.cfg.dw_min is not None
            and self.cfg.dw_max > self.cfg.dw_min
        ) or (
            self.cfg.ent_max is not None
            and self.cfg.ent_min is not None
            and self.cfg.ent_max > self.cfg.ent_min
        )

    def step(
        self, ratio: float
    ) -> tuple[Optional[float], Optional[float], Dict[str, Any]]:
        """提交一次新的推进占比并返回可能更新后的参数与指标。

        Returns:
            (new_distance_weight or None, new_entropy_beta or None, metrics)
        """
        if not self.enabled():
            return None, None, {}
        r = max(0.0, float(ratio))
        self._ratios.append(r)
        if self._ema is None:
            self._ema = r
        else:
            self._ema = self._alpha * r + (1 - self._alpha) * self._ema
        avg_ratio = sum(self._ratios) / len(self._ratios)

        new_dw: Optional[float] = None
        new_ent: Optional[float] = None

        # Distance weight adaptive
        if (
            self.cfg.dw_max is not None
            and self.cfg.dw_min is not None
            and self.cfg.dw_max > self.cfg.dw_min
        ):
            target = self._distance_weight
            if avg_ratio < self.cfg.dw_low:
                target = (
                    self._distance_weight
                    + (self.cfg.dw_max - self._distance_weight) * self.cfg.dw_lr
                )
            elif avg_ratio > self.cfg.dw_high:
                target = (
                    self._distance_weight
                    + (self.cfg.dw_min - self._distance_weight) * self.cfg.dw_lr
                )
            # clamp
            target = max(self.cfg.dw_min, min(self.cfg.dw_max, target))
            if abs(target - self._distance_weight) > 1e-12:
                self._distance_weight = target
                new_dw = target

        # Entropy beta adaptive
        if (
            self.cfg.ent_max is not None
            and self.cfg.ent_min is not None
            and self.cfg.ent_max > self.cfg.ent_min
        ):
            ent_t = self._entropy_beta
            if avg_ratio < self.cfg.ent_low:
                ent_t = (
                    self._entropy_beta
                    + (self.cfg.ent_max - self._entropy_beta) * self.cfg.ent_lr
                )
            elif avg_ratio > self.cfg.ent_high:
                ent_t = (
                    self._entropy_beta
                    + (self.cfg.ent_min - self._entropy_beta) * self.cfg.ent_lr
                )
            ent_t = max(self.cfg.ent_min, min(self.cfg.ent_max, ent_t))
            if abs(ent_t - self._entropy_beta) > 1e-12:
                self._entropy_beta = ent_t
                new_ent = ent_t

        metrics = {
            "adaptive_ratio_avg": round(avg_ratio, 4),
            "adaptive_ratio_ema": round(self._ema, 4),
        }
        if new_dw is not None:
            metrics["adaptive_distance_weight"] = new_dw
        if new_ent is not None:
            metrics["adaptive_entropy_beta"] = new_ent
        return new_dw, new_ent, metrics

    @property
    def distance_weight(self) -> float:
        return self._distance_weight

    @property
    def entropy_beta(self) -> float:
        return self._entropy_beta


__all__ = ["AdaptiveConfig", "AdaptiveScheduler"]
