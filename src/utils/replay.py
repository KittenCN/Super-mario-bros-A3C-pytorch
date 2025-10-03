"""Prioritized experience replay buffer (内存优化版).

改进点 / Improvements:
1. 预分配环形缓冲，避免 Python list + dict 大量碎片化。(原实现: list[dict])
2. 压缩观测为 uint8 (0~255) 存储，采样时再解码为 float32/255：占用约 1/4。
3. 首次 push 时推断 (C,H,W)，惰性分配；支持按环境变量或阈值自动下调容量。
4. 提供环境变量控制：
    - MARIO_PER_COMPRESS=0 禁用 uint8 压缩 (改用 float32)
    - MARIO_PER_MAX_MEM_GB=数值 约束观测缓存目标上限 GiB（默认 2.0）
5. 计算并打印估算内存，给出调参建议。

注：如需继续旧实现，可回退到早期版本或设 capacity 很小再禁用 PER。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch


@dataclass
class ReplaySample:
    observations: torch.Tensor
    actions: torch.Tensor
    target_values: torch.Tensor
    advantages: torch.Tensor
    weights: torch.Tensor
    indices: np.ndarray


class PrioritizedReplay:
    """内存友好的 PER 环形缓冲.

    Push: 批量写入 (B,C,H,W) 观测及 (B,) 动作 / 价值 / 优势。
    Sample: 重要性抽样 + 权重。
    Observations 在 CPU 上以 uint8 (压缩) 或 float32 (禁用压缩) 存储。
    """

    def __init__(
        self,
        capacity: int,
        alpha: float,
        beta_start: float,
        beta_final: float,
        beta_steps: int,
        device: torch.device,
    ) -> None:
        self.requested_capacity = capacity
        self.capacity = capacity  # 可被自适应调整
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_final = beta_final
        self.beta_steps = beta_steps
        self.device = device

        # 压缩控制
        self._compress = os.environ.get("MARIO_PER_COMPRESS", "1").lower() not in {
            "0",
            "false",
            "off",
        }
        self._max_mem_gb = float(
            os.environ.get("MARIO_PER_MAX_MEM_GB", "2.0")
        )  # 仅用于观测主体估算

        # 惰性分配的张量容器
        self.obs_storage: Optional[torch.Tensor] = (
            None  # uint8 或 float32: (capacity,C,H,W)
        )
        self.actions: Optional[torch.Tensor] = None  # long (capacity,)
        self.target_values: Optional[torch.Tensor] = None  # float32 (capacity,)
        self.advantages: Optional[torch.Tensor] = None  # float32 (capacity,)
        self.priorities = np.zeros((self.capacity,), dtype=np.float32)
        self.pos = 0
        self.size = 0
        self.step = 0
        # 统计数据 (采样批次、唯一率、推入次数)
        self.sample_calls = 0
        self.sample_size_accum = 0
        self.last_sample_unique_ratio = 0.0
        self.unique_ratio_accum = 0.0
        self.push_total = 0
        # 配置：是否用 fp16 存储标量（advantages / target_values）节省内存
        self._fp16_scalars = os.environ.get(
            "MARIO_PER_FP16_SCALARS", "1"
        ).lower() not in {"0", "false", "off"}

    def stats(self) -> dict:
        """返回环形缓冲统计信息 + 优先级分布指标。

        注意：优先级分位数在样本数较大时会涉及一次 `np.percentile`，开销 O(n log n)，
        日志周期通常较低（例如每 100 updates）可接受；如需进一步优化可引入随机抽样估计。
        """
        fill_rate = (
            float(self.size) / float(self.capacity) if self.capacity > 0 else 0.0
        )
        avg_batch = (
            (self.sample_size_accum / self.sample_calls)
            if self.sample_calls > 0
            else 0.0
        )
        avg_unique = (
            (self.unique_ratio_accum / self.sample_calls)
            if self.sample_calls > 0
            else 0.0
        )
        prio_view = self.priorities[: self.size]
        if self.size > 0:
            try:
                prio_mean = float(prio_view.mean())
                prio_p50 = float(np.percentile(prio_view, 50))
                prio_p90 = float(np.percentile(prio_view, 90))
                prio_p99 = float(np.percentile(prio_view, 99))
            except Exception:
                prio_mean = prio_p50 = prio_p90 = prio_p99 = 0.0
        else:
            prio_mean = prio_p50 = prio_p90 = prio_p99 = 0.0
        return {
            "capacity": int(self.capacity),
            "size": int(self.size),
            "fill_rate": fill_rate,
            "avg_sample_batch": float(avg_batch),
            "last_sample_unique_ratio": float(self.last_sample_unique_ratio),
            "avg_sample_unique_ratio": float(avg_unique),
            "push_total": int(self.push_total),
            "priority_mean": prio_mean,
            "priority_p50": prio_p50,
            "priority_p90": prio_p90,
            "priority_p99": prio_p99,
        }

    # ----------------- 内部工具 -----------------
    def _maybe_alloc(self, observations: torch.Tensor):
        if self.obs_storage is not None:
            return
        if observations.dim() != 4:
            raise ValueError("Expected observations shape (B,C,H,W)")
        _, c, h, w = observations.shape
        # 估算每条样本的字节数（仅观测部分）
        obs_bytes_per = c * h * w * (1 if self._compress else 4)
        total_bytes = obs_bytes_per * self.capacity
        total_gb = total_bytes / (1024**3)
        # 自适应容量（只对观测主体约束，额外标量忽略）
        if total_gb > self._max_mem_gb:
            # 计算新的容量 (向下取整 >= batch size 或 1024)
            new_cap = int(self._max_mem_gb * (1024**3) // obs_bytes_per)
            min_cap = max(1024, observations.shape[0] * 2)
            if new_cap < min_cap:
                new_cap = min_cap
            if new_cap < self.capacity:
                print(
                    f"[replay][info] shrink capacity {self.capacity} -> {new_cap} due to mem cap {self._max_mem_gb} GiB (requested {self.requested_capacity})"
                )
                self.capacity = new_cap
                self.priorities = np.zeros((self.capacity,), dtype=np.float32)

        obs_dtype = torch.uint8 if self._compress else torch.float32
        self.obs_storage = torch.empty(
            (self.capacity, c, h, w), dtype=obs_dtype, device=torch.device("cpu")
        )
        self.actions = torch.empty(
            (self.capacity,), dtype=torch.long, device=torch.device("cpu")
        )
        scalar_dtype = torch.float16 if self._fp16_scalars else torch.float32
        self.target_values = torch.empty(
            (self.capacity,), dtype=scalar_dtype, device=torch.device("cpu")
        )
        self.advantages = torch.empty(
            (self.capacity,), dtype=scalar_dtype, device=torch.device("cpu")
        )
        eff_bytes = self.capacity * c * h * w * (1 if self._compress else 4)
        eff_gb = eff_bytes / (1024**3)
        mode = "uint8" if self._compress else "float32"
        print(
            f"[replay] allocated obs buffer capacity={self.capacity} shape=({c},{h},{w}) dtype={mode} ~{eff_gb:.2f} GiB (requested={self.requested_capacity})"
        )
        if self._compress:
            print(
                "[replay][hint] set MARIO_PER_COMPRESS=0 可禁用压缩；MARIO_PER_MAX_MEM_GB 调整内存上限。"
            )
        if self._fp16_scalars:
            print(
                "[replay][info] using FP16 storage for advantages/target_values (MARIO_PER_FP16_SCALARS=0 可关闭)"
            )

    # ----------------- 公共 API -----------------
    def __len__(self) -> int:  # pragma: no cover
        return self.size

    def beta(self) -> float:
        fraction = min(1.0, self.step / float(max(1, self.beta_steps)))
        return self.beta_start + fraction * (self.beta_final - self.beta_start)

    def push(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        target_values: torch.Tensor,
        advantages: torch.Tensor,
        priorities: Optional[torch.Tensor] = None,
    ) -> None:
        """批量写入.

        期望 shape:
          observations: (B,C,H,W) float32 in [0,1]
          actions/target_values/advantages: 兼容 (B,) 或 (B,1)
        """
        with torch.no_grad():
            if observations.dim() != 4:
                raise ValueError("observations must be (B,C,H,W)")
            self._maybe_alloc(observations)
            B = observations.shape[0]
            act = actions.view(B)
            tgt = target_values.view(B)
            adv = advantages.view(B)
            if priorities is None:
                priorities = adv.abs()
            prio = priorities.detach().cpu().view(B).numpy()
            # 压缩
            if self._compress:
                obs_comp = (
                    observations.clamp(0, 1).mul(255.0).to(torch.uint8).cpu()
                )
            else:
                obs_comp = observations.detach().cpu()
            act_cpu = act.detach().cpu()
            tgt_cpu = tgt.detach().cpu().to(self.target_values.dtype)
            adv_cpu = adv.detach().cpu().to(self.advantages.dtype)
            if B == 0:
                return
            indices = (torch.arange(B, dtype=torch.long) + self.pos) % self.capacity
            indices = indices.to(dtype=torch.long)
            self.obs_storage.index_copy_(0, indices, obs_comp)  # type: ignore[arg-type]
            self.actions.index_copy_(0, indices, act_cpu)  # type: ignore[arg-type]
            self.target_values.index_copy_(0, indices, tgt_cpu)  # type: ignore[arg-type]
            self.advantages.index_copy_(0, indices, adv_cpu)  # type: ignore[arg-type]
            indices_np = indices.cpu().numpy()
            self.priorities[indices_np] = prio.astype(np.float32) + 1e-5
            self.pos = int((int(self.pos) + B) % self.capacity)
            self.size = min(self.size + B, self.capacity)
            self.push_total += B

    def sample(self, batch_size: int) -> Optional[ReplaySample]:
        if self.size < batch_size or self.obs_storage is None:
            return None
        valid = self.size
        priorities = self.priorities[:valid]
        scaled = priorities**self.alpha
        denom = scaled.sum()
        if denom <= 0:
            probs = np.full(valid, 1.0 / valid, dtype=np.float32)
        else:
            probs = scaled / denom
        indices = np.random.choice(valid, batch_size, p=probs)
        beta = self.beta()
        weights = (valid * probs[indices]) ** (-beta)
        weights = weights / weights.max()
        obs_raw = self.obs_storage[indices]  # type: ignore[index]
        if self._compress:
            obs = obs_raw.float().div_(255.0).to(self.device)
        else:
            obs = obs_raw.to(self.device)
        actions = self.actions[indices].to(self.device)  # type: ignore[index]
        target_values = self.target_values[indices].to(self.device).to(torch.float32)  # type: ignore[index]
        advantages = self.advantages[indices].to(self.device).to(torch.float32)  # type: ignore[index]
        weights_t = torch.tensor(weights, device=self.device, dtype=torch.float32)
        self.step += 1
        unique = len(set(int(x) for x in indices.tolist()))
        self.sample_calls += 1
        self.sample_size_accum += batch_size
        self.last_sample_unique_ratio = unique / float(batch_size)
        self.unique_ratio_accum += self.last_sample_unique_ratio
        return ReplaySample(obs, actions, target_values, advantages, weights_t, indices)

    def sample_detailed(
        self, batch_size: int
    ) -> tuple[Optional[ReplaySample], dict[str, float]]:
        """与 sample 类似，但返回细分阶段耗时 (ms)。

        分段说明:
        - prior: 计算 scaled priorities / 概率分布
        - choice: 执行 np.random.choice 抽样
        - weight: 计算重要性权重并归一化
        - decode: 取出原始观测并解压缩/搬运到目标设备
        - tensor: 构造 actions/values/advantages/weights 张量
        """
        timings: dict[str, float] = {}
        if self.size < batch_size or self.obs_storage is None:
            return None, timings
        import time as _time

        t0 = _time.time()
        valid = self.size
        p_start = _time.time()
        priorities = self.priorities[:valid]
        scaled = priorities**self.alpha
        denom = scaled.sum()
        if denom <= 0:
            probs = np.full(valid, 1.0 / valid, dtype=np.float32)
        else:
            probs = scaled / denom
        timings["prior"] = (_time.time() - p_start) * 1000.0
        c_start = _time.time()
        indices = np.random.choice(valid, batch_size, p=probs)
        timings["choice"] = (_time.time() - c_start) * 1000.0
        w_start = _time.time()
        beta = self.beta()
        weights = (valid * probs[indices]) ** (-beta)
        weights = weights / weights.max()
        timings["weight"] = (_time.time() - w_start) * 1000.0
        d_start = _time.time()
        obs_raw = self.obs_storage[indices]  # type: ignore[index]
        if self._compress:
            obs = obs_raw.float().div_(255.0).to(self.device)
        else:
            obs = obs_raw.to(self.device)
        timings["decode"] = (_time.time() - d_start) * 1000.0
        t_start = _time.time()
        actions = self.actions[indices].to(self.device)  # type: ignore[index]
        target_values = self.target_values[indices].to(self.device).to(torch.float32)  # type: ignore[index]
        advantages = self.advantages[indices].to(self.device).to(torch.float32)  # type: ignore[index]
        weights_t = torch.tensor(weights, device=self.device, dtype=torch.float32)
        timings["tensor"] = (_time.time() - t_start) * 1000.0
        timings["total"] = (_time.time() - t0) * 1000.0
        # 统计更新（与 sample 一致）
        self.step += 1
        unique = len(set(int(x) for x in indices.tolist()))
        self.sample_calls += 1
        self.sample_size_accum += batch_size
        self.last_sample_unique_ratio = unique / float(batch_size)
        self.unique_ratio_accum += self.last_sample_unique_ratio
        sample = ReplaySample(
            obs, actions, target_values, advantages, weights_t, indices
        )
        return sample, timings

    def update_priorities(self, indices: np.ndarray, priorities: torch.Tensor):
        np_prior = priorities.detach().cpu().numpy().flatten()
        for i, idx in enumerate(indices):
            if idx < self.capacity:
                self.priorities[idx] = (
                    float(np_prior[min(i, np_prior.shape[0] - 1)]) + 1e-5
                )
        # 防止全部衰减为 0
        if np.all(self.priorities[: self.size] <= 0):
            self.priorities[: self.size] += 1e-5
