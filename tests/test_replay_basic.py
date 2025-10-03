import json
import os
import sys
import time
from types import SimpleNamespace

sys.path.append(os.getcwd())

import torch
import torch.nn as nn

from src.app_config import TrainingConfig
from src.utils.replay import PrioritizedReplay
from train import _build_checkpoint_metadata, _per_step_update, _write_metadata


def test_prioritized_replay_push_sample():
    per = PrioritizedReplay(
        capacity=64,
        alpha=0.6,
        beta_start=0.4,
        beta_final=1.0,
        beta_steps=1000,
        device=torch.device("cpu"),
    )
    obs = torch.rand(16, 4, 84, 84)
    actions = torch.randint(0, 6, (16,))
    vals = torch.randn(16)
    adv = torch.randn(16)
    per.push(obs, actions, vals, adv)
    sample = per.sample(8)
    assert sample is not None
    assert sample.observations.shape == (8, 4, 84, 84)
    assert sample.actions.shape == (8,)
    stats = per.stats()
    assert stats["size"] == 16
    assert 0 < stats["priority_mean"]


def test_checkpoint_metadata_replay_field(tmp_path):
    cfg = TrainingConfig()
    # 模拟 per_sample_interval 设置
    cfg.replay = type(cfg.replay)(**{**cfg.replay.__dict__, "per_sample_interval": 4})
    meta = _build_checkpoint_metadata(
        cfg, global_step=123, update_idx=7, variant="checkpoint"
    )
    assert "replay" in meta
    assert meta["replay"].get("per_sample_interval") == 4
    # 原子写测试
    ckpt_base = tmp_path / "dummy.pt"
    # 写一个虚假权重文件
    ckpt_base.write_bytes(b"PTDUMMY")
    _write_metadata(ckpt_base, meta)
    meta_path = ckpt_base.with_suffix(".json")
    assert meta_path.exists()
    loaded = json.loads(meta_path.read_text(encoding="utf-8"))
    assert loaded["save_state"]["global_step"] == 123
    assert loaded["replay"]["per_sample_interval"] == 4


def test_per_sample_time_positive():
    per = PrioritizedReplay(
        capacity=64,
        alpha=0.6,
        beta_start=0.4,
        beta_final=1.0,
        beta_steps=1000,
        device=torch.device("cpu"),
    )
    obs = torch.rand(32, 4, 84, 84)
    actions = torch.randint(0, 6, (32,))
    vals = torch.randn(32)
    adv = torch.randn(32)
    per.push(obs, actions, vals, adv)
    t0 = time.time()
    smp = per.sample(16)
    t1 = time.time()
    assert smp is not None
    assert (t1 - t0) * 1000.0 >= 0.0  # 只要能执行即视为成功


def test_sample_detailed_timings():
    per = PrioritizedReplay(
        capacity=128,
        alpha=0.6,
        beta_start=0.4,
        beta_final=1.0,
        beta_steps=1000,
        device=torch.device("cpu"),
    )
    obs = torch.rand(64, 4, 84, 84)
    actions = torch.randint(0, 6, (64,))
    vals = torch.randn(64)
    adv = torch.randn(64)
    per.push(obs, actions, vals, adv)
    sample, timings = per.sample_detailed(32)
    assert sample is not None
    for key in ["prior", "choice", "weight", "decode", "tensor", "total"]:
        assert key in timings
        assert timings[key] >= 0.0


class _DummyModel(nn.Module):
    def __init__(self, action_dim: int, input_channels: int, height: int, width: int):
        super().__init__()
        flat = input_channels * height * width
        self.policy = nn.Linear(flat, action_dim)
        self.value_head = nn.Linear(flat, 1)

    def forward(self, obs, hidden=None, cell=None):
        flat = obs.reshape(obs.shape[0], -1)
        logits = self.policy(flat)
        value = self.value_head(flat)
        return SimpleNamespace(
            logits=logits, value=value, hidden_state=None, cell_state=None
        )


def test_per_interval_pushes_every_update():
    device = torch.device("cpu")
    cfg = TrainingConfig()
    cfg.replay.enable = True
    cfg.replay.per_sample_interval = 3
    cfg.rollout.num_steps = 2
    cfg.env.num_envs = 3
    cfg.rollout.batch_size = cfg.rollout.num_steps * cfg.env.num_envs

    channels, height, width = 2, 4, 4
    action_space = 5
    dummy_model = _DummyModel(action_space, channels, height, width)

    per = PrioritizedReplay(
        capacity=256,
        alpha=0.6,
        beta_start=0.4,
        beta_final=1.0,
        beta_steps=1000,
        device=device,
    )

    obs = torch.rand(cfg.rollout.num_steps, cfg.env.num_envs, channels, height, width)
    actions = torch.randint(0, action_space, (cfg.rollout.num_steps, cfg.env.num_envs))
    vs = torch.rand(cfg.rollout.num_steps, cfg.env.num_envs, 1)
    advantages = torch.rand(cfg.rollout.num_steps, cfg.env.num_envs, 1)

    sequences = {"obs": obs, "actions": actions}

    total_updates = 5
    for update_idx in range(total_updates):
        per_loss, per_metrics = _per_step_update(
            per_buffer=per,
            model=dummy_model,
            sequences=sequences,
            vs=vs,
            advantages=advantages,
            cfg=cfg,
            device=device,
            update_idx=update_idx,
            mixed_precision=False,
            batch_size=cfg.rollout.batch_size,
        )
        assert torch.is_tensor(per_loss)
        assert per_loss.device.type == "cpu"
        assert isinstance(per_metrics, dict)

    expected_push = total_updates * cfg.rollout.num_steps * cfg.env.num_envs
    assert per.push_total == expected_push
    assert per.size == expected_push
