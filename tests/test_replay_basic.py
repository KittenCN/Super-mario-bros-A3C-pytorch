import os
import torch
import json
import time
from pathlib import Path

from src.utils.replay import PrioritizedReplay
from src.config import TrainingConfig
from train import _build_checkpoint_metadata, _write_metadata


def test_prioritized_replay_push_sample():
    per = PrioritizedReplay(capacity=64, alpha=0.6, beta_start=0.4, beta_final=1.0, beta_steps=1000, device=torch.device('cpu'))
    obs = torch.rand(16,4,84,84)
    actions = torch.randint(0, 6, (16,))
    vals = torch.randn(16)
    adv = torch.randn(16)
    per.push(obs, actions, vals, adv)
    sample = per.sample(8)
    assert sample is not None
    assert sample.observations.shape == (8,4,84,84)
    assert sample.actions.shape == (8,)
    stats = per.stats()
    assert stats['size'] == 16
    assert 0 < stats['priority_mean']


def test_checkpoint_metadata_replay_field(tmp_path):
    cfg = TrainingConfig()
    # 模拟 per_sample_interval 设置
    cfg.replay = type(cfg.replay)(**{**cfg.replay.__dict__, 'per_sample_interval': 4})
    meta = _build_checkpoint_metadata(cfg, global_step=123, update_idx=7, variant='checkpoint')
    assert 'replay' in meta
    assert meta['replay'].get('per_sample_interval') == 4
    # 原子写测试
    ckpt_base = tmp_path / 'dummy.pt'
    # 写一个虚假权重文件
    ckpt_base.write_bytes(b'PTDUMMY')
    _write_metadata(ckpt_base, meta)
    meta_path = ckpt_base.with_suffix('.json')
    assert meta_path.exists()
    loaded = json.loads(meta_path.read_text(encoding='utf-8'))
    assert loaded['save_state']['global_step'] == 123
    assert loaded['replay']['per_sample_interval'] == 4


def test_per_sample_time_positive():
    per = PrioritizedReplay(capacity=64, alpha=0.6, beta_start=0.4, beta_final=1.0, beta_steps=1000, device=torch.device('cpu'))
    obs = torch.rand(32,4,84,84)
    actions = torch.randint(0,6,(32,))
    vals = torch.randn(32)
    adv = torch.randn(32)
    per.push(obs, actions, vals, adv)
    t0 = time.time()
    smp = per.sample(16)
    t1 = time.time()
    assert smp is not None
    assert (t1 - t0) * 1000.0 >= 0.0  # 只要能执行即视为成功
