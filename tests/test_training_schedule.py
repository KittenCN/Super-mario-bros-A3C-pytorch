import dataclasses

import torch

from train import build_training_config, compute_returns, parse_args


def test_compute_returns_gae_uses_bootstrap_value():
    args = parse_args([])
    args.total_updates = 10
    cfg = build_training_config(args)
    cfg.rollout = dataclasses.replace(cfg.rollout, use_vtrace=False, gamma=0.9, tau=0.95)

    rewards = torch.tensor([[1.0], [0.0], [2.0]], dtype=torch.float32)
    values = torch.tensor([[0.2], [0.3], [0.4]], dtype=torch.float32)
    dones = torch.tensor([[0.0], [0.0], [1.0]], dtype=torch.float32)
    bootstrap = torch.tensor([0.7], dtype=torch.float32)
    behaviour = torch.zeros_like(rewards)
    target = torch.zeros_like(rewards)

    vs, advantages = compute_returns(
        cfg,
        behaviour_log_probs=behaviour,
        target_log_probs=target,
        rewards=rewards,
        values=values,
        bootstrap_value=bootstrap,
        dones=dones,
    )

    expected_adv = torch.tensor([2.29093981, 1.4280000, 1.6000000]).unsqueeze(-1)
    expected_vs = torch.tensor([2.49093986, 1.7279999, 2.0]).unsqueeze(-1)

    assert torch.allclose(advantages, expected_adv, atol=1e-6)
    assert torch.allclose(vs, expected_vs, atol=1e-6)


def test_scheduler_total_steps_align_with_updates():
    args = parse_args([])
    args.total_updates = 1234
    args.grad_accum = 4
    cfg = build_training_config(args)

    expected_steps = (args.total_updates + args.grad_accum - 1) // args.grad_accum
    assert cfg.scheduler.total_steps == expected_steps
    assert 1 <= cfg.scheduler.warmup_steps <= expected_steps
