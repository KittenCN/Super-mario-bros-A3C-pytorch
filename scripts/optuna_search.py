"""Optuna hyperparameter search for Mario A3C."""

from __future__ import annotations

import argparse
import dataclasses

import optuna

import train


def build_base_args() -> argparse.Namespace:
    args = train.parse_args([])
    args.total_updates = 500
    args.rollout_steps = 32
    args.num_envs = 4
    args.eval_interval = 0
    args.checkpoint_interval = 0
    args.log_interval = 50
    args.no_compile = True
    args.no_amp = True
    args.grad_accum = 1
    args.per = False
    return args


def objective(trial: optuna.Trial) -> float:
    args = build_base_args()
    args.lr = trial.suggest_float("lr", 1e-5, 5e-4, log=True)
    args.hidden_size = trial.suggest_int("hidden_size", 256, 768, step=128)
    args.num_res_blocks = trial.suggest_int("num_res_blocks", 2, 4)
    args.base_channels = trial.suggest_categorical("base_channels", [32, 48, 64])
    args.recurrent_type = trial.suggest_categorical(
        "recurrent_type", ["gru", "lstm", "none"]
    )
    args.use_noisy_linear = trial.suggest_categorical("noisy", [False, True])
    args.entropy_beta = trial.suggest_float("entropy_beta", 0.001, 0.02, log=True)

    cfg = train.build_training_config(args)
    cfg.total_updates = args.total_updates
    cfg.rollout = dataclasses.replace(cfg.rollout, num_steps=args.rollout_steps)

    metrics = train.run_training(cfg, args)
    trial.set_user_attr("metrics", metrics)
    return metrics.get("avg_return", 0.0)


def main():
    parser = argparse.ArgumentParser(description="Run Optuna search for Mario A3C")
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--study-name", type=str, default="mario-a3c-optuna")
    parser.add_argument("--storage", type=str, default=None, help="Optuna storage URL")
    args = parser.parse_args()

    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=args.trials)
    print("Best trial:")
    print(study.best_trial)


if __name__ == "__main__":
    main()
