"""quick_inference.py
快速推理脚本 - 低样板加载并运行模型 (适合 smoke test)。

增强特性 (2025-10-02):
1. 支持 --checkpoint 目录自动解析 (同 inference.py 逻辑)。
2. 灵活 state_dict 加载，兼容 _orig_mod. 前缀。
3. CLI 参数：--episodes, --deterministic, --max-steps, --full-schedule, --device。
4. 输出每 episode steps/sec。
5. 可选保存 state_dict 差异日志 (--issues-log-dir)。
Usage:
  python quick_inference.py --checkpoint trained_models/run01 --episodes 2 --deterministic
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any, List, Tuple

import torch

from src.config import ModelConfig
from src.envs.mario import MarioEnvConfig, MarioVectorEnvConfig, create_vector_env
from src.models import MarioActorCritic

ISSUES_LOG_NAME = "quick_inference_state_dict_issues.log"


def _extract_single_info(raw_info: Any) -> dict:
    if isinstance(raw_info, list):
        if raw_info and isinstance(raw_info[0], dict):
            return raw_info[0]
        return {}
    if isinstance(raw_info, dict):
        return raw_info
    return {}


def _coerce_scalar(x):
    try:
        import numpy as _np
        import torch as _t

        if isinstance(x, _t.Tensor):
            return x.item() if x.numel() == 1 else x
        if isinstance(x, _np.ndarray):
            return x.flatten()[0].item() if x.size == 1 else x
        if hasattr(x, "item"):
            return x.item()
    except Exception:
        pass
    return x


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("快速推理 (轻量 smoke test)")
    p.add_argument(
        "--checkpoint", type=str, required=True, help=".pt 文件或包含 checkpoint 的目录"
    )
    p.add_argument("--episodes", type=int, default=3, help="运行的 episode 数量")
    p.add_argument("--deterministic", action="store_true", help="使用贪婪策略")
    p.add_argument("--max-steps", type=int, default=5000, help="单 episode 最大步数")
    p.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="设备选择",
    )
    p.add_argument(
        "--full-schedule",
        action="store_true",
        help="使用 metadata 中完整 stage_schedule",
    )
    p.add_argument("--issues-log-dir", type=str, default=".", help="保存差异日志目录")
    p.add_argument(
        "--max-missing-keys", type=int, default=20, help="控制打印缺失/多余键数量"
    )
    return p.parse_args()


def _candidate_checkpoints_in_dir(directory: Path) -> List[Path]:
    return sorted(directory.glob("a3c_world*_stage*.pt"))


def _candidate_checkpoints_recursive(base_dir: Path) -> List[Path]:
    return sorted(base_dir.rglob("a3c_world*_stage*.pt"))


def _score_checkpoint(ckpt: Path) -> Tuple[int, int, int, int, Path]:
    meta_p = ckpt.with_suffix(".json")
    variant_pri = 0
    upd = 0
    step = 0
    try:
        if meta_p.exists():
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
            ss = meta.get("save_state", {}) if isinstance(meta, dict) else {}
            upd = int(ss.get("global_update", 0) or 0)
            step = int(ss.get("global_step", 0) or 0)
            variant = str(ss.get("type", "checkpoint"))
            variant_pri = 2 if variant == "latest" else (1 if variant == "checkpoint" else 0)
        else:
            if ckpt.name.endswith("latest.pt"):
                variant_pri = 2
            name = ckpt.stem
            try:
                tail = name.rsplit("_", 1)[-1]
                if tail.isdigit():
                    upd = int(tail)
            except Exception:
                pass
    except Exception:
        pass
    try:
        mtime = int(ckpt.stat().st_mtime_ns)
    except Exception:
        mtime = 0
    return (variant_pri, upd, step, mtime, ckpt)


def _select_best_checkpoint_in_dir_or_recursive(directory: Path, recursive: bool = True) -> Path:
    candidates = _candidate_checkpoints_recursive(directory) if recursive else _candidate_checkpoints_in_dir(directory)
    if not candidates:
        scope = "及子目录 " if recursive else ""
        raise FileNotFoundError(f"目录中未找到 checkpoint: {directory} ({scope}无候选)")
    scored: List[Tuple[int, int, int, int, Path]] = [_score_checkpoint(c) for c in candidates]
    scored.sort(reverse=True)
    return scored[0][4]


def resolve_checkpoint_path(arg_path: str) -> Path:
    p = Path(arg_path)
    if p.is_dir():
        return _select_best_checkpoint_in_dir_or_recursive(p, recursive=True)
    return p


def _flex_load_state_dict(model: torch.nn.Module, state: dict) -> List[str]:
    mk = set(model.state_dict().keys())
    sk = set(state.keys())
    model_pref = any(k.startswith("_orig_mod.") for k in mk)
    state_pref = any(k.startswith("_orig_mod.") for k in sk)
    adj = state
    if model_pref and not state_pref:
        adj = {f"_orig_mod.{k}": v for k, v in state.items()}
    elif not model_pref and state_pref:
        adj = {
            k[len("_orig_mod.") :]: v
            for k, v in state.items()
            if k.startswith("_orig_mod.")
        }
    missing, unexpected = model.load_state_dict(adj, strict=False)
    return list(missing) + [f"unexpected:{u}" for u in unexpected]


def main():
    args = parse_args()
    ckpt_path = resolve_checkpoint_path(args.checkpoint)
    meta_path = ckpt_path.with_suffix(".json")
    if not meta_path.exists():
        raise FileNotFoundError(f"找不到元数据文件: {meta_path}")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[quick] 使用设备: {device}")
    print(f"[quick] 使用 checkpoint: {ckpt_path.name}")

    metadata = json.loads(meta_path.read_text(encoding="utf-8"))

    model_cfg = ModelConfig(**metadata["model"])
    model = MarioActorCritic(model_cfg).to(device)
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    issues = _flex_load_state_dict(model, payload["model"])
    model.eval()
    if issues:
        head = issues[: args.max_missing_keys]
        more = (
            ""
            if len(issues) <= args.max_missing_keys
            else f" ... (+{len(issues)-args.max_missing_keys} more)"
        )
        print(f"[quick][warn] state_dict 差异: {len(issues)} -> {head}{more}")
        try:
            log_dir = Path(args.issues_log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            (log_dir / ISSUES_LOG_NAME).write_text("\n".join(issues), encoding="utf-8")
        except Exception as e:  # pragma: no cover
            print(f"[quick][warn] 写入差异日志失败: {e}")

    ss = payload.get("global_step", 0)
    su = payload.get("global_update", 0)
    print(f"[quick] 模型加载成功: {ckpt_path}")
    print(f"[quick] global_step={ss} global_update={su}")

    # Env
    if args.full_schedule:
        ve = metadata.get("vector_env", {})
        raw_sched = ve.get("stage_schedule") or []
        schedule = (
            [(int(w), int(s)) for (w, s) in raw_sched]
            if raw_sched
            else [(metadata["world"], metadata["stage"])]
        )
    else:
        schedule = [(metadata["world"], metadata["stage"])]
    env_cfg = MarioEnvConfig(
        world=metadata["world"],
        stage=metadata["stage"],
        action_type=metadata["action_type"],
        frame_skip=metadata["frame_skip"],
        frame_stack=metadata["frame_stack"],
    )
    vector_cfg = MarioVectorEnvConfig(
        num_envs=1,
        asynchronous=False,
        stage_schedule=tuple(schedule),
        random_start_stage=False,
        base_seed=42,
        env=env_cfg,
    )
    env = create_vector_env(vector_cfg)

    deterministic = args.deterministic

    try:
        for ep in range(args.episodes):
            print(f"\n=== Episode {ep+1}/{args.episodes} ===")
            obs_np, _ = env.reset()
            obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
            hidden, cell = model.initial_state(1, device)
            total_reward = 0.0
            max_x = 0
            start = time.time()
            for step in range(args.max_steps):
                with torch.no_grad():
                    out = model(obs, hidden, cell)
                    if deterministic:
                        action = torch.argmax(out.logits, dim=-1)
                    else:
                        probs = torch.softmax(out.logits, dim=-1)
                        action = torch.multinomial(probs, num_samples=1).squeeze(-1)
                a_np = action.cpu().numpy()
                obs_np, rew_np, terminated, truncated, raw_info = env.step(a_np)
                # reward shape 兼容
                try:
                    r_scalar = (
                        float(rew_np[0])
                        if hasattr(rew_np, "__len__")
                        else float(rew_np)
                    )
                except Exception:
                    r_scalar = float(rew_np)
                total_reward += r_scalar
                obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
                if out.hidden_state is not None:
                    hidden = out.hidden_state
                    if out.cell_state is not None:
                        cell = out.cell_state
                info_item = _extract_single_info(raw_info)
                if info_item:
                    x_val = info_item.get("x_pos")
                    if x_val is None:
                        metrics = info_item.get("metrics")
                        if isinstance(metrics, dict):
                            x_val = metrics.get("mario_x") or metrics.get("x_pos")
                    if x_val is not None:
                        xv = _coerce_scalar(x_val)
                        try:
                            xv_int = int(xv)
                            max_x = max(max_x, xv_int)
                            if (step + 1) % 100 == 0:
                                print(
                                    f"  step={step+1:4d} x={xv_int:4d} reward={total_reward:7.1f}"
                                )
                        except Exception:
                            if (step + 1) % 100 == 0:
                                print(
                                    f"  step={step+1:4d} x=? raw={xv} reward={total_reward:7.1f}"
                                )
                    else:
                        if (step + 1) % 100 == 0:
                            print(f"  step={step+1:4d} x=NA reward={total_reward:7.1f}")
                    if info_item.get("flag_get", False):
                        print(f"  🏁 通关! steps={step+1} reward={total_reward:.1f}")
                        break
                else:
                    if (step + 1) % 100 == 0:
                        print(f"  step={step+1:4d} reward={total_reward:7.1f}")
                term_flag = (
                    terminated[0] if hasattr(terminated, "__len__") else terminated
                )
                trunc_flag = (
                    truncated[0] if hasattr(truncated, "__len__") else truncated
                )
                if term_flag or trunc_flag:
                    print(
                        f"  ❌ 结束 steps={step+1} max_x={max_x} reward={total_reward:.1f}"
                    )
                    break
            dur = time.time() - start
            sps = (step + 1) / dur if dur > 0 else 0
            print(
                f"[quick] Episode 完成: steps={step+1} reward={total_reward:.1f} max_x={max_x} speed={sps:6.1f} steps/s"
            )
            time.sleep(0.5)
    finally:
        env.close()
        print("\n[quick] 推理完成")


if __name__ == "__main__":
    main()
