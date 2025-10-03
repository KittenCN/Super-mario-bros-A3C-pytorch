#!/usr/bin/env python
"""快速检查 Mario A3C checkpoint 信息。

输出内容:
- 基本路径/存在性
- metadata JSON 是否存在 & 关键字段
- 模型参数统计 (tensor 数量 / 总参数量 / dtype 分布)
- 是否编译保存格式 (检测 _orig_mod 前缀)
- replay 配置摘要 (per_sample_interval 等)
- 最近 state_dict 加载 issues 日志摘要 (若存在)

用法:
  python scripts/inspect_checkpoint.py --ckpt trained_models/run01/a3c_world1_stage1_latest.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch


def parse_args():
    p = argparse.ArgumentParser(description="Inspect Mario A3C checkpoint")
    p.add_argument("--ckpt", required=True, help="Path to .pt checkpoint file")
    p.add_argument(
        "--full-params", action="store_true", help="List every parameter name"
    )
    return p.parse_args()


def load_payload(path: Path):
    try:
        return torch.load(path, map_location="cpu")
    except Exception as e:
        print(f"[inspect][error] failed to torch.load: {e}")
        return {}


def try_metadata(path: Path):
    meta_path = path.with_suffix(".json")
    if meta_path.exists():
        try:
            return json.loads(meta_path.read_text(encoding="utf-8")), meta_path
        except Exception as e:
            print(f"[inspect][warn] metadata parse failed: {e}")
    return None, meta_path


def summarize_state_dict(sd: dict):
    dtypes = Counter()
    param_count = 0
    tensor_count = 0
    compiled_keys = 0
    for k, v in sd.items():
        if hasattr(v, "dtype"):
            dtypes[str(v.dtype)] += 1
            tensor_count += 1
            if hasattr(v, "numel"):
                param_count += v.numel()
        if k.startswith("_orig_mod."):
            compiled_keys += 1
    compiled_saved = compiled_keys > 0
    return {
        "tensor_count": tensor_count,
        "param_count": param_count,
        "dtype_hist": dict(dtypes),
        "compiled_format": compiled_saved,
    }


def main():
    args = parse_args()
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        print(f"[inspect][error] checkpoint not found: {ckpt_path}")
        sys.exit(1)
    payload = load_payload(ckpt_path)
    if not isinstance(payload, dict):
        print("[inspect][error] unexpected checkpoint payload (expected dict)")
        sys.exit(1)
    meta, meta_path = try_metadata(ckpt_path)
    if meta is None:
        print(f"[inspect][warn] metadata sidecar missing or invalid: {meta_path}")
    else:
        env_meta = meta.get("vector_env", {})
        replay_meta = meta.get("replay", {})
        save_state = meta.get("save_state", {})
        print("[inspect] metadata summary:")
        print(
            f"  world={meta.get('world')} stage={meta.get('stage')} action_type={meta.get('action_type')} frame_stack={meta.get('frame_stack')} frame_skip={meta.get('frame_skip')}"
        )
        print(
            f"  num_envs={env_meta.get('num_envs')} async={env_meta.get('asynchronous')} base_seed={env_meta.get('base_seed')}"
        )
        print(
            f"  per_sample_interval={replay_meta.get('per_sample_interval')} replay_capacity={replay_meta.get('capacity')} enable={replay_meta.get('enable')}"
        )
        print(
            f"  save_state: update={save_state.get('global_update')} step={save_state.get('global_step')} type={save_state.get('type')}"
        )
    model_sd = payload.get("model") if isinstance(payload, dict) else None
    if isinstance(model_sd, dict):
        summary = summarize_state_dict(model_sd)
        print("[inspect] model state_dict summary:")
        for k, v in summary.items():
            print(f"  {k}: {v}")
        if args.full_params:
            print("[inspect] parameter names:")
            for name in model_sd.keys():
                print(f"    {name}")
    else:
        print("[inspect][warn] payload missing 'model' key")
    # 查找加载 issues 日志
    issues_files = [
        ckpt_path.parent / "state_dict_load_issues.log",
        ckpt_path.parent / "eval_state_dict_load_issues.log",
    ]
    for f in issues_files:
        if f.exists():
            try:
                tail = f.read_text(encoding="utf-8").strip().splitlines()[-10:]
                print(f"[inspect] tail of {f.name} (last 10 lines):")
                for line in tail:
                    print(f"  {line}")
            except Exception:
                pass


if __name__ == "__main__":
    main()
