#!/usr/bin/env python3
"""回填历史 checkpoint 的 global_step 字段脚本。

使用场景：早期 overlap 采集实现存在 global_step 未正确累加的问题，导致已保存的
`*.json` 元数据中 `save_state.global_step == 0` 而 `save_state.global_update > 0`。
这会影响：
1. 学习率 / 调度器恢复（若依赖 global_step）
2. 训练曲线 / 指标对齐（steps 轴不正确）
3. 后续自动化分析（SPS、Return vs. Steps 混乱）

回填策略：
global_step = global_update * num_envs * rollout_steps

由于旧元数据未持久化 rollout_steps，本脚本通过参数 `--assume-rollout-steps` 采用
“有记录的训练默认值” 假设（默认 64）。脚本会写入顶层键
`reconstructed_step_source` 记录：方法、假设的 rollout_steps、公式、时间戳，以便后续审计。

特性：
- checkpoint 提示：若 metadata 中 `global_update=0` 会尝试读取同名 `.pt` 的 `global_update`。
- 幂等：若已存在 `reconstructed_step_source` 或 global_step>0 则跳过。
- 备份：默认为每个要修改的 JSON 生成同名 `.bak` 文件；已存在备份不会覆盖。
- 递归：扫描指定根目录（默认 trained_models/）。
- Dry run：`--dry-run` 仅打印将执行的修改，不落盘。

示例：
python scripts/backfill_global_step.py --root trained_models --assume-rollout-steps 64

Author: automated agent
"""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import List, Optional

try:
    import torch
except Exception:  # pragma: no cover - optional dependency when only dry-run
    torch = None  # type: ignore[assignment]


def iter_metadata_files(root: Path) -> List[Path]:
    results: List[Path] = []
    for p in root.rglob("*.json"):
        name = p.name
        # 仅粗筛：包含 a3c_world 且不是隐藏文件
        if name.startswith("a3c_world") and name.endswith(".json"):
            results.append(p)
    return sorted(results)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, data: dict) -> None:
    payload = json.dumps(data, indent=2, ensure_ascii=False, sort_keys=False)
    path.write_text(payload, encoding="utf-8")


def _load_checkpoint_metadata(json_path: Path) -> Optional[dict]:
    if torch is None:
        return None
    ckpt_path = json_path.with_suffix(".pt")
    if not ckpt_path.exists():
        # allow latest checkpoint naming pattern
        if json_path.stem.endswith(".json"):
            alt = json_path.parent / (json_path.stem[:-5] + ".pt")
            if alt.exists():
                ckpt_path = alt
            else:
                return None
        else:
            return None
    try:
        try:
            payload = torch.load(ckpt_path, map_location="cpu", weights_only=True)  # type: ignore[call-arg]
        except TypeError:
            payload = torch.load(ckpt_path, map_location="cpu")  # type: ignore[call-arg]
    except Exception:
        return None
    if isinstance(payload, dict):
        return payload
    return None


def backfill_file(
    path: Path,
    rollout_steps: int,
    dry_run: bool,
    backup: bool,
    update_if_lower: bool,
) -> tuple[bool, str]:
    meta = load_json(path)
    save_state = meta.get("save_state", {})
    global_step = int(save_state.get("global_step", 0) or 0)
    global_update = int(save_state.get("global_update", 0) or 0)
    checkpoint_hint = None
    if global_update <= 0:
        payload = _load_checkpoint_metadata(path)
        if payload is not None:
            checkpoint_hint = int(payload.get("global_update", 0) or 0)
            if checkpoint_hint > 0:
                global_update = checkpoint_hint
                save_state["global_update"] = global_update
    if global_update <= 0:
        return False, "no_update_info"
    # 提取 num_envs；无则失败
    try:
        num_envs = int(meta["vector_env"]["num_envs"])  # type: ignore[index]
    except Exception:
        return False, "missing_num_envs"
    new_global_step = global_update * num_envs * rollout_steps
    if global_step > 0:
        if not update_if_lower:
            return False, "already_has_step"
        if new_global_step <= global_step:
            return False, "current_step>=computed"
    if "reconstructed_step_source" in meta and not update_if_lower:
        return False, "already_reconstructed"
    meta["save_state"]["global_step"] = int(new_global_step)
    source = {
        "method": "assumed_rollout_steps",
        "rollout_steps": rollout_steps,
        "formula": "global_update * num_envs * rollout_steps",
        "computed_at": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
    }
    if checkpoint_hint is not None:
        source["checkpoint_global_update"] = checkpoint_hint
        source["method"] = "checkpoint_plus_rollout"
    meta["reconstructed_step_source"] = source
    if dry_run:
        return True, f"would_update:{new_global_step}"
    if backup:
        bak = path.with_suffix(path.suffix + ".bak")
        if not bak.exists():
            bak.write_text(
                json.dumps(load_json(path), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
    dump_json(path, meta)
    return True, f"updated:{new_global_step}"


def main():  # noqa: D401
    parser = argparse.ArgumentParser(description="回填历史 checkpoint global_step")
    parser.add_argument("--root", type=str, default="trained_models", help="扫描根目录")
    parser.add_argument(
        "--assume-rollout-steps",
        type=int,
        default=64,
        help="假设历史训练的 rollout_steps",
    )
    parser.add_argument("--dry-run", action="store_true", help="仅打印将修改的文件")
    parser.add_argument("--no-backup", action="store_true", help="不生成 .bak 备份文件")
    parser.add_argument(
        "--update-if-lower",
        action="store_true",
        help="当计算值大于已有 global_step 时也回填（用于历史残留小步数）",
    )
    args = parser.parse_args()
    root = Path(args.root)
    if not root.exists():
        print(f"[migrate][error] root not found: {root}")
        return 2
    files = iter_metadata_files(root)
    if not files:
        print("[migrate] no metadata files found")
        return 0
    changed = 0
    skipped = 0
    for f in files:
        ok, status = backfill_file(
            f,
            args.assume_rollout_steps,
            args.dry_run,
            backup=not args.no_backup,
            update_if_lower=args.update_if_lower,
        )
        if ok:
            changed += 1
            print(f"[migrate] {f}: {status}")
        else:
            skipped += 1
            print(f"[migrate][skip] {f.name}: {status}")
    print(
        f"[migrate] done changed={changed} skipped={skipped} rollout_steps={args.assume_rollout_steps} dry_run={args.dry_run}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - 脚本入口
    raise SystemExit(main())
