from pathlib import Path

import json
import torch

from scripts import backfill_global_step as backfill


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def test_backfill_uses_checkpoint_hint(tmp_path):
    meta_path = tmp_path / "a3c_world1_stage1_0001000.json"
    save_state = {"global_step": 0, "global_update": 0}
    meta = {
        "save_state": save_state,
        "vector_env": {"num_envs": 4},
    }
    _write_json(meta_path, meta)

    ckpt_payload = {"global_step": 1000, "global_update": 250}
    torch.save(ckpt_payload, meta_path.with_suffix(".pt"))

    changed, status = backfill.backfill_file(
        meta_path,
        rollout_steps=64,
        dry_run=False,
        backup=False,
        update_if_lower=False,
    )

    assert changed is True
    assert status.startswith("updated:")
    updated = backfill.load_json(meta_path)
    assert updated["save_state"]["global_step"] == 250 * 4 * 64
    source = updated["reconstructed_step_source"]
    assert source["method"] == "checkpoint_plus_rollout"
    assert source["checkpoint_global_update"] == 250


def test_backfill_skips_when_no_update(tmp_path):
    meta_path = tmp_path / "a3c_world1_stage1_0000000.json"
    meta = {
        "save_state": {"global_step": 0, "global_update": 0},
        "vector_env": {"num_envs": 4},
    }
    _write_json(meta_path, meta)

    changed, status = backfill.backfill_file(
        meta_path,
        rollout_steps=64,
        dry_run=True,
        backup=False,
        update_if_lower=False,
    )

    assert changed is False
    assert status == "no_update_info"
