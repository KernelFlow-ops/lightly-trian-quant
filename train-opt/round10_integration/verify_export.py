"""Validate Lightly export compatibility and optional infer-opt video loading."""
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from common.bench import assert_gpu_exclusive, env_info, gpu_lock  # noqa: E402
from common.config import DATASET_ROOT  # noqa: E402


def _state_shapes(path: Path) -> tuple[dict[str, tuple[int, ...]], dict]:
    data = torch.load(path, map_location="meta", weights_only=False, mmap=True)
    required = {"model_class_path", "model_init_args", "train_model", "license_info"}
    if set(data) != required:
        raise AssertionError(f"Export keys differ from Lightly: {sorted(data)}")
    return {name: tuple(value.shape) for name, value in data["train_model"].items()}, {
        "model_class_path": data["model_class_path"],
        "model_name": data["model_init_args"].get("model_name"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", type=Path, required=True)
    parser.add_argument("--baseline-export", type=Path, required=True)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--infer-video", action="store_true")
    parser.add_argument("--video-dir", type=Path, default=Path("/root/autodl-tmp"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    for path in (args.export, args.baseline_export):
        if not path.is_file():
            raise FileNotFoundError(path)
    image_path = args.image or next(iter(sorted((DATASET_ROOT / "valid").glob("*.jpg"))), None)
    if image_path is None or not image_path.is_file():
        raise FileNotFoundError("No validation image available for predict")
    output = args.out or ROOT / "round10_integration" / "profile" / f"compat_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    baseline_shapes, baseline_meta = _state_shapes(args.baseline_export)
    current_shapes, current_meta = _state_shapes(args.export)
    if current_shapes != baseline_shapes:
        missing = sorted(baseline_shapes.keys() - current_shapes.keys())
        extra = sorted(current_shapes.keys() - baseline_shapes.keys())
        changed = sorted(name for name in baseline_shapes.keys() & current_shapes.keys() if baseline_shapes[name] != current_shapes[name])
        raise AssertionError(f"Export state_dict differs: missing={missing[:8]}, extra={extra[:8]}, shape_changed={changed[:8]}")
    if current_meta != baseline_meta:
        raise AssertionError(f"Export model metadata differs: {current_meta} vs {baseline_meta}")

    report = {"command": [sys.executable, *sys.argv], "export": str(args.export), "baseline_export": str(args.baseline_export), "state_keys": len(current_shapes), "state_shapes_equal": True, "metadata_equal": True}
    with gpu_lock():
        report["gpu_snapshot_before"] = assert_gpu_exclusive()
        import lightly_train

        model = lightly_train.load_model(args.export, device="cuda")
        prediction = model.predict(image_path)
        report["predict"] = {"image": str(image_path), "shape": list(prediction.shape), "dtype": str(prediction.dtype), "classes": [int(value) for value in torch.unique(prediction).cpu().tolist()]}
        del prediction, model
        gc.collect()
        torch.cuda.empty_cache()
        if args.infer_video:
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            work = Path("/root/autodl-tmp/train-opt-tmp/A") / f"compat_video_{stamp}"
            cache = Path("/root/autodl-tmp/train-opt-tmp/A") / f"compat_cache_{stamp}"
            if work.exists() or cache.exists():
                raise FileExistsError(work if work.exists() else cache)
            command = [sys.executable, str(ROOT.parent / "infer-opt" / "round9_integration" / "video_infer.py"), "--model", str(args.export), "--video-dir", str(args.video_dir), "--out-dir", str(work), "--model-impl", "lightly"]
            child_env = os.environ.copy()
            child_env["TORCH_EXTENSIONS_DIR"] = str(cache / "torch_extensions")
            child_env["TRITON_CACHE_DIR"] = str(cache / "triton_cache")
            completed = subprocess.run(command, capture_output=True, text=True, env=child_env)
            report["infer_video"] = {"command": command, "returncode": completed.returncode, "out_dir": str(work), "stdout_tail": completed.stdout[-2000:], "stderr_tail": completed.stderr[-2000:]}
            if completed.returncode:
                report["infer_video"]["pass"] = False
            else:
                report["infer_video"]["pass"] = True
        report["env"] = env_info()
        report["gpu_snapshot_after"] = assert_gpu_exclusive()
    report["pass"] = report["state_shapes_equal"] and report["metadata_equal"] and bool(report["predict"]["shape"]) and (not args.infer_video or report["infer_video"]["pass"])
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"pass": report["pass"], "log": str(output), "state_keys": report["state_keys"]}, indent=2))
    if not report["pass"]:
        raise RuntimeError("Export compatibility failed; see report")


if __name__ == "__main__":
    main()
