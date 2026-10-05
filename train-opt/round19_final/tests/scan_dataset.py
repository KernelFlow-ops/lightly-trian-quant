"""Round19 dataset check: how many target masks per image does the training pipeline produce (r3 padded loss caps max_targets at 4)?

Three views, all CPU only (no GPU lock needed):
  (a) COCO annotations of the train split: instances per image (information only: the training task is SEMANTIC segmentation, instances merge);
  (b) the semantic mask PNGs actually used (common.data converts COCO -> PNG): distinct pixel values per image, and the set of all values;
  (c) the real training pipeline (Lightly dataset + EoMT transform + collate, i.e. what the model sees: len(binary_masks['labels']) per image),
      1..P passes over the train split with different random augmentation draws.

  python scan_dataset.py --passes 3 --workers 48 --out profile/dataset_targets.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from common.config import DATASET_ROOT, DATA_ROOT, MODEL_NAMES  # noqa: E402


def _uniq(path: str):
    arr = np.asarray(Image.open(path))
    vals = np.unique(arr)
    return path, [int(v) for v in vals], arr.shape


def scan_masks(split: str, workers: int):
    paths = sorted(str(p) for p in (DATA_ROOT / f"{split}_masks").glob("*.png"))
    with Pool(workers) as pool:
        rows = pool.map(_uniq, paths, chunksize=16)
    per = Counter(len(v) for _, v, _ in rows)
    allv = Counter(x for _, v, _ in rows for x in v)
    shapes = Counter(tuple(s) for _, _, s in rows)
    return {"images": len(rows), "distinct_values_per_image": dict(sorted(per.items())), "images_containing_value": {str(k): v for k, v in sorted(allv.items())},
            "mask_shapes": {str(k): v for k, v in shapes.most_common(6)}}


def scan_coco(split: str):
    data = json.loads((DATASET_ROOT / split / "_annotations.coco.json").read_text())
    n_ann = Counter(a["image_id"] for a in data["annotations"])
    per = Counter(n_ann.get(im["id"], 0) for im in data["images"])
    cats = Counter(a["category_id"] for a in data["annotations"])
    crowd = sum(1 for a in data["annotations"] if a.get("iscrowd"))
    return {"images": len(data["images"]), "annotations": len(data["annotations"]), "instances_per_image": dict(sorted(per.items())),
            "max_instances": max(per), "categories": {str(k): v for k, v in cats.items()}, "iscrowd": crowd,
            "categories_defined": [{"id": c["id"], "name": c["name"]} for c in data["categories"]]}


def scan_pipeline(model: str, passes: int, workers: int, batch: int):
    import torch
    from torch.utils.data import DataLoader
    from lightly_train._commands import train_task_helpers as helpers
    from lightly_train._task_models.dinov3_eomt_semantic_segmentation.train_model import DINOv3EoMTSemanticSegmentationTrain as Cls
    from common.data import make_data_args
    from common.model_factory import build_train_model

    data_args = make_data_args()
    total_steps = (3672 // batch) * 20
    _, targs, _ = build_train_model(model, data_args, batch, total_steps)
    dargs = data_args.get_train_args()
    info = list(dargs.list_image_info())
    out = {"passes": [], "images_per_pass": len(info)}
    total = Counter()
    mx = 0
    for p in range(passes):
        ds = dargs.get_dataset_cls()(dataset_args=dargs, image_info=info)
        tr = helpers.get_train_transform(Cls, MODEL_NAMES[model], targs)
        ds.set_transform(tr)
        collate = ds.get_batch_collate_fn_cls()(split="train", transform_args=targs)
        torch.manual_seed(1000 + p)
        loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=workers, collate_fn=collate, prefetch_factor=4, persistent_workers=False)
        cnt, labels = Counter(), Counter()
        t0 = time.time()
        n = 0
        for b in loader:
            for item in b["binary_masks"]:
                k = int(item["masks"].shape[0])
                cnt[k] += 1
                for lb in item["labels"].tolist():
                    labels[int(lb)] += 1
                n += 1
        mx = max(mx, max(cnt))
        total.update(cnt)
        out["passes"].append({"pass": p, "images": n, "targets_per_image": dict(sorted(cnt.items())), "label_counts": dict(sorted(labels.items())), "seconds": time.time() - t0,
                              "img_per_s": n / (time.time() - t0)})
        print("pipeline pass", p, json.dumps(out["passes"][-1]), flush=True)
    out["targets_per_image_total"] = dict(sorted(total.items()))
    out["max_targets_seen"] = mx
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--batch", type=int, default=14)
    ap.add_argument("--model", default="vitb16")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    res = {"command": [sys.executable, *sys.argv]}
    for split in ("train", "valid"):
        res[f"coco_{split}"] = scan_coco(split)
        print(f"coco {split}", json.dumps({k: v for k, v in res[f'coco_{split}'].items() if k != 'categories_defined'}), flush=True)
    for split in ("train", "valid"):
        res[f"png_{split}"] = scan_masks(split, args.workers)
        print(f"png {split}", json.dumps(res[f"png_{split}"]), flush=True)
    res["pipeline_train"] = scan_pipeline(args.model, args.passes, args.workers, args.batch)
    args.out.write_text(json.dumps(res, indent=2))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
