"""Run the whole pipeline for one config: embed -> build-catalog -> build-index -> gen-queries -> label -> snapshot -> train
[-> calibrate] -> estimate -> evaluate. Stages whose artifacts already exist are skipped, so rerunning resumes.

  python run_all.py --config "config path" [--run run1] [--minutes 600] [--device cuda]
"""
import argparse
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
PY = sys.executable


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--run", default="run1")
    ap.add_argument("--minutes", type=float, default=600, help="labeling time budget in minutes")
    ap.add_argument("--device", default=None, help="torch device for train / calibrate (default: config train.device)")
    a = ap.parse_args(argv)
    cfg_path = Path(a.config).resolve()
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    base = cfg_path.parent
    run = (base / cfg["workspace"]).resolve() / "runs" / a.run
    model, est = "judge", "eval"
    dev = ["--device", a.device] if a.device else []

    def cli(*args):
        cmd = [PY, str(HERE / "run.py"), "--config", str(cfg_path), "--run", a.run, *args]
        print("$", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)

    if not (base / cfg["data"]["embeddings"]).resolve().exists():
        cli("embed")
    if not (run / "catalog" / "semantic_catalog.json").exists():
        cli("build-catalog")
    if not (run / "index" / "index.json").exists():
        cli("build-index")
    if not (run / "query_gen" / "queries.json").exists():
        cli("gen-queries")
    if not (run / "snapshots" / "s1").exists():
        cli("label", "--minutes", str(a.minutes))
        cli("snapshot", "--name", "s1")
    if not (run / "models" / model / "meta.json").exists():
        cli("train", "--snapshot", "s1", "--name", model, *dev)
    if cfg["judge"].get("and_judge", "joint") == "per_predicate_calibrated" and not (run / "models" / model / "and_calibration.json").exists():
        cli("calibrate", "--model", model, *dev)
    if not (run / "estimates" / est / "meta.json").exists():
        cli("estimate", "--model", model, "--name", est)
    cli("evaluate", "--name", est)


if __name__ == "__main__":
    main()
