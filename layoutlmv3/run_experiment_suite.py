from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent

SUITES = {
    "audit200_ablation": [
        "task_driven_layoutlmv3_gnn_audit200_mask.json",
        "task_driven_layoutlmv3_gnn_audit200_token.json",
        "task_driven_layoutlmv3_gnn_audit200_none.json",
        "task_driven_layoutlmv3_gnn_audit200_no_dom.json",
    ]
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=sorted(SUITES), default="audit200_ablation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_dir = ROOT_DIR / "configs"

    for config_name in SUITES[args.suite]:
        config_path = config_dir / config_name
        print(f"suite_run_start config={config_path}", flush=True)
        subprocess.run(
            [sys.executable, "-m", "layoutlmv3.train", "--config", str(config_path)],
            check=True,
        )
        print(f"suite_run_done config={config_path}", flush=True)


if __name__ == "__main__":
    main()
