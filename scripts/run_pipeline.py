from __future__ import annotations

import argparse
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run data preparation and training pipeline.")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--stages",
        nargs="+",
        default=["prepare", "train"],
        choices=["prepare", "train"],
        help="Pipeline stages to run.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Dry-run train stage.")
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    if "prepare" in args.stages:
        run([sys.executable, "scripts/01_prepare_data.py", "--config", args.config])
    if "train" in args.stages:
        cmd = [sys.executable, "scripts/02_train.py", "--config", args.config]
        if args.dry_run:
            cmd.append("--dry-run")
        run(cmd)


if __name__ == "__main__":
    main()
