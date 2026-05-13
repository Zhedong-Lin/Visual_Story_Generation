"""Inspect one run directory tree and key json files."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id", help="runs/<run_id>")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    run_dir = project_root / "runs" / args.run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"run not found: {run_dir}")

    print(f"[run] {run_dir}")
    for p in sorted(run_dir.rglob("*")):
        if p.is_file():
            print(" -", p.relative_to(run_dir))

    record = run_dir / "run_record.json"
    if record.exists():
        data = json.loads(record.read_text(encoding="utf-8"))
        print("\n[summary]")
        print("task_type:", data["task_spec"]["task_type"])
        print("num_frames:", data["task_spec"]["num_frames"])
        print("judge:", data.get("judge", {}).get("final_score"))


if __name__ == "__main__":
    main()
