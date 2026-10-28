"""Run one study on several benchmarks (default: all of them).

    python benchmarks/run_all.py <study> [--benchmarks NAME ...] [--seeds SEED ...]

<study> is one of the study scripts in every benchmark directory (nn_size,
domain_importance_adaptation, loss_balancing, optim_f32, optim_f64). Each
benchmark runs as its own process from its directory, exactly as when the
script is started by hand; without --seeds the script's default seed is used.
"""
import argparse
import subprocess
import sys
from pathlib import Path

BENCHMARK_DIR = Path(__file__).resolve().parent
BENCHMARKS = sorted(p.name for p in BENCHMARK_DIR.iterdir() if (p / "models.py").exists())
STUDIES = ["nn_size", "domain_importance_adaptation", "loss_balancing", "optim_f32", "optim_f64"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("study", choices=STUDIES)
    parser.add_argument("--benchmarks", nargs="+", choices=BENCHMARKS, default=BENCHMARKS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[])
    args = parser.parse_args()

    failed = []
    for benchmark in args.benchmarks:
        print(f"=== {benchmark}: {args.study}.py", flush=True)
        cmd = [sys.executable, f"{args.study}.py", *map(str, args.seeds)]
        if subprocess.run(cmd, cwd=BENCHMARK_DIR / benchmark).returncode != 0:
            failed.append(benchmark)
    if failed:
        sys.exit(f"{args.study} failed on: {', '.join(failed)}")


if __name__ == "__main__":
    main()
