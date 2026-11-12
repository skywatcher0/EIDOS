"""Create an isolated environment, install EIDOS and run a dataset example."""
from pathlib import Path
import argparse
import os
import subprocess
import sys
import venv

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=["smoke", "all", "benchmark", "openml"], default="smoke")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--setup-only", action="store_true")
    args = parser.parse_args()
    if not (3, 12) <= sys.version_info[:2] < (3, 14):
        parser.error("Run with 64-bit Python 3.12 or 3.13; Python 3.12 is the locked benchmark environment")
    root = Path(__file__).resolve().parent
    environment = root / ".venv"
    interpreter = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not interpreter.exists():
        venv.EnvBuilder(with_pip=True).create(environment)
    subprocess.run([str(interpreter), "-m", "pip", "install", "--only-binary=:all:", "-r", str(root / "requirements-lock.txt")], check=True)
    subprocess.run([str(interpreter), "-m", "pip", "install", "--no-deps", str(root)], check=True)
    if not args.setup_only:
        command = [str(interpreter), str(root / "run_benchmarks.py"), "--suite", args.suite, "--workers", str(args.workers)]
        if args.output:
            command += ["--output", str(args.output.resolve())]
        subprocess.run(command, check=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
