"""Python 3.12 + this archive are sufficient; no Conda, uv, GPU, R or Julia needed."""
from pathlib import Path
import argparse
import os
import subprocess
import sys
import venv

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite',choices=['smoke','all','benchmark','openml','recovery','conformal'],default='all')
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--output',default=None)
    parser.add_argument('--setup-only',action='store_true')
    args=parser.parse_args()
    if sys.version_info[:2] != (3,12):
        parser.error('Use Python 3.12: py -3.12 bootstrap.py (Windows) or python3.12 bootstrap.py (Linux/macOS).')
    root=Path(__file__).resolve().parent
    env=root/'.venv'
    interpreter=env/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    if not interpreter.exists():
        venv.EnvBuilder(with_pip=True).create(env)
    subprocess.run([str(interpreter),'-m','pip','install','--only-binary=:all:','-r',str(root/'requirements-lock.txt')],check=True)
    if not args.setup_only:
        command=[str(interpreter),str(root/'reproduce.py'),'--suite',args.suite,'--workers',str(args.workers)]
        if args.output: command += ['--output',args.output]
        subprocess.run(command,check=True)
    return 0

if __name__=='__main__':
    raise SystemExit(main())
