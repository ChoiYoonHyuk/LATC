"""Run all four experiment groups in sequence, using installed latc-research."""
from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--preset",choices=["smoke","paper"],default="smoke")
    parser.add_argument("--out",type=Path,default=Path("results/smoke"))
    parser.add_argument("--plots",action="store_true")
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    for rq in ("rq1","rq2","rq3","rq4"):
        command=[sys.executable,"-m","latc","run",str(root/"configs"/f"{rq}_{args.preset}.json"),"--out",str(args.out/rq)]
        if args.plots:
            command.append("--plots")
        subprocess.run(command,check=True)


if __name__ == "__main__":
    main()
