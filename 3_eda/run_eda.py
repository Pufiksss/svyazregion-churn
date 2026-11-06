"""Запуск EDA этапа 3: python run_eda.py [--output папка]"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from eda.pipeline import run
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path)
    ap.add_argument("--output", type=Path)
    a = ap.parse_args()
    run(a.config, a.output)
