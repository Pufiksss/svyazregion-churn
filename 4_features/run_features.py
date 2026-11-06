"""Запуск этапа 4: python run_features.py [--output папка]"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from feat.pipeline import run

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path)
    ap.add_argument("--output", type=Path)
    a = ap.parse_args()
    run(a.config, a.output)
