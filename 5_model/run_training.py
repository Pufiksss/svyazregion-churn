"""Обучение этапа 5: python run_training.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from train import run

if __name__ == "__main__":
    run()
