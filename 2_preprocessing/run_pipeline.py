"""Запуск подготовки данных этапа 2.

    python run_pipeline.py                 # результат в output/
    python run_pipeline.py --output out2   # в другую папку (для проверки воспроизводимости)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from prep.pipeline import run  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Подготовка данных СвязьРегион, этап 2")
    ap.add_argument("--config", type=Path, default=None)
    ap.add_argument("--output", type=Path, default=None)
    a = ap.parse_args()
    run(a.config, a.output)
