"""Загрузка исходных источников без изменения файлов.

Все значения читаются как строки: пустая ячейка остаётся пустой строкой, а не NaN,
чтобы аудит качества видел исходный вид данных.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CSV_KW = dict(sep=";", encoding="utf-8-sig", dtype=str, keep_default_na=False)


@dataclass
class Config:
    raw: dict
    root: Path

    @classmethod
    def load(cls, path: Path | None = None, output_dir: Path | None = None) -> "Config":
        path = path or ROOT / "config.json"
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if output_dir is not None:
            raw["output_dir"] = str(output_dir)
        return cls(raw=raw, root=ROOT)

    def raw_path(self, group: str, name: str) -> Path:
        return self.root / self.raw["raw_dir"] / self.raw[group][name]

    def path(self, key: str) -> Path:
        return self.root / self.raw[key]

    @property
    def out(self) -> Path:
        p = Path(self.raw["output_dir"])
        return p if p.is_absolute() else self.root / p


def verify_checksums(cfg: Config) -> dict:
    """Сверяет SHA-256 исходников с зафиксированным манифестом. Исходники не должны меняться."""
    manifest = cfg.path("checksums")
    base = manifest.parent
    result = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, rel = line.split(maxsplit=1)
        actual = hashlib.sha256((base / rel).read_bytes()).hexdigest()
        result[rel] = actual == digest
    bad = [k for k, ok in result.items() if not ok]
    if bad:
        raise RuntimeError(f"Исходные файлы изменены: {bad}")
    return result


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, **CSV_KW)


def first_statement(sql_text: str) -> str:
    text = re.sub(r"--[^\n]*", "", sql_text)
    return text.split(";")[0].strip()


def read_support_sqlite(db: Path, sql_file: Path) -> pd.DataFrame:
    """Извлекает support_current_raw SQL-запросом в режиме только чтения."""
    query = first_statement(sql_file.read_text(encoding="utf-8"))
    with sqlite3.connect(f"file:{db.resolve()}?mode=ro", uri=True) as conn:
        cur = conn.execute(query)
        cols = [c[0] for c in cur.description]
        rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=cols)
    return df.fillna("").astype(str)


def run_diagnostics(db: Path, sql_file: Path) -> pd.DataFrame:
    """Выполняет именованные запросы из sql/diagnostics.sql (блоки `-- name: ...`)."""
    blocks = re.split(r"--\s*name:\s*", sql_file.read_text(encoding="utf-8"))[1:]
    rows = []
    with sqlite3.connect(f"file:{db.resolve()}?mode=ro", uri=True) as conn:
        for block in blocks:
            name, _, body = block.partition("\n")
            query = first_statement(body)
            rows.append({"query": name.strip(), "value": conn.execute(query).fetchone()[0]})
    return pd.DataFrame(rows)


def load_all(cfg: Config) -> dict[str, pd.DataFrame]:
    sources = {
        "subscribers": read_csv(cfg.raw_path("lesson1", "subscribers")),
        "billing": read_csv(cfg.raw_path("lesson1", "billing")),
        "events": read_csv(cfg.raw_path("lesson1", "events")),
        "support_legacy": read_csv(cfg.raw_path("lesson2", "legacy_csv")),
        "support_current": read_support_sqlite(cfg.raw_path("lesson2", "sqlite"), cfg.path("sql_extract")),
        "schema_support": read_csv(cfg.raw_path("lesson2", "schema")),
    }
    # Резервная CSV-копия не объединяется с SQLite: только сверка, что это та же таблица.
    backup = read_csv(cfg.raw_path("lesson2", "current_csv_backup"))
    cur = sources["support_current"]
    sources["_sqlite_equals_csv_backup"] = (
        backup.shape == cur.shape and bool((backup.values == cur.values).all())
    )
    return sources
