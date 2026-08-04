#!/usr/bin/env python3
"""Удаляет из Trackio метрики оценки (bertscore_*, judge_*) и связанные
артефакты (judge-report), оставляя нетронутыми метрики обучения (loss,
learning_rate, epoch, grad_norm, ...) и артефакты адаптеров (type="model").

Отличие от cleanup_judge_trackio.py: тот чистит только judge_*, этот —
и judge_*, и bertscore_* разом (по умолчанию). Можно сузить/расширить
список префиксов через --prefix.

Безопасность:
  - Перед изменениями делает бэкап .db-файла рядом с оригиналом.
  - Трогает только строки metrics, где среди ключей есть один из --prefix
    (обучение никогда не логирует bertscore_*/judge_* в той же строке —
    это отдельные вызовы trackio.log() из разных скриптов).
  - Артефакты адаптеров (type="model", из finetune.py) не трогает —
    удаляет только type="report" (из llm_as_judge.py).

Запуск (на сервере, тем же пользователем, что запускает app.py/train):
    python scripts/cleanup_eval_trackio.py                 # dry-run
    python scripts/cleanup_eval_trackio.py --apply          # реально удалить
    python scripts/cleanup_eval_trackio.py --apply --prefix judge_   # только judge, не трогать bertscore
"""
import argparse
import os
import shutil
import sqlite3
from pathlib import Path


def resolve_db_path(project: str) -> Path:
    trackio_dir = os.environ.get("TRACKIO_DIR")
    if trackio_dir:
        base = Path(trackio_dir)
    else:
        hf_home = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
        base = Path(hf_home) / "trackio"
    return base / f"{project}.db"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project", default="lora-finetuning")
    p.add_argument("--prefix", action="append", default=None,
                   help="Префикс(ы) метрик для удаления. Можно указать несколько раз. "
                        "По умолчанию: judge_ и bertscore_")
    p.add_argument("--apply", action="store_true",
                   help="Реально удалить. Без этого флага — только dry-run.")
    args = p.parse_args()
    prefixes = args.prefix or ["judge_", "bertscore_"]

    db_path = resolve_db_path(args.project)
    if not db_path.is_file():
        raise SystemExit(f"Не нашёл БД: {db_path}")
    print(f"== БД: {db_path}")
    print(f"== удаляем метрики с префиксами: {prefixes}")

    if args.apply:
        backup_path = db_path.with_suffix(db_path.suffix + ".bak")
        shutil.copy2(db_path, backup_path)
        print(f"== бэкап сохранён: {backup_path}")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    metric_ids: set[int] = set()
    for prefix in prefixes:
        cur.execute("SELECT id, run_name, step FROM metrics WHERE metrics LIKE ?", (f"%{prefix}%",))
        rows = cur.fetchall()
        print(f"\n== метрики с '{prefix}': {len(rows)} строк")
        for row in rows[:5]:
            print(f"   run={row['run_name']!r} step={row['step']}")
        if len(rows) > 5:
            print(f"   ... и ещё {len(rows) - 5}")
        metric_ids.update(row["id"] for row in rows)

    # артефакты-репорты (judge), адаптеры (type="model") не трогаем
    cur.execute("SELECT id, name, type FROM artifacts WHERE type = 'report'")
    artifact_rows = cur.fetchall()
    print(f"\n== артефакты type='report': {len(artifact_rows)}")
    for row in artifact_rows:
        print(f"   id={row['id']} name={row['name']!r}")

    if not args.apply:
        print(f"\n== dry-run: всего к удалению — {len(metric_ids)} метрик, {len(artifact_rows)} артефактов. "
              "Запусти с --apply, чтобы применить.")
        conn.close()
        return

    if metric_ids:
        cur.executemany("DELETE FROM metrics WHERE id = ?", [(i,) for i in metric_ids])
        print(f"\n== удалено метрик: {len(metric_ids)}")

    for row in artifact_rows:
        artifact_id = row["id"]
        cur.execute("SELECT id FROM artifact_versions WHERE artifact_id = ?", (artifact_id,))
        version_ids = [r["id"] for r in cur.fetchall()]
        for version_id in version_ids:
            cur.execute("DELETE FROM artifact_aliases WHERE artifact_version_id = ?", (version_id,))
            cur.execute("DELETE FROM run_artifact_links WHERE artifact_version_id = ?", (version_id,))
        cur.execute("DELETE FROM artifact_versions WHERE artifact_id = ?", (artifact_id,))
        cur.execute("DELETE FROM artifacts WHERE id = ?", (artifact_id,))
    if artifact_rows:
        print(f"== удалено артефактов: {len(artifact_rows)}")

    conn.commit()
    conn.close()
    print("\n== готово. Перезапусти дашборд Trackio, чтобы увидеть изменения.")


if __name__ == "__main__":
    main()
