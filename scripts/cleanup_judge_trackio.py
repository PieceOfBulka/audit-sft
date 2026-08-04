#!/usr/bin/env python3
"""Удаляет старые LLM-as-judge данные (метрики + артефакты-репорты) из Trackio,
не трогая обучение и BERTScore. Нужно один раз прогнать после того, как
llm_as_judge.py перестал логировать по шагам (step=iteration) — старые
per-step метрики иначе продолжат портить графики вперемешку с новыми.

Безопасность:
  - Перед изменениями делает бэкап .db-файла рядом с оригиналом.
  - Трогает ТОЛЬКО строки, где среди залогированных метрик есть judge_*
    (обучение логирует loss/lr/..., bertscore — bertscore_*, они никогда
    не смешиваются с judge_* в одной строке metrics, т.к. это разные
    процессы/вызовы trackio.log()).
  - Артефакты удаляются по имени, оканчивающемуся на "-judge-report"
    (это то имя, под которым llm_as_judge.py логирует репорт).

Запуск (на сервере, тем же пользователем, что запускает app.py/train):
    python scripts/cleanup_judge_trackio.py                 # dry-run, только покажет, что будет удалено
    python scripts/cleanup_judge_trackio.py --apply          # реально удалить
    python scripts/cleanup_judge_trackio.py --project OTHER  # если проект называется не lora-finetuning
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
    p.add_argument("--apply", action="store_true",
                   help="Реально удалить. Без этого флага — только dry-run (покажет, что нашёл).")
    args = p.parse_args()

    db_path = resolve_db_path(args.project)
    if not db_path.is_file():
        raise SystemExit(f"Не нашёл БД: {db_path}")
    print(f"== БД: {db_path}")

    if args.apply:
        backup_path = db_path.with_suffix(db_path.suffix + ".bak")
        shutil.copy2(db_path, backup_path)
        print(f"== бэкап сохранён: {backup_path}")

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # --- метрики: строки, где JSON содержит judge_* ---
    cur.execute("SELECT id, run_name, step, metrics FROM metrics WHERE metrics LIKE '%judge_%'")
    metric_rows = cur.fetchall()
    print(f"\n== метрики с judge_*: {len(metric_rows)} строк")
    for row in metric_rows[:10]:
        print(f"   run={row['run_name']!r} step={row['step']} metrics={row['metrics'][:120]}")
    if len(metric_rows) > 10:
        print(f"   ... и ещё {len(metric_rows) - 10}")

    # --- артефакты: judge-report ---
    cur.execute("SELECT id, name, type FROM artifacts WHERE name LIKE '%-judge-report'")
    artifact_rows = cur.fetchall()
    print(f"\n== артефакты judge-report: {len(artifact_rows)}")
    for row in artifact_rows:
        print(f"   id={row['id']} name={row['name']!r} type={row['type']!r}")

    if not args.apply:
        print("\n== dry-run: ничего не удалено. Запусти с --apply, чтобы применить.")
        conn.close()
        return

    metric_ids = [row["id"] for row in metric_rows]
    if metric_ids:
        cur.executemany("DELETE FROM metrics WHERE id = ?", [(i,) for i in metric_ids])
        print(f"\n== удалено метрик: {cur.rowcount if cur.rowcount != -1 else len(metric_ids)}")

    artifact_ids = [row["id"] for row in artifact_rows]
    for artifact_id in artifact_ids:
        cur.execute("SELECT id FROM artifact_versions WHERE artifact_id = ?", (artifact_id,))
        version_ids = [r["id"] for r in cur.fetchall()]
        for version_id in version_ids:
            cur.execute("DELETE FROM artifact_aliases WHERE artifact_version_id = ?", (version_id,))
            cur.execute("DELETE FROM run_artifact_links WHERE artifact_version_id = ?", (version_id,))
        cur.execute("DELETE FROM artifact_versions WHERE artifact_id = ?", (artifact_id,))
        cur.execute("DELETE FROM artifacts WHERE id = ?", (artifact_id,))
    if artifact_ids:
        print(f"== удалено артефактов: {len(artifact_ids)}")

    conn.commit()
    conn.close()
    print("\n== готово. Перезапусти дашборд Trackio (кнопка 'Открыть/обновить' в app.py), чтобы увидеть изменения.")


if __name__ == "__main__":
    main()
