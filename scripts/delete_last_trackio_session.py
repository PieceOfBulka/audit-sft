#!/usr/bin/env python3
"""Удаляет последнюю (самую свежую) сессию логирования метрик в Trackio для
заданного run'а.

Нужен, когда finetune.py/bertscore.py/llm_as_judge.py по ошибке резюмировались
не в тот run (например run_name ещё не различал какой-то гиперпараметр —
см. common.build_run_name/hparams_hash) и дописали данные ПОВЕРХ уже
существующего run'а. Вместо того чтобы гадать точный timestamp вручную —
скрипт сам находит границу последней сессии по разрыву во времени между
соседними строками метрик (--gap-minutes, по умолчанию 10) и удаляет только
её, оставляя более старые (легитимные) данные того же run_id нетронутыми.

Если нужно не удалить, а перенести последнюю сессию в отдельный run
(например разные target_modules случайно попали в один run_name) —
см. историю проекта: тот же принцип (разрыв по времени), но вместо DELETE —
UPDATE run_id/run_name на новые значения.

Запуск (на сервере, тем же пользователем, что запускает app.py/train):
    python scripts/delete_last_trackio_session.py --run <run_name>                # dry-run
    python scripts/delete_last_trackio_session.py --run <run_name> --apply        # реально удалить
    python scripts/delete_last_trackio_session.py --run-id <run_id> --apply       # если у имени
                                                                                    # несколько run_id
    python scripts/delete_last_trackio_session.py --run <run_name> --gap-minutes 30 --apply
"""
import argparse
import os
import shutil
import sqlite3
from datetime import datetime
from pathlib import Path

# Таблицы схемы trackio (>=0.24, с отдельным run_id) с колонкой времени,
# по которой можно резать на "сессии". artifacts/artifact_versions трогать
# не пытаемся — там нет прямой привязки к run_id по времени, только через
# run_artifact_links (её тоже чистим).
TABLES = [
    ("metrics", "timestamp"),
    ("system_metrics", "timestamp"),
    ("configs", "created_at"),
    ("traces", "timestamp"),
    ("alerts", "timestamp"),
    ("run_artifact_links", "created_at"),
    ("pending_uploads", "created_at"),
]


def resolve_db_path(project: str) -> Path:
    trackio_dir = os.environ.get("TRACKIO_DIR")
    if trackio_dir:
        base = Path(trackio_dir)
    else:
        hf_home = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
        base = Path(hf_home) / "trackio"
    return base / f"{project}.db"


def last_session_start(cur: sqlite3.Cursor, run_id: str, gap_seconds: float) -> str | None:
    cur.execute("SELECT timestamp FROM metrics WHERE run_id = ? ORDER BY timestamp", (run_id,))
    timestamps = [datetime.fromisoformat(r[0]) for r in cur.fetchall()]
    if not timestamps:
        return None
    cutoff = timestamps[0]
    for prev, cur_ts in zip(timestamps, timestamps[1:]):
        if (cur_ts - prev).total_seconds() >= gap_seconds:
            cutoff = cur_ts
    return cutoff.isoformat()


def resolve_run_ids(cur: sqlite3.Cursor, run: str | None, run_id: str | None) -> list[str]:
    if run_id:
        return [run_id]
    cur.execute("SELECT DISTINCT run_id FROM metrics WHERE run_name = ?", (run,))
    run_ids = [r[0] for r in cur.fetchall()]
    if not run_ids:
        raise SystemExit(f"Не нашёл run_id для run_name={run!r}")
    if len(run_ids) > 1:
        print(f"== ВНИМАНИЕ: под именем {run!r} несколько run_id: {run_ids} — обрабатываю каждый отдельно")
    return run_ids


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project", default="lora-finetuning")
    p.add_argument("--run", default=None, help="run_name (может соответствовать нескольким run_id)")
    p.add_argument("--run-id", default=None, help="Точный run_id — приоритетнее --run")
    p.add_argument("--gap-minutes", type=float, default=10,
                   help="Разрыв по времени между соседними записями метрик, который считается "
                        "границей между разными физическими запусками. По умолчанию 10 минут.")
    p.add_argument("--apply", action="store_true",
                   help="Реально удалить. Без этого флага — только dry-run (покажет, что нашёл).")
    args = p.parse_args()
    if not args.run and not args.run_id:
        raise SystemExit("Нужен --run <run_name> или --run-id <run_id>")

    db_path = resolve_db_path(args.project)
    if not db_path.is_file():
        raise SystemExit(f"Не нашёл БД: {db_path}")
    print(f"== БД: {db_path}")

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    gap_seconds = args.gap_minutes * 60

    run_ids = resolve_run_ids(cur, args.run, args.run_id)
    plan: dict[str, str] = {}
    for run_id in run_ids:
        session_start = last_session_start(cur, run_id, gap_seconds)
        if session_start is None:
            print(f"\n== run_id={run_id}: метрик нет, пропускаю")
            continue
        plan[run_id] = session_start
        print(f"\n== run_id={run_id}: последняя сессия начинается с {session_start}")
        for table, ts_col in TABLES:
            cur.execute(f"SELECT COUNT(*) FROM {table} WHERE run_id = ? AND {ts_col} >= ?",
                       (run_id, session_start))
            n = cur.fetchone()[0]
            if n:
                print(f"   {table}: {n} строк")

    if not plan:
        print("\n== нечего удалять.")
        conn.close()
        return

    if not args.apply:
        print("\n== dry-run: ничего не удалено. Запусти с --apply, чтобы применить.")
        conn.close()
        return

    backup_path = db_path.with_suffix(db_path.suffix + ".bak_delete_last")
    shutil.copy2(db_path, backup_path)
    print(f"\n== бэкап: {backup_path}")

    for run_id, session_start in plan.items():
        for table, ts_col in TABLES:
            cur.execute(f"DELETE FROM {table} WHERE run_id = ? AND {ts_col} >= ?", (run_id, session_start))
            if cur.rowcount:
                print(f"   {run_id}/{table}: удалено {cur.rowcount}")

    conn.commit()
    conn.close()
    print("\n== готово. Перезапусти дашборд Trackio, чтобы увидеть изменения.")


if __name__ == "__main__":
    main()
