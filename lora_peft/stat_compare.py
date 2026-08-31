"""Статистическое сравнение прогонов по построчным оценкам LLM-as-judge.

Данные берутся из Trackio (per-example judge_* метрики, step=iteration в
llm_as_judge.py), не из .txt-отчётов — там оценки строго по порядку step,
и это удобнее парсить, чем свободный текст отчёта.

ВАЖНО про парность: llm_as_judge.py по умолчанию (без --shuffle) прогоняет
одни и те же первые N примеров одного домена/сплита в одном порядке для
всех моделей/адаптеров (см. launch_judge() в app.py) — поэтому per-example
оценки двух run'ов, полученные на одном --domain/--eval-on без --shuffle,
парные: i-й элемент в обоих списках — оценка на ОДНОМ И ТОМ ЖЕ вопросе.
Если у --shuffle разные --seed, домены разные или число примеров разное —
данные уже не парные, и Wilcoxon/paired t-test даст бессмысленный результат.
Модуль проверяет только совпадение ДЛИНЫ списков — не содержания вопросов
(Trackio текст вопроса не хранит), так что ответственность за сопоставимость
прогонов — на вызывающем.
"""
import json
import os
import sqlite3
from pathlib import Path

from scipy import stats

METRICS = ("faithfulness", "completeness", "consciousness")


def _db_path(project: str) -> Path:
    trackio_dir = os.environ.get("TRACKIO_DIR")
    if trackio_dir:
        base = Path(trackio_dir)
    else:
        hf_home = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
        base = Path(hf_home) / "trackio"
    return base / f"{project}.db"


def _as_text(value) -> str:
    """metrics в Trackio может лежать в БД как BLOB (bytes), а не TEXT —
    SQLite тогда возвращает bytes из fetchall()."""
    return value.decode("utf-8") if isinstance(value, (bytes, bytearray)) else value


def list_runs(project: str) -> list[str]:
    """Все run_name в проекте, у которых есть хотя бы одна judge-метрика
    (построчная или среднее) — то есть реально гонялся llm_as_judge.py."""
    db = _db_path(project)
    if not db.is_file():
        return []
    conn = sqlite3.connect(db)
    cur = conn.cursor()
    cur.execute("SELECT id, run_name, metrics FROM metrics")
    names = set()
    for _id, run_name, metrics in cur.fetchall():
        if "judge_" in _as_text(metrics):
            names.add(run_name)
    conn.close()
    return sorted(names)


def latest_avg_metrics(project: str, run_name: str, eval_on: str = "test",
                       full: bool = False) -> dict[str, float] | None:
    """Последние залогированные средние judge_*_avg... для одного run'а —
    faithfulness/completeness/consciousness (какие есть) + n (число примеров),
    или None, если таких метрик нет вовсе. "Последние" — потому что судью на
    одном адаптере часто гоняют по нескольку раз, а run.summary/лог метрик
    просто перезаписывает предыдущее значение (см. should_log_trackio_avg)."""
    suffix = "" if eval_on == "test" else f"_{eval_on}"
    suffix += "_full" if full else ""
    keys = {f"judge_{m}_avg{suffix}": m for m in METRICS}
    n_key = f"judge_num_samples{suffix}"

    db = _db_path(project)
    if not db.is_file():
        return None
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT timestamp, metrics FROM metrics WHERE run_name = ? ORDER BY timestamp DESC",
               (run_name,))
    rows = cur.fetchall()
    conn.close()

    for row in rows:
        data = json.loads(_as_text(row["metrics"]))
        found = {m: data[k] for k, m in keys.items() if k in data}
        if found:
            if n_key in data:
                found["n"] = data[n_key]
            return found
    return None


def top_runs(project: str, eval_on: str = "test", full: bool = False,
             metric: str | None = None) -> list[dict]:
    """Топ run'ов по среднему judge-баллу — по одной метрике (metric), или по
    среднему всех трёх, если metric не задан. Учитывает только run'ы, у
    которых реально есть залогированное среднее с нужным суффиксом
    (eval_on/full) — остальные молча пропускаются, не превращаются в 0."""
    if metric is not None and metric not in METRICS:
        raise ValueError(f"metric должен быть одним из {METRICS}, получено {metric!r}")

    rows = []
    for run_name in list_runs(project):
        avgs = latest_avg_metrics(project, run_name, eval_on=eval_on, full=full)
        if not avgs:
            continue
        present = [avgs[m] for m in METRICS if m in avgs]
        if not present:
            continue
        score = avgs[metric] if metric and metric in avgs else sum(present) / len(present)
        rows.append({"run_name": run_name, "score": score, **avgs})

    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows


def per_example_scores(project: str, run_name: str, metric: str, eval_on: str = "test") -> list[float]:
    """Построчные (не средние!) оценки судьи для одного run'а, в порядке step
    (= iteration в llm_as_judge.py). metric — один из METRICS."""
    if metric not in METRICS:
        raise ValueError(f"metric должен быть одним из {METRICS}, получено {metric!r}")
    suffix = "" if eval_on == "test" else f"_{eval_on}"
    key = f"judge_{metric}{suffix}"

    db = _db_path(project)
    if not db.is_file():
        return []
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("SELECT step, metrics FROM metrics WHERE run_name = ? ORDER BY step", (run_name,))
    rows = cur.fetchall()
    conn.close()

    scores = []
    for row in rows:
        data = json.loads(_as_text(row["metrics"]))
        if key in data:  # avg-версия метрики — отдельный ключ judge_..._avg, сюда не попадёт
            scores.append(data[key])
    return scores


def paired_compare(scores_a: list[float], scores_b: list[float]) -> dict:
    """Парное сравнение двух прогонов по одной метрике: Wilcoxon signed-rank
    (непараметрический, основной) + парный t-test (для справки)."""
    if len(scores_a) != len(scores_b):
        raise ValueError(
            f"Разное число примеров: {len(scores_a)} vs {len(scores_b)} — прогоны не парные "
            "(разный --iterations/--shuffle/--domain/--eval-on?)"
        )
    n = len(scores_a)
    if n == 0:
        raise ValueError("Нет данных для сравнения")

    diffs = [a - b for a, b in zip(scores_a, scores_b)]
    result = {
        "n": n,
        "mean_a": sum(scores_a) / n,
        "mean_b": sum(scores_b) / n,
        "mean_diff": sum(diffs) / n,
        "wilcoxon_p": None,
        "ttest_p": None,
    }

    if n >= 2 and any(d != 0 for d in diffs):
        try:
            _, w_p = stats.wilcoxon(scores_a, scores_b)
            result["wilcoxon_p"] = float(w_p)
        except ValueError:
            pass  # например все разности одного знака и n слишком мал для нормальной аппроксимации
    if n >= 2:
        _, t_p = stats.ttest_rel(scores_a, scores_b)
        result["ttest_p"] = float(t_p)

    return result


def friedman_compare(scores_by_run: dict[str, list[float]]) -> dict:
    """Омнибус-тест для 3+ прогонов сразу (Friedman) + попарный Wilcoxon с
    поправкой Holm-Bonferroni на множественные сравнения — только если
    омнибус значим (p<0.05), иначе попарные сравнения не считаются вовсе
    (без этого условия резко растёт риск ложноположительных находок)."""
    names = list(scores_by_run)
    if len(names) < 3:
        raise ValueError("Friedman нужен минимум для 3 прогонов — для двух используй paired_compare")

    lengths = {name: len(scores_by_run[name]) for name in names}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Разное число примеров у прогонов: {lengths}")
    n = lengths[names[0]]
    if n == 0:
        raise ValueError("Нет данных для сравнения")

    arrays = [scores_by_run[name] for name in names]
    stat, p = stats.friedmanchisquare(*arrays)

    pairwise = []
    if p < 0.05:
        import itertools

        pairs = list(itertools.combinations(names, 2))
        raw_ps = []
        for a, b in pairs:
            _, wp = stats.wilcoxon(scores_by_run[a], scores_by_run[b])
            raw_ps.append(float(wp))

        # Holm-Bonferroni: сортируем по возрастанию p, каждому — свой множитель
        # (m - ранг), кумулятивный максимум не даёт скорректированному p упасть
        # ниже уже присвоенного менее значимому сравнению.
        m = len(raw_ps)
        order = sorted(range(m), key=lambda i: raw_ps[i])
        adjusted = [None] * m
        running_max = 0.0
        for rank, idx in enumerate(order):
            adj = min(raw_ps[idx] * (m - rank), 1.0)
            running_max = max(running_max, adj)
            adjusted[idx] = running_max

        for (a, b), raw_p, adj_p in zip(pairs, raw_ps, adjusted):
            pairwise.append({"a": a, "b": b, "p_raw": raw_p, "p_holm": adj_p})

    return {"n": n, "friedman_stat": float(stat), "friedman_p": float(p), "pairwise": pairwise}
