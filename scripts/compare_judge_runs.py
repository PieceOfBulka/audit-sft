#!/usr/bin/env python3
"""Статистическое сравнение прогонов по построчным оценкам LLM-as-judge.

2 прогона -> Wilcoxon signed-rank (парный, основной результат) + парный
t-test (для справки). 3+ прогонов -> Friedman (омнибус) + попарный Wilcoxon
с поправкой Holm-Bonferroni, только если омнибус значим.

ВАЖНО: сравнение осмысленно, только если прогоны реально шли на одних и тех
же примерах в одном порядке — тот же --domain/--eval-on, без --shuffle (или
с тем же --seed) в llm_as_judge.py. См. докстринг lora_peft/stat_compare.py.

Запуск (из корня репозитория):
    python scripts/compare_judge_runs.py --list
    python scripts/compare_judge_runs.py --top                       # топ по всем трём метрикам сразу
    python scripts/compare_judge_runs.py --top --metric faithfulness --full
    python scripts/compare_judge_runs.py --runs run_a,run_b --metric faithfulness
    python scripts/compare_judge_runs.py --runs run_a,run_b,run_c --metric completeness --eval-on train
"""
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lora_peft.common import TRACKIO_PROJECT
from lora_peft.stat_compare import (METRICS, friedman_compare, list_runs, paired_compare,
                                    per_example_scores, top_runs)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project", default=TRACKIO_PROJECT)
    p.add_argument("--runs", default=None, help="Имена run'ов через запятую (2 и больше)")
    p.add_argument("--metric", choices=METRICS, default=None,
                   help="Для --runs: по умолчанию faithfulness. Для --top: по умолчанию среднее "
                        "всех трёх метрик.")
    p.add_argument("--eval-on", choices=["test", "train"], default="test")
    p.add_argument("--full", action="store_true",
                   help="Для --top: брать средние с суффиксом _full (полновесные проверки, n>=50) "
                        "вместо быстрых выборочных.")
    p.add_argument("--top", type=int, nargs="?", const=10, default=None,
                   help="Показать топ-N run'ов по judge-баллу и выйти (по умолчанию 10)")
    p.add_argument("--list", action="store_true", help="Показать все run'ы с judge-данными и выйти")
    args = p.parse_args()
    metric = args.metric

    if args.list:
        runs = list_runs(args.project)
        if not runs:
            print("Не нашёл run'ов с judge-метриками.")
        for name in runs:
            print(name)
        return

    if args.top is not None:
        rows = top_runs(args.project, eval_on=args.eval_on, full=args.full, metric=metric)
        if not rows:
            print("Не нашёл run'ов с усреднёнными judge-метриками для этих условий "
                 f"(eval_on={args.eval_on}, full={args.full}).")
            return
        label = metric or "среднее по 3 метрикам"
        print(f"=== Топ-{args.top} по '{label}' (eval_on={args.eval_on}, full={args.full}) ===\n")
        for i, row in enumerate(rows[:args.top], 1):
            parts = ", ".join(f"{m}={row[m]:.2f}" for m in METRICS if m in row)
            n = f", n={row['n']}" if "n" in row else ""
            print(f"{i:2d}. {row['score']:.3f}  {row['run_name']}  ({parts}{n})")
        return

    if not args.runs:
        raise SystemExit("Нужен --runs run_a,run_b[,run_c,...], --top или --list")

    metric = metric or "faithfulness"

    run_names = [r.strip() for r in args.runs.split(",") if r.strip()]
    if len(run_names) < 2:
        raise SystemExit("Нужно минимум 2 run'а для сравнения")

    scores = {}
    for name in run_names:
        s = per_example_scores(args.project, name, metric, args.eval_on)
        print(f"{name}: n={len(s)}")
        if not s:
            raise SystemExit(f"У run'а {name!r} нет построчных оценок '{metric}' "
                             f"(eval-on={args.eval_on}) — не с чем сравнивать")
        scores[name] = s

    if len(run_names) == 2:
        a, b = run_names
        result = paired_compare(scores[a], scores[b])
        print(f"\n=== {a} vs {b} ({metric}, {args.eval_on}, n={result['n']}) ===")
        print(f"  среднее {a}: {result['mean_a']:.3f}")
        print(f"  среднее {b}: {result['mean_b']:.3f}")
        print(f"  разница:    {result['mean_diff']:+.3f}")
        if result["wilcoxon_p"] is not None:
            marker = "  (значимо, p<0.05)" if result["wilcoxon_p"] < 0.05 else ""
            print(f"  Wilcoxon signed-rank p = {result['wilcoxon_p']:.4f}{marker}")
        else:
            print("  Wilcoxon: недостаточно вариации в разностях (все нули?), пропущено")
        if result["ttest_p"] is not None:
            marker = "  (значимо, p<0.05)" if result["ttest_p"] < 0.05 else ""
            print(f"  Paired t-test p =         {result['ttest_p']:.4f}{marker}")
    else:
        result = friedman_compare(scores)
        print(f"\n=== Friedman ({metric}, {args.eval_on}, n={result['n']}, "
              f"{len(run_names)} прогонов) ===")
        print(f"  chi2={result['friedman_stat']:.3f}, p={result['friedman_p']:.4f}")
        if result["friedman_p"] >= 0.05:
            print("  Значимых различий между прогонами не найдено (p>=0.05) — "
                  "попарные сравнения не считались")
        else:
            print("  Значимо (p<0.05) — попарные сравнения (Wilcoxon, поправка Holm-Bonferroni):")
            for pw in result["pairwise"]:
                marker = " *" if pw["p_holm"] < 0.05 else ""
                print(f"    {pw['a']} vs {pw['b']}: p_raw={pw['p_raw']:.4f} p_holm={pw['p_holm']:.4f}{marker}")


if __name__ == "__main__":
    main()
