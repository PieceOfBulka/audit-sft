#!/usr/bin/env python3
"""Разовая регистрация локальных датасетов (data/*.json) в ClearML Dataset.

Нужно прогнать один раз (или при добавлении нового домена/обновлении
датасета) — после этого finetune.py/bertscore.py/llm_as_judge.py тянут
датасеты через clearml.Dataset.get(...), а не читают локальные файлы
напрямую (см. common.resolve_dataset_path).

Требует настроенного clearml.conf (см. `clearml-init`) с доступом к
self-hosted ClearML-серверу.

Запуск (из корня репозитория):
    python scripts/upload_datasets_to_clearml.py                # все домены из common.DOMAIN_DATASETS
    python scripts/upload_datasets_to_clearml.py --domain zakupki  # только один
"""
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clearml import Dataset

from lora_peft.common import CLEARML_PROJECT, DOMAIN_DATASETS

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def upload_one(domain: str) -> None:
    rel_path = DOMAIN_DATASETS[domain]
    local_path = os.path.join(_ROOT, rel_path)
    if not os.path.isfile(local_path):
        print(f"== ПРОПУСК {domain}: файл не найден на диске: {local_path}")
        return

    print(f"== {domain}: {local_path}")
    ds = Dataset.create(dataset_name=domain, dataset_project=CLEARML_PROJECT)
    ds.add_files(path=local_path)
    ds.upload()
    ds.finalize()
    print(f"   зарегистрирован как ClearML Dataset: project={CLEARML_PROJECT!r} name={domain!r} id={ds.id}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--domain", choices=sorted(DOMAIN_DATASETS), default=None,
                   help="Загрузить только этот домен. Не задано -> все домены из common.DOMAIN_DATASETS")
    args = p.parse_args()

    domains = [args.domain] if args.domain else sorted(DOMAIN_DATASETS)
    for domain in domains:
        upload_one(domain)


if __name__ == "__main__":
    main()
