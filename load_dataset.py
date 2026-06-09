import os

from datasets import load_dataset

# Корень репозитория = каталог этого файла. Путь к данным не зависит от cwd,
# поэтому скрипт можно запускать из любой директории.
ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATASET = os.path.join(ROOT, "data", "claude_answers.json")


def load_train_eval_dataset(dataset_path: str = DEFAULT_DATASET):
    """Read dataset from json file and split it into train and eval parts"""
    # относительный путь трактуем относительно корня репозитория, а не cwd
    if not os.path.isabs(dataset_path):
        dataset_path = os.path.join(ROOT, dataset_path)
    dataset_full = load_dataset("json", data_files=dataset_path)
    # split into train and eval
    return dataset_full["train"].train_test_split(test_size=0.2, seed=42)


if __name__ == "__main__":
    print(load_train_eval_dataset())
