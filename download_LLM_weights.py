"""Download model weights into /weights"""
from huggingface_hub import snapshot_download
import argparse

def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Скачать веса модели с HuggingFace')
    p.add_argument(
        '--model',
        help='Название модели с HF'
    )
    return p.parse_args()

args = parse()
WEIGHTS_DIR = f'weights/{args.model}'

try:
    path = snapshot_download(repo_id=args.model, local_dir=WEIGHTS_DIR)
    print(f'Веса модели {args.model} были скачены в {path}')
except Exception as e:
    print('Во время скачивания весов произошла ошибка:',e)