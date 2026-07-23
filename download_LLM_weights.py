"""Download model weights into /weights"""
from huggingface_hub import snapshot_download

REPO_NAME = 'Qwen/Qwen3-8B'
WEIGHTS_DIR = f'weights/{REPO_NAME}'

try:
    path = snapshot_download(repo_id=REPO_NAME, local_dir=WEIGHTS_DIR)
    print(f'Веса модели {REPO_NAME} были скачены в {path}')
except Exception as e:
    print('Во время скачивания весов произошла ошибка:',e)