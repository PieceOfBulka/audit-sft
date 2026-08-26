#!/usr/bin/env bash
# Запускает команду и сам перезапускает её при появлении новых коммитов в
# origin/<текущая ветка> — чтобы не приходилось руками Ctrl+C -> git pull ->
# заново запускать после каждого пуша. Заодно перезапускает, если процесс
# упал сам (например авария CUDA), без ожидания следующего коммита.
#
# Использование (из корня репозитория):
#   scripts/run_with_autoreload.sh [интервал_сек=60] -- <команда...>
#
# Пример:
#   CUDA_VISIBLE_DEVICES=7 scripts/run_with_autoreload.sh 60 -- uv run python lora_peft/app.py
#
# Обычно запускается внутри tmux, чтобы пережить отключение SSH — см.
# README-комментарий в конце файла или просто:
#   tmux new -s app
#   CUDA_VISIBLE_DEVICES=7 scripts/run_with_autoreload.sh 60 -- uv run python lora_peft/app.py
#   # Ctrl+B, затем D — отсоединиться, процесс продолжит работать
#   tmux attach -t app   # вернуться позже
set -uo pipefail

INTERVAL="${1:-60}"
shift || true
if [ "${1:-}" = "--" ]; then
    shift
fi
CMD=("$@")

if [ "${#CMD[@]}" -eq 0 ]; then
    echo "Использование: $0 [интервал_сек] -- <команда...>" >&2
    exit 1
fi

BRANCH=$(git rev-parse --abbrev-ref HEAD)
PID=""

start() {
    echo "== [$(date '+%H:%M:%S')] запускаю: ${CMD[*]}"
    "${CMD[@]}" &
    PID=$!
}

stop() {
    if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
        echo "== [$(date '+%H:%M:%S')] останавливаю процесс $PID"
        kill "$PID" 2>/dev/null
        wait "$PID" 2>/dev/null
    fi
}

trap 'stop; exit 0' INT TERM

start
while true; do
    sleep "$INTERVAL"

    if ! git fetch origin "$BRANCH" --quiet 2>/dev/null; then
        echo "== [$(date '+%H:%M:%S')] git fetch не удался (сеть?), пробую снова через $INTERVAL с"
        continue
    fi

    LOCAL=$(git rev-parse HEAD)
    REMOTE=$(git rev-parse "origin/$BRANCH")

    if [ "$LOCAL" != "$REMOTE" ]; then
        echo "== [$(date '+%H:%M:%S')] новые коммиты в origin/$BRANCH ($LOCAL -> $REMOTE) — перезапускаю"
        stop
        if git pull --ff-only origin "$BRANCH"; then
            start
        else
            echo "== [$(date '+%H:%M:%S')] git pull --ff-only не удался (расхождение веток?) — процесс НЕ перезапущен, разберись руками"
        fi
    elif ! kill -0 "$PID" 2>/dev/null; then
        echo "== [$(date '+%H:%M:%S')] процесс упал сам (код неизменился) — перезапускаю"
        start
    fi
done
