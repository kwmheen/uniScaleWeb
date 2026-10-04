#!/bin/bash
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8

if [[ ! -x .venv/bin/python ]]; then
  echo "가상환경을 만든 뒤 패키지를 설치합니다."
  if command -v uv >/dev/null 2>&1; then
    uv venv --python 3.11 .venv
    uv pip install -r requirements.txt
  else
    python3 -m venv .venv
    .venv/bin/python -m pip install -r requirements.txt
  fi
fi

.venv/bin/python -c "from serve import ensure_models; ensure_models()"

pids="$(lsof -ti tcp:8765 || true)"
if [[ -n "$pids" ]]; then
  kill $pids >/dev/null 2>&1 || true
fi

(sleep 1 && open "http://127.0.0.1:8765/web/index.html") &
exec .venv/bin/python serve.py
