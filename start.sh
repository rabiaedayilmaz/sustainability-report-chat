#!/usr/bin/env bash
# start.sh — one-shot bootstrap for the sustainability-report RAG service.
#
# What it does (idempotent — safe to re-run):
#   1. Verifies prerequisites (docker, docker compose, .env)
#   2. Brings up the docker-compose stack (Ollama + API)
#   3. Waits for Ollama to report healthy
#   4. Pulls the LLM model named in OLLAMA_MODEL if it isn't already present
#   5. Waits for the API to pass /health
#   6. Optionally re-indexes PDFs into Qdrant   (--reindex)
#   7. Prints a sample curl so you can sanity-check
#
# Usage:
#   ./start.sh                          # boot only (API + Ollama)
#   ./start.sh --reindex                # boot + run scripts/index_pdfs.py inside the API container
#   ./start.sh --recreate               # full re-index (drops the Qdrant collection first)
#   ./start.sh --down                   # tear the stack down
#
set -euo pipefail
cd "$(dirname "$0")"

# ---------------------------------------------------------------- helpers
log()  { printf '\033[1;34m[start]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[start]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[start]\033[0m %s\n' "$*" >&2; exit 1; }

ENV_FILE=".env"
COMPOSE="docker compose"
TIMEOUT_OLLAMA=60     # seconds
TIMEOUT_API=180       # seconds (first boot loads multilingual-e5-large)

# ---------------------------------------------------------------- args
ACTION="up"
for arg in "$@"; do
    case "$arg" in
        --reindex)         ACTION="reindex" ;;
        --recreate)        ACTION="recreate" ;;
        --down)             ACTION="down" ;;
        -h|--help)         sed -n '2,20p' "$0"; exit 0 ;;
        "" ) ;;
        * ) die "Unknown flag: $arg (try --help)" ;;
    esac
done

# ---------------------------------------------------------------- prerequisites
command -v docker >/dev/null || die "docker is required but not on PATH"
$COMPOSE version >/dev/null 2>&1 || die "'docker compose' (v2) is required"

if [[ ! -f "$ENV_FILE" ]]; then
    if [[ -f .env.example ]]; then
        warn "$ENV_FILE not found — copying from .env.example."
        warn "Edit it now to set QDRANT_URL and QDRANT_API_KEY before re-running."
        cp .env.example "$ENV_FILE"
        exit 1
    fi
    die "$ENV_FILE not found and no .env.example to copy from"
fi

# ---------------------------------------------------------------- tear-down
if [[ "$ACTION" == "down" ]]; then
    log "Stopping stack..."
    $COMPOSE down
    log "Done."
    exit 0
fi

# ---------------------------------------------------------------- boot
log "Building + starting containers..."
$COMPOSE up --build -d

log "Waiting for Ollama (max ${TIMEOUT_OLLAMA}s)..."
deadline=$((SECONDS + TIMEOUT_OLLAMA))
until $COMPOSE exec -T ollama ollama list >/dev/null 2>&1; do
    [[ $SECONDS -ge $deadline ]] && die "Ollama did not become ready in ${TIMEOUT_OLLAMA}s"
    sleep 2
done
log "Ollama is up."

# ---------------------------------------------------------------- model pull
MODEL=$(grep -E '^OLLAMA_MODEL=' "$ENV_FILE" | head -1 | cut -d= -f2- | tr -d '"' || true)
MODEL=${MODEL:-qwen3:8b}
if $COMPOSE exec -T ollama ollama list | awk 'NR>1 {print $1}' | grep -qx "$MODEL"; then
    log "Model '$MODEL' already pulled."
else
    log "Pulling model '$MODEL' (one-time, may take a few GB)..."
    $COMPOSE exec -T ollama ollama pull "$MODEL"
fi

# ---------------------------------------------------------------- wait for API
log "Waiting for API /health (max ${TIMEOUT_API}s — first boot loads embedding model)..."
deadline=$((SECONDS + TIMEOUT_API))
until curl -fs http://localhost:8000/health >/dev/null 2>&1; do
    [[ $SECONDS -ge $deadline ]] && die "API did not become healthy in ${TIMEOUT_API}s — check 'docker compose logs api'"
    sleep 3
done
log "API is up."

# ---------------------------------------------------------------- optional reindex
if [[ "$ACTION" == "reindex" || "$ACTION" == "recreate" ]]; then
    flag=""
    [[ "$ACTION" == "recreate" ]] && flag="--recreate"
    log "Running ingest inside the api container ($flag)..."
    $COMPOSE exec -T api python scripts/index_pdfs.py $flag
fi

# ---------------------------------------------------------------- summary
echo
log "Stack is up:"
echo "    API     → http://localhost:8000"
echo "    Docs    → http://localhost:8000/docs"
echo "    Health  → http://localhost:8000/health"
echo "    Ollama  → http://localhost:11434"
echo
log "Try it:"
cat <<'EOF'
    curl -s -X POST http://localhost:8000/ask \
      -H 'Content-Type: application/json' \
      -d '{"question":"What are 2024 emissions targets?"}' | jq .
EOF
