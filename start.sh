#!/usr/bin/env bash
# start.sh — one-shot bootstrap for the sustainability-report RAG service.
#
# Bu versiyon Ollama'nın Docker container içinde değil, HOST makinende
# (Mac'inde) çalıştığını varsayar. Ollama'yı sen başlatırsın, script
# sadece API container'ını ayağa kaldırır.
#
# Ön koşullar:
#   - Mac'inde Ollama uygulaması açık OLMALI  (veya: ollama serve)
#   - İlk kullanımda modeli çek: ollama pull qwen3:8b
#
# Kullanım:
#   ./start.sh                          # sadece API container'ını başlat
#   ./start.sh --reindex                # başlat + PDF'leri index'e ekle
#   ./start.sh --recreate               # başlat + Qdrant koleksiyonunu sıfırla ve yeniden index'le
#   ./start.sh --down                   # stack'i durdur
#
set -euo pipefail
cd "$(dirname "$0")"

# ---------------------------------------------------------------- helpers
log()  { printf '\033[1;34m[start]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[start]\033[0m %s\n' "$*"; }
die()  { printf '\033[1;31m[start]\033[0m %s\n' "$*" >&2; exit 1; }

ENV_FILE=".env"
COMPOSE="docker compose"
TIMEOUT_API=180       # seconds (first boot loads multilingual-e5-large)

# Ollama host'ta çalıştığı için erişim adresi farklı platformlarda değişir:
#   Mac / Windows  → host.docker.internal (Docker Desktop tarafından çözümlenir)
#   Linux          → 172.17.0.1 veya --network=host
OLLAMA_HOST="${OLLAMA_HOST:-http://localhost:11434}"   # host'tan kontrol için
OLLAMA_HOST_FROM_CONTAINER="${OLLAMA_HOST_FROM_CONTAINER:-http://host.docker.internal:11434}"

# ---------------------------------------------------------------- args
ACTION="up"
for arg in "$@"; do
    case "$arg" in
        --reindex)  ACTION="reindex" ;;
        --recreate) ACTION="recreate" ;;
        --down)     ACTION="down" ;;
        -h|--help)  sed -n '2,22p' "$0"; exit 0 ;;
        "")  ;;
        *)   die "Bilinmeyen flag: $arg (--help ile yardım alabilirsin)" ;;
    esac
done

# ---------------------------------------------------------------- prerequisites
command -v docker >/dev/null  || die "docker bulunamadı, PATH'e eklenmemiş"
$COMPOSE version >/dev/null 2>&1 || die "'docker compose' (v2) gerekli"
command -v curl >/dev/null    || die "curl bulunamadı, PATH'e eklenmemiş"

if [[ ! -f "$ENV_FILE" ]]; then
    if [[ -f .env.example ]]; then
        warn "$ENV_FILE bulunamadı — .env.example'dan kopyalanıyor."
        warn "QDRANT_URL ve QDRANT_API_KEY değerlerini düzenleyip tekrar çalıştır."
        cp .env.example "$ENV_FILE"
        exit 1
    fi
    die "$ENV_FILE bulunamadı ve .env.example da yok"
fi

# ---------------------------------------------------------------- tear-down
if [[ "$ACTION" == "down" ]]; then
    log "Stack durduruluyor..."
    $COMPOSE down
    log "Tamam."
    exit 0
fi

# ---------------------------------------------------------------- ollama sağlık kontrolü (host'ta)
log "Host'taki Ollama kontrol ediliyor: $OLLAMA_HOST ..."
if ! curl -fs "${OLLAMA_HOST}/api/tags" >/dev/null 2>&1; then
    die "Ollama yanıt vermiyor ($OLLAMA_HOST). \
Mac'inde Ollama uygulamasını aç veya terminalde 'ollama serve' komutunu çalıştır."
fi
log "Ollama host'ta çalışıyor."

# ---------------------------------------------------------------- model kontrolü (host'ta)
MODEL=$(grep -E '^OLLAMA_MODEL=' "$ENV_FILE" | head -1 | cut -d= -f2- | tr -d '"' || true)
MODEL=${MODEL:-qwen3:8b}

log "Model kontrol ediliyor: '$MODEL' ..."
TAGS_JSON=$(curl -fs "${OLLAMA_HOST}/api/tags")
if echo "$TAGS_JSON" | grep -q "\"${MODEL}\""; then
    log "Model '$MODEL' zaten mevcut."
else
    warn "Model '$MODEL' bulunamadı."
    warn "Lütfen şu komutu çalıştır ve bitince tekrar dene:"
    warn "    ollama pull ${MODEL}"
    die "Gerekli model eksik, işlem durduruluyor."
fi

# ---------------------------------------------------------------- boot
log "Container'lar build ediliyor ve başlatılıyor..."
$COMPOSE up --build -d

# ---------------------------------------------------------------- wait for API
log "API /health bekleniyor (max ${TIMEOUT_API}s — ilk başlatmada embedding modeli yüklenir)..."
deadline=$((SECONDS + TIMEOUT_API))
until curl -fs http://localhost:8000/health >/dev/null 2>&1; do
    [[ $SECONDS -ge $deadline ]] && \
        die "API ${TIMEOUT_API}s içinde sağlıklı hale gelmedi — kontrol: 'docker compose logs api'"
    sleep 3
done
log "API hazır."

# ---------------------------------------------------------------- optional reindex
if [[ "$ACTION" == "reindex" || "$ACTION" == "recreate" ]]; then
    flag=""
    [[ "$ACTION" == "recreate" ]] && flag="--recreate"
    log "Index işlemi başlatılıyor (api container içinde) $flag ..."
    $COMPOSE exec -T api python scripts/index_pdfs.py --phase embed $flag
fi

# ---------------------------------------------------------------- summary
echo
log "Stack çalışıyor:"
printf '    %-10s → %s\n' "API"    "http://localhost:8000"
printf '    %-10s → %s\n' "Docs"   "http://localhost:8000/docs"
printf '    %-10s → %s\n' "Health" "http://localhost:8000/health"
printf '    %-10s → %s\n' "Ollama" "$OLLAMA_HOST (host'ta)"
