# 1) deps + ollama modeli
pip install -r requirements.txt
ollama pull llama3.1

# 2) indeksle (ilk çalıştırma E5-large'i indirir, ~1.3 GB)
python3 scripts/index_pdfs.py --recreate

2026-04-16 15:28:30,529 [INFO] src.pipeline: Ingest complete. pages=1751 chunks=8397 failed_batches=0 collection_total=8397

# 3) sor
python3 scripts/search.py "What are NTT DATA's 2024 emissions targets?"
python3 scripts/interactive.py

docker compose up --build -d
docker compose exec ollama ollama pull qwen3:8b   # one-time, ~5 GB
curl http://localhost:8000/health
curl -X POST http://localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"What are the 2024 emissions targets?"}'

  ./start.sh                # boot
./start.sh --reindex      # boot + index
./start.sh --recreate     # boot + drop+reindex
./start.sh --down         # tear down

# Monitoring ile başlat
./start.sh --with-monitoring

# Veya birlikte
./start.sh --reindex --with-monitoring

# Sadece monitoring eklemek istersen (zaten api+ollama up'sa)
docker compose --profile monitoring up -d

# Hepsini kapat
./start.sh --down            # monitoring profile'ı dahil tüm konteynerleri durdurur