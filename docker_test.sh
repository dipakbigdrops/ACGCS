#!/bin/bash
set -e
echo "=== ACGCS Docker End-to-End Test ==="
echo ""
echo "[1/4] Building Docker image (this may take 10-15 min)..."
docker build -t acgcs-api:test .
echo ""
echo "[2/4] Starting container..."
docker run -d --name acgcs-test -p 8000:8000 acgcs-api:test
echo "Waiting for server to start (up to 120s)..."
for i in $(seq 1 24); do
  if curl -sf http://localhost:8000/health >/dev/null 2>&1; then
    echo "Server is ready."
    break
  fi
  if [ $i -eq 24 ]; then
    echo "Server failed to start. Logs:"
    docker logs acgcs-test
    docker stop acgcs-test
    docker rm acgcs-test
    exit 1
  fi
  sleep 5
done
echo ""
echo "[3/4] Testing endpoints..."
echo "  GET /health"
curl -s http://localhost:8000/health | head -c 200
echo ""
echo "  GET /"
curl -s http://localhost:8000/ | head -c 200
echo ""
echo "  POST /upload-guidelines (with default PDF)"
if [ -f default_guidelines.pdf ]; then
  GUID=$(curl -s -X POST -F "guidelines_pdf=@default_guidelines.pdf" http://localhost:8000/upload-guidelines | python3 -c "import sys,json; print(json.load(sys.stdin).get('guidelines_id',''))")
  if [ -n "$GUID" ]; then
    echo "  Guidelines ID: $GUID"
    echo "  POST /analyze (with guidelines_id)"
    curl -s -X POST -F "guidelines_id=$GUID" -F "creative_file=@tests/creative_27573_cr2.html;type=text/html" http://localhost:8000/analyze | head -c 300
    echo ""
  else
    echo "  Upload failed"
  fi
else
  echo "  Skipped (no default_guidelines.pdf)"
fi
echo ""
echo "[4/4] Stopping container..."
docker stop acgcs-test
docker rm acgcs-test
echo ""
echo "=== Docker test complete ==="
