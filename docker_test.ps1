$ErrorActionPreference = "Stop"
Write-Host "=== ACGCS Docker End-to-End Test ===" -ForegroundColor Cyan
Write-Host ""

Write-Host "[1/4] Building Docker image (this may take 10-15 min)..." -ForegroundColor Yellow
docker build -t acgcs-api:test .
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host ""
Write-Host "[2/4] Starting container..." -ForegroundColor Yellow
docker run -d --name acgcs-test -p 8000:8000 acgcs-api:test
Write-Host "Waiting for server to start (up to 120s)..."
$ready = $false
for ($i = 1; $i -le 24; $i++) {
    try {
        $r = Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing -TimeoutSec 3 -ErrorAction SilentlyContinue
        if ($r.StatusCode -eq 200) {
            Write-Host "Server is ready."
            $ready = $true
            break
        }
    } catch {}
    if ($i -eq 24) {
        Write-Host "Server failed to start. Logs:" -ForegroundColor Red
        docker logs acgcs-test
        docker stop acgcs-test 2>$null
        docker rm acgcs-test 2>$null
        exit 1
    }
    Start-Sleep -Seconds 5
}

Write-Host ""
Write-Host "[3/4] Testing endpoints..." -ForegroundColor Yellow
Write-Host "  GET /health"
try {
    (Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing).Content
} catch { Write-Host $_.Exception.Message }
Write-Host "  GET /"
try {
    (Invoke-WebRequest -Uri "http://localhost:8000/" -UseBasicParsing).Content
} catch { Write-Host $_.Exception.Message }

Write-Host "  To test upload/analyze: run quick_test.py after container is up, or use curl/Postman"

Write-Host ""
Write-Host "[4/4] Stopping container..." -ForegroundColor Yellow
docker stop acgcs-test
docker rm acgcs-test
Write-Host ""
Write-Host "=== Docker test complete ===" -ForegroundColor Green
