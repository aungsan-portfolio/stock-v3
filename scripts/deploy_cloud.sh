#!/usr/bin/env bash
# ── Stock Engine Pro V3 One-Click Cloud Deployment Script ──

set -e

echo "=========================================================="
echo "  Stock Engine Pro V3 — 24/7 Cloud Deployment Setup"
echo "=========================================================="

# 1. Update package index & install Docker if missing
if ! command -v docker &> /dev/null; then
    echo "[1/4] Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    rm get-docker.sh
else
    echo "[1/4] Docker is already installed."
fi

# 2. Install Docker Compose if missing
if ! docker compose version &> /dev/null; then
    echo "[2/4] Installing Docker Compose plugin..."
    apt-get update && apt-get install -y docker-compose-plugin
else
    echo "[2/4] Docker Compose is already installed."
fi

# 3. Check for .env credentials file
if [ ! -f .env ]; then
    if [ -f .env.example ]; then
        echo "[3/4] Creating .env from .env.example..."
        cp .env.example .env
        echo "⚠️  PLEASE UPDATE YOUR .env FILE WITH YOUR ALPACA API KEYS!"
    else
        echo "❌ Error: .env file missing. Please create .env before deploying."
        exit 1
    fi
else
    echo "[3/4] .env file found."
fi

# 4. Build and start containers 24/7
echo "[4/4] Building Docker containers and starting 24/7 background services..."
docker compose down --remove-orphans || true
docker compose up -d --build

SERVER_IP=$(curl -s https://api.ipify.org || hostname -I | awk '{print $1}')

echo "=========================================================="
echo "🎉 SUCCESS: 24/7 Cloud Deployment Complete!"
echo "=========================================================="
echo "📱 Mobile Web Dashboard Access : http://${SERVER_IP}:5050"
echo "🤖 Services Running in Background:"
echo "   - Daytrade Bot  : ACTIVE (24/7 Intraday Loop)"
echo "   - Swing Bot     : ACTIVE (24/7 30-min ML Loop)"
echo "   - Web Dashboard : ACTIVE (Port 5050)"
echo "=========================================================="
echo "Commands for managing your Cloud Bots:"
echo "  - View Logs       : docker compose logs -f"
echo "  - Restart Bots    : docker compose restart"
echo "  - Stop Bots       : docker compose down"
echo "=========================================================="
