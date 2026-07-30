# ☁️ Stock Engine Pro V3 — 24/7 Cloud Deployment Guide

This guide details how to host **Stock Engine Pro V3** (Daytrade Bot + Swing Bot + Web Dashboard) in the cloud **24/7**, allowing you to monitor and control your paper trading from your mobile phone without keeping your home PC turned on.

---

## 🎯 Recommended Cloud Providers

| Provider | Cost | Best For |
|---|---|---|
| **AWS (Amazon Web Services)** | **FREE for 12 months** (t3.micro / t2.micro) | 100% Free 24/7 hosting |
| **DigitalOcean** | **$4 - $6 / month** (Basic Droplet) | Simplest 1-click Linux VPS |
| **Hetzner** | **~€3.80 / month** (CX22) | High performance EU/US VPS |

---

## 🚀 Quick 5-Minute Deployment Steps

### Step 1: Create a Linux VPS (Ubuntu 22.04 LTS)
1. Sign up on **AWS Free Tier** or **DigitalOcean**.
2. Create an **Ubuntu 22.04 LTS** virtual machine.
3. Note your VPS **Public IP address** (e.g. `159.65.123.45`).

---

### Step 2: Connect to your VPS via SSH
Open Terminal (macOS/Linux) or PowerShell (Windows) and run:
```bash
ssh root@YOUR_SERVER_IP
```

---

### Step 3: Clone Repository & Run One-Click Deployment
Run the following commands on your server:

```bash
# 1. Clone repository
git clone https://github.com/H2o2026/stock_engine_pro_v3_production_ready.git stock-v3
cd stock-v3

# 2. Copy and edit .env file with your Alpaca paper keys
cp .env.example .env
nano .env
```

Fill in your `.env` keys:
```env
DAYTRADE_APCA_API_KEY_ID=PK...
DAYTRADE_APCA_API_SECRET_KEY=...
SWING_APCA_API_KEY_ID=PK...
SWING_APCA_API_SECRET_KEY=...
```

Run the one-click installer:
```bash
bash scripts/deploy_cloud.sh
```

---

## 📱 Mobile Phone Access

Once deployment completes, open your mobile browser (Safari / Chrome) and go to:

```text
http://YOUR_SERVER_IP:5050
```

Bookmark this URL on your phone's home screen! You can now monitor positions, signals, and controls 24/7 from anywhere in the world.

---

## 🛠️ Essential Cloud Management Commands

| Action | Command |
|---|---|
| **View Live Bot Logs** | `docker compose logs -f` |
| **View Specific Bot Log** | `docker compose logs -f daytrade_bot` |
| **Restart All Services** | `docker compose restart` |
| **Stop All Services** | `docker compose down` |
| **Update Code & Redeploy** | `git pull && docker compose up -d --build` |
