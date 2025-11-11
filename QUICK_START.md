# 🚀 Borsa MCP - Hızlı Başlangıç Kılavuzu

## 3 Farklı Deployment Yöntemi

---

## 📦 1. Railway ile Deploy (EN KOLAY - 15 dakika)

### Adım 1: Railway Hesabı Oluştur
1. https://railway.app/ adresine git
2. "Login with GitHub" ile giriş yap

### Adım 2: Projeyi Deploy Et
1. Railway dashboard'da **"New Project"** butonuna tıkla
2. **"Deploy from GitHub repo"** seç
3. Repository'yi seç: `botfusions/borsa-mcp`
4. Branch seç: `claude/security-improvements-011CV2tLmRbkzcvbPWij5WZ3`
5. Railway otomatik olarak deploy edecek

### Adım 3: Domain Oluştur
1. Deployed projeye tıkla
2. **Settings** > **Networking** > **Generate Domain**
3. Public URL'ini al: `https://borsa-mcp-production.up.railway.app`

### Adım 4: Test Et
```bash
# Health check
curl https://borsa-mcp-production.up.railway.app/health

# Tools listesi
curl https://borsa-mcp-production.up.railway.app/mcp/tools

# API test
curl -X POST https://borsa-mcp-production.up.railway.app/analyze/technical/THYAO \
  -H "Content-Type: application/json" \
  -d '{"period": "1mo"}'
```

### Environment Variables (Opsiyonel)
Railway'de **Variables** sekmesinden ekle:
```
PORT=9000
PYTHONUNBUFFERED=1
LOG_LEVEL=INFO
```

**✅ Bitti! API'niz yayında.**

---

## 🎨 2. Render ile Deploy (15 dakika)

### Adım 1: Render Hesabı
1. https://render.com/ - Sign up with GitHub
2. Dashboard'a git

### Adım 2: Web Service Oluştur
1. **"New +"** > **"Web Service"**
2. **"Connect a repository"** > `botfusions/borsa-mcp` seç
3. Ayarları doldur:
   - **Name:** `borsa-mcp-api`
   - **Region:** Frankfurt (Europe)
   - **Branch:** `claude/security-improvements-011CV2tLmRbkzcvbPWij5WZ3`
   - **Runtime:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`

### Adım 3: Environment Variables
```
PYTHON_VERSION=3.11.0
PORT=10000
```

### Adım 4: Deploy
1. **"Create Web Service"** butonuna tıkla
2. 5-10 dakika bekle (ilk deploy biraz uzun sürer)
3. Public URL: `https://borsa-mcp-api.onrender.com`

### Test
```bash
curl https://borsa-mcp-api.onrender.com/health
```

**💡 Not:** Ücretsiz tier 750 saat/ay sunuyor. 15 dakika inaktivite sonrası uyuyor, ilk istek 30 saniye sürebilir.

---

## 🖥️ 3. VPS ile Deploy (60 dakika)

### Gereksinimler
- Ubuntu 22.04 LTS server
- Root erişimi
- Domain (opsiyonel, SSL için gerekli)

### Tek Komutla Kurulum
```bash
# SSH ile VPS'e bağlan
ssh root@YOUR_VPS_IP

# Deploy script'ini indir ve çalıştır
curl -sSL https://raw.githubusercontent.com/botfusions/borsa-mcp/main/deploy-vps.sh | bash
```

### Manuel Kurulum
Eğer script'i indirip çalıştırmak istemiyorsanız:

```bash
# 1. Sistemi güncelle
apt update && apt upgrade -y

# 2. Gerekli paketleri kur
apt install -y python3.11 python3.11-venv python3-pip nginx certbot python3-certbot-nginx git ufw

# 3. Kullanıcı oluştur
adduser borsa
usermod -aG sudo borsa
su - borsa

# 4. Projeyi klonla
cd ~
git clone https://github.com/botfusions/borsa-mcp.git
cd borsa-mcp
git checkout claude/security-improvements-011CV2tLmRbkzcvbPWij5WZ3

# 5. Virtual environment oluştur
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 6. Systemd service oluştur (root olarak)
exit  # borsa kullanıcısından çık
nano /etc/systemd/system/borsa-mcp.service
```

Service dosyası içeriği:
```ini
[Unit]
Description=Borsa MCP FastAPI Server
After=network.target

[Service]
Type=simple
User=borsa
WorkingDirectory=/home/borsa/borsa-mcp
Environment="PATH=/home/borsa/borsa-mcp/venv/bin"
ExecStart=/home/borsa/borsa-mcp/venv/bin/uvicorn main:app --host 0.0.0.0 --port 9000 --workers 2
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
# 7. Service'i başlat
systemctl daemon-reload
systemctl enable borsa-mcp
systemctl start borsa-mcp
systemctl status borsa-mcp

# 8. Firewall yapılandır
ufw allow 22/tcp
ufw allow 80/tcp
ufw allow 443/tcp
ufw enable

# 9. Nginx yapılandır
nano /etc/nginx/sites-available/borsa-mcp
```

Nginx config:
```nginx
server {
    listen 80;
    server_name YOUR_DOMAIN.com;

    location / {
        proxy_pass http://127.0.0.1:9000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_buffering off;
        proxy_cache off;
    }
}
```

```bash
# 10. Nginx'i aktifleştir
ln -s /etc/nginx/sites-available/borsa-mcp /etc/nginx/sites-enabled/
nginx -t
systemctl restart nginx

# 11. SSL sertifikası (Let's Encrypt)
certbot --nginx -d YOUR_DOMAIN.com
```

### Test
```bash
# Local test
curl http://localhost:9000/health

# External test
curl https://YOUR_DOMAIN.com/health
```

---

## 🐳 4. Docker ile Deploy (20 dakika)

### Tek Sunucu (Sadece API)
```bash
# Image'i build et
docker build -t borsa-mcp .

# Container'ı çalıştır
docker run -d \
  --name borsa-mcp \
  -p 9000:9000 \
  --restart unless-stopped \
  borsa-mcp

# Logları kontrol et
docker logs -f borsa-mcp
```

### Docker Compose (Önerilen)
```bash
# Compose ile başlat
docker-compose up -d

# Logları kontrol et
docker-compose logs -f

# Durdur
docker-compose down
```

### Test
```bash
curl http://localhost:9000/health
```

---

## ✅ Deployment Sonrası Kontroller

### 1. Health Check
```bash
curl https://your-api.com/health

# Beklenen yanıt:
{
  "status": "healthy",
  "timestamp": "2025-11-11T12:00:00"
}
```

### 2. API Documentation
Tarayıcıda aç: `https://your-api.com/docs`

### 3. MCP Tools
```bash
curl https://your-api.com/mcp/tools

# 4 tool görmeli:
# - borsa_technical_analysis
# - maestro_full_analysis
# - market_sentiment_analysis
# - market_overview
```

### 4. Test Request
```bash
# Market overview
curl https://your-api.com/market/overview

# Technical analysis
curl -X POST https://your-api.com/analyze/technical/THYAO \
  -H "Content-Type: application/json" \
  -d '{"period": "1mo", "interval": "1d"}'
```

---

## 🔧 Sorun Giderme

### Railway/Render Deploy Hataları

**Hata: "Build failed"**
```bash
# Çözüm: requirements.txt kontrol
pip freeze > requirements.txt
git add requirements.txt
git commit -m "Update requirements"
git push
```

**Hata: "Application timeout"**
```bash
# Çözüm: Health check endpoint kontrol et
curl https://your-app.railway.app/health
# Eğer 500 dönüyorsa logs kontrol et
```

### VPS Endpoint Sorunları

**Hata: "Connection refused"**
```bash
# 1. Service çalışıyor mu kontrol et
sudo systemctl status borsa-mcp

# 2. Port dinliyor mu kontrol et
sudo lsof -i :9000

# 3. Firewall kontrol et
sudo ufw status

# 4. Nginx kontrol et
sudo nginx -t
sudo systemctl status nginx
```

**Hata: "502 Bad Gateway"**
```bash
# Uvicorn başlamadı, logs kontrol et
sudo journalctl -u borsa-mcp -n 50

# Nginx logs
sudo tail -f /var/log/nginx/error.log
```

**Hata: "SSL certificate problem"**
```bash
# Let's Encrypt yenileme
sudo certbot renew
sudo systemctl reload nginx
```

### Docker Sorunları

**Container başlamıyor**
```bash
# Logs kontrol et
docker logs borsa-mcp

# Interactive mode ile test et
docker run -it --rm -p 9000:9000 borsa-mcp bash
# Container içinde:
python main.py
```

**Port already in use**
```bash
# Port'u kim kullanıyor bul
sudo lsof -i :9000
# Process'i durdur
docker stop borsa-mcp
# Veya farklı port kullan
docker run -p 8000:9000 borsa-mcp
```

---

## 📊 Performans İzleme

### Uptime Monitoring
1. https://uptimerobot.com/ - Ücretsiz 50 monitor
2. Health check URL ekle: `https://your-api.com/health`
3. Interval: 5 dakika

### Error Tracking
```bash
# Sentry kurulumu
pip install sentry-sdk[fastapi]
```

`main.py` içine ekle:
```python
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration

sentry_sdk.init(
    dsn="YOUR_SENTRY_DSN",
    integrations=[FastApiIntegration()],
)
```

### Logs
```bash
# Railway: Dashboard > Logs
# Render: Dashboard > Logs
# VPS:
sudo journalctl -u borsa-mcp -f
tail -f /home/borsa/borsa-mcp/logs/app.log
```

---

## 🎯 Önerilen Deployment Stratejisi

### Geliştirme (Development)
→ **Local** - `uvicorn main:app --reload`

### Test (Staging)
→ **Railway veya Render** - Hızlı deploy, kolay test

### Canlı (Production)
→ **VPS (DigitalOcean/Hetzner)** - Tam kontrol, cost-effective

---

## 💰 Maliyet Karşılaştırması

| Platform | Ücretsiz Tier | Ücretli Plan | Önerilen |
|----------|---------------|--------------|----------|
| **Railway** | $5 kredi | $5-20/ay | Test için |
| **Render** | 750 saat | $7-25/ay | Prototype için |
| **DigitalOcean** | $200 kredi (60 gün) | $6/ay | Production için ⭐ |
| **Hetzner** | ❌ | €4.5/ay | En ucuz |
| **AWS Lightsail** | 3 ay ücretsiz | $5/ay | AWS ekosistemi için |

---

## 📞 Destek

- **Dokümantasyon:** [DEPLOYMENT_SECURITY_REPORT.md](./DEPLOYMENT_SECURITY_REPORT.md)
- **Issues:** https://github.com/botfusions/borsa-mcp/issues

---

**Son Güncelleme:** 2025-11-11
