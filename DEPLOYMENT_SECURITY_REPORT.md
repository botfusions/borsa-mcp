# 🔒 Borsa MCP - Güvenlik ve Deployment Raporu

**Tarih:** 2025-11-11
**Proje:** borsa-mcp v2.0.0
**Hazırlayan:** Claude AI Security Analysis

---

## 📋 İçindekiler
1. [Güvenlik Analizi](#güvenlik-analizi)
2. [Deployment Seçenekleri](#deployment-seçenekleri)
3. [Endpoint Sorunları ve Çözümleri](#endpoint-sorunları-ve-çözümleri)
4. [Adım Adım Deployment Kılavuzu](#adım-adım-deployment-kılavuzu)
5. [Öneriler ve İyileştirmeler](#öneriler-ve-iyileştirmeler)

---

## 🔴 1. Güvenlik Analizi

### Kritik Güvenlik Sorunları

#### 1.1 SSL Doğrulama Kapalı (CRITICAL)
**Durum:** `borsa_mcp_server.py` dosyasında SSL doğrulaması global olarak devre dışı
```python
# MEVCUT KOD (GÜVENLİK RİSKİ!)
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['PYTHONHTTPSVERIFY'] = '0'
```

**Risk:** Man-in-the-middle (MITM) saldırıları
**Çözüm:** Provider seviyesinde seçici SSL bypass

#### 1.2 CORS Tamamen Açık (HIGH)
**Durum:** `main.py` - Tüm originlere izin verilmiş
```python
# MEVCUT KOD (GÜVENLİK RİSKİ!)
allow_origins=["*"]
```

**Risk:** CSRF saldırıları, yetkisiz API kullanımı
**Çözüm:** Belirli domainlere kısıtlama + environment variable

#### 1.3 Authentication/Authorization YOK (HIGH)
**Durum:** FastAPI endpoint'leri herkese açık
**Risk:**
- API abuse
- Rate limiting yok
- DoS saldırılarına açık
**Çözüm:** API Key veya JWT authentication

#### 1.4 Rate Limiting YOK (MEDIUM)
**Durum:** Sınırsız istek yapılabilir
**Risk:** Resource exhaustion, abuse
**Çözüm:** `slowapi` ile rate limiting

#### 1.5 Hassas Bilgi Loglanıyor (MEDIUM)
**Durum:** `logs/` klasörüne tüm işlemler yazılıyor
**Risk:** Credential leak (eğer gelecekte API key eklenirse)
**Çözüm:** Log sanitization

#### 1.6 Error Handling Zayıf (LOW)
**Durum:** Stack trace'ler kullanıcıya gösteriliyor
**Risk:** Information disclosure
**Çözüm:** Generic error messages

### Güvenlik Skoru
```
MEVCUT DURUM: 4.2/10 (Düşük)
HEDEF: 8.5/10 (İyi)
```

---

## 🚀 2. Deployment Seçenekleri

### Neden Vercel ve VPS'de Sorun Yaşandı?

#### ❌ Vercel'de Neden Çalışmadı?
1. **MCP Stdio Protokolü:** Vercel serverless - stdin/stdout yok
2. **Port Binding:** Vercel otomatik port atar, 9000 kullanılamaz
3. **Timeout:** Serverless function timeout (10-60 saniye)
4. **WebSocket/SSE Sınırlamaları:** Edge runtime SSE'yi tam desteklemez

**Sonuç:** Vercel sadece REST API için kullanılabilir, MCP sunucusu için uygun değil

#### ⚠️ VPS'de Endpoint Alamamanızın Nedenleri
1. **Firewall:** Port 9000 kapalı olabilir
2. **Uvicorn Binding:** `127.0.0.1` yerine `0.0.0.0` kullanılmalı
3. **Reverse Proxy Yok:** Nginx/Caddy yapılandırması eksik
4. **SSL/HTTPS Yok:** Modern tarayıcılar HTTP'ye izin vermez
5. **Process Management Yok:** Python process'i duruyor

### ✅ Önerilen Deployment Platformları

| Platform | MCP Stdio | FastAPI | Maliyet | Zorluk | Önerilen |
|----------|-----------|---------|---------|--------|----------|
| **Railway** | ❌ | ✅ | $5-20/ay | Kolay | ⭐⭐⭐⭐⭐ |
| **Render** | ❌ | ✅ | $7-25/ay | Kolay | ⭐⭐⭐⭐⭐ |
| **DigitalOcean** | ✅ | ✅ | $6-12/ay | Orta | ⭐⭐⭐⭐ |
| **Hetzner** | ✅ | ✅ | €4-8/ay | Orta | ⭐⭐⭐⭐ |
| **AWS Lightsail** | ✅ | ✅ | $5-10/ay | Orta | ⭐⭐⭐⭐ |
| **Fly.io** | ❌ | ✅ | $3-15/ay | Kolay | ⭐⭐⭐ |
| **Vercel** | ❌ | ⚠️ | Ücretsiz | Kolay | ⭐⭐ |

**En İyi Seçenekler:**

### 🥇 1. Railway (EN KOLAY - ÖNERİLEN)
- Otomatik SSL
- GitHub entegrasyonu
- Auto-deploy
- Environment variables
- Logs dashboard
- Public URL otomatik

### 🥈 2. Render (İKİNCİ EN İYİ)
- Ücretsiz tier (750 saat/ay)
- Otomatik SSL
- GitHub auto-deploy
- Health checks

### 🥉 3. DigitalOcean/Hetzner (EN UCUZ)
- Tam kontrol
- MCP stdio destekler
- Daha fazla yapılandırma gerekli

---

## 🛠️ 3. Endpoint Sorunları ve Çözümleri

### VPS'de Karşılaşılan Tipik Sorunlar

#### Sorun 1: "Connection Refused"
```bash
# Neden: Uvicorn 127.0.0.1'e bind olmuş
# Çözüm:
uvicorn main:app --host 0.0.0.0 --port 9000
```

#### Sorun 2: "No route to host"
```bash
# Neden: Firewall port'u blokluyor
# Çözüm (Ubuntu/Debian):
sudo ufw allow 9000
sudo ufw status

# Çözüm (CentOS/RHEL):
sudo firewall-cmd --permanent --add-port=9000/tcp
sudo firewall-cmd --reload
```

#### Sorun 3: "SSL Required"
```bash
# Neden: Tarayıcı HTTPS istiyor
# Çözüm: Nginx + Let's Encrypt
```

#### Sorun 4: Process Duruyor
```bash
# Neden: SSH bağlantısı kopunca process kapanıyor
# Çözüm: systemd service veya PM2
```

---

## 📦 4. Adım Adım Deployment Kılavuzu

### SEÇENEK A: Railway ile Deploy (EN KOLAY - 15 dakika)

#### Adım 1: Railway Hazırlığı
```bash
# Proje dosyalarını hazırla
cat > railway.json << 'EOF'
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "NIXPACKS"
  },
  "deploy": {
    "startCommand": "uvicorn main:app --host 0.0.0.0 --port $PORT",
    "healthcheckPath": "/health",
    "healthcheckTimeout": 100
  }
}
EOF

# Procfile oluştur
echo "web: uvicorn main:app --host 0.0.0.0 --port \$PORT" > Procfile
```

#### Adım 2: Railway'e Deploy
1. https://railway.app/ adresine git
2. "New Project" > "Deploy from GitHub repo"
3. Repository'yi seç: `botfusions/borsa-mcp`
4. Environment Variables ekle:
   - `PORT`: 9000
   - `PYTHONUNBUFFERED`: 1
5. "Deploy" butonuna bas
6. Domain oluştur: Settings > Generate Domain
7. Public URL'i al: `https://borsa-mcp-production.up.railway.app`

**Bitti!** 🎉

#### Test Et
```bash
curl https://borsa-mcp-production.up.railway.app/health
curl https://borsa-mcp-production.up.railway.app/mcp/tools
```

---

### SEÇENEK B: Render ile Deploy (15 dakika)

#### Adım 1: render.yaml Oluştur
```yaml
services:
  - type: web
    name: borsa-mcp-api
    runtime: python
    buildCommand: "pip install -r requirements.txt"
    startCommand: "uvicorn main:app --host 0.0.0.0 --port $PORT"
    envVars:
      - key: PYTHON_VERSION
        value: 3.11.0
      - key: PORT
        value: 9000
    healthCheckPath: /health
```

#### Adım 2: Render'a Deploy
1. https://render.com/ - Sign up
2. "New" > "Web Service"
3. Connect GitHub repo
4. Name: `borsa-mcp-api`
5. Build Command: `pip install -r requirements.txt`
6. Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
7. "Create Web Service"

**Public URL:** `https://borsa-mcp-api.onrender.com`

---

### SEÇENEK C: VPS ile Production Deploy (60 dakika)

#### Sistem Gereksinimleri
- Ubuntu 22.04 LTS
- 2 CPU, 2GB RAM
- 20GB disk
- Public IP

#### Adım 1: Sunucu Hazırlığı
```bash
# SSH ile bağlan
ssh root@YOUR_VPS_IP

# Sistem güncellemesi
apt update && apt upgrade -y

# Python 3.11 kurulumu
apt install -y python3.11 python3.11-venv python3-pip nginx certbot python3-certbot-nginx

# Non-root kullanıcı oluştur
adduser borsa
usermod -aG sudo borsa
su - borsa
```

#### Adım 2: Uygulama Kurulumu
```bash
# Proje klonla
cd ~
git clone https://github.com/botfusions/borsa-mcp.git
cd borsa-mcp
git checkout claude/security-improvements-011CV2tLmRbkzcvbPWij5WZ3

# Virtual environment
python3.11 -m venv venv
source venv/bin/activate

# Bağımlılıkları yükle
pip install --upgrade pip
pip install -r requirements.txt
```

#### Adım 3: Systemd Service
```bash
# Service dosyası oluştur
sudo nano /etc/systemd/system/borsa-mcp.service
```

İçeriği yapıştır:
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
# Service'i aktifleştir
sudo systemctl daemon-reload
sudo systemctl enable borsa-mcp
sudo systemctl start borsa-mcp
sudo systemctl status borsa-mcp
```

#### Adım 4: Nginx Reverse Proxy
```bash
# Nginx yapılandırması
sudo nano /etc/nginx/sites-available/borsa-mcp
```

İçeriği yapıştır:
```nginx
server {
    listen 80;
    server_name YOUR_DOMAIN.com;  # Değiştir!

    location / {
        proxy_pass http://127.0.0.1:9000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # SSE için önemli
        proxy_buffering off;
        proxy_cache off;
    }
}
```

```bash
# Nginx'i aktifleştir
sudo ln -s /etc/nginx/sites-available/borsa-mcp /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

#### Adım 5: SSL Sertifikası (Let's Encrypt)
```bash
# Certbot ile SSL
sudo certbot --nginx -d YOUR_DOMAIN.com

# Otomatik yenileme
sudo systemctl status certbot.timer
```

#### Adım 6: Firewall
```bash
# UFW yapılandırması
sudo ufw allow 22/tcp   # SSH
sudo ufw allow 80/tcp   # HTTP
sudo ufw allow 443/tcp  # HTTPS
sudo ufw enable
sudo ufw status
```

#### Test Et
```bash
# Health check
curl https://YOUR_DOMAIN.com/health

# API test
curl https://YOUR_DOMAIN.com/mcp/tools
```

---

## 💡 5. Öneriler ve İyileştirmeler

### Acil İyileştirmeler (1-2 Gün)

#### 1. SSL Doğrulama Düzeltmesi
```python
# borsa_mcp_server.py - SSL bypass'ı kaldır
# providers/*.py - Her provider'da seçici SSL
import httpx

client = httpx.AsyncClient(
    verify=True,  # SSL doğrulamasını aç
    timeout=60.0
)
```

#### 2. CORS Kısıtlaması
```python
# main.py
ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS",
    "https://yourdomain.com,https://n8n.yourdomain.com"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

#### 3. API Key Authentication
```python
# main.py - Basit API Key
from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader

API_KEY = os.getenv("BORSA_API_KEY")
api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return api_key

# Endpoint'lere ekle
@app.post("/mcp/messages", dependencies=[Depends(verify_api_key)])
async def handle_mcp_message(...):
    ...
```

#### 4. Rate Limiting
```python
# requirements.txt'e ekle: slowapi==0.1.9

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/mcp/messages")
@limiter.limit("30/minute")  # Dakikada 30 istek
async def handle_mcp_message(request: Request, ...):
    ...
```

### Orta Vadeli İyileştirmeler (1 Hafta)

#### 5. Environment Variables
```bash
# .env.example oluştur
PORT=9000
LOG_LEVEL=INFO
ALLOWED_ORIGINS=https://yourdomain.com
BORSA_API_KEY=your_secure_key_here
RATE_LIMIT=30/minute
ENABLE_DOCS=false  # Production'da Swagger'ı kapat
```

#### 6. Docker Compose (Production)
```yaml
# docker-compose.yml
version: '3.8'
services:
  borsa-mcp:
    build: .
    ports:
      - "9000:9000"
    environment:
      - PORT=9000
      - BORSA_API_KEY=${BORSA_API_KEY}
      - ALLOWED_ORIGINS=${ALLOWED_ORIGINS}
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

#### 7. Monitoring ve Logging
```python
# Sentry entegrasyonu
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration

sentry_sdk.init(
    dsn=os.getenv("SENTRY_DSN"),
    integrations=[FastApiIntegration()],
    traces_sample_rate=0.1,
)
```

### Uzun Vadeli İyileştirmeler (1 Ay)

#### 8. CI/CD Pipeline (GitHub Actions)
```yaml
# .github/workflows/deploy.yml
name: Deploy to Production
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Deploy to Railway
        env:
          RAILWAY_TOKEN: ${{ secrets.RAILWAY_TOKEN }}
        run: |
          npm install -g @railway/cli
          railway up
```

#### 9. Database (Cache Layer)
```python
# Redis ile caching
import redis.asyncio as redis

redis_client = redis.from_url(
    os.getenv("REDIS_URL", "redis://localhost:6379"),
    encoding="utf-8",
    decode_responses=True
)

# Provider'larda kullan
@lru_cache(maxsize=1000)
async def get_stock_data(ticker: str):
    cached = await redis_client.get(f"stock:{ticker}")
    if cached:
        return json.loads(cached)
    # ... API çağrısı
    await redis_client.setex(f"stock:{ticker}", 300, json.dumps(data))
    return data
```

#### 10. Load Balancing (Multiple Workers)
```bash
# Gunicorn ile production deployment
gunicorn main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:9000 \
  --timeout 120 \
  --access-logfile - \
  --error-logfile -
```

---

## 📊 Deployment Karşılaştırma Tablosu

| Özellik | Railway | Render | VPS (DigitalOcean) |
|---------|---------|--------|-------------------|
| **Kurulum Süresi** | 15 dk | 15 dk | 60 dk |
| **Zorluk** | ⭐ Kolay | ⭐ Kolay | ⭐⭐⭐ Orta |
| **Aylık Maliyet** | $5-20 | $7-25 (ücretsiz tier var) | $6-12 |
| **SSL Otomatik** | ✅ | ✅ | ❌ (Manuel) |
| **Auto-Deploy** | ✅ | ✅ | ❌ (CI/CD gerekli) |
| **Logs Dashboard** | ✅ | ✅ | ❌ (Manuel setup) |
| **Monitoring** | ✅ | ✅ | ❌ (3rd party tool gerekli) |
| **MCP Stdio Desteği** | ❌ | ❌ | ✅ |
| **Scaling** | Kolay | Kolay | Manuel |
| **Backup** | Otomatik | Otomatik | Manuel |

---

## 🎯 Tavsiyeler

### Hızlı Başlangıç İçin (Bugün)
1. **Railway kullanın** - En hızlı ve sorunsuz
2. Sadece FastAPI'yi deploy edin (MCP stdio değil)
3. Public URL'i alın ve test edin

### Production İçin (1 Hafta)
1. **VPS (DigitalOcean/Hetzner)** - Tam kontrol
2. Nginx + SSL yapılandırması
3. Systemd service
4. Monitoring ekleyin (UptimeRobot, Sentry)

### Güvenlik İçin (Acil)
1. SSL bypass'ı kaldırın
2. CORS'u kısıtlayın
3. API Key ekleyin
4. Rate limiting aktifleştirin

---

## ✅ Checklist

### Deployment Öncesi
- [ ] `requirements.txt` güncel
- [ ] `.env.example` oluşturuldu
- [ ] `Procfile` veya `railway.json` eklendi
- [ ] Health check endpoint test edildi
- [ ] CORS ayarları yapılandırıldı
- [ ] SSL sertifikası hazır (VPS için)

### Deployment Sonrası
- [ ] Public URL çalışıyor
- [ ] `/health` endpoint 200 dönüyor
- [ ] `/mcp/tools` tool listesi gösteriyor
- [ ] SSE endpoint `/mcp` bağlanıyor
- [ ] Logs kontrol edildi
- [ ] Performance test yapıldı (load testing)
- [ ] Monitoring kuruldu
- [ ] Backup stratejisi belirlendi

### Güvenlik Kontrolleri
- [ ] SSL/HTTPS aktif
- [ ] API Key authentication var
- [ ] Rate limiting çalışıyor
- [ ] CORS kısıtlanmış
- [ ] Logs sanitized
- [ ] Error messages generic
- [ ] Firewall kuralları aktif
- [ ] SSH key-based authentication (VPS)

---

## 📞 Destek ve Kaynaklar

### Dokümantasyon
- **Railway:** https://docs.railway.app/
- **Render:** https://render.com/docs
- **FastAPI:** https://fastapi.tiangolo.com/
- **MCP Protocol:** https://modelcontextprotocol.io/

### Monitoring Araçları
- **Uptime:** https://uptimerobot.com/ (Ücretsiz)
- **Logs:** https://betterstack.com/logs (Ücretsiz tier)
- **Errors:** https://sentry.io/ (Ücretsiz 5K events)

### Load Testing
```bash
# Apache Bench ile test
ab -n 1000 -c 10 https://your-api.com/health

# wrk ile test
wrk -t12 -c400 -d30s https://your-api.com/mcp/tools
```

---

## 🚨 Yaygın Hatalar ve Çözümleri

### Hata 1: "ModuleNotFoundError"
```bash
# Çözüm: requirements.txt eksik paket var
pip freeze > requirements.txt
git add requirements.txt
git commit -m "Update requirements"
```

### Hata 2: "Port already in use"
```bash
# Port 9000 kullanımda
lsof -i :9000
kill -9 <PID>
# Veya farklı port kullan
uvicorn main:app --port 8000
```

### Hata 3: "Connection timeout"
```bash
# Uvicorn timeout artır
uvicorn main:app --timeout-keep-alive 120
```

### Hata 4: "SSL Certificate verify failed"
```bash
# Geçici çözüm (development için)
export PYTHONHTTPSVERIFY=0
# Kalıcı çözüm: SSL doğrulamasını providerda düzelt
```

---

## 📈 Performans Optimizasyonları

### 1. Uvicorn Workers
```bash
# CPU core sayısı kadar worker
uvicorn main:app --workers 4 --host 0.0.0.0 --port 9000
```

### 2. Redis Caching
```python
# Sık kullanılan dataları cache'le
from redis import Redis
redis_client = Redis(host='localhost', port=6379, decode_responses=True)

@app.get("/market/overview")
async def market_overview():
    cached = redis_client.get("market:overview")
    if cached:
        return json.loads(cached)
    data = await fetch_market_data()
    redis_client.setex("market:overview", 60, json.dumps(data))
    return data
```

### 3. Connection Pooling
```python
# httpx client pool
client = httpx.AsyncClient(
    limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)
)
```

---

## 🎉 Sonuç

Bu proje **Railway** veya **Render** ile 15 dakikada deploy edilebilir. VPS tercih ederseniz 1 saat içinde production-ready hale gelebilir. Güvenlik iyileştirmeleri ile birlikte profesyonel bir API servisi olur.

**Önerilen Aksiyon Planı:**
1. ✅ Railway'e hemen deploy et (15 dk) - Test için
2. ✅ Güvenlik yamalarını uygula (1 gün) - CORS, API Key, Rate Limit
3. ✅ VPS setup yap (1 hafta) - Production için
4. ✅ Monitoring ekle (1 hafta) - Sentry, Uptime Robot
5. ✅ CI/CD pipeline kur (1 ay) - Otomasyonlar

**Başarılar! 🚀**
