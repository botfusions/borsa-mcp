#!/bin/bash
# Borsa MCP - VPS Deployment Script
# Usage: bash deploy-vps.sh

set -e

echo "🚀 Borsa MCP VPS Deployment Script"
echo "===================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}Please run as root (sudo bash deploy-vps.sh)${NC}"
    exit 1
fi

# Get non-root username
read -p "Enter non-root username (default: borsa): " USERNAME
USERNAME=${USERNAME:-borsa}

echo -e "${GREEN}Step 1/8: System Update${NC}"
apt update && apt upgrade -y

echo -e "${GREEN}Step 2/8: Install Dependencies${NC}"
apt install -y python3.11 python3.11-venv python3-pip nginx certbot python3-certbot-nginx git ufw

echo -e "${GREEN}Step 3/8: Create User${NC}"
if id "$USERNAME" &>/dev/null; then
    echo "User $USERNAME already exists"
else
    adduser --disabled-password --gecos "" $USERNAME
    usermod -aG sudo $USERNAME
    echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME
fi

echo -e "${GREEN}Step 4/8: Clone Repository${NC}"
su - $USERNAME << 'EOSU'
cd ~
if [ -d "borsa-mcp" ]; then
    echo "Repository already exists, pulling latest changes..."
    cd borsa-mcp
    git pull
else
    git clone https://github.com/botfusions/borsa-mcp.git
    cd borsa-mcp
fi

# Checkout correct branch
git checkout claude/security-improvements-011CV2tLmRbkzcvbPWij5WZ3 || git checkout main

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "✅ Application installed successfully"
EOSU

echo -e "${GREEN}Step 5/8: Create Systemd Service${NC}"
cat > /etc/systemd/system/borsa-mcp.service << EOF
[Unit]
Description=Borsa MCP FastAPI Server
After=network.target

[Service]
Type=simple
User=$USERNAME
WorkingDirectory=/home/$USERNAME/borsa-mcp
Environment="PATH=/home/$USERNAME/borsa-mcp/venv/bin"
ExecStart=/home/$USERNAME/borsa-mcp/venv/bin/uvicorn main:app --host 0.0.0.0 --port 9000 --workers 2
Restart=always
RestartSec=10
StandardOutput=append:/home/$USERNAME/borsa-mcp/logs/app.log
StandardError=append:/home/$USERNAME/borsa-mcp/logs/error.log

[Install]
WantedBy=multi-user.target
EOF

# Create logs directory
mkdir -p /home/$USERNAME/borsa-mcp/logs
chown -R $USERNAME:$USERNAME /home/$USERNAME/borsa-mcp/logs

# Enable and start service
systemctl daemon-reload
systemctl enable borsa-mcp
systemctl start borsa-mcp

echo -e "${GREEN}Step 6/8: Configure Firewall${NC}"
ufw --force enable
ufw allow 22/tcp   # SSH
ufw allow 80/tcp   # HTTP
ufw allow 443/tcp  # HTTPS
ufw status

echo -e "${GREEN}Step 7/8: Configure Nginx${NC}"
read -p "Enter your domain name (or press Enter to skip SSL): " DOMAIN

if [ -z "$DOMAIN" ]; then
    echo "Skipping domain configuration, using IP only"
    cat > /etc/nginx/sites-available/borsa-mcp << 'EOF'
server {
    listen 80 default_server;
    server_name _;

    location / {
        proxy_pass http://127.0.0.1:9000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # SSE support
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 86400;
    }
}
EOF
else
    cat > /etc/nginx/sites-available/borsa-mcp << EOF
server {
    listen 80;
    server_name $DOMAIN;

    location / {
        proxy_pass http://127.0.0.1:9000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;

        # SSE support
        proxy_buffering off;
        proxy_cache off;
        proxy_read_timeout 86400;
    }
}
EOF
fi

# Enable site
rm -f /etc/nginx/sites-enabled/default
ln -sf /etc/nginx/sites-available/borsa-mcp /etc/nginx/sites-enabled/
nginx -t && systemctl restart nginx

echo -e "${GREEN}Step 8/8: SSL Certificate (Optional)${NC}"
if [ ! -z "$DOMAIN" ]; then
    read -p "Install SSL certificate with Let's Encrypt? (y/n): " INSTALL_SSL
    if [ "$INSTALL_SSL" = "y" ]; then
        certbot --nginx -d $DOMAIN --non-interactive --agree-tos --register-unsafely-without-email || echo "SSL installation failed, continuing..."
    fi
fi

echo ""
echo -e "${GREEN}================================${NC}"
echo -e "${GREEN}🎉 Deployment Complete!${NC}"
echo -e "${GREEN}================================${NC}"
echo ""
echo "Service Status:"
systemctl status borsa-mcp --no-pager

echo ""
echo "Test your API:"
if [ -z "$DOMAIN" ]; then
    SERVER_IP=$(hostname -I | awk '{print $1}')
    echo "  curl http://$SERVER_IP/health"
    echo "  curl http://$SERVER_IP/mcp/tools"
    echo ""
    echo "Access API at: http://$SERVER_IP"
else
    echo "  curl https://$DOMAIN/health"
    echo "  curl https://$DOMAIN/mcp/tools"
    echo ""
    echo "Access API at: https://$DOMAIN"
fi

echo ""
echo "Useful commands:"
echo "  sudo systemctl status borsa-mcp   # Check status"
echo "  sudo systemctl restart borsa-mcp  # Restart service"
echo "  sudo journalctl -u borsa-mcp -f   # View logs"
echo "  tail -f /home/$USERNAME/borsa-mcp/logs/app.log  # Application logs"
echo ""
echo -e "${YELLOW}⚠️  Remember to:${NC}"
echo "  1. Update CORS settings in main.py"
echo "  2. Add API key authentication"
echo "  3. Configure environment variables"
echo "  4. Set up monitoring (Sentry, UptimeRobot)"
echo ""
