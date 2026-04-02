# EnterpriseRAG Deployment Guide

Production deployment guide for EnterpriseRAG on workstations, servers, and production environments.

---

## Table of Contents

1. [Deployment Options](#deployment-options)
2. [Prerequisites](#prerequisites)
3. [Single Workstation Deployment](#single-workstation-deployment)
4. [Server Deployment](#server-deployment)
5. [Production Configuration](#production-configuration)
6. [Security Hardening](#security-hardening)
7. [Performance Tuning](#performance-tuning)
8. [Monitoring & Logging](#monitoring--logging)
9. [Backup & Recovery](#backup--recovery)
10. [Scaling Considerations](#scaling-considerations)

---

## Deployment Options

### Option 1: Single Workstation (Recommended for Field Use)

**Best for:**
- Field engineers in remote locations
- Offline operation requirements
- 10-100 documents
- 1-10 concurrent users

**Architecture:**
```
┌─────────────────────────────────────┐
│      Workstation/Laptop             │
│                                     │
│  ┌─────────────┐                    │
│  │  Browser    │                    │
│  │ (localhost) │                    │
│  └──────┬──────┘                    │
│         │                           │
│  ┌──────▼──────────────────┐        │
│  │  EnterpriseRAG          │        │
│  │  - FastAPI Backend      │        │
│  │  - React Frontend       │        │
│  └──────┬──────────────────┘        │
│         │                           │
│  ┌──────▼──────┐  ┌──────────┐     │
│  │JadeVectorDB │  │  Ollama  │     │
│  └─────────────┘  └──────────┘     │
└─────────────────────────────────────┘
```

### Option 2: Shared Server (Multi-User)

**Best for:**
- Central maintenance department
- 100-1000 documents
- 10-50 concurrent users
- Shared document library

**Architecture:**
```
┌────────────────────────────────────────────────┐
│            Network                             │
│                                                │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐  │
│  │ Laptop 1  │  │ Laptop 2  │  │ Laptop 3  │  │
│  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘  │
│        │              │              │        │
│        └──────────────┼──────────────┘        │
│                       │                       │
└───────────────────────┼───────────────────────┘
                        │
          ┌─────────────▼────────────────┐
          │    Application Server        │
          │                              │
          │  ┌────────────────────────┐  │
          │  │  EnterpriseRAG         │  │
          │  │  - Backend (FastAPI)   │  │
          │  │  - Frontend (React)    │  │
          │  └────────┬───────────────┘  │
          │           │                  │
          │  ┌────────▼───────┐          │
          │  │ JadeVectorDB   │          │
          │  └────────────────┘          │
          │                              │
          │  ┌────────────────┐          │
          │  │    Ollama      │          │
          │  └────────────────┘          │
          └──────────────────────────────┘
```

### Option 3: Docker Deployment

**Best for:**
- Easy deployment and updates
- Consistent environments
- Isolated installations

---

## Prerequisites

### Hardware Requirements

**Minimum (10-100 documents):**
- CPU: 4 cores (Intel i5 or equivalent)
- RAM: 8GB
- Storage: 50GB SSD
- Network: Optional (for server deployment)

**Recommended (100-1000 documents):**
- CPU: 8 cores (Intel i7 or equivalent)
- RAM: 16GB
- Storage: 100GB SSD
- Network: 1 Gbps (for server deployment)

**High Performance (1000+ documents):**
- CPU: 16+ cores
- RAM: 32GB+
- Storage: 256GB+ NVMe SSD
- GPU: Optional (for faster LLM inference)

### Software Requirements

**Operating System:**
- Linux (Ubuntu 20.04+, Debian 11+, CentOS 8+)
- macOS 11+
- Windows 10+ (with WSL2 for best performance)

**Required Software:**
- Python 3.9 or higher
- Node.js 18 or higher
- Ollama (latest version)
- JadeVectorDB (built from source)
- Git

---

## Single Workstation Deployment

### Step 1: System Preparation

```bash
# Update system
sudo apt update && sudo apt upgrade -y  # Debian/Ubuntu
# or
brew update && brew upgrade  # macOS

# Install Python 3.9+
sudo apt install python3 python3-pip python3-venv  # Linux
brew install python@3.9  # macOS

# Install Node.js 18+
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install -y nodejs  # Linux
brew install node@18  # macOS

# Install Git
sudo apt install git  # Linux
brew install git  # macOS
```

### Step 2: Install Dependencies

**Install Ollama:**
```bash
# Linux
curl https://ollama.ai/install.sh | sh

# macOS
brew install ollama

# Verify
ollama --version

# Pull required models
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

**Install JadeVectorDB:**
```bash
# Clone and build JadeVectorDB
git clone https://github.com/yourusername/JadeVectorDB.git
cd JadeVectorDB/backend
make
cd build
./jadevectordb &

# Verify
curl http://localhost:8080/health
```

### Step 3: Install EnterpriseRAG

```bash
# Clone repository (or copy files)
git clone https://github.com/yourusername/EnterpriseRAG.git
cd EnterpriseRAG

# Run setup
./scripts/setup.sh

# Configure for production
cd backend
cp .env.example .env
nano .env
# Set: MODE=production
# Verify other settings

cd ..
```

### Step 4: Initial Configuration

**Edit `backend/.env`:**
```bash
# Operation Mode
MODE=production

# API Server (change for network access)
API_HOST=0.0.0.0  # Listen on all interfaces
API_PORT=8000

# Frontend URL (change if accessing remotely)
FRONTEND_URL=http://localhost:5173

# JadeVectorDB
JADEVECTORDB_URL=http://localhost:8080
JADEVECTORDB_DATABASE_ID=maintenance_docs

# Ollama
OLLAMA_URL=http://localhost:11434
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
OLLAMA_LLM_MODEL=llama3.2:3b

# Performance tuning
CHUNK_SIZE=512
CHUNK_OVERLAP=50
TOP_K=5
MAX_UPLOAD_SIZE_MB=50
```

### Step 5: Create Database

```bash
# Create maintenance_docs database in JadeVectorDB
curl -X POST http://localhost:8080/api/v1/databases \
  -H "Content-Type: application/json" \
  -d '{
    "id": "maintenance_docs",
    "dimension": 384,
    "index_type": "HNSW",
    "index_config": {
      "M": 16,
      "ef_construction": 200
    }
  }'

# Verify
curl http://localhost:8080/api/v1/databases/maintenance_docs
```

### Step 6: Start System

```bash
# Start all services
./scripts/start.sh

# Or manually:
# Terminal 1: JadeVectorDB (if not already running)
cd /path/to/JadeVectorDB/backend/build
./jadevectordb

# Terminal 2: Backend
cd /path/to/EnterpriseRAG/backend
source venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8000

# Terminal 3: Frontend
cd /path/to/EnterpriseRAG/frontend
npm run dev
```

### Step 7: Verify Installation

```bash
# Health check
curl http://localhost:8000/api/health

# Expected response:
{
  "status": "healthy",
  "mode": "production",
  "components": {
    "database": "healthy",
    "llm": "healthy",
    "embeddings": "healthy"
  }
}

# Access UI
open http://localhost:5173  # macOS
xdg-open http://localhost:5173  # Linux
```

### Step 8: Upload Initial Documents

1. Navigate to http://localhost:5173/admin/documents
2. Upload your first maintenance manual
3. Wait for processing (30-60 seconds)
4. Test a query on the main page

---

## Server Deployment

### Network Access Configuration

**For multi-user access:**

1. **Update Backend Configuration:**
```bash
# backend/.env
API_HOST=0.0.0.0  # Listen on all network interfaces
API_PORT=8000

# Change frontend URL to server IP
FRONTEND_URL=http://192.168.1.100:5173  # Use actual server IP
```

2. **Update Frontend Configuration:**
```javascript
// frontend/.env (create if doesn't exist)
VITE_API_URL=http://192.168.1.100:8000
```

3. **Configure Firewall:**
```bash
# Ubuntu/Debian with ufw
sudo ufw allow 8000/tcp  # Backend API
sudo ufw allow 5173/tcp  # Frontend (or use nginx on 80/443)
sudo ufw enable

# CentOS/RHEL with firewalld
sudo firewall-cmd --permanent --add-port=8000/tcp
sudo firewall-cmd --permanent --add-port=5173/tcp
sudo firewall-cmd --reload
```

### Production Web Server (Nginx)

**Install Nginx:**
```bash
sudo apt install nginx  # Debian/Ubuntu
sudo yum install nginx  # CentOS/RHEL
```

**Configure Nginx:**
```nginx
# /etc/nginx/sites-available/enterpriserag
server {
    listen 80;
    server_name your-server.example.com;

    # Frontend
    location / {
        proxy_pass http://localhost:5173;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_cache_bypass $http_upgrade;
    }

    # Backend API
    location /api {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Increase timeouts for document processing
        proxy_read_timeout 300s;
        proxy_connect_timeout 300s;
        proxy_send_timeout 300s;

        # Increase max upload size
        client_max_body_size 50M;
    }
}
```

**Enable site:**
```bash
sudo ln -s /etc/nginx/sites-available/enterpriserag /etc/nginx/sites-enabled/
sudo nginx -t  # Test configuration
sudo systemctl reload nginx
```

### Systemd Service (Auto-start)

**Backend Service:**
```ini
# /etc/systemd/system/enterpriserag-backend.service
[Unit]
Description=EnterpriseRAG Backend
After=network.target

[Service]
Type=simple
User=youruser
WorkingDirectory=/path/to/EnterpriseRAG/backend
Environment="PATH=/path/to/EnterpriseRAG/backend/venv/bin"
ExecStart=/path/to/EnterpriseRAG/backend/venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Frontend Service:**
```ini
# /etc/systemd/system/enterpriserag-frontend.service
[Unit]
Description=EnterpriseRAG Frontend
After=network.target

[Service]
Type=simple
User=youruser
WorkingDirectory=/path/to/EnterpriseRAG/frontend
ExecStart=/usr/bin/npm run dev -- --host 0.0.0.0
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

**Enable services:**
```bash
sudo systemctl daemon-reload
sudo systemctl enable enterpriserag-backend
sudo systemctl enable enterpriserag-frontend
sudo systemctl start enterpriserag-backend
sudo systemctl start enterpriserag-frontend

# Check status
sudo systemctl status enterpriserag-backend
sudo systemctl status enterpriserag-frontend
```

---

## Production Configuration

### Performance Tuning

**Backend Configuration (`backend/.env`):**
```bash
# Increase for larger documents
MAX_UPLOAD_SIZE_MB=100

# Adjust chunk size based on content
CHUNK_SIZE=512  # 256, 512, or 1024

# Increase for more context
TOP_K=10  # Default 5, max 20

# Reduce for faster responses (less accuracy)
TOP_K=3

# Timeouts
EMBEDDING_TIMEOUT=60
LLM_TIMEOUT=120
SEARCH_TIMEOUT=30
```

**Ollama Performance:**
```bash
# Use quantized models for speed
ollama pull llama3.2:3b-q4_0  # 4-bit quantization

# Or use smaller model
ollama pull llama3.2:1b

# Set concurrent requests
export OLLAMA_NUM_PARALLEL=4
export OLLAMA_MAX_LOADED_MODELS=2
```

**JadeVectorDB Tuning:**
```json
// Increase for better recall (slower build)
{
  "M": 32,  // Default: 16
  "ef_construction": 400  // Default: 200
}

// Or decrease for faster build (less accuracy)
{
  "M": 8,
  "ef_construction": 100
}
```

### Resource Limits

**Systemd Limits:**
```ini
# In service file [Service] section
MemoryLimit=8G
CPUQuota=400%  # 4 cores max
```

**Ulimit Configuration:**
```bash
# /etc/security/limits.conf
youruser soft nofile 65536
youruser hard nofile 65536
```

---

## Security Hardening

### Authentication (Production)

**Add Basic Auth to Nginx:**
```nginx
# Install htpasswd tool
sudo apt install apache2-utils

# Create password file
sudo htpasswd -c /etc/nginx/.htpasswd admin

# Update nginx config
location / {
    auth_basic "EnterpriseRAG Admin";
    auth_basic_user_file /etc/nginx/.htpasswd;
    proxy_pass http://localhost:5173;
}
```

**Add JWT Authentication to FastAPI:**
```python
# backend/main.py
from fastapi import Depends, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    if credentials.credentials != "your-secret-token":
        raise HTTPException(status_code=401, detail="Invalid token")
    return credentials.credentials

# Protect admin endpoints
@app.post("/api/admin/documents/upload", dependencies=[Depends(verify_token)])
async def upload_document(...):
    ...
```

### HTTPS Setup

**With Let's Encrypt:**
```bash
# Install certbot
sudo apt install certbot python3-certbot-nginx

# Get certificate
sudo certbot --nginx -d your-server.example.com

# Auto-renewal (added automatically)
sudo certbot renew --dry-run
```

### File Permissions

```bash
# Set ownership
sudo chown -R youruser:yourgroup /path/to/EnterpriseRAG

# Restrict .env file
chmod 600 backend/.env

# Restrict uploads directory
chmod 755 backend/uploads
```

### Firewall Rules

```bash
# Only allow necessary ports
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 80/tcp   # HTTP
sudo ufw allow 443/tcp  # HTTPS
sudo ufw enable

# Don't expose backend directly
# Access through nginx only
```

---

## Monitoring & Logging

### Application Logging

**Configure Python Logging:**
```python
# backend/utils/logger.py
import logging
from logging.handlers import RotatingFileHandler

logger = logging.getLogger("enterpriserag")
logger.setLevel(logging.INFO)

handler = RotatingFileHandler(
    "logs/enterpriserag.log",
    maxBytes=10485760,  # 10MB
    backupCount=5
)
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
handler.setFormatter(formatter)
logger.addHandler(handler)
```

**Log Important Events:**
```python
# In service files
logger.info(f"Document uploaded: {filename}")
logger.info(f"Processing started: {doc_id}")
logger.error(f"Processing failed: {doc_id}, error: {str(e)}")
logger.info(f"Query processed: {question}, confidence: {confidence}")
```

### System Monitoring

**Install monitoring tools:**
```bash
# htop for system resources
sudo apt install htop

# Monitor processes
htop

# Monitor disk space
df -h
du -sh backend/uploads/

# Monitor logs
tail -f backend/logs/enterpriserag.log
```

**Create monitoring script:**
```bash
#!/bin/bash
# monitor.sh

while true; do
    clear
    echo "=== EnterpriseRAG Status ==="
    echo ""

    echo "Backend:"
    systemctl status enterpriserag-backend | grep Active

    echo "Frontend:"
    systemctl status enterpriserag-frontend | grep Active

    echo ""
    echo "Health:"
    curl -s http://localhost:8000/api/health | jq

    echo ""
    echo "Stats:"
    curl -s http://localhost:8000/api/stats | jq

    echo ""
    echo "Resources:"
    free -h | grep Mem
    df -h | grep -E "/$|/home"

    sleep 30
done
```

---

## Backup & Recovery

### Automated Backup Script

```bash
#!/bin/bash
# backup.sh - Run daily via cron

DATE=$(date +%Y%m%d)
BACKUP_DIR="/backups/enterpriserag"
RETENTION_DAYS=30

mkdir -p "$BACKUP_DIR"

# Backup uploads
echo "Backing up uploads..."
tar -czf "$BACKUP_DIR/uploads_$DATE.tar.gz" \
    /path/to/EnterpriseRAG/backend/uploads/

# Backup configuration
echo "Backing up config..."
cp /path/to/EnterpriseRAG/backend/.env \
   "$BACKUP_DIR/env_$DATE"

# Backup JadeVectorDB
echo "Backing up vector database..."
tar -czf "$BACKUP_DIR/jadevectordb_$DATE.tar.gz" \
    /path/to/JadeVectorDB/data/

# Remove old backups
find "$BACKUP_DIR" -name "*.tar.gz" -mtime +$RETENTION_DAYS -delete
find "$BACKUP_DIR" -name "env_*" -mtime +$RETENTION_DAYS -delete

echo "Backup complete: $DATE"
```

**Setup cron job:**
```bash
# Run backup daily at 2 AM
crontab -e

# Add:
0 2 * * * /path/to/backup.sh >> /var/log/enterpriserag-backup.log 2>&1
```

### Disaster Recovery

**Recovery Procedure:**
```bash
# 1. Stop services
sudo systemctl stop enterpriserag-backend
sudo systemctl stop enterpriserag-frontend

# 2. Restore uploads
tar -xzf backups/enterpriserag/uploads_YYYYMMDD.tar.gz \
    -C /path/to/EnterpriseRAG/backend/

# 3. Restore configuration
cp backups/enterpriserag/env_YYYYMMDD \
   /path/to/EnterpriseRAG/backend/.env

# 4. Restore vector database
tar -xzf backups/enterpriserag/jadevectordb_YYYYMMDD.tar.gz \
    -C /path/to/JadeVectorDB/

# 5. Restart services
sudo systemctl start enterpriserag-backend
sudo systemctl start enterpriserag-frontend

# 6. Verify
curl http://localhost:8000/api/health
```

---

## Scaling Considerations

### When to Scale

**Signs you need to scale:**
- Query response time >10 seconds
- Document processing takes >10 minutes
- System resources consistently >80%
- Multiple users experiencing slowness

### Vertical Scaling (Upgrade Hardware)

**Priority order:**
1. **More RAM**: Fastest improvement for LLM
2. **Faster CPU**: Helps with processing
3. **SSD Storage**: Faster database access
4. **GPU**: Optional, significant LLM speedup

### Horizontal Scaling (Future)

**Current limitations:**
- Single-instance design
- Shared vector database
- No load balancing

**To support horizontal scaling (future enhancement):**
- Add load balancer (nginx)
- Separate backend instances
- Shared JadeVectorDB cluster
- Distributed Ollama instances
- Session management

---

## Maintenance Schedule

### Daily
- Check system health
- Monitor disk space
- Review error logs

### Weekly
- Review performance metrics
- Check backup success
- Update documents as needed
- Test disaster recovery

### Monthly
- Update software dependencies
- Review security logs
- Optimize vector database
- Archive old logs

### Quarterly
- Full system backup
- Security audit
- Performance tuning
- User training refresh

---

**Version 1.0** | Last Updated: March 28, 2026

**For Support:** Refer to USER_GUIDE.md, ADMIN_GUIDE.md, and TROUBLESHOOTING.md
