# EnterpriseRAG Troubleshooting Guide

Quick solutions to common problems.

---

## Table of Contents

1. [Quick Diagnostics](#quick-diagnostics)
2. [Installation Issues](#installation-issues)
3. [Startup Issues](#startup-issues)
4. [Upload & Processing Issues](#upload--processing-issues)
5. [Query Issues](#query-issues)
6. [Performance Issues](#performance-issues)
7. [Production Mode Issues](#production-mode-issues)
8. [Database Issues](#database-issues)
9. [Network Issues](#network-issues)
10. [Recovery Procedures](#recovery-procedures)

---

## Quick Diagnostics

### System Health Check

Run these commands to quickly diagnose issues:

```bash
# 1. Check if backend is running
curl http://localhost:8000/api/health

# 2. Check if frontend is accessible
curl http://localhost:5173

# 3. Check system stats
curl http://localhost:8000/api/stats

# 4. Check if Ollama is running (production mode)
curl http://localhost:11434/api/tags

# 5. Check if JadeVectorDB is running (production mode)
curl http://localhost:8080/health
```

### Component Status Matrix

| Component | Check | Expected | If Failed |
|-----------|-------|----------|-----------|
| Backend | `curl localhost:8000/api/health` | HTTP 200 | [Backend won't start](#backend-wont-start) |
| Frontend | Browser to `localhost:5173` | Page loads | [Frontend won't start](#frontend-wont-start) |
| Ollama | `curl localhost:11434/api/tags` | JSON response | [Ollama issues](#ollama-issues) |
| JadeVectorDB | `curl localhost:8080/health` | HTTP 200 | [JadeVectorDB issues](#jadevectordb-issues) |

---

## Installation Issues

### Setup Script Fails

**Error: "python3: command not found"**
```bash
# Install Python 3.9+
# Ubuntu/Debian:
sudo apt update && sudo apt install python3 python3-pip python3-venv

# macOS:
brew install python@3.9

# Verify
python3 --version  # Should show 3.9+
```

**Error: "node: command not found"**
```bash
# Install Node.js 18+
# Ubuntu/Debian:
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt install -y nodejs

# macOS:
brew install node@18

# Verify
node --version  # Should show 18+
```

**Error: "pip install failed"**
```bash
# Clear pip cache and retry
cd backend
source venv/bin/activate
pip cache purge
pip install --no-cache-dir -r requirements.txt

# If specific package fails, install separately:
pip install <package-name>
```

**Error: "npm install failed"**
```bash
# Clear npm cache
cd frontend
rm -rf node_modules package-lock.json
npm cache clean --force
npm install

# If still fails, try:
npm install --legacy-peer-deps
```

### Permission Errors

**Error: "Permission denied" during setup**
```bash
# Make scripts executable
chmod +x scripts/setup.sh scripts/start.sh

# Fix ownership
sudo chown -R $USER:$USER EnterpriseRAG/

# Retry setup
./scripts/setup.sh
```

---

## Startup Issues

### Backend Won't Start

**Error: "Address already in use" (Port 8000)**
```bash
# Find process using port 8000
lsof -i :8000

# Kill the process
kill -9 <PID>

# Or use different port in .env:
API_PORT=8001
```

**Error: "ModuleNotFoundError"**
```bash
# Ensure virtual environment is activated
cd backend
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt

# Verify installation
pip list | grep fastapi
```

**Error: "Cannot load .env file"**
```bash
# Create .env from example
cd backend
cp .env.example .env

# Edit with your settings
nano .env  # or vim, or any editor
```

**Error: "Import error: relative import"**
```bash
# Start from project root, not backend directory
cd /path/to/EnterpriseRAG
./scripts/start.sh

# Or use correct module syntax
cd backend
python -m uvicorn main:app
```

### Frontend Won't Start

**Error: "Cannot find module"**
```bash
cd frontend
rm -rf node_modules
npm install
npm run dev
```

**Error: "Port 5173 already in use"**
```bash
# Find and kill process
lsof -i :5173
kill -9 <PID>

# Or change port in vite.config.js:
server: {
  port: 5174
}
```

**Error: "EACCES: permission denied"**
```bash
# Fix npm permissions
sudo chown -R $USER ~/.npm
cd frontend
npm install
```

### Start Script Issues

**Error: "Backend not set up"**
```bash
# Run setup first
./scripts/setup.sh

# Then start
./scripts/start.sh
```

**Error: "Backend started but immediately exits"**
```bash
# Check for Python errors
cd backend
source venv/bin/activate
python -c "import fastapi, uvicorn, pydantic"

# If import fails, reinstall
pip install -r requirements.txt
```

---

## Upload & Processing Issues

### Upload Fails

**Error: "Invalid file type"**
- **Cause**: File is not PDF or DOCX
- **Solution**: Convert file to PDF or DOCX format
- **Verify**: File extension is `.pdf` or `.docx` (case-insensitive)

**Error: "File too large"**
- **Cause**: File exceeds 50MB default limit
- **Solution**:
  ```bash
  # Edit backend/.env
  MAX_UPLOAD_SIZE_MB=100

  # Restart backend
  ```

**Error: "Upload request failed"**
- **Cause**: Backend not running or network issue
- **Check**:
  ```bash
  curl http://localhost:8000/api/health
  ```
- **Solution**: Restart backend

### Processing Stuck

**Status: "Processing" for >10 minutes**
- **Normal**: Large documents (>200 pages) can take 5-10 minutes
- **Check backend logs**: Look for errors
- **Verify Ollama**: `curl http://localhost:11434/api/tags`
- **Solution if stuck**:
  ```bash
  # Restart backend (stops stuck process)
  # Delete document from admin panel
  # Re-upload
  ```

**Status: "Failed"**
- **Check error message** in document details
- **Common causes**:
  1. **Text extraction failed**: Scanned PDF without OCR
  2. **Ollama error**: Embedding model not available
  3. **Database error**: JadeVectorDB connection issue
  4. **Memory error**: File too large for available RAM

**Text extraction failed**
```bash
# Verify PDF has text (not just images)
pdftotext test.pdf test.txt
cat test.txt  # Should show extracted text

# If empty, PDF is scanned - needs OCR:
# Use Adobe Acrobat, pdfsandwich, or tesseract
```

**Embedding generation failed**
```bash
# Check Ollama has embedding model
ollama list | grep nomic-embed-text

# If not present:
ollama pull nomic-embed-text

# Test embedding generation
curl http://localhost:11434/api/embeddings -d '{
  "model": "nomic-embed-text",
  "prompt": "test"
}'
```

### Upload Not Appearing

**Document not in list after upload**
- **Refresh page**: Click refresh button
- **Check backend logs**: May have failed silently
- **Verify file was uploaded**:
  ```bash
  ls backend/uploads/
  ```

---

## Query Issues

### No Results Found

**Error: "No relevant information found"**

**Diagnosis**:
1. Check document is uploaded and status="complete"
2. Verify device type filter (try "All Devices")
3. Try simpler, broader question
4. Check if document actually contains answer

**Solutions**:
```bash
# 1. List all documents
curl http://localhost:8000/api/admin/documents

# 2. Check stats
curl http://localhost:8000/api/stats
# Should show total_chunks > 0

# 3. Try test query
curl -X POST http://localhost:8000/api/query \
  -H "Content-Type: application/json" \
  -d '{"question":"test","device_type":"all","top_k":5}'
```

### Low Quality Answers

**Confidence score <60%**
- **Review sources**: Are they relevant?
- **Try different question wording**
- **Increase top_k**: More sources may help
- **Check document quality**: Is content clear and structured?

**Answer seems wrong**
- **Verify sources**: Check page numbers in original manual
- **Check device type**: Filter may have selected wrong equipment
- **Review question**: Was it ambiguous?
- **Report if persistent**: May indicate document issue

### Query Takes Too Long

**Response time >10 seconds**

**Normal**: 2-4 seconds
**Acceptable**: Up to 6 seconds
**Too slow**: >10 seconds

**Solutions**:
```bash
# 1. Reduce top_k
# In query form, select "3" instead of "10"

# 2. Check system resources
top  # Look for high CPU/memory usage

# 3. Check Ollama response time
time curl http://localhost:11434/api/generate \
  -d '{"model":"llama3.2:3b","prompt":"test","stream":false}'
# Should complete in <3 seconds

# 4. Check JadeVectorDB response time
time curl http://localhost:8080/health
# Should respond in <100ms
```

---

## Performance Issues

### Slow System Overall

**Symptoms**: All operations slow

**Diagnosis**:
```bash
# Check CPU and memory
top

# Check disk I/O
iostat -x 1

# Check disk space
df -h
```

**Solutions**:
1. **High CPU**:
   - Close other applications
   - Reduce concurrent uploads
   - Use smaller Ollama model

2. **High Memory**:
   - Restart backend
   - Reduce chunk size in config
   - Process smaller documents

3. **Low Disk Space**:
   - Delete old documents
   - Clean up uploads: `rm backend/uploads/*`
   - Run index compaction

### Backend Memory Leak

**Symptoms**: Memory usage grows over time

**Check**:
```bash
# Monitor backend process
ps aux | grep uvicorn
# Watch RSS (resident memory)
```

**Solution**:
```bash
# Restart backend periodically
# Add to cron for automatic restart:
0 2 * * * cd /path/to/EnterpriseRAG && ./restart_backend.sh
```

### Database Growing Too Large

**Symptoms**: Disk space filling up

**Check**:
```bash
# Check vector DB size
du -sh /path/to/jadevectordb/data

# Check upload directory
du -sh backend/uploads/

# Check stats
curl http://localhost:8000/api/stats
```

**Solutions**:
1. **Delete outdated documents** via admin panel
2. **Archive old uploads**: `mv backend/uploads/old_* /archive/`
3. **Run index compaction** (automatic if >10% deleted)
4. **Clear old .env backups** if any

---

## Production Mode Issues

### Ollama Issues

**Error: "Connection refused" to localhost:11434**

**Check if running**:
```bash
ollama list
# Should show installed models

# If not running, start Ollama
# (Method varies by OS)
```

**Error: "Model not found"**
```bash
# List models
ollama list

# Pull required models
ollama pull llama3.2:3b
ollama pull nomic-embed-text

# Verify
ollama list | grep -E "llama3.2:3b|nomic-embed-text"
```

**Error: "Ollama running but requests fail"**
```bash
# Test directly
curl http://localhost:11434/api/tags

# If timeout, check firewall
# If JSON error, Ollama may be corrupted - reinstall
```

### JadeVectorDB Issues

**Error: "Connection refused" to localhost:8080**

**Check if running**:
```bash
curl http://localhost:8080/health

# If fails, start JadeVectorDB
cd /path/to/JadeVectorDB/backend/build
./jadevectordb
```

**Error: "Database not found"**
```bash
# Create database using JadeVectorDB API
curl -X POST http://localhost:8080/api/v1/databases \
  -H "Content-Type: application/json" \
  -d '{
    "id": "maintenance_docs",
    "dimension": 384,
    "index_type": "HNSW"
  }'
```

**Error: "Insert failed"**
- Check vector dimension matches (384 for nomic-embed-text)
- Verify database exists
- Check JadeVectorDB logs

### Mode Switching Issues

**Switching from mock to production doesn't work**
```bash
# 1. Stop backend (Ctrl+C)

# 2. Edit .env
cd backend
nano .env
# Change: MODE=production

# 3. Verify dependencies
curl http://localhost:11434/api/tags
curl http://localhost:8080/health

# 4. Restart
cd ../
./scripts/start.sh
```

---

## Database Issues

### Vector Database Corruption

**Symptoms**: Queries fail, inserts fail

**Recovery**:
```bash
# 1. Stop backend

# 2. Backup current DB
cp -r /path/to/jadevectordb/data /backup/

# 3. Stop JadeVectorDB
# Kill process or Ctrl+C

# 4. Restart JadeVectorDB
cd /path/to/JadeVectorDB/backend/build
./jadevectordb

# 5. If still fails, recreate database
curl -X DELETE http://localhost:8080/api/v1/databases/maintenance_docs
curl -X POST http://localhost:8080/api/v1/databases \
  -d '{"id":"maintenance_docs","dimension":384,"index_type":"HNSW"}'

# 6. Re-upload documents
```

### Metadata Inconsistency

**Symptoms**: Document list shows wrong info

**In production, metadata is stored in memory by default**

**Recovery**:
```bash
# Restart backend (resets memory store)
# Re-upload affected documents
```

---

## Network Issues

### CORS Errors in Browser

**Error**: "Access to fetch blocked by CORS policy"

**Solution**:
```python
# Verify CORS config in backend/main.py
# Should allow frontend URL

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Proxy Issues

**Frontend can't reach backend**

**Check vite.config.js**:
```javascript
server: {
  proxy: {
    '/api': {
      target: 'http://localhost:8000',
      changeOrigin: true,
    },
  },
}
```

### Port Conflicts

**Ports already in use**
```bash
# Check what's using ports
lsof -i :8000  # Backend
lsof -i :5173  # Frontend
lsof -i :8080  # JadeVectorDB
lsof -i :11434 # Ollama

# Change port in config or kill process
```

---

## Recovery Procedures

### Complete System Reset

**When**: System is completely broken

**Steps**:
```bash
# 1. Stop all processes
pkill -f uvicorn
pkill -f vite

# 2. Backup important data
mkdir -p ~/EnterpriseRAG_backup
cp -r backend/uploads ~/EnterpriseRAG_backup/
cp backend/.env ~/EnterpriseRAG_backup/

# 3. Clean installation
cd EnterpriseRAG
rm -rf backend/venv frontend/node_modules

# 4. Reinstall
./scripts/setup.sh

# 5. Restore config
cp ~/EnterpriseRAG_backup/.env backend/

# 6. Restart
./scripts/start.sh
```

### Restore from Backup

**Restore uploaded documents**:
```bash
# Stop backend
# Copy backup
cp -r /backup/uploads/* backend/uploads/

# Documents need to be re-uploaded through UI to regenerate vectors
```

**Restore vector database**:
```bash
# Stop JadeVectorDB
# Restore data directory
cp -r /backup/jadevectordb/data /path/to/jadevectordb/

# Restart JadeVectorDB
```

### Emergency Contact Procedure

**If all else fails**:

1. **Document the issue**:
   - Error messages
   - Steps that led to error
   - System configuration
   - Logs

2. **Create issue report**:
   ```
   System: EnterpriseRAG v1.0
   OS: [Your OS]
   Mode: [mock/production]
   Error: [Description]
   Steps: [How to reproduce]
   Logs: [Relevant log entries]
   ```

3. **Contact support**:
   - System administrator
   - IT support team
   - Development team (if available)

---

## Diagnostic Commands Reference

### Quick Health Check
```bash
#!/bin/bash
# health_check.sh - Run when something's wrong

echo "=== EnterpriseRAG Health Check ==="

echo -n "Backend: "
curl -s http://localhost:8000/api/health | grep -q "healthy" && echo "✓ OK" || echo "✗ FAIL"

echo -n "Frontend: "
curl -s http://localhost:5173 > /dev/null && echo "✓ OK" || echo "✗ FAIL"

echo -n "Ollama: "
curl -s http://localhost:11434/api/tags > /dev/null && echo "✓ OK" || echo "✗ FAIL"

echo -n "JadeVectorDB: "
curl -s http://localhost:8080/health > /dev/null && echo "✓ OK" || echo "✗ FAIL"

echo ""
echo "=== System Stats ==="
curl -s http://localhost:8000/api/stats | grep -E "total_documents|total_chunks|total_queries"

echo ""
echo "=== Disk Space ==="
df -h | grep -E "Filesystem|/$"
```

### Log Collection
```bash
#!/bin/bash
# collect_logs.sh - Gather diagnostic info

DATE=$(date +%Y%m%d_%H%M%S)
LOGDIR="logs_$DATE"
mkdir -p "$LOGDIR"

# System info
uname -a > "$LOGDIR/system_info.txt"
python3 --version >> "$LOGDIR/system_info.txt"
node --version >> "$LOGDIR/system_info.txt"

# Backend logs
cp backend/logs/* "$LOGDIR/" 2>/dev/null

# Process list
ps aux | grep -E "uvicorn|vite|ollama|jadevectordb" > "$LOGDIR/processes.txt"

# Port usage
lsof -i :8000,5173,8080,11434 > "$LOGDIR/ports.txt"

# Health checks
curl -s http://localhost:8000/api/health > "$LOGDIR/health.json"
curl -s http://localhost:8000/api/stats > "$LOGDIR/stats.json"

echo "Logs collected in: $LOGDIR"
```

---

## Prevention Tips

### Before Problems Occur

1. **Regular backups**: Daily backups of uploads and vector DB
2. **Monitor disk space**: Set up alerts at 80% full
3. **Test queries regularly**: Ensure system works before users need it
4. **Keep dependencies updated**: But test updates in development first
5. **Document customizations**: Note any config changes
6. **Train users properly**: Reduce support burden

### Monitoring Checklist

**Daily**:
- [ ] System accessible
- [ ] No error messages in admin panel
- [ ] New uploads processing correctly

**Weekly**:
- [ ] Check disk space
- [ ] Review query logs for issues
- [ ] Test sample queries
- [ ] Backup uploads

**Monthly**:
- [ ] Full health check
- [ ] Review system stats trends
- [ ] Check for software updates
- [ ] Test disaster recovery procedure

---

**Version 1.0** | Last Updated: March 28, 2026

**Need Help?** Check USER_GUIDE.md and ADMIN_GUIDE.md for detailed documentation.
