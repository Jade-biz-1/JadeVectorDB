# EnterpriseRAG - Quick Start Guide

Get EnterpriseRAG up and running in 5 minutes!

## Prerequisites

- **Python 3.9+**
- **Node.js 18+**
- That's it for mock mode!

## Installation (One Command)

```bash
cd EnterpriseRAG
./scripts/setup.sh
```

This will:
- Create Python virtual environment
- Install all dependencies
- Set up configuration files
- Create necessary directories

## Start the Application

```bash
./scripts/start.sh
```

This starts both:
- **Backend API** on http://localhost:8000
- **Frontend UI** on http://localhost:5173

## Access the Application

Open your browser to: **http://localhost:5173**

### Main Interface

1. **Query Page** (default): Ask questions about maintenance
   - Type your question
   - Select device type (optional)
   - Choose number of sources
   - Get answers with citations!

2. **Documents Page**: Upload and manage documents
   - Upload PDF/DOCX files
   - View processing status
   - Delete outdated documents

## Try It Out (Mock Mode)

In mock mode, the system returns simulated responses:

1. Go to http://localhost:5173
2. Try example questions:
   - "How do I replace the hydraulic fluid?"
   - "What is the maintenance schedule?"
   - "How to troubleshoot overheating?"
3. See realistic answers with source citations
4. View system statistics at the bottom

## Production Mode (Optional)

For real RAG with your documents:

1. **Start JadeVectorDB**:
   ```bash
   cd /path/to/JadeVectorDB/backend/build
   ./jadevectordb
   ```

2. **Install Ollama models**:
   ```bash
   ollama pull llama3.2:3b
   ollama pull nomic-embed-text
   ```

3. **Configure backend**:
   Edit `backend/.env`:
   ```bash
   MODE=production
   JADEVECTORDB_URL=http://localhost:8080
   OLLAMA_URL=http://localhost:11434
   ```

4. **Restart**:
   ```bash
   ./scripts/start.sh
   ```

5. **Upload documents**:
   - Go to http://localhost:5173/admin/documents
   - Upload your maintenance manuals
   - Wait for processing (30-60 seconds per PDF)
   - Ask questions about your actual documentation!

## API Documentation

Interactive API docs: http://localhost:8000/docs

### Key Endpoints

- `POST /api/query` - Ask questions
- `POST /api/admin/documents/upload` - Upload documents
- `GET /api/admin/documents` - List all documents
- `DELETE /api/admin/documents/{id}` - Delete document
- `GET /api/stats` - System statistics
- `GET /api/health` - Health check

## Stopping the Application

Press **Ctrl+C** in the terminal where you ran `start.sh`

## Troubleshooting

### Backend won't start
```bash
# Check if port 8000 is in use
lsof -i :8000

# Check Python version
python3 --version  # Should be 3.9+
```

### Frontend won't start
```bash
# Check Node version
node --version  # Should be 18+

# Clear cache and reinstall
cd frontend
rm -rf node_modules package-lock.json
npm install
```

### "Module not found" errors
```bash
# Reinstall backend dependencies
cd backend
source venv/bin/activate
pip install -r requirements.txt
```

## Next Steps

1. **Read the full README**: `README.md`
2. **Configure for production**: Edit `backend/.env`
3. **Upload your documents**: Use the admin panel
4. **Customize**: Modify `backend/utils/config.py` for settings
5. **Deploy**: See `docs/DEPLOYMENT.md`

## Support

- Issues: Check `README.md` troubleshooting section
- API Docs: http://localhost:8000/docs
- Full Documentation: See `README.md`

---

**Happy querying!** 🚀
