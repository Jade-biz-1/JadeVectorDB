# EnterpriseRAG Administrator Guide

**For System Administrators and Documentation Managers**

This guide covers document management, system administration, and maintenance of the EnterpriseRAG system.

---

## Table of Contents

1. [Overview](#overview)
2. [Accessing the Admin Panel](#accessing-the-admin-panel)
3. [Document Management](#document-management)
4. [System Configuration](#system-configuration)
5. [Monitoring & Maintenance](#monitoring--maintenance)
6. [Troubleshooting](#troubleshooting)
7. [Best Practices](#best-practices)

---

## Overview

### Administrator Responsibilities

As an EnterpriseRAG administrator, you are responsible for:

1. **Document Library Management**
   - Uploading new maintenance documentation
   - Organizing documents by device type
   - Removing outdated documents
   - Maintaining document quality

2. **System Maintenance**
   - Monitoring system health
   - Managing storage and performance
   - Handling processing errors
   - Backing up the vector database

3. **User Support**
   - Identifying documentation gaps
   - Adding requested manuals
   - Addressing content quality issues
   - Training new users

### System Architecture Overview

```
┌──────────────────────────────────────────────────────┐
│                  EnterpriseRAG System                │
├──────────────────────────────────────────────────────┤
│                                                      │
│  Frontend (React)                                    │
│    - Query Interface (for engineers)                 │
│    - Admin Panel (for you)                           │
│                                                      │
│  Backend (FastAPI)                                   │
│    - Document Processing Pipeline                    │
│    - Query Processing                                │
│    - Vector Search                                   │
│                                                      │
│  Storage                                             │
│    - JadeVectorDB (vectors + metadata)              │
│    - Uploaded files (backend/uploads/)              │
│    - Configuration (.env)                            │
│                                                      │
│  LLM/Embeddings                                      │
│    - Ollama (local LLM + embeddings)                │
│                                                      │
└──────────────────────────────────────────────────────┘
```

---

## Accessing the Admin Panel

### URL
Navigate to: **http://localhost:5173/admin/documents**

### Admin Interface Components

1. **Document Upload Section**
   - File selection (PDF/DOCX)
   - Device type selector
   - Upload button with progress

2. **Document List**
   - All indexed documents
   - Status indicators
   - Action buttons (delete, reprocess)
   - System statistics

3. **System Stats Dashboard**
   - Total documents indexed
   - Total vector chunks
   - Processing status
   - System health

---

## Document Management

### Uploading Documents

#### Step-by-Step Upload Process

1. **Navigate to Admin Panel**
   ```
   http://localhost:5173/admin/documents
   ```

2. **Prepare Your Document**
   - ✅ **Supported formats**: PDF, DOCX
   - ✅ **File size**: Up to 50MB (configurable)
   - ✅ **Content**: Text-based (not scanned images)
   - ✅ **File name**: Use descriptive names (e.g., "HP3000_hydraulic_pump_manual_v2.pdf")

3. **Select File**
   - Click "Select PDF or DOCX file" button
   - Choose your document
   - File name and size will display

4. **Choose Device Type**
   - Select from dropdown:
     - Hydraulic Pump
     - Air Compressor
     - Conveyor System
     - Other
   - This helps users filter searches

5. **Upload**
   - Click "Upload Document"
   - You'll see: "Document uploaded successfully. Processing..."
   - Document appears in list with status "processing"

#### What Happens During Processing

```
Upload → Text Extraction → Chunking → Embedding Generation → Vector Storage
  ↓            ↓              ↓              ↓                    ↓
1 sec      5-15 sec       5-10 sec      10-30 sec            5-10 sec
```

**Total Time:** 30-60 seconds for typical maintenance manual (100-200 pages)

**Processing Steps:**
1. **Text Extraction** (5-15 sec)
   - PDF: PyMuPDF extracts text, preserves page numbers
   - DOCX: python-docx extracts formatted text
   - Tables and lists preserved

2. **Semantic Chunking** (5-10 sec)
   - Splits document into ~512-token chunks
   - Keeps related content together
   - 50-token overlap between chunks
   - Preserves section headers

3. **Embedding Generation** (10-30 sec)
   - Ollama generates 384-dim vectors
   - Each chunk gets a vector representation
   - Captures semantic meaning

4. **Vector Storage** (5-10 sec)
   - Stores in JadeVectorDB
   - Adds metadata (doc name, page, section, device type)
   - Indexes for fast retrieval

#### Monitoring Upload Progress

**Status Indicators:**
- 🟡 **Pending**: Queued for processing
- 🔵 **Processing**: Currently being processed
- 🟢 **Complete**: Ready for queries
- 🔴 **Failed**: Error occurred (check logs)

**To Check Status:**
1. Refresh the document list
2. Look at the status badge
3. Check "Chunks" column (empty until complete)

#### Handling Upload Errors

**Common Errors:**

| Error | Cause | Solution |
|-------|-------|----------|
| "Invalid file type" | Not PDF/DOCX | Convert to supported format |
| "File too large" | Exceeds 50MB | Split into smaller files or increase limit |
| "Processing failed" | Corrupted file | Try re-uploading or repair file |
| "Text extraction failed" | Scanned PDF without OCR | Run OCR first or use text-based PDF |

**Checking Error Details:**
1. Click on failed document in list
2. Look at error message in details
3. Check backend logs: `backend/logs/` (if logging enabled)

### Managing Documents

#### Viewing Document Details

Each document in the list shows:
- **Filename**: Original file name
- **Device Type**: Category assigned at upload
- **Chunks**: Number of text segments created
- **Status**: Processing state
- **Uploaded**: Date and time of upload

#### Deleting Documents

**When to Delete:**
- Document is outdated (replaced by newer version)
- Wrong document uploaded
- Document contains errors
- Equipment no longer in use

**Deletion Process:**

1. **Click Delete Button**
   - Find document in list
   - Click 🗑️ Delete button

2. **Review Confirmation Dialog**
   ```
   Delete "hydraulic_pump_manual.pdf"?

   This will permanently remove:
   • Document metadata
   • 156 vector chunks from JadeVectorDB
   • All index entries

   This action cannot be undone.
   ```

3. **Confirm Deletion**
   - Click "OK" to proceed
   - System deletes:
     - All vector chunks
     - Index entries (automatic update)
     - Metadata records

4. **Automatic Optimization**
   - If >10% of vectors deleted, system triggers index compaction
   - Runs in background (no interruption to users)
   - Reclaims storage space

**Important Notes:**
- ✅ Deletion is immediate for users (no longer in search results)
- ✅ Index updates automatically (no manual reindexing)
- ✅ Background compaction optimizes performance
- ⚠️ **Cannot be undone** - verify before deleting
- 💡 Consider backing up important documents before deletion

#### Reprocessing Documents

**When to Reprocess:**
- Updated document with same name
- Changed chunking strategy
- Previous processing had errors
- Want to update metadata

**How to Reprocess:**
1. Delete old version
2. Upload new version with same metadata
3. Monitor processing status

*Note: Direct reprocessing endpoint exists but currently returns "not implemented". Use delete + re-upload workflow.*

### Document Organization Best Practices

#### File Naming Convention

**Use descriptive, consistent names:**

✅ **Good Examples:**
- `HP-3000_hydraulic_pump_maintenance_manual_v2.1.pdf`
- `AC-500_air_compressor_troubleshooting_guide.pdf`
- `CVR-200_conveyor_system_operators_manual.docx`

❌ **Avoid:**
- `manual.pdf` (too generic)
- `doc_final_FINAL_v3_NEW.pdf` (confusing)
- `scan001.pdf` (no context)

**Naming Pattern:**
```
[Model]_[Equipment-Type]_[Document-Type]_[Version].pdf

Examples:
HP-3000_hydraulic-pump_maintenance_v2.1.pdf
AC-500_air-compressor_service-manual_2025.pdf
```

#### Device Type Classification

**Standard Categories:**
- **Hydraulic Pump**: All hydraulic systems
- **Air Compressor**: Compressed air equipment
- **Conveyor System**: Belt, roller, chain conveyors
- **Other**: Miscellaneous equipment

**Tips:**
- Be consistent across similar equipment
- Use "Other" sparingly - better to add specific categories
- Document your classification scheme

#### Version Control

**Track Document Versions:**
1. Include version in filename
2. Delete old versions when uploading new
3. Keep version history log (external)
4. Note significant changes

**Example Log:**
```
HP-3000 Manual:
- v1.0 (2023-05): Initial upload
- v2.0 (2024-08): Updated safety procedures
- v2.1 (2026-03): Added troubleshooting section
```

---

## System Configuration

### Configuration File: `backend/.env`

#### Key Settings

```bash
# Operation Mode
MODE=production  # or 'mock' for demo

# JadeVectorDB
JADEVECTORDB_URL=http://localhost:8080
JADEVECTORDB_DATABASE_ID=maintenance_docs

# Ollama
OLLAMA_URL=http://localhost:11434
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
OLLAMA_LLM_MODEL=llama3.2:3b

# Document Processing
CHUNK_SIZE=512          # Tokens per chunk
CHUNK_OVERLAP=50        # Overlap between chunks
TOP_K=5                 # Default sources per query

# Upload Limits
MAX_UPLOAD_SIZE_MB=50
```

#### Adjusting Settings

**Chunk Size:**
- **256 tokens**: Smaller, more precise chunks (more chunks per doc)
- **512 tokens**: ✅ **Default** - Good balance
- **1024 tokens**: Larger context, fewer chunks

**When to Change:**
- 256: Very detailed manuals with short procedures
- 512: Most maintenance documentation
- 1024: High-level documents, narrative content

**Top-K (Sources per Query):**
- **3**: Quick answers, focused
- **5**: ✅ **Default** - Recommended
- **10**: Comprehensive, but may overwhelm

**Upload Size Limit:**
- Default: 50MB
- Increase if needed for large manuals
- Consider splitting very large files

**To Apply Changes:**
1. Edit `backend/.env`
2. Restart backend: `./scripts/start.sh`

### Mode Configuration

#### Mock Mode (Demo/Testing)
```bash
MODE=mock
```
**Use When:**
- Demonstrating system capabilities
- Testing UI without dependencies
- Training new users
- No JadeVectorDB or Ollama available

**Behavior:**
- Returns simulated responses
- No actual document processing
- Fast response times (<500ms)
- Pre-programmed answers

#### Production Mode (Real Use)
```bash
MODE=production
JADEVECTORDB_URL=http://localhost:8080
OLLAMA_URL=http://localhost:11434
```
**Requirements:**
- JadeVectorDB running on specified URL
- Ollama with required models installed
- Sufficient storage for vectors

**Behavior:**
- Real document processing
- Actual vector search
- LLM-generated answers
- Response times 2-4 seconds

---

## Monitoring & Maintenance

### System Statistics Dashboard

**Access:** Bottom of Query page or Admin panel

**Metrics:**
- **Total Documents**: Documents successfully indexed
- **Total Chunks**: Vector chunks in database
- **Total Queries**: Questions asked by users
- **Uptime**: Time since backend started
- **Status**: System health (healthy/degraded)
- **DB Status**: JadeVectorDB connection
- **LLM Status**: Ollama connection

### Health Checks

**Manual Health Check:**
```bash
curl http://localhost:8000/api/health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2026-03-28T10:30:00Z",
  "mode": "production",
  "components": {
    "database": "healthy",
    "llm": "healthy",
    "embeddings": "healthy"
  }
}
```

**Status Meanings:**
- **healthy**: All systems operational
- **degraded**: Some components down
- **unavailable**: System not responding

### Storage Management

#### Disk Space Monitoring

**Check Upload Directory:**
```bash
du -sh EnterpriseRAG/backend/uploads/
```

**Check JadeVectorDB Storage:**
```bash
# See JadeVectorDB documentation
# Typical: 1000 chunks = ~10MB
```

**Estimates:**
- 100-page PDF → ~150-200 chunks → ~1-2MB vectors
- 1000 documents → ~150,000 chunks → ~1.5GB vectors

#### Cleanup Strategies

**When Storage Gets Full:**
1. Delete outdated documents (triggers compaction)
2. Archive old uploads: `mv uploads/old_* archive/`
3. Run manual compaction if >10% deleted
4. Consider adding storage

### Backups

#### What to Backup

1. **Vector Database**
   - JadeVectorDB data directory
   - Frequency: Daily

2. **Uploaded Documents**
   - `backend/uploads/` directory
   - Frequency: After each upload

3. **Configuration**
   - `backend/.env` file
   - Frequency: After changes

4. **Document Metadata**
   - In production, stored in database
   - Included in vector DB backup

#### Backup Commands

**Quick Backup Script:**
```bash
#!/bin/bash
# backup.sh - Run daily

DATE=$(date +%Y%m%d)
BACKUP_DIR="backups/$DATE"

mkdir -p "$BACKUP_DIR"

# Backup uploads
cp -r backend/uploads "$BACKUP_DIR/"

# Backup config
cp backend/.env "$BACKUP_DIR/"

# Backup vector DB (if local)
# See JadeVectorDB backup procedures

echo "Backup completed: $BACKUP_DIR"
```

**Restore Procedure:**
1. Stop backend
2. Restore files from backup
3. Restart backend
4. Verify health check

### Index Compaction

#### What is Compaction?

**Purpose:** Physically remove deleted vectors and optimize index

**When It Runs:**
- Automatically when >10% of vectors deleted
- Can be triggered manually
- Runs in background

**Duration:** 5-30 minutes depending on database size

#### Manual Compaction

**Check Deletion Percentage:**
```bash
curl http://localhost:8000/api/stats
```

**Trigger Compaction:**
*Note: Currently automatic only. Manual trigger requires direct JadeVectorDB API call.*

**Monitor Compaction:**
- Check backend logs
- System remains responsive
- No user-visible impact

---

## Troubleshooting

### Common Issues and Solutions

#### Upload Issues

**Problem: "File too large"**
- **Solution**: Increase `MAX_UPLOAD_SIZE_MB` in `.env` or split file

**Problem: "Text extraction failed"**
- **Cause**: Scanned PDF without OCR
- **Solution**:
  1. Run OCR on PDF first
  2. Use text-based version
  3. Convert to DOCX

**Problem: "Processing stuck at 'processing' status"**
- **Cause**: Backend crash, Ollama down, or network issue
- **Solution**:
  1. Check backend logs
  2. Verify Ollama is running: `curl http://localhost:11434/api/tags`
  3. Restart backend
  4. Delete stuck document and re-upload

#### Query Issues

**Problem: "No relevant information found"**
- **Cause**: Document not indexed or query mismatch
- **Solution**:
  1. Verify document uploaded and status="complete"
  2. Try broader search terms
  3. Check device type filter
  4. Review document content quality

**Problem: "Low confidence scores (<60%)"**
- **Cause**: Poor document quality or ambiguous query
- **Solution**:
  1. Review source documents for clarity
  2. Add supplementary documentation
  3. Train users on effective questions

#### System Issues

**Problem: "Backend won't start"**
```bash
# Check if port in use
lsof -i :8000

# Check Python environment
cd backend
source venv/bin/activate
python -c "import fastapi"

# Check logs
tail -f backend/logs/error.log
```

**Problem: "Frontend shows 'Connection refused'"**
- **Cause**: Backend not running
- **Solution**: Start backend: `cd backend && uvicorn main:app`

**Problem: "Ollama not responding"**
```bash
# Check Ollama status
ollama list

# Restart Ollama
# (Varies by OS - see Ollama docs)
```

**Problem: "JadeVectorDB connection failed"**
```bash
# Check if running
curl http://localhost:8080/health

# Restart JadeVectorDB
cd /path/to/JadeVectorDB/backend/build
./jadevectordb
```

### Log Files

**Backend Logs:**
- Location: `backend/logs/` (if configured)
- Or check terminal where backend is running

**Key Log Messages:**
```
INFO: Document uploaded: doc_abc123
INFO: Processing started: doc_abc123
INFO: Chunking complete: 156 chunks
INFO: Embedding generation complete
INFO: Vector storage complete
INFO: Document processing complete: doc_abc123
ERROR: Failed to process document: [error details]
```

### Performance Issues

**Slow Upload Processing:**
- Normal: 30-60 seconds per document
- Check Ollama performance
- Monitor CPU usage
- Consider upgrading hardware

**Slow Query Responses:**
- Normal: 2-4 seconds
- Reduce top-k value
- Check JadeVectorDB performance
- Monitor system resources

---

## Best Practices

### Content Quality

**Document Preparation:**
1. ✅ Use text-based PDFs (not scanned images)
2. ✅ Include table of contents and section headers
3. ✅ Ensure consistent formatting
4. ✅ Remove unnecessary pages (blank, covers)
5. ✅ Verify document is latest version

**Content Guidelines:**
- Clear, structured procedures
- Numbered steps for maintenance tasks
- Technical specifications in tables
- Consistent terminology
- Page numbers visible

### Library Management

**Organization:**
- Use consistent naming conventions
- Assign correct device types
- Delete old versions promptly
- Document version changes
- Keep external inventory

**Quality Control:**
1. Test queries after upload
2. Verify answers match source material
3. Check source citations are accurate
4. Review low-confidence queries
5. Identify documentation gaps

### User Support

**Monitoring User Queries:**
- Review query logs regularly
- Identify common questions
- Find documentation gaps
- Note frequently requested topics
- Add missing manuals

**Continuous Improvement:**
1. Collect user feedback
2. Add requested documentation
3. Improve existing content
4. Update outdated manuals
5. Train users on effective questions

### Security

**Access Control:**
- Protect admin panel (add authentication in production)
- Restrict document upload rights
- Control who can delete documents
- Monitor admin actions

**Data Protection:**
- Regular backups
- Secure `.env` file (contains credentials)
- Don't share API keys
- Use HTTPS in production

---

## Maintenance Schedule

### Daily Tasks
- [ ] Check system health dashboard
- [ ] Monitor upload status for new documents
- [ ] Review any error notifications

### Weekly Tasks
- [ ] Review query logs for issues
- [ ] Check storage usage
- [ ] Test sample queries
- [ ] Backup uploaded documents

### Monthly Tasks
- [ ] Review document library for outdated content
- [ ] Update manuals with new versions
- [ ] Analyze user feedback
- [ ] Check for system updates
- [ ] Review and clean up old uploads

### Quarterly Tasks
- [ ] Full system backup
- [ ] Performance optimization review
- [ ] User training refresh
- [ ] Documentation quality audit
- [ ] System configuration review

---

## Support and Resources

### Documentation
- **User Guide**: `docs/USER_GUIDE.md`
- **Deployment Guide**: `docs/DEPLOYMENT.md`
- **Main README**: `README.md`
- **API Documentation**: http://localhost:8000/docs

### Getting Help

**System Issues:**
- Check this guide first
- Review troubleshooting section
- Check backend logs
- Consult IT support

**Content Issues:**
- Review USER_GUIDE.md
- Test with different queries
- Verify source documents
- Contact documentation team

### Useful Commands

```bash
# Start system
./scripts/start.sh

# Stop system
# Press Ctrl+C in terminal

# Restart backend only
cd backend
source venv/bin/activate
uvicorn main:app --reload

# Check system health
curl http://localhost:8000/api/health

# View stats
curl http://localhost:8000/api/stats

# List documents
curl http://localhost:8000/api/admin/documents
```

---

## Appendix: Technical Details

### Document Processing Pipeline

```python
# Simplified processing flow
def process_document(file_path, device_type):
    # 1. Extract text
    text = extract_text(file_path)  # PDF/DOCX → text

    # 2. Chunk semantically
    chunks = chunk_text(text, size=512, overlap=50)

    # 3. Generate embeddings
    embeddings = []
    for chunk in chunks:
        emb = ollama.embed(chunk.text)  # 384-dim vector
        embeddings.append(emb)

    # 4. Store in vector DB
    for chunk, embedding in zip(chunks, embeddings):
        jadevectordb.insert(
            vector=embedding,
            metadata={
                'text': chunk.text,
                'page': chunk.page,
                'section': chunk.section,
                'device_type': device_type
            }
        )
```

### Query Processing Pipeline

```python
# Simplified query flow
def query(question, device_type, top_k):
    # 1. Embed question
    query_embedding = ollama.embed(question)

    # 2. Search vectors
    results = jadevectordb.search(
        vector=query_embedding,
        top_k=top_k,
        filter={'device_type': device_type}
    )

    # 3. Build context
    context = "\n\n".join([r.metadata['text'] for r in results])

    # 4. Generate answer
    prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    answer = ollama.generate(prompt)

    # 5. Extract sources
    sources = [r.metadata for r in results]

    return answer, sources
```

---

**Version 1.0** | Last Updated: March 28, 2026
**For Support:** Contact System Administrator or IT Team
