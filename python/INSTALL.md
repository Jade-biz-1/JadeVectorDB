# Installation Guide - Re-ranking Server

Quick guide to set up the Python re-ranking server for JadeVectorDB.

## Prerequisites

- Python 3.9 or higher
- pip (Python package manager)
- 2GB free disk space (for models)
- 1GB RAM minimum (2GB+ recommended)

## Step 1: Check Python Version

```bash
python3 --version
# Should output: Python 3.9.x or higher
```

## Step 2: Install Dependencies

### Option A: Using pip directly

```bash
cd /home/deepak/Public/JadeVectorDB/python
pip3 install -r requirements.txt
```

### Option B: Using virtual environment (recommended)

```bash
cd /home/deepak/Public/JadeVectorDB/python

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate  # On Windows

# Install dependencies
pip install -r requirements.txt
```

## Step 3: Verify Installation

```bash
python3 -c "from sentence_transformers import CrossEncoder; print('✓ sentence-transformers installed')"
```

Expected output:
```
✓ sentence-transformers installed
```

## Step 4: Test the Server

### Quick Test (Interactive)

```bash
python3 reranking_server.py
```

Wait for model to load (~5 seconds), then type:

```json
{"query": "test", "documents": ["doc1", "doc2"]}
```

You should see a response with scores.

Press Ctrl+C to exit.

### Automated Test

```bash
python3 test_reranking_server.py
```

Expected output:
```
================================================================================
JadeVectorDB Re-ranking Server Test
================================================================================

[1/6] Starting reranking server...
✓ Server process started

[2/6] Waiting for model to load...
✓ Model loaded: cross-encoder/ms-marco-MiniLM-L-6-v2
  Load time: 2345ms

[3/6] Testing heartbeat...
✓ Heartbeat OK: status=alive

[4/6] Testing re-ranking...
✓ Re-ranking completed
  Latency: 123.45ms
  Documents: 5
  Per-document: 24.69ms

  Ranked results:
    1. Score: 0.950 - Machine learning is a branch of artificial intelligence...
    2. Score: 0.820 - Deep learning is a subset of machine learning...
    3. Score: 0.780 - Artificial intelligence enables computers...

  ✓ All validations passed

[5/6] Testing edge cases...
✓ Empty documents handled correctly

[6/6] Testing statistics...
✓ Statistics retrieved:
  Requests processed: 2
  Average latency: 61.73ms

[Cleanup] Shutting down server...
✓ Server shut down gracefully

================================================================================
✅ All tests passed!
================================================================================
```

### Example Usage

```bash
python3 example_usage.py
```

This will show a practical example of re-ranking search results.

## Step 5: GPU Support (Optional, for Production)

For better performance, install GPU-enabled PyTorch:

### NVIDIA GPU (CUDA 11.8)

```bash
pip3 install torch --index-url https://download.pytorch.org/whl/cu118
```

### NVIDIA GPU (CUDA 12.1)

```bash
pip3 install torch --index-url https://download.pytorch.org/whl/cu121
```

### Verify GPU

```bash
python3 -c "import torch; print('GPU available:', torch.cuda.is_available())"
```

## Troubleshooting

### Issue: ModuleNotFoundError: No module named 'sentence_transformers'

**Solution**: Install dependencies
```bash
pip3 install sentence-transformers
```

### Issue: Model download fails

**Solution**: Check internet connection and try manual download
```bash
python3 -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"
```

### Issue: Out of memory

**Solutions**:
1. Close other applications
2. Use smaller batch size: `--batch-size 16`
3. Use smaller model (already using the smallest)

### Issue: Permission denied

**Solution**: Make scripts executable
```bash
chmod +x *.py
```

## Next Steps

Once installed and tested:

1. **Review architecture**: See `docs/architecture.md` for deployment options
2. **C++ integration**: Implement RerankingService (T16.10)
3. **Production deployment**: Configure environment variables

## Model Information

### Default Model

- **Name**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Size**: ~90MB
- **Performance**: Good balance of speed and quality
- **Latency**: ~2-3ms per document (CPU)

### Model Storage

Models are cached in:
- Linux/Mac: `~/.cache/huggingface/`
- Windows: `C:\Users\<user>\.cache\huggingface\`

First run downloads the model (~90MB). Subsequent runs use cached model.

## Support

- **Documentation**: `python/README.md`
- **Architecture**: `docs/architecture.md`
- **Task tracking**: `TasksTracking/16-hybrid-search-reranking-analytics.md`
- **Issues**: Create GitHub issue with error logs
