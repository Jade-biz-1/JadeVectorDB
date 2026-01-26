# JadeVectorDB Re-ranking Server

Python-based re-ranking server using cross-encoder models for improved search precision.

## Overview

This server provides cross-encoder based re-ranking for JadeVectorDB search results. It communicates with the main C++ application via stdin/stdout using a JSON protocol.

**Architecture**: Python Subprocess (Phase 1)
- **Deployment**: Subprocess spawned by JadeVectorDB
- **Communication**: stdin/stdout JSON IPC
- **Best For**: Single-node and small cluster deployments

See `docs/architecture.md` for full architecture documentation and future phases.

## Installation

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)

### Install Dependencies

```bash
cd python
pip install -r requirements.txt
```

For GPU support (recommended for production):

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## Usage

### Command Line

```bash
python reranking_server.py [--model MODEL_NAME] [--batch-size BATCH_SIZE]
```

**Options**:
- `--model`: Cross-encoder model name or path (default: `cross-encoder/ms-marco-MiniLM-L-6-v2`)
- `--batch-size`: Batch size for inference (default: 32)

**Environment Variables**:
- `RERANKING_MODEL_PATH`: Model name or path
- `RERANKING_BATCH_SIZE`: Batch size
- `RERANKING_LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)

### Example

```bash
# Run with default model
python reranking_server.py

# Run with custom model
python reranking_server.py --model cross-encoder/ms-marco-MiniLM-L-12-v2

# Run with environment variables
export RERANKING_MODEL_PATH=cross-encoder/ms-marco-MiniLM-L-12-v2
export RERANKING_BATCH_SIZE=64
export RERANKING_LOG_LEVEL=DEBUG
python reranking_server.py
```

## Protocol

The server uses a line-based JSON protocol over stdin/stdout.

### Request Format

**Re-ranking Request**:
```json
{"query": "search query text", "documents": ["document 1", "document 2", "document 3"]}
```

**Heartbeat Request**:
```json
{"type": "heartbeat"}
```

**Statistics Request**:
```json
{"type": "stats"}
```

**Shutdown Request**:
```json
{"type": "shutdown"}
```

### Response Format

**Re-ranking Response**:
```json
{
  "scores": [0.95, 0.78, 0.62],
  "latency_ms": 123.45,
  "num_documents": 3
}
```

**Heartbeat Response**:
```json
{
  "type": "heartbeat",
  "status": "alive",
  "requests_processed": 42
}
```

**Ready Signal** (sent after model loads):
```json
{
  "type": "ready",
  "model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
  "load_time_ms": 2345.67
}
```

**Error Response**:
```json
{
  "error": "error message",
  "code": "ERROR_CODE",
  "timestamp": "2026-01-09T12:34:56.789Z"
}
```

### Error Codes

- `MODEL_LOAD_ERROR`: Failed to load the model
- `INVALID_REQUEST`: Request missing required fields
- `INVALID_JSON`: Malformed JSON
- `INFERENCE_ERROR`: Model inference failed
- `PROCESSING_ERROR`: General processing error
- `SERVER_ERROR`: Unexpected server error

## Interactive Testing

You can test the server interactively using stdin:

```bash
python reranking_server.py
```

Then type JSON requests:

```
{"query": "What is machine learning?", "documents": ["ML is a field of AI", "Python is a programming language", "Deep learning uses neural networks"]}
{"type": "heartbeat"}
{"type": "stats"}
{"type": "shutdown"}
```

## Performance

### Latency

- **Model loading**: ~2-5 seconds (one-time)
- **Per-document inference**: ~2-3ms (CPU), ~0.5-1ms (GPU)
- **100 documents**: ~150-300ms total (CPU), ~50-100ms (GPU)

### Memory Usage

- **Model**: ~200-500MB
- **Runtime overhead**: ~50-100MB
- **Total**: ~500MB-1GB

### Throughput

- **CPU**: ~300-500 document pairs per second
- **GPU**: ~1000-2000 document pairs per second

## Models

### Available Models

**Fast (Default)**:
- `cross-encoder/ms-marco-MiniLM-L-6-v2`: Fast, good quality
- Size: ~90MB
- Latency: ~2-3ms per document (CPU)

**Better Quality**:
- `cross-encoder/ms-marco-MiniLM-L-12-v2`: Better quality, slower
- Size: ~130MB
- Latency: ~4-6ms per document (CPU)

**Best Quality**:
- `cross-encoder/ms-marco-TinyBERT-L-6`: Best quality, slowest
- Size: ~250MB
- Latency: ~8-12ms per document (CPU)

### Custom Models

You can use any HuggingFace cross-encoder model:

```bash
python reranking_server.py --model your-username/your-model
```

Or load from local path:

```bash
python reranking_server.py --model /path/to/model
```

## Troubleshooting

### Model Download Issues

If model download fails:

1. Check internet connection
2. Verify HuggingFace model name
3. Try manual download:
   ```bash
   python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"
   ```

### Out of Memory

If you encounter OOM errors:

1. Reduce batch size: `--batch-size 16`
2. Use smaller model: `--model cross-encoder/ms-marco-MiniLM-L-6-v2`
3. Close other applications

### Performance Issues

For better performance:

1. **Use GPU**: Install CUDA-enabled PyTorch
2. **Increase batch size**: `--batch-size 64` (if memory allows)
3. **Use smaller model**: Trade quality for speed

### Connection Issues

If C++ subprocess can't communicate:

1. Check Python version: `python3 --version` (must be 3.9+)
2. Verify dependencies: `pip list | grep sentence-transformers`
3. Test standalone: `echo '{"query":"test","documents":["doc"]}' | python reranking_server.py`

## Integration with JadeVectorDB

The C++ application spawns this server as a subprocess. See:
- `backend/src/search/reranking_service.h/cpp`: C++ integration
- `docs/architecture.md`: Architecture documentation
- `TasksTracking/16-hybrid-search-reranking-analytics.md`: Implementation plan

## Development

### Running Tests

```bash
cd python
python -m pytest test_reranking_server.py -v
```

### Logging

Set log level for debugging:

```bash
export RERANKING_LOG_LEVEL=DEBUG
python reranking_server.py
```

Logs go to stderr (not stdout) to avoid interfering with JSON protocol.

### Monitoring

Monitor the server in real-time:

```bash
# In one terminal, run server
python reranking_server.py 2>server.log

# In another terminal, watch logs
tail -f server.log
```

## Future Enhancements

See `docs/architecture.md` for planned improvements:

**Phase 2**: Dedicated Re-ranking Service
- gRPC-based microservice
- Better for large distributed clusters
- GPU sharing across nodes
- Independent scaling

**Phase 3**: ONNX Runtime
- Native C++ inference
- Maximum performance
- Simplified deployment

## License

Part of JadeVectorDB project. See main repository LICENSE file.
