# RAG Use Case: Maintenance Documentation Q&A System

## Executive Summary

This document outlines the architecture and implementation strategy for a **Retrieval-Augmented Generation (RAG)** system designed to help field engineers and mechanics access maintenance documentation for devices, instruments, and machines. The system uses JadeVectorDB as the vector database, Ollama for local LLM inference, and operates entirely offline for data privacy and cost efficiency.

**Target Scale**: Medium (100-1000 documents, 1000-10000 pages)
**Deployment**: Local workstation/laptop
**Language**: English only
**LLM**: Ollama (local, offline)

---

## Table of Contents

1. [Use Case Overview](#use-case-overview)
2. [System Architecture](#system-architecture)
3. [Component Details](#component-details)
4. [JadeVectorDB Capability Assessment](#jadevectordb-capability-assessment)
5. [Required Enhancements](#required-enhancements)
6. [Technology Stack](#technology-stack)
7. [Implementation Roadmap](#implementation-roadmap)
8. [Best Practices](#best-practices)
9. [Performance Considerations](#performance-considerations)
10. [Cost Analysis](#cost-analysis)
11. [References](#references)

---

## Use Case Overview

### Problem Statement

Field engineers and mechanics need quick access to specific maintenance procedures, troubleshooting steps, and technical specifications buried within hundreds of PDF and DOCX documents. Traditional keyword search is inefficient and often misses semantically relevant information.

### Solution

A RAG-based Q&A system that:
1. **Ingests** PDF and DOCX maintenance documents
2. **Chunks** documents into semantically meaningful segments
3. **Embeds** text chunks into high-dimensional vectors
4. **Stores** vectors and metadata in JadeVectorDB
5. **Retrieves** relevant context based on user questions
6. **Generates** accurate, context-aware answers using a local LLM

### User Workflow

```
┌─────────────────────────────────────────────────────────────┐
│                      User Journey                           │
└─────────────────────────────────────────────────────────────┘

Field Engineer → Asks Question → System Searches Docs →
  → Retrieves Relevant Sections → LLM Generates Answer →
    → Engineer Gets Answer + Source References
```

### Key Benefits

- **Offline Operation**: No internet required, works in remote field locations
- **Data Privacy**: Sensitive maintenance docs never leave local infrastructure
- **Zero Ongoing Costs**: No API fees for embeddings or LLM inference
- **Fast Responses**: Local processing with sub-second retrieval
- **Source Attribution**: Every answer includes references to source documents

---

## System Architecture

### High-Level Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                        RAG System Architecture                     │
└────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────┐
│                     1. DOCUMENT INGESTION PIPELINE                 │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌─────────┐      ┌──────────────┐      ┌─────────────────┐        │
│  │ PDF/    │ ───> │ Text         │ ───> │ Semantic        │        │
│  │ DOCX    │      │ Extraction   │      │ Chunking        │        │
│  │ Files   │      │ (PyMuPDF,    │      │ (LangChain)     │        │
│  └─────────┘      │  python-docx)│      └─────────────────┘        │
│                   └──────────────┘               │                 │
│                                                  │                 │
│                                                  v                 │
│                            ┌──────────────────────────────┐        │
│                            │ Metadata Extraction          │        │
│                            │ - Document name              │        │
│                            │ - Section/chapter            │        │
│                            │ - Page numbers               │        │
│                            │ - Device/machine type        │        │
│                            └──────────────────────────────┘        │
│                                       │                            │
└───────────────────────────────────────┼────────────────────────────┘
                                        │
                                        v
┌───────────────────────────────────────┼────────────────────────────┐
│                     2. EMBEDDING GENERATION                        │
├───────────────────────────────────────┼────────────────────────────┤
│                                       │                            │
│                            ┌──────────────────────┐                │
│                            │ Local Embedding Model│                │
│                            │ (e5-small or         │                │
│                            │  nomic-embed-text    │                │
│                            │  via Ollama)         │                │
│                            └──────────────────────┘                │
│                                       │                            │
│                                       v                            │
│                            ┌──────────────────────┐                │
│                            │ 384-dim vectors      │                │
│                            └──────────────────────┘                │
└───────────────────────────────────────┼────────────────────────────┘
                                        │
                                        v
┌───────────────────────────────────────┼────────────────────────────┐
│                     3. VECTOR STORAGE (JadeVectorDB)               │
├───────────────────────────────────────┼────────────────────────────┤
│                                       │                            │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │                    JadeVectorDB                            │    │
│  │                                                            │    │
│  │  Database: "maintenance_docs"                              │    │
│  │  Dimension: 384                                            │    │
│  │  Index Type: HNSW (for fast similarity search)             │    │
│  │                                                            │    │
│  │  Storage:                                                  │    │
│  │  ┌──────────────────────┬──────────────────────────────┐   │    │
│  │  │ Vector Embeddings    │ Metadata                     │   │    │
│  │  │ (384-dim floats)     │ - doc_id                     │   │    │
│  │  │                      │ - doc_name                   │   │    │
│  │  │                      │ - chunk_id                   │   │    │
│  │  │                      │ - page_numbers               │   │    │
│  │  │                      │ - section                    │   │    │
│  │  │                      │ - text (original chunk)      │   │    │
│  │  │                      │ - device_type                │   │    │
│  │  └──────────────────────┴──────────────────────────────┘   │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                     4. QUERY PROCESSING PIPELINE                     │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  User Question                                                       │
│       │                                                              │
│       v                                                              │
│  ┌─────────────────────┐                                            │
│  │ Query Embedding     │  (Same embedding model)                    │
│  └─────────────────────┘                                            │
│       │                                                              │
│       v                                                              │
│  ┌─────────────────────────────────────────────┐                    │
│  │ Similarity Search in JadeVectorDB           │                    │
│  │ - Cosine similarity                         │                    │
│  │ - Retrieve top-k chunks (k=5-10)           │                    │
│  │ - Optional: metadata filtering              │                    │
│  │   (device_type, section, etc.)             │                    │
│  └─────────────────────────────────────────────┘                    │
│       │                                                              │
│       v                                                              │
│  ┌─────────────────────────────────────────────┐                    │
│  │ Retrieved Context Chunks                     │                    │
│  │ - Chunk 1 (similarity: 0.89)                │                    │
│  │ - Chunk 2 (similarity: 0.85)                │                    │
│  │ - Chunk 3 (similarity: 0.82)                │                    │
│  │ - ...                                        │                    │
│  └─────────────────────────────────────────────┘                    │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                     5. ANSWER GENERATION (LLM)                       │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────┐                    │
│  │ Prompt Construction                          │                    │
│  │                                              │                    │
│  │ System: You are a helpful maintenance        │                    │
│  │         assistant...                         │                    │
│  │                                              │                    │
│  │ Context: [Retrieved chunks 1-5]             │                    │
│  │                                              │                    │
│  │ Question: [User question]                    │                    │
│  │                                              │                    │
│  │ Instructions: Answer based on context,       │                    │
│  │              cite sources, indicate if       │                    │
│  │              information is not found        │                    │
│  └─────────────────────────────────────────────┘                    │
│       │                                                              │
│       v                                                              │
│  ┌─────────────────────────────────────────────┐                    │
│  │ Ollama (Local LLM)                           │                    │
│  │ - llama3.2 or mistral                        │                    │
│  │ - Runs locally, no internet needed          │                    │
│  └─────────────────────────────────────────────┘                    │
│       │                                                              │
│       v                                                              │
│  ┌─────────────────────────────────────────────┐                    │
│  │ Generated Answer + Source Citations          │                    │
│  │                                              │                    │
│  │ Answer: "To reset the XYZ-100, follow       │                    │
│  │          these steps: 1) Turn off power...  │                    │
│  │                                              │                    │
│  │ Sources:                                     │                    │
│  │ - XYZ-100_Manual.pdf, Page 45                │                    │
│  │ - Troubleshooting_Guide.docx, Section 3.2   │                    │
│  └─────────────────────────────────────────────┘                    │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                     6. USER INTERFACE                                │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Option 1: Web UI (Streamlit/Gradio)                               │
│  ┌────────────────────────────────────────┐                        │
│  │ ┌────────────────────────────────────┐ │                        │
│  │ │ Question: [Text input box]         │ │                        │
│  │ └────────────────────────────────────┘ │                        │
│  │ [Ask] [Clear]                           │                        │
│  │                                         │                        │
│  │ Answer:                                 │                        │
│  │ ┌────────────────────────────────────┐ │                        │
│  │ │ [Generated answer with formatting] │ │                        │
│  │ │                                    │ │                        │
│  │ │ Sources:                           │ │                        │
│  │ │ • Doc1.pdf, Page 45               │ │                        │
│  │ │ • Doc2.docx, Section 3.2          │ │                        │
│  │ └────────────────────────────────────┘ │                        │
│  └────────────────────────────────────────┘                        │
│                                                                      │
│  Option 2: CLI Interface (for power users)                         │
│  $ rag-query "How to reset XYZ-100?"                               │
│                                                                      │
│  Option 3: REST API (for integration with existing tools)          │
│  POST /api/v1/query                                                 │
│  { "question": "How to reset XYZ-100?" }                           │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
Document Ingestion (One-time setup):
PDF/DOCX → Parse → Chunk → Embed → Store in JadeVectorDB

Query Processing (Runtime):
User Question → Embed → Search JadeVectorDB → Retrieve Context →
  → Send to Ollama → Generate Answer → Display to User
```

---

## Component Details

### 1. Document Processing Pipeline

#### Text Extraction

**For PDF Files:**
- **Primary**: PyMuPDF (fitz) - Fast, accurate, handles complex layouts
- **Alternative**: pdfplumber - Good for tables and structured data
- **Fallback**: OCR with Tesseract for scanned PDFs

**For DOCX Files:**
- **Primary**: python-docx - Official Python library for Word documents
- **Features**: Extract text, preserve formatting, extract tables

#### Chunking Strategy

Based on [2025 RAG best practices](https://medium.com/@adnanmasood/chunking-strategies-for-retrieval-augmented-generation-rag-a-comprehensive-guide-5522c4ea2a90), **semantic chunking** is recommended for maintenance documentation:

**Approach: Hybrid Semantic Chunking**

```python
Chunking Parameters:
- Base chunk size: 512 tokens (~400 words)
- Overlap: 50 tokens (10%) to preserve context
- Respect boundaries: Sections, paragraphs, numbered lists
- Preserve structure: Keep procedure steps together
```

**Why This Works:**
- Maintenance docs have clear structure (sections, procedures)
- Semantic coherence ensures complete procedure steps stay together
- Overlap prevents information loss at chunk boundaries
- Optimized size balances context vs. specificity

**Implementation:**
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=50,
    length_function=len,
    separators=[
        "\n\n\n",  # Section breaks
        "\n\n",    # Paragraph breaks
        "\n",      # Line breaks
        ". ",      # Sentences
        " ",       # Words
        ""         # Characters
    ]
)
```

#### Metadata Extraction

Each chunk includes rich metadata for filtering and source attribution:

```json
{
  "vector_id": "doc_123_chunk_45",
  "vector": [0.123, 0.456, ...],  // 384-dim embedding
  "metadata": {
    "doc_id": "XYZ-100_Manual_v2.3",
    "doc_name": "XYZ-100 Service Manual",
    "doc_type": "pdf",
    "file_path": "/docs/manuals/XYZ-100_Manual.pdf",
    "chunk_id": 45,
    "page_numbers": [23, 24],
    "section": "Chapter 3: Troubleshooting",
    "subsection": "3.2 Reset Procedures",
    "device_type": "XYZ-100",
    "device_category": "Diagnostic Equipment",
    "text": "To reset the XYZ-100 device, follow these steps: 1) Turn off power...",
    "chunk_length": 487,
    "created_at": "2025-01-02T10:30:00Z"
  }
}
```

### 2. Embedding Generation

#### Recommended Model: E5-Small

Based on [2025 benchmarks](https://supermemory.ai/blog/best-open-source-embedding-models-benchmarked-and-ranked/), **E5-Small** is optimal for this use case:

**Specifications:**
- **Parameters**: 118M
- **Dimensions**: 384
- **Max tokens**: 512
- **Performance**: 100% Top-5 accuracy, <30ms latency
- **Size**: ~500MB on disk

**Why E5-Small?**
- Excellent accuracy for RAG tasks
- Fast inference on laptop CPUs
- Small memory footprint (ideal for local deployment)
- 384 dimensions balance accuracy and storage efficiency

**Alternative: Nomic-Embed-Text (via Ollama)**
- Pre-integrated with Ollama
- Competitive performance
- Slightly larger (768 dimensions)

#### Integration Options

**Option 1: Direct via Sentence Transformers (Recommended)**
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('intfloat/e5-small-v2')
embeddings = model.encode(texts, convert_to_numpy=True)
```

**Option 2: Via Ollama API**
```python
import ollama

response = ollama.embeddings(
    model='nomic-embed-text',
    prompt='Your text here'
)
embedding = response['embedding']
```

### 3. Vector Storage (JadeVectorDB)

#### Database Configuration

```json
{
  "name": "maintenance_docs",
  "description": "Maintenance documentation for field engineers",
  "vectorDimension": 384,
  "indexType": "HNSW",
  "distance_metric": "cosine",
  "config": {
    "hnsw_m": 16,
    "hnsw_ef_construction": 200,
    "hnsw_ef_search": 100
  }
}
```

#### Storage Structure

**For 1000 documents, ~10,000 pages:**

Estimated metrics:
- Chunks: ~50,000 (5 chunks/page average)
- Vectors: 50,000 × 384 dimensions × 4 bytes = ~77 MB (vectors only)
- Metadata: ~50,000 × 2 KB = ~100 MB
- **Total storage**: ~200-300 MB (highly manageable)

#### Index Type: HNSW

JadeVectorDB's HNSW (Hierarchical Navigable Small World) index is ideal for this use case:

**Advantages:**
- Sub-50ms search for 1M vectors (per JadeVectorDB specs)
- Excellent recall (>95% at top-10)
- Memory-efficient graph structure
- Incremental updates (add new docs without full reindex)

**Search Performance:**
- For 50,000 chunks: <10ms latency
- Recall@10: >95%
- Throughput: 1000+ queries/second

### 4. Query Processing

#### Query Enhancement

Before embedding the user question, apply these enhancements:

**1. Query Expansion (Optional)**
```python
# Expand abbreviations common in maintenance docs
expansions = {
    "PSU": "power supply unit",
    "HMI": "human machine interface",
    "PLC": "programmable logic controller"
}
```

**2. Instruction Prefix (for E5 models)**
```python
# E5 models benefit from instruction prefix
query_prefix = "query: "
expanded_query = query_prefix + user_question
```

#### Retrieval Strategy

**Hybrid Retrieval (Recommended for 2025):**

```python
# 1. Vector similarity search
vector_results = jadevectordb.search(
    database_id="maintenance_docs",
    query_vector=query_embedding,
    top_k=10,
    threshold=0.7  # Minimum similarity
)

# 2. Optional: Metadata filtering
filtered_results = [
    r for r in vector_results
    if r['metadata']['device_type'] == user_selected_device
]

# 3. Re-rank by recency (if time-sensitive)
sorted_results = sorted(
    filtered_results,
    key=lambda x: (x['similarity'], x['metadata']['created_at']),
    reverse=True
)

# 4. Select top-5 for context
top_chunks = sorted_results[:5]
```

**Context Window Management:**

```python
# LLMs have limited context (e.g., 4K tokens for Llama 3.2)
# Budget allocation:
# - System prompt: ~200 tokens
# - Retrieved context: ~2000 tokens (400 tokens × 5 chunks)
# - User question: ~50 tokens
# - Answer generation: ~1750 tokens (reserved)
# Total: ~4000 tokens

max_context_tokens = 2000
selected_chunks = []
current_tokens = 0

for chunk in top_chunks:
    chunk_tokens = estimate_tokens(chunk['text'])
    if current_tokens + chunk_tokens <= max_context_tokens:
        selected_chunks.append(chunk)
        current_tokens += chunk_tokens
    else:
        break
```

### 5. Answer Generation (Ollama)

#### LLM Selection

**Recommended Models (via Ollama):**

| Model | Size | Context | Speed | Best For |
|-------|------|---------|-------|----------|
| **llama3.2:3b** | 2GB | 8K tokens | Fast | Laptops, quick answers |
| **mistral:7b** | 4GB | 8K tokens | Medium | Balanced performance |
| **llama3.1:8b** | 4.7GB | 128K tokens | Medium | Long context, detailed answers |

**For your use case (laptop deployment):** **llama3.2:3b** is recommended
- Small enough to run smoothly on 8GB RAM
- Fast inference (~50 tokens/sec on CPU)
- Sufficient quality for technical Q&A

#### Prompt Engineering

**System Prompt:**
```python
system_prompt = """You are a helpful maintenance assistant for field engineers and mechanics.

Your role:
- Answer questions based ONLY on the provided maintenance documentation
- Provide step-by-step instructions clearly and concisely
- Always cite source documents (document name and page number)
- If the information is not in the provided context, say "I don't have that information in the available documentation"
- Use technical terminology accurately
- Format procedures as numbered lists for clarity

Guidelines:
- Be precise and safety-conscious
- Highlight any warnings or cautions mentioned in the documentation
- If multiple procedures exist, present the official/recommended one first
"""
```

**User Prompt Template:**
```python
user_prompt_template = """Context from maintenance documentation:

{context_chunks}

Question: {user_question}

Please provide a detailed answer based on the context above. Include source references (document name and page number) for your answer.

Answer:"""
```

**Full Prompt Construction:**
```python
context_text = "\n\n---\n\n".join([
    f"Source: {chunk['metadata']['doc_name']}, Page {chunk['metadata']['page_numbers']}\n"
    f"Section: {chunk['metadata']['section']}\n\n"
    f"{chunk['metadata']['text']}"
    for chunk in retrieved_chunks
])

full_prompt = user_prompt_template.format(
    context_chunks=context_text,
    user_question=user_question
)
```

#### Ollama Integration

```python
import ollama

response = ollama.chat(
    model='llama3.2:3b',
    messages=[
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': full_prompt}
    ],
    options={
        'temperature': 0.1,  # Low for factual accuracy
        'top_p': 0.9,
        'max_tokens': 1024
    }
)

answer = response['message']['content']
```

### 6. User Interface

#### Option 1: Streamlit Web UI (Recommended)

**Advantages:**
- Rapid development (< 100 lines of code)
- Professional-looking interface
- Easy to deploy locally
- Supports history, file upload for new docs

**Example Interface:**
```python
import streamlit as st

st.title("🔧 Maintenance Documentation Q&A")

# Device filter (optional)
device_filter = st.selectbox(
    "Filter by device (optional)",
    options=["All"] + get_unique_devices()
)

# Question input
question = st.text_area("Ask a question about maintenance procedures:")

if st.button("Get Answer"):
    with st.spinner("Searching documentation..."):
        # 1. Embed query
        query_embedding = embed_text(question)

        # 2. Search JadeVectorDB
        results = search_database(query_embedding, device_filter)

        # 3. Generate answer
        answer = generate_answer(question, results)

        # 4. Display
        st.success("Answer:")
        st.markdown(answer['text'])

        st.info("Sources:")
        for source in answer['sources']:
            st.write(f"• {source['doc_name']}, Page {source['page']}")
```

#### Option 2: CLI Interface

For field engineers who prefer command-line:

```bash
$ rag-query "How to calibrate XYZ-100 sensor?"

Searching documentation...
Found 5 relevant sections.

Answer:
To calibrate the XYZ-100 sensor, follow these steps:

1. Ensure the device is powered off
2. Connect the calibration probe to port A3
3. Power on while holding the CAL button
4. Wait for the LED to turn green (approximately 30 seconds)
5. Follow the on-screen prompts to complete calibration

⚠️  WARNING: Do not disconnect the calibration probe until the process is complete.

Sources:
• XYZ-100_Service_Manual.pdf, Page 67
• Calibration_Procedures.docx, Section 4.2

Query time: 0.8s
```

#### Option 3: REST API

For integration with existing maintenance management systems:

```python
from fastapi import FastAPI

app = FastAPI()

@app.post("/api/v1/query")
async def query_documents(request: QueryRequest):
    # 1. Embed query
    embedding = embed_text(request.question)

    # 2. Search
    results = jadevectordb.search(
        database_id="maintenance_docs",
        query_vector=embedding,
        top_k=10
    )

    # 3. Generate answer
    answer = ollama_generate(request.question, results)

    return {
        "answer": answer,
        "sources": extract_sources(results),
        "confidence": calculate_confidence(results)
    }
```

---

## JadeVectorDB Capability Assessment

### Current Capabilities (✅ Ready to Use)

Based on the documentation review, JadeVectorDB provides:

1. **✅ Vector Storage**
   - High-performance storage with memory-mapped files
   - 384-dimension vectors (perfect for E5-small)
   - Metadata storage (critical for source attribution)

2. **✅ Similarity Search**
   - Cosine similarity (ideal for text embeddings)
   - HNSW indexing for fast retrieval
   - Top-k search with threshold filtering
   - Sub-50ms search for 1M vectors (spec from README)

3. **✅ Metadata Filtering**
   - Complex filters with AND/OR logic
   - Range queries for numeric metadata
   - Essential for device-type filtering

4. **✅ Batch Operations**
   - Batch vector insertion
   - Critical for initial document ingestion
   - Bulk retrieval for multi-query scenarios

5. **✅ REST API**
   - Well-documented HTTP API
   - Easy integration with Python pipeline
   - Multiple client libraries (Python CLI, Shell CLI)

6. **✅ Persistence**
   - Hybrid SQLite + memory-mapped storage
   - ACID guarantees for metadata
   - WAL (Write-Ahead Logging) for durability

7. **✅ Database Management**
   - Multi-database support (can separate by department/device type)
   - Custom configurations per database
   - Schema validation

8. **✅ Authentication & Security**
   - JWT-based authentication
   - API key management
   - RBAC (Role-Based Access Control)
   - Audit logging

### Capability Gaps (⚠️ Requires Custom Development)

The following components are **NOT** part of JadeVectorDB and need to be developed:

1. **⚠️ Document Processing Pipeline**
   - PDF/DOCX parsing
   - Text extraction
   - Chunking logic
   - Metadata extraction
   - **Action**: Build separate Python pipeline

2. **⚠️ Embedding Generation**
   - JadeVectorDB has `/v1/embeddings/generate` endpoint, but:
     - Current implementation details unclear from docs
     - May not support E5-small or Ollama models
   - **Action**: Use external embedding service (Sentence Transformers or Ollama)

3. **⚠️ RAG Orchestration Layer**
   - Query processing logic
   - Context assembly
   - LLM integration
   - Answer formatting
   - **Action**: Build custom Python application

4. **⚠️ User Interface**
   - No built-in Q&A interface
   - Web frontend is for database management, not RAG
   - **Action**: Build Streamlit/FastAPI UI

5. **⚠️ LLM Integration**
   - No Ollama integration
   - No prompt management
   - **Action**: Direct Ollama API integration

### Enhancement Recommendations

While not critical for MVP, these enhancements would improve the system:

1. **Hybrid Search** (Vector + Keyword)
   - Current: Pure vector similarity
   - Enhancement: Add BM25 keyword search, combine scores
   - Benefit: Better handling of exact model numbers, part codes

2. **Re-ranking**
   - Current: Simple cosine similarity ranking
   - Enhancement: Add cross-encoder re-ranking
   - Benefit: Improved relevance of top results

3. **Query Analytics**
   - Track common questions
   - Identify documentation gaps
   - Monitor answer quality

4. **Document Update Tracking**
   - Version control for documents
   - Incremental re-indexing
   - Change notifications

5. **Feedback Loop**
   - User ratings on answers
   - Collect ground-truth Q&A pairs
   - Fine-tune retrieval thresholds

---

## Required Enhancements

### 1. Document Ingestion Service

**Purpose**: Parse PDFs/DOCX, chunk, embed, and store in JadeVectorDB

**Components:**

```python
# document_processor.py

class DocumentProcessor:
    def __init__(self, jadevectordb_client, embedding_model):
        self.db_client = jadevectordb_client
        self.embedder = embedding_model

    def process_pdf(self, pdf_path: str) -> List[Dict]:
        """Extract text from PDF, chunk, and prepare for storage."""
        import fitz  # PyMuPDF

        doc = fitz.open(pdf_path)
        chunks = []

        for page_num, page in enumerate(doc):
            text = page.get_text()
            page_chunks = self.chunk_text(text, page_num + 1)
            chunks.extend(page_chunks)

        return chunks

    def process_docx(self, docx_path: str) -> List[Dict]:
        """Extract text from DOCX, chunk, and prepare for storage."""
        from docx import Document

        doc = Document(docx_path)
        full_text = "\n".join([para.text for para in doc.paragraphs])
        chunks = self.chunk_text(full_text)

        return chunks

    def chunk_text(self, text: str, page_num: int = None) -> List[Dict]:
        """Apply semantic chunking strategy."""
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=512,
            chunk_overlap=50,
            separators=["\n\n\n", "\n\n", "\n", ". ", " ", ""]
        )

        chunks = splitter.split_text(text)

        return [
            {
                'text': chunk,
                'page_num': page_num,
                'chunk_id': i
            }
            for i, chunk in enumerate(chunks)
        ]

    def embed_and_store(self, document_path: str, metadata: Dict):
        """Full pipeline: parse → chunk → embed → store."""

        # 1. Parse
        if document_path.endswith('.pdf'):
            chunks = self.process_pdf(document_path)
        elif document_path.endswith('.docx'):
            chunks = self.process_docx(document_path)
        else:
            raise ValueError(f"Unsupported file type: {document_path}")

        # 2. Embed
        texts = [chunk['text'] for chunk in chunks]
        embeddings = self.embedder.encode(texts)

        # 3. Store in JadeVectorDB
        vectors = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            vector = {
                'id': f"{metadata['doc_id']}_chunk_{i}",
                'values': embedding.tolist(),
                'metadata': {
                    **metadata,
                    'chunk_id': i,
                    'text': chunk['text'],
                    'page_num': chunk.get('page_num'),
                    'chunk_length': len(chunk['text'])
                }
            }
            vectors.append(vector)

        # Batch insert
        self.db_client.batch_store_vectors(
            database_id="maintenance_docs",
            vectors=vectors
        )

        return len(vectors)
```

**Batch Processing Script:**

```python
# ingest_documents.py

from pathlib import Path
from document_processor import DocumentProcessor
from sentence_transformers import SentenceTransformer
from jadevectordb import JadeVectorDBClient

def ingest_all_documents(docs_directory: str):
    # Initialize
    db_client = JadeVectorDBClient("http://localhost:8080")
    embedder = SentenceTransformer('intfloat/e5-small-v2')
    processor = DocumentProcessor(db_client, embedder)

    # Find all PDFs and DOCX files
    docs_path = Path(docs_directory)
    files = list(docs_path.glob("**/*.pdf")) + list(docs_path.glob("**/*.docx"))

    print(f"Found {len(files)} documents to process...")

    for file_path in files:
        print(f"Processing: {file_path.name}")

        # Extract metadata from filename or directory structure
        metadata = {
            'doc_id': file_path.stem,
            'doc_name': file_path.name,
            'doc_type': file_path.suffix[1:],  # pdf or docx
            'file_path': str(file_path),
            'device_type': extract_device_type(file_path),  # Custom logic
            'created_at': datetime.now().isoformat()
        }

        try:
            num_chunks = processor.embed_and_store(str(file_path), metadata)
            print(f"  ✓ Stored {num_chunks} chunks")
        except Exception as e:
            print(f"  ✗ Error: {e}")

    print("Ingestion complete!")

if __name__ == "__main__":
    ingest_all_documents("/path/to/maintenance/docs")
```

### 2. RAG Query Service

**Purpose**: Handle user queries, retrieve context, generate answers

```python
# rag_service.py

from sentence_transformers import SentenceTransformer
from jadevectordb import JadeVectorDBClient
import ollama

class RAGService:
    def __init__(self):
        self.db_client = JadeVectorDBClient("http://localhost:8080")
        self.embedder = SentenceTransformer('intfloat/e5-small-v2')
        self.llm_model = "llama3.2:3b"

    def query(self, question: str, device_filter: str = None, top_k: int = 5):
        """Full RAG pipeline."""

        # 1. Embed question
        query_embedding = self.embedder.encode(f"query: {question}")

        # 2. Search JadeVectorDB
        search_results = self.db_client.search(
            database_id="maintenance_docs",
            query_vector=query_embedding.tolist(),
            top_k=top_k * 2,  # Retrieve more for filtering
            threshold=0.65
        )

        # 3. Optional: Filter by device
        if device_filter:
            search_results = [
                r for r in search_results
                if r['metadata'].get('device_type') == device_filter
            ]

        # 4. Take top-k after filtering
        top_results = search_results[:top_k]

        # 5. Build context
        context = self._build_context(top_results)

        # 6. Generate answer with Ollama
        answer = self._generate_answer(question, context)

        # 7. Format response
        return {
            'answer': answer,
            'sources': self._extract_sources(top_results),
            'confidence': self._calculate_confidence(top_results)
        }

    def _build_context(self, results: List[Dict]) -> str:
        """Assemble retrieved chunks into LLM context."""
        context_parts = []

        for i, result in enumerate(results):
            meta = result['metadata']
            context_parts.append(
                f"--- Source {i+1} ---\n"
                f"Document: {meta['doc_name']}\n"
                f"Page: {meta.get('page_num', 'N/A')}\n"
                f"Section: {meta.get('section', 'N/A')}\n\n"
                f"{meta['text']}\n"
            )

        return "\n\n".join(context_parts)

    def _generate_answer(self, question: str, context: str) -> str:
        """Use Ollama to generate answer."""

        system_prompt = """You are a helpful maintenance assistant for field engineers.
Answer questions based ONLY on the provided documentation.
Always cite sources with document name and page number.
If information is not in the context, say so clearly.
Format procedures as numbered lists."""

        user_prompt = f"""Context from maintenance documentation:

{context}

Question: {question}

Provide a detailed answer based on the context above. Include source references.

Answer:"""

        response = ollama.chat(
            model=self.llm_model,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ],
            options={'temperature': 0.1}
        )

        return response['message']['content']

    def _extract_sources(self, results: List[Dict]) -> List[Dict]:
        """Extract source citations."""
        sources = []
        for result in results:
            meta = result['metadata']
            sources.append({
                'doc_name': meta['doc_name'],
                'page': meta.get('page_num', 'N/A'),
                'section': meta.get('section', 'N/A'),
                'similarity': result.get('similarity', 0.0)
            })
        return sources

    def _calculate_confidence(self, results: List[Dict]) -> str:
        """Simple confidence heuristic."""
        if not results:
            return "No relevant information found"

        avg_similarity = sum(r.get('similarity', 0) for r in results) / len(results)

        if avg_similarity > 0.85:
            return "High"
        elif avg_similarity > 0.70:
            return "Medium"
        else:
            return "Low"
```

### 3. Web Interface (Streamlit)

**Purpose**: User-friendly interface for field engineers

```python
# app.py

import streamlit as st
from rag_service import RAGService

st.set_page_config(page_title="Maintenance Q&A", page_icon="🔧", layout="wide")

# Initialize RAG service (cached)
@st.cache_resource
def load_rag_service():
    return RAGService()

rag = load_rag_service()

# UI Layout
st.title("🔧 Maintenance Documentation Q&A System")
st.markdown("Ask questions about device maintenance, troubleshooting, and procedures.")

# Sidebar: Filters
with st.sidebar:
    st.header("Filters")

    device_filter = st.selectbox(
        "Device Type (Optional)",
        options=["All Devices", "XYZ-100", "ABC-200", "DEF-300"]  # Dynamic from DB
    )

    top_k = st.slider("Number of sources to retrieve", 3, 10, 5)

    st.markdown("---")
    st.info("💡 Tip: Be specific in your questions for better results.")

# Main area: Question input
question = st.text_area(
    "Your Question:",
    placeholder="e.g., How do I reset the XYZ-100 after a power failure?",
    height=100
)

col1, col2, col3 = st.columns([1, 1, 4])

with col1:
    ask_button = st.button("🔍 Get Answer", type="primary")

with col2:
    clear_button = st.button("Clear")

if clear_button:
    st.rerun()

if ask_button and question:
    with st.spinner("🔎 Searching documentation and generating answer..."):
        # Query RAG system
        device = None if device_filter == "All Devices" else device_filter
        result = rag.query(question, device_filter=device, top_k=top_k)

        # Display answer
        st.success("✅ Answer Generated")

        # Answer box
        st.markdown("### 📝 Answer")
        st.markdown(result['answer'])

        # Confidence indicator
        confidence = result['confidence']
        confidence_color = {
            'High': '🟢',
            'Medium': '🟡',
            'Low': '🔴'
        }
        st.markdown(f"**Confidence:** {confidence_color.get(confidence, '⚪')} {confidence}")

        # Sources
        st.markdown("### 📚 Sources")
        for i, source in enumerate(result['sources'], 1):
            with st.expander(f"Source {i}: {source['doc_name']} (Similarity: {source['similarity']:.2%})"):
                st.write(f"**Page:** {source['page']}")
                st.write(f"**Section:** {source['section']}")

        # Feedback (optional)
        st.markdown("---")
        st.markdown("### 📊 Was this answer helpful?")
        col_fb1, col_fb2, col_fb3 = st.columns([1, 1, 4])
        with col_fb1:
            st.button("👍 Yes")
        with col_fb2:
            st.button("👎 No")

# Session history (optional)
if 'history' not in st.session_state:
    st.session_state.history = []

if ask_button and question:
    st.session_state.history.append({
        'question': question,
        'answer': result['answer'][:100] + "..."  # Truncated
    })

# Display history in sidebar
with st.sidebar:
    if st.session_state.history:
        st.markdown("---")
        st.header("Recent Questions")
        for i, item in enumerate(reversed(st.session_state.history[-5:]), 1):
            with st.expander(f"{i}. {item['question'][:30]}..."):
                st.write(item['answer'])
```

**To run:**
```bash
streamlit run app.py
```

### 4. CLI Tool (Optional)

```python
#!/usr/bin/env python3
# rag_cli.py

import click
from rag_service import RAGService

@click.command()
@click.argument('question')
@click.option('--device', '-d', default=None, help='Filter by device type')
@click.option('--top-k', '-k', default=5, help='Number of sources to retrieve')
def query(question, device, top_k):
    """Query the maintenance documentation."""

    click.echo(f"\n🔍 Searching for: {question}\n")

    rag = RAGService()
    result = rag.query(question, device_filter=device, top_k=top_k)

    # Print answer
    click.echo("📝 Answer:")
    click.echo("─" * 80)
    click.echo(result['answer'])
    click.echo("─" * 80)

    # Print sources
    click.echo("\n📚 Sources:")
    for i, source in enumerate(result['sources'], 1):
        click.echo(f"  {i}. {source['doc_name']}, Page {source['page']} "
                   f"(Similarity: {source['similarity']:.1%})")

    # Print confidence
    click.echo(f"\n💡 Confidence: {result['confidence']}\n")

if __name__ == '__main__':
    query()
```

**Usage:**
```bash
chmod +x rag_cli.py
./rag_cli.py "How to reset XYZ-100?"
./rag_cli.py "Calibration procedure" --device XYZ-100 --top-k 8
```

---

## Technology Stack

### Complete Stack Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Technology Stack                         │
└─────────────────────────────────────────────────────────────┘

Layer 1: Data Storage
├─ JadeVectorDB (vector database)
│  ├─ HNSW indexing
│  ├─ SQLite persistence
│  └─ Memory-mapped vector storage
└─ File system (original PDFs/DOCX)

Layer 2: Embedding & Retrieval
├─ Sentence Transformers (embedding generation)
│  └─ Model: intfloat/e5-small-v2
└─ JadeVectorDB Python Client (vector search)

Layer 3: Language Model
└─ Ollama (local LLM serving)
   └─ Model: llama3.2:3b

Layer 4: Document Processing
├─ PyMuPDF (fitz) - PDF parsing
├─ python-docx - DOCX parsing
├─ LangChain - Text chunking
└─ NumPy - Vector operations

Layer 5: Orchestration
└─ Custom Python RAG service
   ├─ Query processing
   ├─ Context assembly
   └─ Response formatting

Layer 6: User Interface
├─ Option 1: Streamlit (web UI)
├─ Option 2: Click (CLI)
└─ Option 3: FastAPI (REST API)
```

### Detailed Dependencies

**Core Dependencies:**

```txt
# requirements.txt

# Vector database client
# (Assuming JadeVectorDB has a Python client, otherwise use requests)
requests==2.31.0

# Embedding models
sentence-transformers==2.3.1
torch==2.1.2  # CPU version for laptop deployment

# LLM integration
ollama==0.1.6

# Document processing
PyMuPDF==1.23.8  # PDF parsing
python-docx==1.1.0  # DOCX parsing
langchain==0.1.0  # Text splitting
langchain-community==0.0.10

# Web UI (choose one)
streamlit==1.29.0  # Option 1: Web UI
fastapi==0.109.0  # Option 3: REST API
uvicorn==0.27.0  # For FastAPI

# CLI (optional)
click==8.1.7

# Utilities
numpy==1.26.3
pandas==2.1.4
tqdm==4.66.1  # Progress bars for batch processing
python-dotenv==1.0.0  # Configuration management
```

**System Requirements:**

```yaml
Minimum (Laptop):
  CPU: 4 cores, 2.5 GHz
  RAM: 8 GB
  Storage: 10 GB free space
  OS: Linux, macOS, Windows 10+

Recommended (Workstation):
  CPU: 8 cores, 3.0 GHz
  RAM: 16 GB
  Storage: 50 GB SSD
  OS: Linux Ubuntu 20.04+

Software:
  Python: 3.9+
  Ollama: Latest version
  JadeVectorDB: As per project build instructions
```

### Installation Script

```bash
#!/bin/bash
# setup.sh - Setup RAG system

echo "🚀 Setting up Maintenance Documentation Q&A System"

# 1. Install Python dependencies
echo "📦 Installing Python packages..."
pip install -r requirements.txt

# 2. Install Ollama (if not already installed)
if ! command -v ollama &> /dev/null; then
    echo "📥 Installing Ollama..."
    curl https://ollama.ai/install.sh | sh
else
    echo "✓ Ollama already installed"
fi

# 3. Pull LLM model
echo "🤖 Downloading Llama 3.2 (3B) model..."
ollama pull llama3.2:3b

# 4. Download embedding model
echo "📊 Downloading E5-Small embedding model..."
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/e5-small-v2')"

# 5. Start JadeVectorDB (assumes it's already built)
echo "🗄️  Starting JadeVectorDB..."
cd backend/build
./jadevectordb &
JADEVECTORDB_PID=$!
echo "JadeVectorDB started with PID: $JADEVECTORDB_PID"

# 6. Create database
echo "📁 Creating maintenance_docs database..."
sleep 2  # Wait for JadeVectorDB to start
python -c "
from jadevectordb import JadeVectorDBClient
client = JadeVectorDBClient('http://localhost:8080')
client.create_database(
    name='maintenance_docs',
    description='Maintenance documentation for field engineers',
    vectorDimension=384,
    indexType='HNSW'
)
print('✓ Database created successfully')
"

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Place your PDF/DOCX files in ./docs/"
echo "  2. Run: python ingest_documents.py ./docs"
echo "  3. Start UI: streamlit run app.py"
echo ""
```

---

## Implementation Roadmap

### Phase 1: Foundation (Week 1)

**Goals:**
- Set up development environment
- Get JadeVectorDB running
- Validate basic vector storage and retrieval

**Tasks:**

1. **Environment Setup**
   - [ ] Install Python 3.9+
   - [ ] Install Ollama
   - [ ] Build and start JadeVectorDB
   - [ ] Create Python virtual environment

2. **Dependency Installation**
   - [ ] Install all Python packages
   - [ ] Download E5-Small embedding model
   - [ ] Pull Llama 3.2 model via Ollama

3. **JadeVectorDB Configuration**
   - [ ] Create `maintenance_docs` database
   - [ ] Configure HNSW index (M=16, ef_construction=200)
   - [ ] Test basic vector insert/search operations

4. **Validation**
   - [ ] Store 10 sample vectors
   - [ ] Run similarity search
   - [ ] Verify metadata retrieval

**Deliverables:**
- Working JadeVectorDB instance
- Configured environment
- Basic search validation

---

### Phase 2: Document Processing (Week 2)

**Goals:**
- Build document ingestion pipeline
- Process sample documents
- Validate chunking strategy

**Tasks:**

1. **PDF Processing**
   - [ ] Implement `process_pdf()` function
   - [ ] Test with sample maintenance PDFs
   - [ ] Handle multi-column layouts
   - [ ] Extract page numbers correctly

2. **DOCX Processing**
   - [ ] Implement `process_docx()` function
   - [ ] Test with sample Word documents
   - [ ] Preserve section headers
   - [ ] Handle tables and lists

3. **Chunking Implementation**
   - [ ] Implement semantic chunking with LangChain
   - [ ] Tune chunk size (test 256, 512, 1024 tokens)
   - [ ] Validate chunk overlap
   - [ ] Ensure procedure steps stay together

4. **Metadata Extraction**
   - [ ] Extract document name, type
   - [ ] Identify device type from filename/content
   - [ ] Track page numbers per chunk
   - [ ] Detect section headers

5. **Batch Ingestion**
   - [ ] Create `ingest_documents.py` script
   - [ ] Process 10-20 sample documents
   - [ ] Monitor progress with tqdm
   - [ ] Log errors and successes

**Deliverables:**
- Document processing pipeline
- 100+ chunks stored in JadeVectorDB
- Metadata validation report

---

### Phase 3: RAG Core (Week 3)

**Goals:**
- Implement query processing
- Integrate Ollama for answer generation
- Test end-to-end RAG flow

**Tasks:**

1. **Embedding Service**
   - [ ] Create `embed_query()` function
   - [ ] Add query prefix for E5 model
   - [ ] Test embedding latency (<100ms target)

2. **Retrieval Service**
   - [ ] Implement vector similarity search
   - [ ] Add metadata filtering (device type)
   - [ ] Tune top-k parameter (test 5, 10, 15)
   - [ ] Validate retrieval relevance

3. **LLM Integration**
   - [ ] Set up Ollama client
   - [ ] Design system prompt
   - [ ] Implement context assembly
   - [ ] Test answer generation

4. **RAG Service Class**
   - [ ] Create `RAGService` class
   - [ ] Implement full `query()` method
   - [ ] Add source extraction
   - [ ] Calculate confidence scores

5. **Testing**
   - [ ] Create test question set (20 questions)
   - [ ] Validate answer accuracy
   - [ ] Check source attribution
   - [ ] Measure end-to-end latency (<3s target)

**Deliverables:**
- Working RAG pipeline
- Test results for 20 questions
- Performance benchmarks

---

### Phase 4: User Interface (Week 4)

**Goals:**
- Build Streamlit web UI
- Add filtering and configuration
- Implement feedback collection

**Tasks:**

1. **Basic UI**
   - [ ] Create Streamlit app structure
   - [ ] Add question input box
   - [ ] Display answers with formatting
   - [ ] Show source citations

2. **Filters & Configuration**
   - [ ] Add device type filter
   - [ ] Top-k slider
   - [ ] Model selection (if multiple LLMs)

3. **User Experience**
   - [ ] Loading spinners
   - [ ] Error handling and messages
   - [ ] Clear button
   - [ ] Question history

4. **Feedback System**
   - [ ] Thumbs up/down buttons
   - [ ] Log feedback to file
   - [ ] Optional: comment box

5. **Deployment**
   - [ ] Package as Docker container (optional)
   - [ ] Create startup script
   - [ ] Write user documentation

**Deliverables:**
- Production-ready web UI
- User documentation
- Deployment guide

---

### Phase 5: Optimization & Polish (Week 5)

**Goals:**
- Improve retrieval accuracy
- Optimize performance
- Add monitoring

**Tasks:**

1. **Retrieval Tuning**
   - [ ] Experiment with chunk sizes
   - [ ] Test different top-k values
   - [ ] Add re-ranking (optional)
   - [ ] Implement hybrid search (optional)

2. **Performance Optimization**
   - [ ] Cache embeddings for common queries
   - [ ] Optimize batch processing
   - [ ] Reduce LLM latency (quantization?)
   - [ ] Monitor memory usage

3. **Quality Improvements**
   - [ ] Refine system prompt
   - [ ] Add query preprocessing
   - [ ] Handle abbreviations
   - [ ] Improve source formatting

4. **Monitoring**
   - [ ] Log all queries and responses
   - [ ] Track answer quality metrics
   - [ ] Monitor system performance
   - [ ] Set up alerts for errors

5. **Documentation**
   - [ ] Write admin guide
   - [ ] Create troubleshooting FAQ
   - [ ] Document common queries
   - [ ] Identify documentation gaps

**Deliverables:**
- Optimized system
- Monitoring dashboard
- Complete documentation

---

### Phase 6: Production Deployment (Week 6)

**Goals:**
- Full document ingestion
- User training
- Production launch

**Tasks:**

1. **Full Ingestion**
   - [ ] Ingest all 100-1000 documents
   - [ ] Validate completeness
   - [ ] Create document index
   - [ ] Backup vector database

2. **User Training**
   - [ ] Create training materials
   - [ ] Run pilot with 5-10 engineers
   - [ ] Collect feedback
   - [ ] Refine based on feedback

3. **Production Launch**
   - [ ] Deploy to production server/workstation
   - [ ] Set up automatic backups
   - [ ] Configure monitoring
   - [ ] Create support process

4. **Iteration Plan**
   - [ ] Weekly review of query logs
   - [ ] Monthly documentation updates
   - [ ] Quarterly system optimization
   - [ ] Continuous feedback collection

**Deliverables:**
- Production system
- Trained users
- Support documentation
- Maintenance plan

---

## Best Practices

### 1. Document Chunking

Based on [recent RAG research](https://stackoverflow.blog/2024/12/27/breaking-up-is-hard-to-do-chunking-in-rag-applications/), follow these best practices:

**✅ DO:**
- Use semantic chunking for structured documents (maintenance manuals)
- Preserve document structure (keep procedures together)
- Use 50-100 token overlap to prevent context loss
- Test multiple chunk sizes (256, 512, 1024) and evaluate
- Keep chunk metadata rich (document, section, page, device type)

**❌ DON'T:**
- Use fixed-size chunking without overlap
- Split procedure steps across chunks
- Make chunks too small (<100 tokens) or too large (>1000 tokens)
- Ignore document structure (headings, lists, tables)

### 2. Embedding Selection

**✅ DO:**
- Use E5-Small for balance of accuracy and speed
- Add "query:" prefix for E5 models
- Normalize embeddings before storage (if not auto-normalized)
- Use same embedding model for indexing and querying
- Consider model size vs. laptop resources

**❌ DON'T:**
- Mix embedding models (e.g., index with E5, query with nomic)
- Use unnecessarily large models (e.g., 768-dim when 384-dim suffices)
- Forget to update embeddings when documents change

### 3. Retrieval Strategies

**✅ DO:**
- Start with top-k=5, adjust based on answer quality
- Set similarity threshold (e.g., 0.65) to filter low-quality results
- Use metadata filters when user specifies device type
- Re-rank results by recency if documents have versions
- Provide source citations with page numbers

**❌ DON'T:**
- Retrieve too many chunks (>10) - confuses LLM
- Ignore metadata - it's crucial for source attribution
- Trust low similarity scores (<0.6) - usually irrelevant

### 4. Prompt Engineering

**✅ DO:**
- Instruct LLM to cite sources
- Tell LLM to say "I don't know" if context lacks info
- Use low temperature (0.1-0.2) for factual accuracy
- Format procedures as numbered lists
- Include safety warnings from documentation

**❌ DON'T:**
- Let LLM hallucinate when context is insufficient
- Use high temperature (>0.5) - causes unreliable answers
- Omit system prompt - it guides LLM behavior
- Forget to include context in prompt

### 5. System Maintenance

**✅ DO:**
- Log all queries for quality monitoring
- Review low-confidence answers weekly
- Update documents incrementally (don't re-index everything)
- Collect user feedback systematically
- Back up JadeVectorDB regularly

**❌ DON'T:**
- Ignore user feedback
- Let document index get stale (>6 months without update)
- Skip testing after adding new documents
- Neglect performance monitoring

### 6. User Experience

**✅ DO:**
- Show source documents and page numbers
- Indicate confidence level
- Provide loading feedback (spinners)
- Allow filtering by device type
- Show recent question history

**❌ DON'T:**
- Hide sources - transparency builds trust
- Give vague answers without citations
- Make users wait >5 seconds without feedback
- Overwhelm with too many options

---

## Performance Considerations

### Expected Performance Metrics

Based on JadeVectorDB specs and component benchmarks:

| Metric | Target | Expected |
|--------|--------|----------|
| **Document Ingestion** | | |
| PDF parsing speed | >10 pages/sec | 15-20 pages/sec (PyMuPDF) |
| Chunking speed | >1000 chunks/sec | 500-800 chunks/sec (LangChain) |
| Embedding generation | >100 chunks/sec | 50-100 chunks/sec (E5-small, CPU) |
| Batch insert to DB | >1000 vectors/sec | 500-1000 vectors/sec (JadeVectorDB) |
| **Full ingestion (10,000 pages)** | <30 min | 15-25 min |
| | | |
| **Query Processing** | | |
| Query embedding | <100ms | 30-50ms (E5-small) |
| Vector search (50k chunks) | <50ms | 10-20ms (HNSW) |
| LLM answer generation | <2s | 1.5-3s (Llama 3.2, CPU) |
| **End-to-end latency** | <3s | 2-4s |
| | | |
| **Storage** | | |
| Vectors (50k × 384 dims) | <100 MB | ~77 MB |
| Metadata (50k chunks) | <200 MB | ~100-150 MB |
| **Total database size** | <500 MB | ~200-300 MB |
| | | |
| **Memory Usage** | | |
| JadeVectorDB | <500 MB | ~300-400 MB |
| Embedding model | <500 MB | ~400 MB (E5-small) |
| LLM model | <3 GB | ~2 GB (Llama 3.2:3b) |
| Python application | <500 MB | ~200-300 MB |
| **Total RAM** | <6 GB | 3-4 GB |

### Optimization Strategies

#### For Ingestion Speed

1. **Parallel Processing**
   ```python
   from multiprocessing import Pool

   def process_file(file_path):
       # Process single file
       pass

   with Pool(processes=4) as pool:
       pool.map(process_file, file_list)
   ```

2. **Batch Embeddings**
   ```python
   # Embed in batches of 32
   for i in range(0, len(texts), 32):
       batch = texts[i:i+32]
       embeddings = model.encode(batch)
   ```

3. **Batch Database Inserts**
   ```python
   # Insert 100 vectors at a time
   batch_size = 100
   for i in range(0, len(vectors), batch_size):
       batch = vectors[i:i+batch_size]
       db_client.batch_store_vectors(database_id, batch)
   ```

#### For Query Speed

1. **Embedding Caching**
   ```python
   from functools import lru_cache

   @lru_cache(maxsize=1000)
   def embed_cached(query: str):
       return embedder.encode(query)
   ```

2. **Connection Pooling**
   ```python
   # Reuse JadeVectorDB client connection
   @st.cache_resource
   def get_db_client():
       return JadeVectorDBClient("http://localhost:8080")
   ```

3. **Reduce Top-K**
   - Start with top-k=5 instead of 10
   - Only retrieve more if confidence is low

4. **LLM Optimization**
   ```python
   # Use smaller, faster models for simple queries
   # Use quantized models (Q4_K_M) for speed
   ollama pull llama3.2:3b-q4_K_M
   ```

#### For Memory Management

1. **Lazy Loading**
   - Don't load all documents into memory
   - Stream from disk as needed

2. **Model Quantization**
   - Use 4-bit quantized LLMs for lower memory usage
   - E5-small already efficient (118M params)

3. **Garbage Collection**
   ```python
   import gc

   # After large batch operations
   gc.collect()
   ```

### Scaling Considerations

**If System Grows Beyond Medium Scale:**

| Scaling Path | When | Solution |
|-------------|------|----------|
| **More documents** | >10,000 docs (>500k chunks) | Enable JadeVectorDB distributed mode with sharding |
| **More users** | >10 concurrent users | Deploy multiple LLM replicas, load balance queries |
| **Faster retrieval** | Need <5ms search | Use GPU for embedding, upgrade to distributed DB |
| **Lower memory** | <4GB RAM available | Use quantized LLM (3-bit), smaller embedding model |

**JadeVectorDB Distributed Mode:**

Per the documentation, JadeVectorDB has full distributed capabilities (currently disabled by default):

- **Sharding**: Distribute 500k+ vectors across multiple nodes
- **Replication**: High availability with data redundancy
- **Load balancing**: Distribute query load across cluster

To enable (when needed):
```bash
export JADEVECTORDB_ENABLE_SHARDING=true
export JADEVECTORDB_ENABLE_REPLICATION=true
export JADEVECTORDB_NUM_SHARDS=4
```

---

## Cost Analysis

### One-Time Setup Costs

| Item | Cost | Notes |
|------|------|-------|
| **Development Time** | | |
| Developer (120 hours) | $12,000 | @$100/hr, 6 weeks |
| Testing & QA | $2,000 | User testing, refinement |
| Documentation | $1,000 | User guides, admin docs |
| **Hardware** | | |
| Laptop/Workstation | $0 | Use existing hardware |
| Additional RAM (if needed) | $100 | 16GB upgrade |
| Storage (if needed) | $50 | 1TB SSD |
| **Software & Models** | | |
| JadeVectorDB | $0 | Open source |
| Ollama | $0 | Open source |
| Python libraries | $0 | Open source |
| LLM models | $0 | Local, open weights |
| Embedding models | $0 | Open source |
| **TOTAL SETUP** | **~$15,150** | |

### Ongoing Operational Costs

| Item | Monthly Cost | Annual Cost | Notes |
|------|--------------|-------------|-------|
| **Infrastructure** | | | |
| Hosting (local) | $0 | $0 | Runs on existing hardware |
| Cloud hosting (optional) | $0 | $0 | Not needed for local deployment |
| Electricity | ~$5 | ~$60 | Negligible for laptop |
| **API Costs** | | | |
| Embedding API | $0 | $0 | Local E5-small model |
| LLM API | $0 | $0 | Local Llama via Ollama |
| **Maintenance** | | | |
| Document updates | $200 | $2,400 | 2 hrs/month @$100/hr |
| System monitoring | $100 | $1,200 | 1 hr/month |
| User support | $200 | $2,400 | Ad-hoc assistance |
| **TOTAL ANNUAL** | **~$505/mo** | **~$6,060** | |

### Cost Comparison: Local RAG vs. Cloud API

**Cloud-based Alternative (e.g., OpenAI API):**

Assumptions:
- 100 queries/day
- Average 5 chunks @ 400 tokens each = 2000 tokens context
- 300 tokens output per query
- Embeddings: 50,000 chunks initially, 500 new/month

| Service | Usage | Cost/Month | Annual Cost |
|---------|-------|------------|-------------|
| Embeddings (initial) | 50,000 chunks × 384 tokens | $1.92 | (one-time) |
| Embeddings (ongoing) | 500 chunks/month | $0.19 | $2.28 |
| LLM queries | 3000 queries × (2000 input + 300 output) | $207 | $2,484 |
| **TOTAL** | | **~$209/month** | **~$2,506/year** |

**Local RAG System:**
- **Year 1**: $15,150 (setup) + $6,060 (ops) = **$21,210**
- **Year 2+**: $6,060/year (ops only)

**Break-even Analysis:**
- Cloud: $2,506/year ongoing
- Local: $21,210 year 1, $6,060 year 2+
- **Break-even**: ~8 years if only considering direct costs

**BUT: Key Advantages of Local System:**
- **Data privacy**: Sensitive maintenance docs never leave premises
- **Offline operation**: Works in remote field locations (critical!)
- **No usage limits**: Unlimited queries, no rate limiting
- **Predictable costs**: No surprise bills from usage spikes
- **Customization**: Full control over models, prompts, data

**For your use case (field engineers in remote locations), local deployment is strongly recommended regardless of cost.**

---

## References

### Research & Best Practices

1. [Chunking Strategies for RAG - Comprehensive Guide (2025)](https://medium.com/@adnanmasood/chunking-strategies-for-retrieval-augmented-generation-rag-a-comprehensive-guide-5522c4ea2a90)
2. [Breaking up is hard to do: Chunking in RAG applications - Stack Overflow](https://stackoverflow.blog/2024/12/27/breaking-up-is-hard-to-do-chunking-in-rag-applications/)
3. [Complete Guide to Building a Robust RAG Pipeline 2025 - DhiWise](https://www.dhiwise.com/post/build-rag-pipeline-guide)
4. [Enhancing Retrieval-Augmented Generation: Best Practices (arXiv 2025)](https://arxiv.org/abs/2501.07391)

### Embedding Models

5. [13 Best Embedding Models in 2025 - Elephas](https://elephas.app/blog/best-embedding-models)
6. [Best Open-Source Embedding Models Benchmarked - Supermemory](https://supermemory.ai/blog/best-open-source-embedding-models-benchmarked-and-ranked/)
7. [Sentence Transformers Documentation](https://sbert.net/)

### Ollama & RAG Integration

8. [RAG with Ollama and LangChain: Complete Document Q&A System 2025 - Markaicode](https://markaicode.com/rag-ollama-langchain-document-qa-system-2025/)
9. [Building RAG Applications with Ollama and Python - Collabnix](https://collabnix.com/building-rag-applications-with-ollama-and-python-complete-2025-tutorial/)
10. [Build a Local AI RAG App with Ollama and Python - Medium](https://auscunningham.medium.com/build-a-local-ai-rag-app-with-ollama-and-python-96f9df9c2a3e)

### Document Processing

11. [Best Python PDF to Text Parser Libraries: A 2026 Evaluation - Unstract](https://unstract.com/blog/evaluating-python-pdf-to-text-libraries/)
12. [I Tested 7 Python PDF Extractors (2025 Edition) - DEV Community](https://dev.to/onlyoneaman/i-tested-7-python-pdf-extractors-so-you-dont-have-to-2025-edition-akm)

### JadeVectorDB Documentation

13. JadeVectorDB README.md - Project overview and capabilities
14. JadeVectorDB BOOTSTRAP.md - Developer guide and architecture
15. JadeVectorDB docs/COMPLETE_BUILD_SYSTEM_SETUP.md - Build and deployment

---

## Appendix A: Sample Code Repository Structure

```
rag-maintenance-qa/
├── README.md
├── requirements.txt
├── setup.sh
├── .env.example
│
├── config/
│   └── rag_config.yaml
│
├── src/
│   ├── __init__.py
│   ├── document_processor.py      # PDF/DOCX parsing and chunking
│   ├── embedding_service.py        # E5-small embedding generation
│   ├── vector_store.py             # JadeVectorDB client wrapper
│   ├── rag_service.py              # Main RAG orchestration
│   └── utils.py                    # Helper functions
│
├── scripts/
│   ├── ingest_documents.py         # Batch document ingestion
│   ├── create_database.py          # Initialize JadeVectorDB
│   └── test_pipeline.py            # End-to-end testing
│
├── ui/
│   ├── streamlit_app.py            # Web interface
│   ├── cli.py                      # Command-line interface
│   └── api.py                      # FastAPI REST API (optional)
│
├── tests/
│   ├── test_document_processor.py
│   ├── test_embedding.py
│   ├── test_rag_service.py
│   └── test_questions.json         # Test Q&A pairs
│
├── data/
│   ├── raw_docs/                   # Original PDFs/DOCX
│   ├── processed/                  # Parsed and chunked data
│   └── logs/                       # Query logs, feedback
│
└── docs/
    ├── SETUP.md                    # Setup instructions
    ├── USER_GUIDE.md               # End-user documentation
    └── ADMIN_GUIDE.md              # System administration
```

---

## Appendix B: Quick Start Commands

```bash
# 1. Clone repository (assuming you've created this structure)
git clone <repo-url>
cd rag-maintenance-qa

# 2. Run setup script
chmod +x setup.sh
./setup.sh

# 3. Place your documents
mkdir -p data/raw_docs
cp /path/to/your/pdfs/*.pdf data/raw_docs/
cp /path/to/your/docx/*.docx data/raw_docs/

# 4. Ingest documents
python scripts/ingest_documents.py data/raw_docs/

# 5. Test with a question
python scripts/test_pipeline.py

# 6. Launch web UI
streamlit run ui/streamlit_app.py

# Or use CLI
python ui/cli.py "How to reset XYZ-100?"
```

---

## Appendix C: Sample Test Questions

Create `tests/test_questions.json`:

```json
[
  {
    "id": 1,
    "question": "How do I reset the XYZ-100 after a power failure?",
    "expected_docs": ["XYZ-100_Manual.pdf"],
    "expected_sections": ["Troubleshooting", "Reset Procedures"]
  },
  {
    "id": 2,
    "question": "What is the calibration procedure for the ABC-200 sensor?",
    "expected_docs": ["ABC-200_Calibration_Guide.docx"],
    "expected_sections": ["Calibration", "Maintenance"]
  },
  {
    "id": 3,
    "question": "What safety precautions should I take when servicing the DEF-300?",
    "expected_docs": ["DEF-300_Safety_Manual.pdf"],
    "expected_sections": ["Safety", "Warnings"]
  },
  {
    "id": 4,
    "question": "How often should I replace the filter on the XYZ-100?",
    "expected_docs": ["XYZ-100_Manual.pdf", "Maintenance_Schedule.docx"],
    "expected_sections": ["Maintenance Schedule", "Filter Replacement"]
  },
  {
    "id": 5,
    "question": "What error code E42 means on the ABC-200?",
    "expected_docs": ["ABC-200_Error_Codes.pdf"],
    "expected_sections": ["Error Codes", "Diagnostics"]
  }
]
```

---

## Conclusion

This RAG system design leverages JadeVectorDB's robust vector storage and retrieval capabilities alongside local LLMs (Ollama) to create a **privacy-preserving, cost-effective, and offline-capable** maintenance documentation Q&A system for field engineers and mechanics.

### Key Takeaways

**✅ JadeVectorDB is Well-Suited for This Use Case:**
- Excellent vector storage with HNSW indexing
- Rich metadata support for source attribution
- Proven performance (sub-50ms for 1M vectors)
- Persistent storage with ACID guarantees
- Battle-tested architecture

**⚠️ Custom Development Required:**
- Document processing pipeline (PDF/DOCX parsing, chunking)
- Embedding generation (via Sentence Transformers or Ollama)
- RAG orchestration layer
- User interface (Streamlit/CLI/API)
- LLM integration (Ollama)

**📊 Expected Outcomes:**
- **Retrieval Latency**: <20ms for 50,000 chunks
- **End-to-End Response**: 2-4 seconds
- **Storage**: ~300MB for 50,000 chunks
- **Accuracy**: >85% with proper chunking and top-k tuning
- **Cost**: Zero ongoing API costs

**🚀 Next Steps:**
1. Review and approve this architecture
2. Begin Phase 1: Foundation setup
3. Process sample documents to validate pipeline
4. Iterate based on field engineer feedback

---

**Document Version**: 1.0
**Date**: January 2, 2026
**Author**: RAG System Design Team
**Status**: Ready for Implementation
