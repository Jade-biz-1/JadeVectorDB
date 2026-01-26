# RAG System: Chunking and Search Explained

## Table of Contents

1. [Understanding Chunking: One Vector = One Text Chunk](#understanding-chunking-one-vector--one-text-chunk)
2. [How Search Works: From Question to Answer](#how-search-works-from-question-to-answer)
3. [What the User Sees: Complete Example](#what-the-user-sees-complete-example)
4. [Technical Deep Dive](#technical-deep-dive)

---

## Understanding Chunking: One Vector = One Text Chunk

### Key Concept

**One Vector = One Text Chunk (NOT Entire Pages)**

Each vector represents **ONE chunk of text**, typically:
- **~400-500 words** (512 tokens)
- **NOT** an entire page or multiple pages
- **NOT** an entire document

### Why Multiple Page Numbers?

When you see `"page_numbers": [23, 24]`, it means:
- **This specific chunk** happens to span across pages 23 and 24
- The chunk started on page 23 and continued onto page 24
- We preserve semantic meaning, so we don't split chunks at page boundaries

### Visual Example: How One Document Becomes Many Chunks

```
┌─────────────────────────────────────────────────────────┐
│              XYZ-100 Service Manual                     │
│                  (50 pages total)                       │
└─────────────────────────────────────────────────────────┘

Page 23:
┌──────────────────────────────────────────────┐
│ Chapter 3: Troubleshooting                   │
│                                              │
│ 3.1 Common Issues                            │ ◄── Chunk 43
│ [text about common issues...]                │     Vector ID: chunk_43
│                                              │     Pages: [23]
│ 3.2 Reset Procedures                         │
│ To reset the XYZ-100 device, follow          │ ◄── Chunk 44
│ these steps:                                 │     Vector ID: chunk_44
│ 1) Turn off power                            │     Pages: [23, 24]
│ 2) Disconnect all cables                     │     (continues on next page)
└──────────────────────────────────────────────┘

Page 24:
┌──────────────────────────────────────────────┐
│ 3) Wait 30 seconds                           │ ◄── Still Chunk 44
│ 4) Reconnect cables                          │     (from previous page)
│ 5) Turn on power                             │
│                                              │
│ 3.3 Calibration Procedures                   │ ◄── Chunk 45
│ [text about calibration...]                  │     Vector ID: chunk_45
│                                              │     Pages: [24]
│ WARNING: Do not attempt calibration          │ ◄── Chunk 46
│ without proper training. Incorrect           │     Vector ID: chunk_46
│ calibration can damage the device...         │     Pages: [24, 25]
└──────────────────────────────────────────────┘

Page 25:
┌──────────────────────────────────────────────┐
│ and void your warranty. Contact support      │ ◄── Still Chunk 46
│ if you have questions.                       │     (from previous page)
│                                              │
│ 3.4 Advanced Diagnostics                     │ ◄── Chunk 47
│ [text about diagnostics...]                  │     Vector ID: chunk_47
│                                              │     Pages: [25]
└──────────────────────────────────────────────┘

Result: Pages 23-25 (3 pages) → 5 chunks → 5 vectors stored in database
```

### Actual Data Stored in JadeVectorDB

From the example above, here's what gets stored:

```json
// Chunk 43: All on page 23
{
  "vector_id": "XYZ-100_Manual_chunk_43",
  "vector": [0.234, 0.567, 0.891, ...],  // 384 numbers
  "metadata": {
    "doc_name": "XYZ-100 Service Manual",
    "doc_type": "pdf",
    "page_numbers": [23],
    "section": "3.1 Common Issues",
    "device_type": "XYZ-100",
    "text": "Common issues include power failures, sensor errors, and connectivity problems. Most can be resolved by following the reset procedure in Section 3.2.",
    "chunk_id": 43,
    "chunk_length": 187
  }
}

// Chunk 44: Spans pages 23-24
{
  "vector_id": "XYZ-100_Manual_chunk_44",
  "vector": [0.123, 0.456, 0.789, ...],  // 384 numbers
  "metadata": {
    "doc_name": "XYZ-100 Service Manual",
    "doc_type": "pdf",
    "page_numbers": [23, 24],  // ← Spans two pages!
    "section": "3.2 Reset Procedures",
    "device_type": "XYZ-100",
    "text": "To reset the XYZ-100 device, follow these steps: 1) Turn off power using the main switch on the back panel. 2) Disconnect all cables including power, data, and sensor connections. 3) Wait 30 seconds to allow capacitors to discharge. 4) Reconnect cables in reverse order: sensors first, then data, then power. 5) Turn on power and wait for the initialization sequence to complete (green LED).",
    "chunk_id": 44,
    "chunk_length": 412
  }
}

// Chunk 45: All on page 24
{
  "vector_id": "XYZ-100_Manual_chunk_45",
  "vector": [0.789, 0.012, 0.345, ...],  // 384 numbers
  "metadata": {
    "doc_name": "XYZ-100 Service Manual",
    "doc_type": "pdf",
    "page_numbers": [24],
    "section": "3.3 Calibration Procedures",
    "device_type": "XYZ-100",
    "text": "Calibration procedures require specialized equipment including a certified calibration probe (Part #CAL-100) and calibration software version 2.3 or later. Only trained technicians should perform calibration.",
    "chunk_id": 45,
    "chunk_length": 234
  }
}

// Chunk 46: Spans pages 24-25
{
  "vector_id": "XYZ-100_Manual_chunk_46",
  "vector": [0.345, 0.678, 0.901, ...],  // 384 numbers
  "metadata": {
    "doc_name": "XYZ-100 Service Manual",
    "doc_type": "pdf",
    "page_numbers": [24, 25],  // ← Spans two pages!
    "section": "3.3 Calibration Procedures",
    "device_type": "XYZ-100",
    "text": "WARNING: Do not attempt calibration without proper training. Incorrect calibration can damage the device, affect measurement accuracy, and void your warranty. Contact technical support at 1-800-SUPPORT if you have questions about calibration procedures.",
    "chunk_id": 46,
    "chunk_length": 287
  }
}

// Chunk 47: All on page 25
{
  "vector_id": "XYZ-100_Manual_chunk_47",
  "vector": [0.567, 0.234, 0.890, ...],  // 384 numbers
  "metadata": {
    "doc_name": "XYZ-100 Service Manual",
    "doc_type": "pdf",
    "page_numbers": [25],
    "section": "3.4 Advanced Diagnostics",
    "device_type": "XYZ-100",
    "text": "Advanced diagnostics mode can be accessed by holding the DIAG button during startup. This mode provides access to internal sensor readings, error logs, and system health metrics.",
    "chunk_id": 47,
    "chunk_length": 215
  }
}
```

### Key Points About Chunking

1. **Chunking ignores page boundaries** (by design!)
   - We chunk by semantic meaning, not by pages
   - If a procedure spans 2 pages, we keep it together in one chunk
   - This preserves context and improves retrieval quality

2. **One page = Multiple chunks**
   - A typical page (500 words) might produce **2-3 chunks**
   - So pages 23-25 (3 pages) produced **5 chunks/vectors** in our example

3. **Page numbers in metadata**
   - `"page_numbers": [23]` - chunk is entirely on page 23
   - `"page_numbers": [23, 24]` - chunk starts on 23, ends on 24
   - `"page_numbers": [23, 24, 25]` - chunk spans 3 pages (rare, but possible for long procedures)

4. **Why keep procedures together?**
   - Better retrieval: complete procedure in one chunk
   - Better answers: LLM sees all steps together
   - Better user experience: no partial instructions

---

## How Search Works: From Question to Answer

Now let's walk through what happens when a field engineer asks a question.

### Complete User Journey

```
Field Engineer Question
        ↓
    Embedding
        ↓
  Vector Search
        ↓
  Find Top Matches
        ↓
   Retrieve Context
        ↓
    Send to LLM
        ↓
  Generate Answer
        ↓
   Show to User
```

### Detailed Step-by-Step Process

Let's follow a real example: **"How do I reset the XYZ-100 after a power failure?"**

---

#### **Step 1: User Asks Question**

**User Interface (Streamlit web app):**
```
┌────────────────────────────────────────────────────┐
│ 🔧 Maintenance Documentation Q&A                   │
├────────────────────────────────────────────────────┤
│                                                    │
│ Question:                                          │
│ ┌────────────────────────────────────────────┐   │
│ │ How do I reset the XYZ-100 after a power   │   │
│ │ failure?                                   │   │
│ └────────────────────────────────────────────┘   │
│                                                    │
│        [🔍 Get Answer]  [Clear]                   │
│                                                    │
└────────────────────────────────────────────────────┘
```

**What happens internally:**
```python
user_question = "How do I reset the XYZ-100 after a power failure?"
```

---

#### **Step 2: Convert Question to Vector (Embedding)**

**Process:**
The question is converted into a 384-dimensional vector using the same embedding model (E5-Small) that was used to embed the documents.

```python
from sentence_transformers import SentenceTransformer

# Load embedding model (same one used for documents)
embedder = SentenceTransformer('intfloat/e5-small-v2')

# Add prefix for E5 models (improves retrieval)
query_with_prefix = "query: How do I reset the XYZ-100 after a power failure?"

# Convert to vector
query_embedding = embedder.encode(query_with_prefix)
# Result: [0.234, -0.123, 0.567, ..., 0.891]  # 384 numbers
```

**Why this works:**
- Questions about "reset" will have vectors similar to text containing "reset procedures"
- "XYZ-100" matches documents about that specific device
- "power failure" is semantically similar to "power issues", "power loss", etc.

---

#### **Step 3: Search JadeVectorDB for Similar Vectors**

**Process:**
The query vector is compared to all 50,000 document chunk vectors using cosine similarity.

```python
# Search JadeVectorDB
search_results = jadevectordb.search(
    database_id="maintenance_docs",
    query_vector=query_embedding.tolist(),  # Our 384-dim vector
    top_k=10,  # Get top 10 most similar chunks
    threshold=0.65  # Minimum similarity score (0-1 scale)
)
```

**What JadeVectorDB does:**
1. Uses HNSW (Hierarchical Navigable Small World) index
2. Compares query vector to all stored vectors
3. Finds top-10 most similar chunks in ~10-20 milliseconds
4. Returns chunks sorted by similarity score

**Results returned:**

```python
[
    {
        "vector_id": "XYZ-100_Manual_chunk_44",
        "similarity": 0.89,  # 89% similar!
        "metadata": {
            "doc_name": "XYZ-100 Service Manual",
            "page_numbers": [23, 24],
            "section": "3.2 Reset Procedures",
            "device_type": "XYZ-100",
            "text": "To reset the XYZ-100 device, follow these steps: 1) Turn off power using the main switch on the back panel. 2) Disconnect all cables including power, data, and sensor connections. 3) Wait 30 seconds to allow capacitors to discharge. 4) Reconnect cables in reverse order: sensors first, then data, then power. 5) Turn on power and wait for the initialization sequence to complete (green LED).",
            "chunk_id": 44
        }
    },
    {
        "vector_id": "XYZ-100_Manual_chunk_43",
        "similarity": 0.82,  # 82% similar
        "metadata": {
            "doc_name": "XYZ-100 Service Manual",
            "page_numbers": [23],
            "section": "3.1 Common Issues",
            "device_type": "XYZ-100",
            "text": "Common issues include power failures, sensor errors, and connectivity problems. Most can be resolved by following the reset procedure in Section 3.2.",
            "chunk_id": 43
        }
    },
    {
        "vector_id": "XYZ-100_Troubleshooting_chunk_12",
        "similarity": 0.78,  # 78% similar
        "metadata": {
            "doc_name": "XYZ-100 Troubleshooting Guide",
            "page_numbers": [5],
            "section": "Power Issues",
            "device_type": "XYZ-100",
            "text": "After a power failure, the XYZ-100 may need to be reset. See the service manual for complete reset procedures. Always check that backup power is disconnected before resetting.",
            "chunk_id": 12
        }
    },
    {
        "vector_id": "XYZ-100_Manual_chunk_46",
        "similarity": 0.71,
        "metadata": {
            "doc_name": "XYZ-100 Service Manual",
            "page_numbers": [24, 25],
            "section": "3.3 Calibration Procedures",
            "text": "WARNING: Do not attempt calibration without proper training...",
            "chunk_id": 46
        }
    },
    {
        "vector_id": "General_Maintenance_chunk_89",
        "similarity": 0.68,
        "metadata": {
            "doc_name": "General Maintenance Guidelines",
            "page_numbers": [15],
            "section": "Device Resets",
            "text": "General reset procedures for most devices involve power cycling...",
            "chunk_id": 89
        }
    }
    // ... 5 more results ...
]
```

**Notice:**
- **Chunk 44** (Reset Procedures) has the highest similarity (0.89)
- **Chunk 43** (mentions power failures and points to reset) is second (0.82)
- Results are automatically sorted by similarity

---

#### **Step 4: Filter and Select Top Results**

**Process:**
Take the top 5 most relevant chunks (you can configure this).

```python
# Select top 5 for context
top_chunks = search_results[:5]

# Optional: Filter by device type if user specified
# (in this case, "XYZ-100" was in the question, so all results match)
```

---

#### **Step 5: Build Context for LLM**

**Process:**
Assemble the retrieved chunks into a formatted context string.

```python
context_parts = []

for i, result in enumerate(top_chunks, 1):
    meta = result['metadata']
    context_parts.append(
        f"--- Source {i} ---\n"
        f"Document: {meta['doc_name']}\n"
        f"Page(s): {', '.join(map(str, meta['page_numbers']))}\n"
        f"Section: {meta['section']}\n\n"
        f"{meta['text']}\n"
    )

context = "\n\n".join(context_parts)
```

**Assembled Context:**

```
--- Source 1 ---
Document: XYZ-100 Service Manual
Page(s): 23, 24
Section: 3.2 Reset Procedures

To reset the XYZ-100 device, follow these steps: 1) Turn off power using the main switch on the back panel. 2) Disconnect all cables including power, data, and sensor connections. 3) Wait 30 seconds to allow capacitors to discharge. 4) Reconnect cables in reverse order: sensors first, then data, then power. 5) Turn on power and wait for the initialization sequence to complete (green LED).


--- Source 2 ---
Document: XYZ-100 Service Manual
Page(s): 23
Section: 3.1 Common Issues

Common issues include power failures, sensor errors, and connectivity problems. Most can be resolved by following the reset procedure in Section 3.2.


--- Source 3 ---
Document: XYZ-100 Troubleshooting Guide
Page(s): 5
Section: Power Issues

After a power failure, the XYZ-100 may need to be reset. See the service manual for complete reset procedures. Always check that backup power is disconnected before resetting.


--- Source 4 ---
Document: XYZ-100 Service Manual
Page(s): 24, 25
Section: 3.3 Calibration Procedures

WARNING: Do not attempt calibration without proper training. Incorrect calibration can damage the device, affect measurement accuracy, and void your warranty. Contact technical support at 1-800-SUPPORT if you have questions about calibration procedures.


--- Source 5 ---
Document: General Maintenance Guidelines
Page(s): 15
Section: Device Resets

General reset procedures for most devices involve power cycling and waiting for capacitor discharge. Always consult device-specific manuals for detailed procedures.
```

---

#### **Step 6: Create Prompt for LLM**

**Process:**
Combine system instructions, context, and user question into a single prompt.

```python
system_prompt = """You are a helpful maintenance assistant for field engineers and mechanics.

Your role:
- Answer questions based ONLY on the provided maintenance documentation
- Provide step-by-step instructions clearly and concisely
- Always cite source documents (document name and page number)
- If the information is not in the provided context, say "I don't have that information in the available documentation"
- Use technical terminology accurately
- Format procedures as numbered lists for clarity
- Highlight any warnings or cautions mentioned in the documentation
"""

user_prompt = f"""Context from maintenance documentation:

{context}

Question: {user_question}

Please provide a detailed answer based on the context above. Include source references (document name and page number) for your answer.

Answer:"""
```

---

#### **Step 7: Send to LLM (Ollama)**

**Process:**
Send the prompt to Llama 3.2 (3B) running locally via Ollama.

```python
import ollama

response = ollama.chat(
    model='llama3.2:3b',
    messages=[
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': user_prompt}
    ],
    options={
        'temperature': 0.1,  # Low = factual, high = creative
        'top_p': 0.9,
        'max_tokens': 1024
    }
)

answer_text = response['message']['content']
```

**What the LLM receives:**
- System instructions (how to behave)
- 5 relevant text chunks with source info
- The user's question
- Instruction to cite sources

**LLM Processing:**
- Reads all the context
- Identifies the most relevant information (Source 1 has the complete procedure)
- Generates a clear, step-by-step answer
- Cites the source documents

---

#### **Step 8: LLM Generates Answer**

**Generated Answer:**

```
To reset the XYZ-100 device after a power failure, follow this procedure:

**Reset Steps:**

1. **Turn off power** using the main switch on the back panel
2. **Disconnect all cables** including:
   - Power connections
   - Data connections
   - Sensor connections
3. **Wait 30 seconds** to allow capacitors to discharge
4. **Reconnect cables** in reverse order:
   - Sensors first
   - Data cables next
   - Power cable last
5. **Turn on power** and wait for the initialization sequence to complete (indicated by green LED)

**Important Notes:**
- Always check that backup power is disconnected before resetting
- After a power failure, the device may need to be reset to restore normal operation

**Sources:**
- XYZ-100 Service Manual, Pages 23-24, Section 3.2 (Reset Procedures)
- XYZ-100 Troubleshooting Guide, Page 5 (Power Issues)
- XYZ-100 Service Manual, Page 23, Section 3.1 (Common Issues)
```

---

#### **Step 9: Display to User**

**User Interface (Streamlit web app):**

```
┌────────────────────────────────────────────────────────────────┐
│ 🔧 Maintenance Documentation Q&A                               │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│ Question:                                                      │
│ ┌──────────────────────────────────────────────────────────┐ │
│ │ How do I reset the XYZ-100 after a power failure?       │ │
│ └──────────────────────────────────────────────────────────┘ │
│                                                                │
│ ✅ Answer Generated (in 2.3 seconds)                          │
│                                                                │
│ ┌──────────────────────────────────────────────────────────┐ │
│ │ 📝 Answer                                                │ │
│ │                                                          │ │
│ │ To reset the XYZ-100 device after a power failure,      │ │
│ │ follow this procedure:                                   │ │
│ │                                                          │ │
│ │ **Reset Steps:**                                         │ │
│ │                                                          │ │
│ │ 1. **Turn off power** using the main switch on the      │ │
│ │    back panel                                            │ │
│ │ 2. **Disconnect all cables** including:                 │ │
│ │    - Power connections                                   │ │
│ │    - Data connections                                    │ │
│ │    - Sensor connections                                  │ │
│ │ 3. **Wait 30 seconds** to allow capacitors to discharge │ │
│ │ 4. **Reconnect cables** in reverse order:               │ │
│ │    - Sensors first                                       │ │
│ │    - Data cables next                                    │ │
│ │    - Power cable last                                    │ │
│ │ 5. **Turn on power** and wait for initialization        │ │
│ │    (green LED indicates completion)                      │ │
│ │                                                          │ │
│ │ **Important Notes:**                                     │ │
│ │ - Always check that backup power is disconnected        │ │
│ │   before resetting                                       │ │
│ │ - After a power failure, the device may need to be      │ │
│ │   reset to restore normal operation                      │ │
│ │                                                          │ │
│ └──────────────────────────────────────────────────────────┘ │
│                                                                │
│ 💡 Confidence: 🟢 High                                        │
│                                                                │
│ ┌──────────────────────────────────────────────────────────┐ │
│ │ 📚 Sources                                               │ │
│ │                                                          │ │
│ │ ▼ Source 1: XYZ-100 Service Manual (Similarity: 89%)    │ │
│ │   ├─ Pages: 23-24                                        │ │
│ │   └─ Section: 3.2 Reset Procedures                      │ │
│ │                                                          │ │
│ │ ▼ Source 2: XYZ-100 Troubleshooting Guide (Sim: 78%)    │ │
│ │   ├─ Page: 5                                             │ │
│ │   └─ Section: Power Issues                              │ │
│ │                                                          │ │
│ │ ▼ Source 3: XYZ-100 Service Manual (Similarity: 82%)    │ │
│ │   ├─ Page: 23                                            │ │
│ │   └─ Section: 3.1 Common Issues                         │ │
│ │                                                          │ │
│ └──────────────────────────────────────────────────────────┘ │
│                                                                │
│ ───────────────────────────────────────────────────────────── │
│ 📊 Was this answer helpful?                                   │
│                                                                │
│    [👍 Yes]  [👎 No]  [💬 Add Comment]                        │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## What the User Sees: Complete Example

### Mobile/Tablet View (for field engineers)

```
┌─────────────────────────────────────┐
│ 🔧 Maintenance Q&A                  │
├─────────────────────────────────────┤
│                                     │
│ Your Question:                      │
│ ┌─────────────────────────────────┐ │
│ │ How do I reset the XYZ-100     │ │
│ │ after a power failure?         │ │
│ └─────────────────────────────────┘ │
│                                     │
│ Filter: [XYZ-100 ▼]  [🔍 Search]  │
│                                     │
├─────────────────────────────────────┤
│ ✅ Answer (2.3s) 🟢 High Confidence │
├─────────────────────────────────────┤
│                                     │
│ 📝 Reset Procedure:                │
│                                     │
│ 1️⃣ Turn off power (back panel)    │
│                                     │
│ 2️⃣ Disconnect all cables:          │
│    • Power                          │
│    • Data                           │
│    • Sensors                        │
│                                     │
│ 3️⃣ Wait 30 seconds                 │
│    (capacitor discharge)            │
│                                     │
│ 4️⃣ Reconnect in order:             │
│    • Sensors first                  │
│    • Data next                      │
│    • Power last                     │
│                                     │
│ 5️⃣ Power on, wait for green LED   │
│                                     │
│ ⚠️  Important:                      │
│ Check backup power is disconnected  │
│                                     │
├─────────────────────────────────────┤
│ 📚 Sources (tap to expand)          │
│                                     │
│ ▶ XYZ-100 Service Manual           │
│   Pages 23-24, Section 3.2          │
│                                     │
│ ▶ XYZ-100 Troubleshooting Guide    │
│   Page 5, Power Issues              │
│                                     │
├─────────────────────────────────────┤
│ Helpful? [👍] [👎]  Share: [📤]    │
│                                     │
│ [Ask Another Question]              │
│                                     │
└─────────────────────────────────────┘
```

### CLI Output (command line)

```bash
$ rag-query "How do I reset the XYZ-100 after a power failure?"

🔍 Searching documentation...
Found 10 relevant sections in 2.1 seconds.

───────────────────────────────────────────────────────────────
📝 ANSWER
───────────────────────────────────────────────────────────────

To reset the XYZ-100 device after a power failure, follow this
procedure:

Reset Steps:

1. Turn off power using the main switch on the back panel

2. Disconnect all cables including:
   - Power connections
   - Data connections
   - Sensor connections

3. Wait 30 seconds to allow capacitors to discharge

4. Reconnect cables in reverse order:
   - Sensors first
   - Data cables next
   - Power cable last

5. Turn on power and wait for the initialization sequence to
   complete (indicated by green LED)

Important Notes:
• Always check that backup power is disconnected before resetting
• After a power failure, the device may need to be reset to
  restore normal operation

───────────────────────────────────────────────────────────────
📚 SOURCES
───────────────────────────────────────────────────────────────

[1] XYZ-100 Service Manual, Pages 23-24, Section 3.2
    Reset Procedures (Similarity: 89%)

[2] XYZ-100 Troubleshooting Guide, Page 5
    Power Issues (Similarity: 78%)

[3] XYZ-100 Service Manual, Page 23, Section 3.1
    Common Issues (Similarity: 82%)

───────────────────────────────────────────────────────────────
💡 Confidence: HIGH 🟢
───────────────────────────────────────────────────────────────

Query time: 2.3 seconds
Helpful? Rate this answer: rag-query --rate <query-id> [1-5]
```

---

## Technical Deep Dive

### Performance Breakdown

**Total Response Time: ~2-4 seconds**

| Step | Operation | Time | Details |
|------|-----------|------|---------|
| 1 | Question input | 0ms | User types question |
| 2 | Embedding generation | 30-50ms | E5-Small converts text to 384-dim vector |
| 3 | Vector search | 10-20ms | HNSW search through 50,000 chunks |
| 4 | Context assembly | 5-10ms | Format top 5 chunks into prompt |
| 5 | LLM processing | 1500-3000ms | Llama 3.2:3b generates answer (~50 tokens/sec) |
| 6 | Display rendering | 10-20ms | Streamlit renders UI |
| **TOTAL** | | **~2-4 seconds** | End-to-end response time |

### Storage Requirements

**For 10,000 pages (1000 documents):**

| Component | Calculation | Size |
|-----------|-------------|------|
| Original documents | 1000 PDFs @ ~1MB each | ~1 GB |
| Text chunks | 50,000 chunks @ ~2KB text each | ~100 MB |
| Vector embeddings | 50,000 × 384 × 4 bytes (float32) | ~77 MB |
| Metadata | 50,000 × ~2KB each | ~100 MB |
| HNSW index | ~15% overhead on vectors | ~12 MB |
| SQLite database | Metadata + indexes | ~120 MB |
| **TOTAL** | | **~1.4 GB** |

### Understanding Vector Embeddings and Storage

**Critical Concept: Embeddings are Fixed-Size**

A common question about the storage calculation is: "Do we need to multiply the vector storage by the number of tokens in each chunk?"

**Answer: NO! Here's why:**

#### How Embeddings Work

Embedding models convert **variable-length text** into **fixed-size vectors**. This is a fundamental concept in RAG systems.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('intfloat/e5-small-v2')

# Short text (5 tokens)
text1 = "Reset the device"
embedding1 = model.encode(text1)
print(embedding1.shape)  # Output: (384,)

# Long text (512 tokens)
text2 = """To reset the XYZ-100 device after a power failure,
follow these detailed steps: First, ensure all connections
are secure. Second, disconnect the power supply for at least
30 seconds... [continues for 512 tokens total]"""
embedding2 = model.encode(text2)
print(embedding2.shape)  # Output: (384,)

# BOTH embeddings are EXACTLY 384 dimensions!
```

**Key Point:** Regardless of input length (1 token or 512 tokens), the output is **always 384 dimensions** for E5-Small-v2.

#### Storage Calculation Breakdown

**For 50,000 chunks:**

**1. Vector Embeddings (Semantic Representation):**
```
Each chunk → 384 float32 values → 384 × 4 bytes = 1,536 bytes
50,000 chunks → 50,000 × 1,536 bytes = 76,800,000 bytes ≈ 77 MB
```

**Important:** This is ONLY the numerical embeddings (the 384 numbers), NOT the original text.

**2. Original Text (Stored in Metadata):**
```
Each chunk → ~512 characters → ~512 bytes
50,000 chunks → 50,000 × 512 bytes = 25,600,000 bytes ≈ 26 MB
```

**3. Other Metadata (Document info):**
```json
{
  "doc_name": "XYZ-100 Manual",        // ~50 bytes
  "doc_id": "doc_123",                 // ~20 bytes
  "page_numbers": [23, 24],            // ~20 bytes
  "section": "Chapter 3",              // ~50 bytes
  "device_type": "XYZ-100",            // ~20 bytes
  "chunk_id": 45,                      // ~8 bytes
  // + database overhead              // ~200 bytes
  // Total: ~370 bytes per chunk
}
```

```
50,000 chunks → 50,000 × 370 bytes = 18,500,000 bytes ≈ 19 MB
```

#### Complete Storage Picture

| Component | What It Stores | Calculation | Size |
|-----------|----------------|-------------|------|
| **Vector Embeddings** | Semantic meaning as 384 numbers | 50,000 × 384 × 4 bytes | **77 MB** |
| **Chunk Text** | Original text content | 50,000 × 512 bytes | **26 MB** |
| **Other Metadata** | Doc name, pages, section, etc. | 50,000 × 370 bytes | **19 MB** |
| **HNSW Index** | Graph structure for fast search | ~15% overhead | **12 MB** |
| **SQLite Overhead** | Database indexes, headers | Variable | **20 MB** |
| **TOTAL** | | | **~154 MB** |

#### Why Token Count Doesn't Affect Vector Storage

**The 512 tokens/chunk specification affects:**

1. ✅ **Input limit** for the embedding model (E5-Small can handle up to 512 tokens)
2. ✅ **Text storage** in metadata (~512 bytes for 512 characters)

**The 512 tokens/chunk does NOT affect:**

1. ❌ **Embedding dimensions** (always 384, regardless of input length)
2. ❌ **Vector storage size** (always 384 × 4 = 1,536 bytes per chunk)

#### How Embedding Models Compress Information

Think of embedding models as **semantic compressors**:

```
┌─────────────────────────────────────────────────────┐
│ INPUT: Variable-Length Text                         │
│ "To reset the XYZ-100 device, follow these         │
│  steps: 1) Turn off power... [512 tokens]"         │
│                                                     │
│                      ↓                              │
│                                                     │
│            Neural Network Processing                │
│         (E5-Small: 118M parameters,                │
│          12 layers, 12 attention heads)            │
│                                                     │
│                      ↓                              │
│                                                     │
│ OUTPUT: Fixed-Size Semantic Vector                 │
│ [0.234, -0.123, 0.567, ..., 0.891]                │
│          (exactly 384 numbers)                      │
│                                                     │
│ Storage: 384 × 4 bytes = 1,536 bytes              │
└─────────────────────────────────────────────────────┘
```

The embedding is a **compressed semantic fingerprint** of the text, not a token-by-token encoding. This is what makes vector search efficient - all vectors are the same size, making similarity calculations uniform and fast.

#### Example: Storage for Different Scenarios

**Scenario 1: Short Chunks (256 tokens average)**
- Vector storage: 50,000 × 384 × 4 = **77 MB** (same!)
- Text storage: 50,000 × 256 = **13 MB** (half)
- Total: ~109 MB

**Scenario 2: Long Chunks (512 tokens average)**
- Vector storage: 50,000 × 384 × 4 = **77 MB** (same!)
- Text storage: 50,000 × 512 = **26 MB** (double)
- Total: ~122 MB

**Scenario 3: Using Different Embedding Model (1024 dimensions)**
- Vector storage: 50,000 × 1,024 × 4 = **205 MB** (dimensions changed!)
- Text storage: 50,000 × 512 = **26 MB** (same)
- Total: ~250 MB

**Notice:** Vector storage only changes with embedding dimensions, NOT with chunk length!

#### Why This Matters for RAG Systems

1. **Predictable Storage:** You can calculate exact storage needs based on number of chunks and embedding dimensions

2. **Consistent Performance:** All vectors are same size, so search time is consistent regardless of chunk length

3. **Efficient Scaling:** Doubling the number of documents doubles storage linearly, not exponentially

4. **Memory Planning:** For 50,000 chunks with 384-dim embeddings, you need ~77 MB for vectors + ~50 MB for metadata = **~130 MB total** (very manageable!)

### Accuracy Metrics

**Expected Retrieval Quality:**

| Metric | Target | Meaning |
|--------|--------|---------|
| **Recall@5** | >90% | Relevant chunk in top 5 results 90% of time |
| **Recall@10** | >95% | Relevant chunk in top 10 results 95% of time |
| **Precision** | >80% | 80% of returned chunks are relevant |
| **Answer Quality** | >85% | 85% of answers rated helpful by users |

### Why This Works So Well

1. **Semantic Understanding**
   - "reset" matches "reset procedures", "reboot", "restart", "power cycle"
   - "power failure" matches "power loss", "outage", "electrical issue"
   - Model understands synonyms and related concepts

2. **Context Preservation**
   - Complete procedures stay together (not split across chunks)
   - Related information (warnings, notes) included in same chunk
   - Page-spanning content handled gracefully

3. **Source Attribution**
   - Every answer includes exact document and page references
   - Users can verify information in original docs
   - Builds trust and credibility

4. **Fast Retrieval**
   - HNSW index: sub-50ms for 1M vectors
   - For 50K chunks: ~10-20ms search time
   - Parallel processing possible for multiple queries

5. **Privacy & Offline**
   - All processing happens locally
   - No internet required
   - Sensitive maintenance docs never leave premises
   - Works in remote field locations

---

## Comparison: Traditional Search vs. RAG

### Traditional Keyword Search

**User Query:** "How do I reset the XYZ-100 after a power failure?"

**Traditional Search Results:**
```
1. Page 23: Contains "XYZ-100" and "reset" (keyword match)
2. Page 156: Contains "power failure" (keyword match)
3. Page 87: Contains "XYZ-100" (partial match)
4. Page 203: Contains "reset" (partial match)
```

**Problems:**
- ❌ No understanding of meaning
- ❌ Misses synonyms ("reboot", "power cycle")
- ❌ No ranking by relevance
- ❌ User must read multiple pages
- ❌ May miss relevant info that doesn't have exact keywords

### RAG System (Vector Search + LLM)

**User Query:** "How do I reset the XYZ-100 after a power failure?"

**RAG Results:**
```
1. Chunk 44: Complete reset procedure (89% similarity)
   - Understands "reset" semantically
   - Knows this is about XYZ-100
   - Recognizes power-related context

2. Chunk 43: Context about when reset is needed (82% similarity)
   - Mentions power failures explicitly
   - Points to reset procedure

3. Direct answer with step-by-step instructions
4. Source citations for verification
```

**Advantages:**
- ✅ Understands meaning, not just keywords
- ✅ Finds semantically similar content
- ✅ Ranks by relevance (similarity score)
- ✅ Provides direct answer, not just search results
- ✅ Cites sources for verification
- ✅ Works with natural language questions

---

## Example Queries and Results

### Example 1: Simple Procedure Question

**Question:** "How often should I calibrate the XYZ-100?"

**Retrieved Chunks:**
- Calibration schedule (similarity: 0.91)
- Calibration procedures overview (similarity: 0.84)
- Maintenance schedule table (similarity: 0.79)

**Answer:**
```
The XYZ-100 should be calibrated every 6 months or after any major
repair work. More frequent calibration may be required in harsh
environments or high-precision applications.

Calibration should also be performed if:
• Measurement accuracy appears degraded
• Device has been subjected to extreme temperatures
• After firmware updates

Source: XYZ-100 Service Manual, Page 67, Section 5.3 (Calibration Schedule)
```

### Example 2: Troubleshooting Question

**Question:** "XYZ-100 shows error code E42, what does that mean?"

**Retrieved Chunks:**
- Error code E42 definition (similarity: 0.94)
- Error code troubleshooting steps (similarity: 0.88)
- Related error codes (similarity: 0.76)

**Answer:**
```
Error code E42 indicates a sensor communication failure.

Troubleshooting Steps:
1. Check that all sensor cables are properly connected
2. Verify sensor power LED is illuminated
3. Try reseating the sensor connector at both ends
4. If problem persists, test with a known-good sensor
5. Contact support if error continues after sensor replacement

⚠️  WARNING: Do not operate the device with sensor communication
errors as this may result in inaccurate readings.

Sources:
• XYZ-100 Error Code Reference, Page 12, Section E42
• XYZ-100 Troubleshooting Guide, Page 34, Sensor Issues
```

### Example 3: Safety Question

**Question:** "What safety precautions should I take when working on the DEF-300?"

**Retrieved Chunks:**
- DEF-300 safety warnings (similarity: 0.93)
- General safety procedures (similarity: 0.81)
- Electrical safety guidelines (similarity: 0.78)

**Answer:**
```
Important safety precautions for DEF-300 service:

⚠️  CRITICAL SAFETY WARNINGS:

1. **Electrical Hazard:**
   - Disconnect ALL power sources before opening enclosure
   - Wait 5 minutes for capacitors to discharge
   - Verify zero voltage with multimeter

2. **High Voltage:**
   - Device contains components operating at 480V
   - Only qualified electricians should service high-voltage sections
   - Use insulated tools rated for 1000V

3. **Personal Protective Equipment:**
   - Safety glasses (ANSI Z87.1 rated)
   - Insulated gloves (Class 00 minimum)
   - Non-conductive shoes

4. **Lock-Out/Tag-Out:**
   - Follow facility LOTO procedures
   - Use personal lock
   - Verify lock cannot be removed by others

⚠️  DANGER: Failure to follow these precautions can result in
serious injury or death.

Sources:
• DEF-300 Safety Manual, Pages 1-3, Critical Safety Information
• DEF-300 Service Manual, Page 8, Section 1.4 (Safety Procedures)
```

### Example 4: Question Not in Documentation

**Question:** "Can I use the XYZ-100 underwater?"

**Retrieved Chunks:**
- Operating environment specifications (similarity: 0.68)
- Moisture protection ratings (similarity: 0.64)
- General usage guidelines (similarity: 0.59)

**Answer:**
```
I don't have specific information about underwater use of the XYZ-100
in the available documentation.

Based on the available information:
• The XYZ-100 is rated IP54 (dust and splash resistant)
• It is NOT rated for submersion or underwater use
• Operating environment: Indoor use, 0-50°C, <85% humidity

⚠️  RECOMMENDATION: Do NOT use the XYZ-100 underwater. Contact the
manufacturer for specific applications requiring water resistance or
submersion.

For aquatic environments, consider devices with IP67 or IP68 ratings.

Sources:
• XYZ-100 Service Manual, Page 4, Environmental Specifications
• XYZ-100 Service Manual, Page 6, Operating Conditions

💡 Note: This answer is based on environmental specifications. For
definitive guidance on underwater use, contact technical support at
1-800-SUPPORT.
```

**Notice:** The system correctly indicates when information is not available and provides relevant context based on what IS in the documentation.

---

## Summary

### Key Takeaways

1. **Chunking:**
   - One vector = one text chunk (~500 words)
   - NOT one page, NOT entire document
   - Chunks can span multiple pages for context preservation

2. **Search Process:**
   - Question → Vector → Search → Top matches → LLM → Answer
   - Takes 2-4 seconds end-to-end
   - Returns exact source citations

3. **User Experience:**
   - Direct answers, not just search results
   - Step-by-step procedures when appropriate
   - Source documents and page numbers for verification
   - Works offline, fully private

4. **Why It Works:**
   - Semantic understanding (not just keywords)
   - Context preservation (complete procedures)
   - Fast retrieval (HNSW indexing)
   - Accurate answers (LLM with grounded context)

### Questions?

This system is specifically designed for field engineers and mechanics who need:
- ✅ **Fast answers** to specific technical questions
- ✅ **Accurate information** from official documentation
- ✅ **Source verification** to trust the answers
- ✅ **Offline capability** for remote locations
- ✅ **Easy access** via web or mobile interface

The combination of JadeVectorDB (vector search) + Ollama (local LLM) + proper chunking strategy creates a powerful, private, and practical solution for maintenance documentation access.

---

**Document Version:** 1.0
**Date:** January 2, 2026
**Related Documents:**
- RAG_USECASE.md - Complete technical architecture
- BOOTSTRAP.md - JadeVectorDB developer guide
- README.md - Project overview
