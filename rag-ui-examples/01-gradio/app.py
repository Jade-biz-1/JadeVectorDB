#!/usr/bin/env python3
"""
RAG UI Alternative #1: Gradio
Fast migration from Streamlit, better chat interface

Usage:
    pip install gradio
    python app.py
    Open: http://localhost:7860
"""

import gradio as gr
import json
from datetime import datetime
from typing import List, Dict, Tuple

# Mock RAG service (replace with your actual implementation)
class MockRAGService:
    """Simulates your RAG service for demo purposes"""

    def __init__(self):
        self.query_history = []

    def query(self, question: str, device_type: str = "All", top_k: int = 5) -> Dict:
        """
        Simulate RAG query
        In production, this would:
        1. Generate embedding from question
        2. Search JadeVectorDB for similar chunks
        3. Send context + question to Ollama LLM
        4. Return answer with sources
        """
        # Simulate processing
        import time
        time.sleep(0.5)  # Simulate embedding + search + LLM

        # Mock response
        answer = f"""Based on the maintenance documentation for {device_type if device_type != 'All' else 'the equipment'}:

**Procedure:**
1. First, ensure the equipment is powered off and disconnected
2. Remove the access panel using a 10mm socket wrench
3. Inspect the internal components for wear or damage
4. Apply lubricant to moving parts as specified in section 4.2
5. Reassemble and test operation

**Important:** Always follow safety protocols outlined in the safety manual before performing maintenance.

This procedure should take approximately 30-45 minutes depending on equipment condition."""

        sources = [
            {
                "doc_name": f"{device_type}_Maintenance_Manual_v3.2.pdf" if device_type != "All" else "General_Maintenance_Guide.pdf",
                "page_numbers": "23-25",
                "section": "Chapter 4: Routine Maintenance",
                "relevance_score": 0.89,
                "excerpt": "Standard maintenance procedure for routine inspections..."
            },
            {
                "doc_name": "Safety_Guidelines_2025.pdf",
                "page_numbers": "12",
                "section": "Section 2: Equipment Safety",
                "relevance_score": 0.76,
                "excerpt": "Always disconnect power before accessing internal components..."
            }
        ]

        result = {
            "answer": answer,
            "sources": sources,
            "query": question,
            "device_type": device_type,
            "timestamp": datetime.now().isoformat(),
            "confidence": 0.85
        }

        self.query_history.append(result)
        return result

    def get_history(self) -> List[Dict]:
        return self.query_history


# Initialize RAG service
rag_service = MockRAGService()


def format_sources(sources: List[Dict]) -> str:
    """Format source citations for display"""
    if not sources:
        return "No sources found"

    formatted = "### 📚 Sources:\n\n"
    for i, source in enumerate(sources, 1):
        formatted += f"""**{i}. {source['doc_name']}**
- Pages: {source['page_numbers']}
- Section: {source['section']}
- Relevance: {source['relevance_score']:.0%}
- Excerpt: "{source['excerpt']}"

"""
    return formatted


def query_rag(question: str, device_type: str, top_k: int, history: List) -> Tuple:
    """
    Process user query and return answer with sources

    Args:
        question: User's question
        device_type: Filter by device type
        top_k: Number of chunks to retrieve
        history: Chat history

    Returns:
        Tuple of (answer, sources_formatted, updated_history)
    """
    if not question.strip():
        return "", "Please enter a question", history

    # Query RAG service
    result = rag_service.query(question, device_type, top_k)

    # Format response
    answer = result["answer"]
    sources = format_sources(result["sources"])

    # Update chat history
    history = history or []
    history.append({
        "role": "user",
        "content": question,
        "timestamp": datetime.now().strftime("%H:%M:%S")
    })
    history.append({
        "role": "assistant",
        "content": answer,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "confidence": result["confidence"]
    })

    return answer, sources, history


def format_chat_history(history: List) -> str:
    """Format chat history for display"""
    if not history:
        return "No conversation history yet"

    formatted = ""
    for msg in history:
        role_icon = "👤" if msg["role"] == "user" else "🤖"
        time = msg.get("timestamp", "")
        confidence = msg.get("confidence", None)

        formatted += f"**{role_icon} {msg['role'].title()}** ({time})"
        if confidence:
            formatted += f" - Confidence: {confidence:.0%}"
        formatted += f"\n\n{msg['content']}\n\n---\n\n"

    return formatted


def clear_history():
    """Clear chat history"""
    return [], "", ""


# Build Gradio Interface
with gr.Blocks(theme=gr.themes.Soft(), title="Maintenance Documentation Q&A") as demo:

    # Header
    gr.Markdown("""
    # 🔧 Maintenance Documentation Q&A System

    Ask questions about equipment maintenance procedures, troubleshooting steps, and technical specifications.
    All answers are based on your local maintenance documentation library.

    **Features:**
    - 🔌 Fully offline operation
    - 📚 Source citations with page numbers
    - 🎯 Device-specific filtering
    - 💬 Conversation history
    """)

    # State for chat history
    chat_history = gr.State([])

    with gr.Row():
        # Left column: Query input
        with gr.Column(scale=2):
            question_input = gr.Textbox(
                label="Your Question",
                placeholder="Example: How do I perform routine maintenance on the hydraulic pump?",
                lines=3
            )

            with gr.Row():
                device_dropdown = gr.Dropdown(
                    choices=["All", "Hydraulic Pump", "Air Compressor", "Generator", "CNC Machine", "Conveyor Belt"],
                    value="All",
                    label="Device Type Filter"
                )

                top_k_slider = gr.Slider(
                    minimum=1,
                    maximum=15,
                    value=5,
                    step=1,
                    label="Number of Chunks to Retrieve"
                )

            with gr.Row():
                submit_btn = gr.Button("🔍 Search Documentation", variant="primary")
                clear_btn = gr.Button("🗑️ Clear History")

            # Example questions
            gr.Examples(
                examples=[
                    ["How do I replace the air filter?", "Air Compressor", 5],
                    ["What is the recommended maintenance schedule?", "All", 5],
                    ["Troubleshooting steps for error code E47", "CNC Machine", 7],
                    ["Safety precautions for hydraulic systems", "Hydraulic Pump", 5],
                ],
                inputs=[question_input, device_dropdown, top_k_slider]
            )

        # Right column: Results
        with gr.Column(scale=3):
            answer_output = gr.Markdown(
                label="Answer",
                value="*Answers will appear here*"
            )

            with gr.Accordion("📚 Source Documents", open=True):
                sources_output = gr.Markdown(
                    value="*Source citations will appear here*"
                )

    # Chat history at bottom
    with gr.Accordion("💬 Conversation History", open=False):
        history_output = gr.Markdown(value="No conversation history yet")

    # Footer
    gr.Markdown("""
    ---
    **System Status:** ✅ Connected to JadeVectorDB | 🤖 Ollama LLM: llama3.2:3b | 🔌 Offline Mode

    *Powered by JadeVectorDB + Ollama*
    """)

    # Event handlers
    def submit_and_update(question, device, top_k, history):
        answer, sources, updated_history = query_rag(question, device, top_k, history)
        history_display = format_chat_history(updated_history)
        return answer, sources, updated_history, history_display, ""

    submit_btn.click(
        fn=submit_and_update,
        inputs=[question_input, device_dropdown, top_k_slider, chat_history],
        outputs=[answer_output, sources_output, chat_history, history_output, question_input]
    )

    # Also submit on Enter (Shift+Enter for newline)
    question_input.submit(
        fn=submit_and_update,
        inputs=[question_input, device_dropdown, top_k_slider, chat_history],
        outputs=[answer_output, sources_output, chat_history, history_output, question_input]
    )

    clear_btn.click(
        fn=clear_history,
        inputs=[],
        outputs=[chat_history, answer_output, history_output]
    )


if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 Starting Gradio RAG Interface")
    print("="*60)
    print("\n📍 Access the UI at: http://localhost:7860")
    print("🔌 Running in offline mode (no internet required)")
    print("\n💡 Tip: Press Ctrl+C to stop the server\n")

    demo.launch(
        server_name="0.0.0.0",  # Allow access from network
        server_port=7860,
        share=False,  # Keep it local (no public URL)
        show_error=True
    )
