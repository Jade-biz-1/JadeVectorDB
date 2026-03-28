#!/usr/bin/env python3
"""
RAG UI Alternative #2: Flask + Jinja Templates
Clean, production-ready web interface with full control

Usage:
    pip install flask
    python app.py
    Open: http://localhost:5000
"""

from flask import Flask, render_template, request, jsonify
from datetime import datetime
from typing import Dict, List
import json

app = Flask(__name__)

# Mock RAG service (replace with your actual implementation)
class MockRAGService:
    """Simulates your RAG service for demo purposes"""

    def __init__(self):
        self.query_count = 0

    def query(self, question: str, device_type: str = "all", top_k: int = 5) -> Dict:
        """
        Simulate RAG query
        In production, this would call your actual RAG pipeline
        """
        import time
        time.sleep(0.3)  # Simulate processing

        self.query_count += 1

        device_name = device_type.replace("_", " ").title() if device_type != "all" else "the equipment"

        answer = f"""Based on the maintenance documentation for {device_name}:

**Maintenance Procedure:**

1. **Safety First**: Ensure equipment is powered off and properly locked out/tagged out
2. **Access**: Remove the access panel using appropriate tools (typically 10mm socket)
3. **Inspection**: Visually inspect all components for:
   - Wear patterns
   - Fluid leaks
   - Corrosion
   - Loose connections
4. **Lubrication**: Apply manufacturer-specified lubricant to moving parts
5. **Reassembly**: Replace all covers and fasteners to proper torque specifications
6. **Testing**: Perform operational test before returning to service

**Estimated Time**: 30-45 minutes

**Required Tools**: Socket set, torque wrench, inspection mirror

⚠️ **Safety Warning**: Always follow LOTO (Lockout/Tagout) procedures before maintenance."""

        sources = [
            {
                "doc_name": f"{device_type.upper()}_Maintenance_Manual_v3.2.pdf" if device_type != "all" else "General_Maintenance_Guide.pdf",
                "page_numbers": "23-25",
                "section": "Chapter 4: Routine Maintenance",
                "relevance": 0.89,
                "excerpt": "Standard maintenance procedure for routine inspections and preventive maintenance tasks..."
            },
            {
                "doc_name": "Safety_Guidelines_2025.pdf",
                "page_numbers": "12-14",
                "section": "Section 2: Equipment Safety",
                "relevance": 0.76,
                "excerpt": "Always disconnect power and verify zero energy state before accessing internal components..."
            },
            {
                "doc_name": f"{device_type.upper()}_Technical_Specs.pdf" if device_type != "all" else "Equipment_Specifications.pdf",
                "page_numbers": "8",
                "section": "Maintenance Schedule",
                "relevance": 0.68,
                "excerpt": "Recommended maintenance intervals and procedures for optimal equipment performance..."
            }
        ]

        return {
            "success": True,
            "answer": answer,
            "sources": sources,
            "query": question,
            "device_type": device_type,
            "timestamp": datetime.now().isoformat(),
            "confidence": 0.85,
            "query_number": self.query_count
        }

    def get_stats(self) -> Dict:
        """Get system statistics"""
        return {
            "total_queries": self.query_count,
            "db_status": "connected",
            "llm_status": "ready",
            "total_documents": 127,
            "total_chunks": 3842
        }


# Initialize RAG service
rag_service = MockRAGService()


@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')


@app.route('/api/query', methods=['POST'])
def query():
    """Handle RAG query request"""
    try:
        data = request.get_json()

        question = data.get('question', '').strip()
        device_type = data.get('device_type', 'all')
        top_k = int(data.get('top_k', 5))

        if not question:
            return jsonify({
                "success": False,
                "error": "Question is required"
            }), 400

        # Query RAG service
        result = rag_service.query(question, device_type, top_k)

        return jsonify(result)

    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/api/stats', methods=['GET'])
def stats():
    """Get system statistics"""
    return jsonify(rag_service.get_stats())


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    })


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Starting Flask RAG Interface")
    print("="*60)
    print("\n📍 Access the UI at: http://localhost:5000")
    print("🔌 Running in offline mode (no internet required)")
    print("\n💡 Tip: Press Ctrl+C to stop the server\n")

    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True  # Set to False in production
    )
