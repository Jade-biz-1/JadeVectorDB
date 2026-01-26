#!/usr/bin/env python3
"""
Example usage of the re-ranking server.

Demonstrates how to interact with the reranking server from Python.
"""

import subprocess
import json
import sys


def main():
    """Example: Re-rank search results for a query."""

    print("=" * 80)
    print("JadeVectorDB Re-ranking Server - Example Usage")
    print("=" * 80)

    # Sample query and documents (e.g., from a vector search)
    query = "How does vector search work?"

    documents = [
        "Vector search uses embeddings to find semantically similar content.",
        "A database is a structured collection of data stored electronically.",
        "Semantic search understands the intent behind queries using embeddings.",
        "SQL databases use tables, rows, and columns to organize data.",
        "Vector databases like JadeVectorDB store and search high-dimensional vectors.",
        "Machine learning models convert text into numerical vector representations.",
    ]

    print(f"\nQuery: {query}")
    print(f"\nDocuments to re-rank: {len(documents)}")
    for i, doc in enumerate(documents, 1):
        print(f"  {i}. {doc}")

    # Start the reranking server
    print("\n" + "-" * 80)
    print("Starting reranking server...")
    print("-" * 80)

    process = subprocess.Popen(
        [sys.executable, "reranking_server.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )

    try:
        # Wait for ready signal
        print("Waiting for model to load...")
        ready_line = process.stdout.readline()
        ready = json.loads(ready_line.strip())

        if ready.get("type") == "ready":
            print(f"✓ Model loaded: {ready.get('model')}")
            print(f"  Load time: {ready.get('load_time_ms', 0):.0f}ms\n")
        else:
            print(f"❌ Unexpected response: {ready}")
            return

        # Send re-ranking request
        print("-" * 80)
        print("Sending re-ranking request...")
        print("-" * 80)

        request = {
            "query": query,
            "documents": documents
        }

        request_json = json.dumps(request) + "\n"
        process.stdin.write(request_json)
        process.stdin.flush()

        # Receive response
        response_line = process.stdout.readline()
        response = json.loads(response_line.strip())

        if "error" in response:
            print(f"❌ Error: {response['error']}")
            return

        scores = response["scores"]
        latency_ms = response["latency_ms"]

        print(f"\n✓ Re-ranking completed in {latency_ms:.2f}ms")
        print(f"  Per-document latency: {latency_ms/len(documents):.2f}ms")

        # Display ranked results
        print("\n" + "=" * 80)
        print("Re-ranked Results (by relevance):")
        print("=" * 80)

        # Sort documents by score
        ranked = sorted(zip(scores, documents), reverse=True)

        for i, (score, doc) in enumerate(ranked, 1):
            # Visual indicator based on score
            if score > 0.8:
                indicator = "🟢"  # High relevance
            elif score > 0.5:
                indicator = "🟡"  # Medium relevance
            else:
                indicator = "🔴"  # Low relevance

            print(f"\n{i}. {indicator} Score: {score:.4f}")
            print(f"   {doc}")

        # Highlight top result
        print("\n" + "-" * 80)
        print(f"🏆 Most Relevant: {ranked[0][1]}")
        print(f"   Score: {ranked[0][0]:.4f}")
        print("-" * 80)

        # Shutdown server
        shutdown = {"type": "shutdown"}
        process.stdin.write(json.dumps(shutdown) + "\n")
        process.stdin.flush()

        process.wait(timeout=5)
        print("\n✓ Server shut down\n")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=2)


if __name__ == "__main__":
    main()
