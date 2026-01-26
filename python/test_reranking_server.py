#!/usr/bin/env python3
"""
Test script for the reranking server.

Tests the reranking server by sending sample requests and validating responses.
"""

import subprocess
import json
import time
import sys


def send_request(process, request):
    """Send a JSON request to the server and get response."""
    json_str = json.dumps(request) + "\n"
    process.stdin.write(json_str)
    process.stdin.flush()

    # Read response
    response_line = process.stdout.readline()
    if not response_line:
        return None

    return json.loads(response_line.strip())


def test_reranking_server():
    """Test the reranking server functionality."""
    print("=" * 80)
    print("JadeVectorDB Re-ranking Server Test")
    print("=" * 80)

    # Start the server process
    print("\n[1/6] Starting reranking server...")
    try:
        process = subprocess.Popen(
            [sys.executable, "reranking_server.py"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1
        )
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        return False

    print("✓ Server process started")

    try:
        # Wait for ready signal
        print("\n[2/6] Waiting for model to load...")
        start_time = time.time()
        ready_response = process.stdout.readline()

        if not ready_response:
            print("❌ No ready signal received")
            return False

        ready_data = json.loads(ready_response.strip())
        load_time = time.time() - start_time

        if ready_data.get("type") == "ready":
            print(f"✓ Model loaded: {ready_data.get('model')}")
            print(f"  Load time: {ready_data.get('load_time_ms', 0):.0f}ms")
        else:
            print(f"❌ Unexpected response: {ready_data}")
            return False

        # Test heartbeat
        print("\n[3/6] Testing heartbeat...")
        heartbeat_request = {"type": "heartbeat"}
        heartbeat_response = send_request(process, heartbeat_request)

        if heartbeat_response and heartbeat_response.get("type") == "heartbeat":
            print(f"✓ Heartbeat OK: status={heartbeat_response.get('status')}")
        else:
            print(f"❌ Heartbeat failed: {heartbeat_response}")
            return False

        # Test re-ranking with sample data
        print("\n[4/6] Testing re-ranking...")
        query = "What is machine learning?"
        documents = [
            "Machine learning is a branch of artificial intelligence that focuses on building systems that learn from data.",
            "Python is a high-level programming language used for web development and data science.",
            "Deep learning is a subset of machine learning that uses neural networks with multiple layers.",
            "Database systems store and manage large amounts of data efficiently.",
            "Artificial intelligence enables computers to perform tasks that typically require human intelligence."
        ]

        rerank_request = {"query": query, "documents": documents}
        rerank_response = send_request(process, rerank_request)

        if rerank_response and "scores" in rerank_response:
            scores = rerank_response["scores"]
            latency = rerank_response.get("latency_ms", 0)

            print(f"✓ Re-ranking completed")
            print(f"  Latency: {latency:.2f}ms")
            print(f"  Documents: {len(documents)}")
            print(f"  Per-document: {latency/len(documents):.2f}ms")
            print(f"\n  Ranked results:")

            # Sort by score and display top 3
            ranked = sorted(zip(scores, documents), reverse=True)
            for i, (score, doc) in enumerate(ranked[:3], 1):
                print(f"    {i}. Score: {score:.3f} - {doc[:60]}...")

            # Validate scores
            if not all(isinstance(s, (int, float)) for s in scores):
                print(f"❌ Invalid score types")
                return False

            if len(scores) != len(documents):
                print(f"❌ Score count mismatch: {len(scores)} != {len(documents)}")
                return False

            print("\n  ✓ All validations passed")

        else:
            print(f"❌ Re-ranking failed: {rerank_response}")
            return False

        # Test empty documents
        print("\n[5/6] Testing edge cases...")
        empty_request = {"query": "test", "documents": []}
        empty_response = send_request(process, empty_request)

        if empty_response and empty_response.get("scores") == []:
            print(f"✓ Empty documents handled correctly")
        else:
            print(f"❌ Empty documents test failed: {empty_response}")
            return False

        # Test statistics
        print("\n[6/6] Testing statistics...")
        stats_request = {"type": "stats"}
        stats_response = send_request(process, stats_request)

        if stats_response and stats_response.get("type") == "stats":
            print(f"✓ Statistics retrieved:")
            print(f"  Requests processed: {stats_response.get('requests_processed', 0)}")
            print(f"  Average latency: {stats_response.get('avg_latency_ms', 0):.2f}ms")
        else:
            print(f"❌ Statistics test failed: {stats_response}")
            return False

        # Shutdown server
        print("\n[Cleanup] Shutting down server...")
        shutdown_request = {"type": "shutdown"}
        send_request(process, shutdown_request)

        # Wait for process to exit
        process.wait(timeout=5)
        print("✓ Server shut down gracefully")

        print("\n" + "=" * 80)
        print("✅ All tests passed!")
        print("=" * 80)
        return True

    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Ensure process is terminated
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=2)


if __name__ == "__main__":
    success = test_reranking_server()
    sys.exit(0 if success else 1)
