#!/usr/bin/env python3
"""
JadeVectorDB Re-ranking Server

A subprocess-based re-ranking server that uses cross-encoder models to improve
search result quality. Communicates via stdin/stdout using JSON protocol.

Architecture: Python Subprocess (Phase 1)
- Designed for single-node and small cluster deployments
- See docs/architecture.md for full architecture documentation

Usage:
    python reranking_server.py [--model MODEL_NAME] [--batch-size BATCH_SIZE]

Environment Variables:
    RERANKING_MODEL_PATH: Path or name of the cross-encoder model
    RERANKING_BATCH_SIZE: Batch size for inference (default: 32)
    RERANKING_LOG_LEVEL: Logging level (DEBUG, INFO, WARNING, ERROR)

Protocol:
    Request: {"query": "text", "documents": ["doc1", "doc2", ...]}
    Response: {"scores": [0.85, 0.72, ...], "latency_ms": 123}
    Error: {"error": "error message", "code": "ERROR_CODE"}
    Heartbeat: {"type": "heartbeat"}
"""

import sys
import json
import time
import signal
import logging
import argparse
import os
from typing import List, Dict, Any, Optional
from datetime import datetime

# Logging setup
LOG_LEVEL = os.environ.get('RERANKING_LOG_LEVEL', 'INFO')
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='[%(asctime)s] [%(levelname)s] [reranking_server] %(message)s',
    stream=sys.stderr  # Log to stderr to avoid interfering with stdout communication
)
logger = logging.getLogger(__name__)


class RerankingServer:
    """Cross-encoder based re-ranking server."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 batch_size: int = 32):
        """
        Initialize the re-ranking server.

        Args:
            model_name: HuggingFace model name or path
            batch_size: Batch size for inference
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = None
        self.running = True
        self.request_count = 0
        self.total_latency_ms = 0

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

        logger.info(f"Initializing RerankingServer with model: {model_name}")
        logger.info(f"Batch size: {batch_size}")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self.running = False
        self._log_stats()
        sys.exit(0)

    def _log_stats(self):
        """Log performance statistics."""
        if self.request_count > 0:
            avg_latency = self.total_latency_ms / self.request_count
            logger.info(f"Performance stats:")
            logger.info(f"  Total requests: {self.request_count}")
            logger.info(f"  Average latency: {avg_latency:.2f}ms")

    def load_model(self) -> bool:
        """
        Load the cross-encoder model.

        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            start_time = time.time()
            logger.info(f"Loading model: {self.model_name}")

            # Import here to catch import errors
            try:
                from sentence_transformers import CrossEncoder
            except ImportError as e:
                logger.error(f"Failed to import sentence_transformers: {e}")
                logger.error("Please install: pip install sentence-transformers")
                return False

            # Load the model
            self.model = CrossEncoder(self.model_name, max_length=512)

            load_time = (time.time() - start_time) * 1000
            logger.info(f"Model loaded successfully in {load_time:.0f}ms")

            # Log model information
            logger.info(f"Model max length: {self.model.max_length}")

            # Send ready signal
            self._send_response({"type": "ready", "model": self.model_name, "load_time_ms": load_time})

            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}", exc_info=True)
            self._send_error(f"Model loading failed: {str(e)}", "MODEL_LOAD_ERROR")
            return False

    def _send_response(self, data: Dict[str, Any]):
        """Send a JSON response to stdout."""
        try:
            json_str = json.dumps(data)
            print(json_str, flush=True)
        except Exception as e:
            logger.error(f"Failed to send response: {e}")

    def _send_error(self, message: str, error_code: str):
        """Send an error response."""
        self._send_response({
            "error": message,
            "code": error_code,
            "timestamp": datetime.now().isoformat()
        })

    def process_request(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process a re-ranking request.

        Args:
            request: Dictionary with 'query' and 'documents' keys

        Returns:
            Response dictionary with scores, or None on error
        """
        try:
            # Validate request
            if "query" not in request:
                self._send_error("Missing 'query' field in request", "INVALID_REQUEST")
                return None

            if "documents" not in request:
                self._send_error("Missing 'documents' field in request", "INVALID_REQUEST")
                return None

            query = request["query"]
            documents = request["documents"]

            if not isinstance(documents, list):
                self._send_error("'documents' must be a list", "INVALID_REQUEST")
                return None

            if len(documents) == 0:
                # Empty documents, return empty scores
                return {"scores": [], "latency_ms": 0}

            # Log request info
            logger.debug(f"Processing request: query_len={len(query)}, num_docs={len(documents)}")

            # Perform inference
            start_time = time.time()

            # Create query-document pairs
            pairs = [[query, doc] for doc in documents]

            # Predict scores in batches
            try:
                scores = self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False)
                scores = scores.tolist()  # Convert numpy array to list
            except Exception as e:
                logger.error(f"Inference failed: {e}", exc_info=True)
                self._send_error(f"Inference error: {str(e)}", "INFERENCE_ERROR")
                return None

            latency_ms = (time.time() - start_time) * 1000

            # Update stats
            self.request_count += 1
            self.total_latency_ms += latency_ms

            # Log performance
            per_doc_latency = latency_ms / len(documents) if len(documents) > 0 else 0
            logger.debug(f"Inference completed: {latency_ms:.2f}ms total, {per_doc_latency:.2f}ms per document")

            # Return response
            response = {
                "scores": scores,
                "latency_ms": round(latency_ms, 2),
                "num_documents": len(documents)
            }

            return response

        except Exception as e:
            logger.error(f"Request processing failed: {e}", exc_info=True)
            self._send_error(f"Processing error: {str(e)}", "PROCESSING_ERROR")
            return None

    def run(self):
        """Main server loop - read requests from stdin and respond to stdout."""
        logger.info("Server started, waiting for requests on stdin...")
        logger.info("Send heartbeat by sending: {\"type\": \"heartbeat\"}")
        logger.info("Send shutdown by sending: {\"type\": \"shutdown\"}")

        # Send initial heartbeat
        self._send_response({"type": "heartbeat", "status": "alive"})

        while self.running:
            try:
                # Read line from stdin
                line = sys.stdin.readline()

                if not line:
                    # EOF reached, exit gracefully
                    logger.info("EOF reached on stdin, shutting down...")
                    break

                line = line.strip()
                if not line:
                    continue

                # Parse JSON request
                try:
                    request = json.loads(line)
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON received: {e}")
                    self._send_error(f"Invalid JSON: {str(e)}", "INVALID_JSON")
                    continue

                # Handle special request types
                request_type = request.get("type")

                if request_type == "heartbeat":
                    # Respond to heartbeat
                    self._send_response({
                        "type": "heartbeat",
                        "status": "alive",
                        "requests_processed": self.request_count
                    })
                    continue

                if request_type == "shutdown":
                    # Graceful shutdown requested
                    logger.info("Shutdown requested via command")
                    self._log_stats()
                    self._send_response({"type": "shutdown", "status": "acknowledged"})
                    break

                if request_type == "stats":
                    # Return statistics
                    avg_latency = (self.total_latency_ms / self.request_count) if self.request_count > 0 else 0
                    self._send_response({
                        "type": "stats",
                        "requests_processed": self.request_count,
                        "total_latency_ms": self.total_latency_ms,
                        "avg_latency_ms": round(avg_latency, 2)
                    })
                    continue

                # Process re-ranking request
                response = self.process_request(request)
                if response is not None:
                    self._send_response(response)

            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received, shutting down...")
                break
            except Exception as e:
                logger.error(f"Unexpected error in main loop: {e}", exc_info=True)
                self._send_error(f"Server error: {str(e)}", "SERVER_ERROR")

        # Cleanup
        logger.info("Server shutting down...")
        self._log_stats()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="JadeVectorDB Re-ranking Server")
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("RERANKING_MODEL_PATH", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
        help="Cross-encoder model name or path"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=int(os.environ.get("RERANKING_BATCH_SIZE", "32")),
        help="Batch size for inference"
    )

    args = parser.parse_args()

    # Create and initialize server
    server = RerankingServer(model_name=args.model, batch_size=args.batch_size)

    # Load model
    if not server.load_model():
        logger.error("Failed to initialize server, exiting...")
        sys.exit(1)

    # Run server
    try:
        server.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
