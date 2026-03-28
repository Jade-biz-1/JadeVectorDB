#!/bin/bash
# EnterpriseRAG Start Script

set -e

echo "========================================"
echo "  Starting EnterpriseRAG"
echo "========================================"
echo ""

# Check if setup was run
if [ ! -d "backend/venv" ]; then
    echo "❌ Backend not set up. Please run ./scripts/setup.sh first"
    exit 1
fi

if [ ! -d "frontend/node_modules" ]; then
    echo "❌ Frontend not set up. Please run ./scripts/setup.sh first"
    exit 1
fi

# Load environment
if [ -f backend/.env ]; then
    export $(cat backend/.env | grep -v '^#' | xargs)
fi

MODE=${MODE:-mock}
echo "Starting in $MODE mode..."
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Shutting down..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null || true
    exit 0
}

trap cleanup SIGINT SIGTERM

# Start backend
echo "Starting backend..."
cd backend
source venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
cd ..

echo "✅ Backend started (PID: $BACKEND_PID)"
echo "   API: http://localhost:8000"
echo "   Docs: http://localhost:8000/docs"
echo ""

# Wait for backend to be ready
echo "Waiting for backend to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:8000/api/health > /dev/null 2>&1; then
        echo "✅ Backend is ready"
        break
    fi
    sleep 1
done
echo ""

# Start frontend
echo "Starting frontend..."
cd frontend
npm run dev &
FRONTEND_PID=$!
cd ..

echo "✅ Frontend started (PID: $FRONTEND_PID)"
echo "   UI: http://localhost:5173"
echo ""

echo "========================================"
echo "  EnterpriseRAG is running!"
echo "========================================"
echo ""
echo "Mode: $MODE"
echo "Web Interface: http://localhost:5173"
echo "API Documentation: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Wait for processes
wait $BACKEND_PID $FRONTEND_PID
