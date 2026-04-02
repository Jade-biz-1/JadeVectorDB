#!/bin/bash
# EnterpriseRAG Setup Script

set -e

echo "========================================"
echo "  EnterpriseRAG Setup"
echo "========================================"
echo ""

# Check for required tools
echo "Checking prerequisites..."

if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed"
    exit 1
fi

if ! command -v node &> /dev/null; then
    echo "❌ Node.js is required but not installed"
    exit 1
fi

echo "✅ Prerequisites OK"
echo ""

# Setup backend
echo "Setting up backend..."
cd backend

echo "  Creating Python virtual environment..."
python3 -m venv venv

echo "  Activating virtual environment..."
source venv/bin/activate

echo "  Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "  Creating .env file from template..."
if [ ! -f .env ]; then
    cp .env.example .env
    echo "  ✅ Created .env file - please edit if needed"
else
    echo "  ⚠️  .env file already exists - skipping"
fi

cd ..
echo "✅ Backend setup complete"
echo ""

# Setup frontend
echo "Setting up frontend..."
cd frontend

echo "  Installing Node dependencies..."
npm install

cd ..
echo "✅ Frontend setup complete"
echo ""

# Create uploads directory
echo "Creating uploads directory..."
mkdir -p backend/uploads
echo "✅ Uploads directory created"
echo ""

echo "========================================"
echo "  Setup Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo ""
echo "1. Review backend/.env configuration"
echo "   - For demo: MODE=mock (default, no dependencies)"
echo "   - For production: MODE=production (requires JadeVectorDB + Ollama)"
echo ""
echo "2. Start the application:"
echo "   ./scripts/start.sh"
echo ""
echo "3. Access the web interface:"
echo "   http://localhost:5173"
echo ""
echo "4. API documentation:"
echo "   http://localhost:8000/docs"
echo ""
