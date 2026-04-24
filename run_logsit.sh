#!/bin/bash
cd /home/ytbob/projects/logsit_agent
source venv/bin/activate

# Kill any existing processes on port 7870
lsof -ti:7870 | xargs kill -9 2>/dev/null || true

# Set a free port
export GRADIO_SERVER_PORT=7870

echo "🚀 Starting Logsit Agent on port 7870..."
echo "🌐 Open: http://localhost:7870"
echo "📱 Loading page..."

# Run the application
python app/main.py