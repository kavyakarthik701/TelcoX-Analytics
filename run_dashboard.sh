#!/bin/bash
# Script to launch the redesigned TelcoX Analytics Maturity Dashboard

cd "$(dirname "$0")"

# Ensure temp directory for Streamlit cache/config
export STREAMLIT_CONFIG_DIR="/tmp/streamlit_config"
mkdir -p "$STREAMLIT_CONFIG_DIR"

if [ -d "venv" ]; then
    echo "Starting redesigned TelcoX Analytics Maturity Dashboard..."
    ./venv/bin/streamlit run app.py --server.port 8501 --server.headless true --browser.gatherUsageStats false
else
    streamlit run app.py --server.port 8501 --server.headless true --browser.gatherUsageStats false
fi
