#!/usr/bin/env bash
echo "======================================================="
echo "  NETWORLD-AI ONE-CLICK LAUNCH SCRIPT (LINUX/MACOS)"
echo "  SIH 2026 NTRO PS 26153"
echo "======================================================="
echo ""

if [ ! -d "venv" ]; then
    echo "[INFO] Creating Python virtual environment..."
    python3 -m venv venv
fi

echo "[INFO] Activating virtual environment..."
source venv/bin/activate

echo "[INFO] Installing / Updating dependencies..."
pip install -q -r requirements.txt

echo "[INFO] Generating sample dataset..."
python src/generate_sample_data.py

echo "[INFO] Executing Model Training..."
python src/train.py

echo "[INFO] Executing Model Evaluation..."
python src/evaluate.py

echo "[INFO] Launching Interactive Streamlit Dashboard..."
streamlit run app/streamlit_app.py
