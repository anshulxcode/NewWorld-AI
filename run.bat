@echo off
echo =======================================================
echo   NETWORLD-AI ONE-CLICK LAUNCH SCRIPT (WINDOWS)
echo   SIH 2026 NTRO PS 26153
echo =======================================================
echo.

IF NOT EXIST venv (
    echo [INFO] Creating Python virtual environment...
    python -m venv venv
)

echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat

echo [INFO] Installing / Updating dependencies...
pip install -q -r requirements.txt

echo [INFO] Generating sample dataset if needed...
python src/generate_sample_data.py

echo [INFO] Executing Model Training...
python src/train.py

echo [INFO] Executing Model Evaluation...
python src/evaluate.py

echo [INFO] Launching Interactive Streamlit Dashboard...
streamlit run app/streamlit_app.py

pause
