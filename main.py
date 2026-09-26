"""
Main entry point for running the Audio Hit Predictor Streamlit application.
Execute with:
    streamlit run app.py
or:
    python main.py
"""
import sys
import subprocess

if __name__ == "__main__":
    print("Launching Audio Hit Predictor Streamlit application...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
