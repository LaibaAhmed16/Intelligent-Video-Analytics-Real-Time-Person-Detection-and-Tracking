
# 🔧 Installation Guide — YOLOv8 + DeepSORT Tracker

This guide explains how to set up and run the YOLOv8 + DeepSORT person tracking web app.

---


✅ 1. Set Up Python Environment

Recommended: Python 3.9+

Using venv
python -m venv venv
source venv/bin/activate      # On Linux/Mac
venv\Scripts\activate         # On Windows

Or using Conda
conda create -n tracker python=3.9 -y
conda activate tracker

✅ 2. Install Dependencies
pip install --upgrade pip
pip install -r requirements.txt

✅ 3. (Optional) Install GPU Support for PyTorch

👉 Find the right command here: https://pytorch.org/get-started/locally/

Example for CUDA 11.8:

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

✅ 4. Run the Application
streamlit run app.py

Open the local URL (e.g. http://localhost:8501) in your browser.
