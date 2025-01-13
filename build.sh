#!/bin/bash
set -e

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
# Dynamically find the Python version directory inside .venv/lib
PYTHON_LIB_PATH=$(find .venv/lib -type d -name "python3.*" -print -quit)
AHRS_UTILS_PATH="$PYTHON_LIB_PATH/site-packages/ahrs/utils"
python -m PyInstaller --add-data "$AHRS_UTILS_PATH:ahrs/utils" --onefile app.py
tar -czvf dist/dist.tar.gz dist/main
