# You can name this file something like build.ps1
# Run it in PowerShell: .\build.ps1

# Stop on any error
$ErrorActionPreference = "Stop"

# Create a virtual environment in .venv
py -3.11 -m venv .venv

# Activate the virtual environment
. .\.venv\Scripts\Activate.ps1

# Upgrade pip and install requirements
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# For Windows, ahrs/utils should be located in:
#     .venv\Lib\site-packages\ahrs\utils
$AHRS_UTILS_PATH = "./.venv/Lib/site-packages/ahrs/utils"
# Confirm that directory exists
if (-not (Test-Path $AHRS_UTILS_PATH)) {
    Write-Error "Could not find 'ahrs/utils' in .venv/Lib/site-packages. Make sure AHRS is installed."
        exit 1
}

# Build with PyInstaller, adding the data folder
python -m PyInstaller --add-data "$AHRS_UTILS_PATH;ahrs/utils" app.py

# Create the tar.gz archive (requires tar in PATH)
tar -czvf dist/dist.tar.gz dist/app
