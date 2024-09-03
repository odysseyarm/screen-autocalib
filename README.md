
# Ody-Autocalib

Ody-Autocalib is a Python-based program that requires Python 3.11. This guide will walk you through the steps to download and install Python 3.11, set up a virtual environment, install the necessary dependencies, and run the program.

## Prerequisites

- Python 3.11 installed (see instructions below)
- Git (optional, for cloning the repository)

## Installing Python 3.11

### Windows

1. **Download Python 3.11:**
   - Visit the official Python website: [Python 3.11 Downloads](https://www.python.org/downloads/release/python-3119/)
   - Download the appropriate installer for your operating system.

2. **Install Python 3.11:**
   - Run the installer and follow the on-screen instructions.
   - Make sure to check the box that says "Add Python to PATH" if this is your first python installation.
   - Verify the installation by opening a command prompt or PowerShell and typing:
     ```sh
     py -3.11 --version
     ```
     This should display the installed version of Python 3.11.

### Linux

1. **Update package list:**
   ```sh
   sudo apt update
   ```

2. **Install prerequisites:**
   ```sh
   sudo apt install -y software-properties-common
   ```

3. **Add the deadsnakes PPA (if using Ubuntu/Debian):**
   ```sh
   sudo add-apt-repository ppa:deadsnakes/ppa
   ```

4. **Install Python 3.11:**
   ```sh
   sudo apt install -y python3.11 python3.11-venv python3.11-dev
   ```

5. **Verify the installation:**
   ```sh
   python3.11 --version
   ```

## Setting Up a Virtual Environment

To avoid conflicts with other Python projects, it's recommended to use a virtual environment.

### Windows

1. **Navigate to your project directory:**
   ```sh
   cd path\to\ody-autocalib
   ```

2. **Create a virtual environment:**
   Use the following command to create a virtual environment in your project directory:
   ```sh
   py -3.11 -m venv .venv
   ```

3. **Activate the virtual environment:**

   - **For PowerShell:**
     ```sh
     .\.venv\Scripts\Activate.ps1
     ```

   - **For Command Prompt:**
     ```sh
     .\.venv\Scripts\Activate.bat
     ```

   Once activated, your command prompt should show that you're working within the `.venv` environment.

### Linux

1. **Navigate to your project directory:**
   ```sh
   cd path/to/ody-autocalib
   ```

2. **Create a virtual environment:**
   Use the following command to create a virtual environment in your project directory:
   ```sh
   python3.11 -m venv .venv
   ```

3. **Activate the virtual environment:**
   ```sh
   source .venv/bin/activate
   ```

   Once activated, your terminal prompt should show that you're working within the `.venv` environment.

## Installing Dependencies

1. **Install the required packages:**
   With the virtual environment activated, install the dependencies listed in the `requirements.txt` file:
   ```sh
   pip install -r requirements.txt
   ```

## Running the Program

1. **Run the application:**
   After installing the dependencies, you can run the `app.py` script:
   ```sh
   python app.py
   ```

## Deactivating the Virtual Environment

When you're done, you can deactivate the virtual environment by simply typing:
```sh
deactivate
```

## Troubleshooting

- If you encounter issues with permissions while activating the virtual environment in PowerShell on Windows, you may need to change the execution policy:
  ```sh
  Set-ExecutionPolicy RemoteSigned -Scope Process
  ```

- Ensure that you are using Python 3.11 by explicitly invoking `py -3.11` on Windows or `python3.11` on Linux.
