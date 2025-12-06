import subprocess
import sys
import os

def run_command(command, description):
    """
    Executes a shell command and handles potential errors.
    """
    print(f"\n[INFO] Starting: {description}...")
    try:
        subprocess.check_call(command)
        print(f"[SUCCESS] Completed: {description}")
    except subprocess.CalledProcessError:
        print(f"\n[ERROR] Failed during: {description}")
        print("Please check your internet connection or python environment.")
        sys.exit(1)

def main():
    print("===================================================")
    print("   PASSENGER ATTRIBUTE RECOGNITION - SETUP TOOL    ")
    print("===================================================")
    
    # Step 1: Install base dependencies from requirements.txt
    # This ensures tools like 'numpy', 'cython', and 'gdown' are ready.
    run_command(
        [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
        "Installing base dependencies from requirements.txt"
    )

    # Step 2: Install the ReID library (Torchreid) from source
    # We use '--no-build-isolation' to force pip to use the numpy/cython we just installed.
    # This fixes the common 'gdown' or 'numpy' missing errors during build.
    print("\n[INFO] Installing Deep-Person-Reid (Torchreid) engine...")
    print("       Note: This might take a few minutes depending on your internet speed.")
    
    torchreid_url = "git+https://github.com/KaiyangZhou/deep-person-reid.git"
    
    run_command(
        [sys.executable, "-m", "pip", "install", "--no-build-isolation", torchreid_url],
        "Installing Torchreid with no-build-isolation"
    )

    print("\n===================================================")
    print("   SETUP COMPLETED SUCCESSFULLY! 🚀")
    print("===================================================")
    print("You can now run the pipeline using: python main.py")

if __name__ == "__main__":
    main()