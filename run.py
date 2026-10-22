"""Quickstart Launcher: verifies files and runs SpotON immediately."""
import os, sys, subprocess

if __name__ == "__main__":
    if not os.path.exists("data/Picture1.png"):
        print("Generating baseline camera frame...")
        subprocess.run([sys.executable, "make_sample.py"], check=True)
    
    print("\n=======================================================")
    print("Launching SpotON Smart Multi-Zone Parking Dashboard")
    print("=======================================================\n")
    subprocess.run([sys.executable, "app.py"])
