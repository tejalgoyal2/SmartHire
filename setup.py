#!/usr/bin/env python3
"""
Setup script for Resume-Job Matching System
Run this script to install dependencies and set up the environment
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    
    packages = [
        "pandas>=1.5.0",
        "numpy>=1.21.0",
        "scikit-learn>=1.2.0",
        "requests>=2.28.0",
        "beautifulsoup4>=4.11.0",
        "nltk>=3.8.0",
        "lxml>=4.9.0",
        "html5lib>=1.1",
        "streamlit>=1.25.0",
        "plotly>=5.15.0"
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ Installed {package}")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package}")
            return False
    
    return True

def download_nltk_data():
    """Download required NLTK data"""
    print("Downloading NLTK data...")
    
    import nltk
    
    try:
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        nltk.download('wordnet', quiet=True)
        print("✅ NLTK data downloaded successfully")
        return True
    except Exception as e:
        print(f"❌ Failed to download NLTK data: {e}")
        return False

def check_csv_file():
    """Check if resume CSV file exists"""
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    
    if not csv_files:
        print("⚠️  No CSV files found in current directory")
        print("Please download your resume dataset and place it in this folder")
        return False
    
    print(f"✅ Found CSV files: {csv_files}")
    return True

def main():
    """Main setup function"""
    print("=" * 50)
    print("Resume-Job Matching System Setup")
    print("=" * 50)
    
    # Step 1: Install requirements
    if not install_requirements():
        print("❌ Failed to install requirements. Please check your Python environment.")
        return
    
    # Step 2: Download NLTK data
    if not download_nltk_data():
        print("❌ Failed to download NLTK data. You may need to do this manually.")
        return
    
    # Step 3: Check for CSV file
    check_csv_file()
    
    print("\n" + "=" * 50)
    print("Setup Complete!")
    print("=" * 50)
    
    print("\nNext steps:")
    print("1. Make sure your resume CSV file is in the current directory")
    print("2. Run the command-line version: python resume_matcher.py")
    print("3. Or run the web version: streamlit run streamlit_app.py")
    print("\nFor the web version, open your browser and go to: http://localhost:8501")

if __name__ == "__main__":
    main()
