#!/usr/bin/env python3
"""
Setup script for Comprehensive Vaping Analysis
This script helps set up the environment and verify the installation
"""

import subprocess
import sys
from pathlib import Path
import os

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    if not requirements_file.exists():
        print("❌ requirements.txt not found!")
        return False
    
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install requirements: {e}")
        return False

def check_data_files():
    """Check if required data files exist"""
    print("Checking for required data files...")
    
    parent_dir = Path(__file__).parent.parent
    required_files = ["hstexas.csv", "variable_names.csv"]
    
    missing_files = []
    for file in required_files:
        file_path = parent_dir / file
        if file_path.exists():
            file_size = file_path.stat().st_size / (1024 * 1024)  # MB
            print(f"✅ {file} found ({file_size:.1f} MB)")
        else:
            print(f"❌ {file} not found in {parent_dir}")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n⚠️  Missing files: {missing_files}")
        print("Please ensure these files are in the parent directory.")
        return False
    else:
        print("✅ All required data files found!")
        return True

def create_output_directory():
    """Create output directory if it doesn't exist"""
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    print(f"✅ Output directory ready: {output_dir}")

def run_test_suite():
    """Run the test suite"""
    print("\nRunning test suite...")
    
    test_script = Path(__file__).parent / "test_analysis.py"
    
    if not test_script.exists():
        print("❌ Test script not found!")
        return False
    
    try:
        result = subprocess.run([sys.executable, str(test_script)], 
                              capture_output=True, text=True)
        
        print("Test Output:")
        print("-" * 40)
        print(result.stdout)
        
        if result.stderr:
            print("Errors/Warnings:")
            print("-" * 40)
            print(result.stderr)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Failed to run tests: {e}")
        return False

def main():
    """Main setup function"""
    print("="*60)
    print("COMPREHENSIVE VAPING ANALYSIS - SETUP")
    print("="*60)
    
    # Step 1: Install requirements
    if not install_requirements():
        print("\n❌ Setup failed at requirements installation.")
        return
    
    # Step 2: Check data files
    if not check_data_files():
        print("\n❌ Setup failed at data file check.")
        return
    
    # Step 3: Create output directory
    create_output_directory()
    
    # Step 4: Run tests
    print("\n" + "="*60)
    print("RUNNING VERIFICATION TESTS")
    print("="*60)
    
    if run_test_suite():
        print("\n🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Run the complete analysis: python main_analysis.py")
        print("2. Or run individual phases as needed")
        print("3. Check the README.md for detailed usage instructions")
    else:
        print("\n⚠️  Setup completed but some tests failed.")
        print("Check the test output above for issues.")
        print("You may still be able to run parts of the analysis.")

if __name__ == "__main__":
    main()