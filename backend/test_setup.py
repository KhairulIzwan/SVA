"""
Test script to verify the development environment setup
Run this after completing the setup to ensure everything works
"""

import sys
import subprocess
import importlib

def check_python_version():
    """Check if Python version is 3.11+"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 11:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - Good!")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Need 3.11+")
        return False

def check_package(package_name, import_name=None):
    """Check if a package is installed and can be imported"""
    if import_name is None:
        import_name = package_name
    
    try:
        module = importlib.import_module(import_name)
        print(f"✅ {package_name} - Installed")
        return True
    except ImportError:
        print(f"❌ {package_name} - Not found")
        return False

def main():
    print("🔍 SVA Environment Setup Verification")
    print("=" * 40)
    
    # Check Python version
    python_ok = check_python_version()
    
    # Check critical packages
    print("\n📦 Checking critical packages...")
    packages = [
        ("torch", "torch"),
        ("whisper", "whisper"), 
        ("opencv", "cv2"),
        ("ffmpeg-python", "ffmpeg"),
        ("fastapi", "fastapi"),
        ("numpy", "numpy"),
        ("pillow", "PIL"),
    ]
    
    all_packages_ok = True
    for package_name, import_name in packages:
        if not check_package(package_name, import_name):
            all_packages_ok = False
    
    print("\n" + "=" * 40)
    if python_ok and all_packages_ok:
        print("🎉 Environment setup successful!")
        print("\n📋 Next steps:")
        print("1. Add a test video to data/videos/")
        print("2. Run: python test_transcription.py")
        print("3. Start building your first AI feature!")
    else:
        print("⚠️  Some issues found. Please run:")
        print("pip install -r ../requirements.txt")
        return False
    
    return True

if __name__ == "__main__":
    main()