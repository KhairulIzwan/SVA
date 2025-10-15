#!/bin/bash

# Setup script for SVA project (Linux)
echo "🚀 Setting up SVA (Smart Video Assistant) project..."

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Check Python installation
echo -e "${YELLOW}Checking Python installation...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python3 not found! Please install Python 3.11+ first.${NC}"
    exit 1
fi

python3 --version
echo -e "${GREEN}✅ Python found${NC}"

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo -e "${RED}❌ requirements.txt not found! Make sure you're in the SVA project root.${NC}"
    exit 1
fi

# Virtual environment setup (required for Ubuntu 24.04+)
echo -e "${YELLOW}Setting up virtual environment...${NC}"
if [ -d "venv" ]; then
    echo -e "${GREEN}✅ Virtual environment already exists${NC}"
else
    echo -e "${YELLOW}Creating Python virtual environment...${NC}"
    python3 -m venv venv
    echo -e "${GREEN}✅ Virtual environment created${NC}"
fi

# Activate virtual environment
echo -e "${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate
echo -e "${GREEN}✅ Virtual environment activated${NC}"

# Upgrade pip
echo -e "${YELLOW}Installing pip if needed...${NC}"
if ! command -v pip3 &> /dev/null; then
    sudo apt-get install -y python3-pip
fi

echo -e "${YELLOW}Upgrading pip...${NC}"
python3 -m pip install --upgrade pip

# Install system dependencies for audio/video processing
echo -e "${YELLOW}Checking system dependencies...${NC}"
if command -v apt-get &> /dev/null; then
    echo -e "${YELLOW}Installing system dependencies (Ubuntu/Debian)...${NC}"
    sudo apt-get update
    sudo apt-get install -y ffmpeg libsndfile1 libgl1-mesa-dev python3-dev
elif command -v yum &> /dev/null; then
    echo -e "${YELLOW}Installing system dependencies (RHEL/CentOS)...${NC}"
    sudo yum install -y ffmpeg libsndfile mesa-libGL python3-devel
else
    echo -e "${YELLOW}⚠️  Please install ffmpeg and libsndfile manually${NC}"
fi

# Install Python dependencies
echo -e "${YELLOW}Installing Python dependencies...${NC}"
pip install -r requirements.txt

echo -e "${GREEN}✅ Dependencies installation complete!${NC}"

# Create directory structure
echo -e "${YELLOW}Creating project directory structure...${NC}"
mkdir -p data/videos
mkdir -p data/models
mkdir -p logs
mkdir -p backend/mcp_servers
mkdir -p backend/models
mkdir -p frontend
mkdir -p scripts

echo -e "${GREEN}✅ Directory structure created${NC}"

# Download initial models (optional)
echo -e "${YELLOW}Would you like to download initial AI models? (y/n)${NC}"
read -r download_models

if [ "$download_models" = "y" ] || [ "$download_models" = "Y" ]; then
    echo -e "${YELLOW}Downloading Whisper base model...${NC}"
    python -c "import whisper; whisper.load_model('base')"
    echo -e "${GREEN}✅ Whisper model downloaded${NC}"
fi

# Test setup
echo -e "${YELLOW}Testing setup...${NC}"
python3 -c "
import sys
print(f'Python version: {sys.version}')
try:
    import torch
    print(f'PyTorch version: {torch.__version__}')
    import cv2
    print(f'OpenCV version: {cv2.__version__}')
    import whisper
    print('Whisper: ✅')
    import transformers
    print(f'Transformers version: {transformers.__version__}')
    print('🎉 All core dependencies working!')
except ImportError as e:
    print(f'❌ Import error: {e}')
"

echo -e "${GREEN}🎉 Project setup complete! Next steps:${NC}"
echo -e "${CYAN}1. Add a test video to data/videos/ folder${NC}"
echo -e "${CYAN}2. Test with: python backend/test_setup.py${NC}"
echo -e "${CYAN}3. Start building MCP servers${NC}"
echo -e "${CYAN}4. Virtual environment is already activated!${NC}"