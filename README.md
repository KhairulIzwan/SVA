# SVA - Smart Video Assistant 🎬

A fully local AI desktop application that analyzes and queries short video files completely offline. No internet connection required!

## 🌟 Features

- **🎤 Speech-to-Text**: Extract spoken content with HuggingFace Whisper
- **👁️ Visual Text Recognition**: Detect and extract on-screen text with TrOCR
- **🎯 Object Detection**: Identify objects using DETR models
- **📊 Content Analysis**: Intelligent thematic analysis and summarization
- **📄 Report Generation**: Professional PDF, PowerPoint, and text reports
- **💻 Desktop Interface**: Modern chat-based UI built with Tauri + React
- **🔒 Fully Offline**: All processing happens locally with no internet dependency

## 📋 Requirements

### System Requirements
- **OS**: Linux, macOS, or Windows
- **RAM**: Minimum 8GB (16GB recommended for better performance)
- **Storage**: 5GB free space for models and dependencies
- **CPU**: Multi-core processor recommended for faster analysis

### Software Prerequisites
- **Python**: 3.10 or higher
- **Node.js**: 16.x or higher
- **Rust**: Latest stable version (for Tauri)
- **FFmpeg**: For video processing

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/KhairulIzwan/SVA.git
cd SVA
```

### 2. Backend Setup (Python + AI Models)
```bash
# Create Python virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Install additional report generation libraries
pip install reportlab python-pptx

# Test backend setup
cd backend
python test_setup.py
```

### 3. Frontend Setup (Tauri + React)
```bash
cd frontend

# Install Node.js dependencies
npm install

# Install Tauri CLI (if not already installed)
npm install -g @tauri-apps/cli

# Build frontend
npm run build
```

### 4. Install System Dependencies

#### Ubuntu/Debian:
```bash
sudo apt update
sudo apt install ffmpeg build-essential pkg-config libssl-dev
```

#### macOS:
```bash
brew install ffmpeg
```

#### Windows:
- Download FFmpeg from https://ffmpeg.org/download.html
- Add FFmpeg to your system PATH

## 🎯 Quick Start

### Option 1: Development Mode
```bash
# Terminal 1: Start backend server
cd backend
source ../venv/bin/activate
python grpc_server_fixed_new.py

# Terminal 2: Start frontend
cd frontend
npm run tauri dev
```

### Option 2: Production Build
```bash
# Build the complete application
cd frontend
npm run tauri build

# Run the built application
# The executable will be in src-tauri/target/release/
```

## 📖 Usage Guide

### 1. **Upload Video**
- Click "Upload Video" in the chat interface
- Select a video file (MP4, AVI, MOV supported)
- Maximum recommended duration: 2-3 minutes for optimal performance

### 2. **Ask Questions**
Use natural language to query your video:

- **"Extract all text from this video"** - Gets spoken and visual text
- **"What objects can you see?"** - Detects and lists objects
- **"Analyze this presentation"** - Comprehensive analysis
- **"Transcribe the speech"** - Audio-to-text conversion only
- **"What's written on screen?"** - Visual text recognition only

### 3. **Generate Reports**
After analysis, use the report buttons:
- **📄 PDF Report**: Professional formatted report
- **📊 PowerPoint**: Presentation-ready slides
- **📝 Text Report**: Simple text summary

### 4. **View Results**
- Analysis appears in real-time in the chat
- Reports are saved to `backend/generated_reports/`
- All conversations are preserved in `backend/chat_storage/`

## 📚 Examples & Sample Outputs

### 📹 **Example Input Files**

We provide sample videos to help you test SVA's capabilities:

#### 🎥 **Sample Test Video**
![SVA Analysis Demo](sva_demo.gif)

**Included Test Files:**
```bash
data/videos/
├── test_video.mp4          # Main demo video (30s business presentation)
└── sample_video           # Additional test content
```

### 💬 **Sample Queries & Responses**

![SVA Chat Interface](https://github.com/KhairulIzwan/SVA/blob/main/Screenshot%202025-10-17%20093237.png)

*Example of SVA's chat interface showing real video analysis results*

### 📄 **Generated Report Examples**

![SVA Generated Reports](https://github.com/KhairulIzwan/SVA/blob/main/Screenshot%202025-10-17%20093905.png)

*Example of SVA's report generation interface and sample output formats*

## 📊 Project Status & Summary

### ✅ **What Works Well**

#### **Core Functionality**
- **🎤 Speech Recognition**: Excellent accuracy with HuggingFace Whisper models
- **👁️ Object Detection**: Reliable identification of people, objects, and scenes
- **📖 Text Extraction**: Good performance on clear, readable on-screen text
- **💬 Chat Interface**: Smooth, responsive desktop application experience
- **📄 Report Generation**: Professional PDF, PowerPoint, and text outputs

#### **Technical Achievements**
- **🔒 100% Offline Operation**: No internet dependency after initial setup
- **🚀 Local AI Models**: All processing using HuggingFace transformers
- **📱 Modern UI**: Tauri + React providing native desktop experience
- **🗃️ Data Persistence**: Chat history and analysis results properly stored
- **⚡ Real-time Processing**: Live analysis feedback in chat interface

#### **User Experience**
- **🎯 Natural Language Queries**: Intuitive conversation-based interaction
- **📊 Professional Reports**: Export-ready documents for sharing
- **🔄 Multi-format Support**: PDF, PowerPoint, and text report options
- **📁 File Management**: Organized storage of videos, chats, and reports

### ⚠️ **Known Limitations**

#### **Content Recognition Challenges**
- **📝 Text Recognition**: Struggles with handwritten text, stylized fonts, or low-resolution content
- **🎤 Audio Quality**: Performance degrades with background noise or multiple speakers
- **🌐 Language Support**: Primarily optimized for English content
- **📏 Video Length**: Processing time increases significantly with longer videos (>3 minutes)

#### **Technical Constraints**
- **🖥️ Resource Usage**: High RAM consumption during analysis (8GB+ recommended)
- **⏱️ Processing Speed**: Analysis can take 30-90 seconds depending on video complexity
- **📦 Model Size**: Initial download requires ~2GB for all AI models
- **🔧 Setup Complexity**: Multiple dependencies (Python, Node.js, Rust, FFmpeg)

#### **User Interface Areas**
- **📋 Report Context**: Generated reports include all analysis data, not just user-requested content
- **🔄 Session Management**: Limited ability to separate different video analyses in same chat
- **📂 File Upload**: Original video filenames not always preserved in processing pipeline
- **🎯 Query Specificity**: Reports don't filter content based on specific user requests

### 🚧 **Encountered Challenges**

#### **Integration Complexity**
- **🔗 Multi-technology Stack**: Coordinating Python backend, Rust Tauri, and React frontend
- **📡 gRPC Communication**: Managing protocol buffers and cross-language communication
- **🏗️ Build Process**: Complex build pipeline with multiple compilation steps
- **🔧 Dependency Management**: Handling conflicts between Python, Node.js, and system libraries

#### **AI Model Challenges**
- **📥 Model Loading**: Large model files causing slow initial startup
- **🎯 Confidence Calibration**: Balancing detection sensitivity vs. false positives
- **🔀 Multi-modal Fusion**: Combining results from different AI models effectively
- **⚡ Performance Optimization**: Balancing accuracy with processing speed

#### **Data Flow Issues**
- **🆔 Session Tracking**: Maintaining context across multiple user interactions
- **📊 Content Extraction**: Parsing complex analysis results for report generation
- **🎬 Video Processing**: Handling various video formats and resolutions consistently
- **💾 State Management**: Synchronizing frontend state with backend processing

### 🔮 **Potential Improvements**

#### **Short-term Enhancements (1-2 weeks)**
- **🎯 Context-Aware Reports**: Generate reports based on specific user queries only
- **📂 Better File Management**: Preserve original video filenames throughout processing
- **🔄 Session Separation**: Clear boundaries between different video analysis sessions
- **⚡ Performance Optimization**: Implement model caching and parallel processing

#### **Medium-term Features (1-2 months)**
- **🌐 Multi-language Support**: Expand beyond English for global users
- **📊 Advanced Analytics**: Sentiment analysis, topic modeling, and content categorization
- **🎨 UI/UX Improvements**: Drag-and-drop uploads, progress indicators, batch processing
- **📱 Export Options**: Email integration, cloud save, custom report templates

#### **Long-term Vision (3-6 months)**
- **🧠 Advanced AI Models**: Integration with newer, more capable models
- **🎬 Video Editing Integration**: Basic editing capabilities within the application
- **📈 Analytics Dashboard**: Usage patterns, accuracy metrics, performance insights
- **🔌 Plugin System**: Extensible architecture for custom analysis modules

### 🎯 **What Could Be Achieved with More Time**

#### **With Additional 2-4 Weeks**

## 🔧 Configuration

### Model Configuration
Models are automatically downloaded on first use:
- **Speech Recognition**: `openai/whisper-small`
- **Visual Text**: `microsoft/trocr-base-printed`
- **Object Detection**: `facebook/detr-resnet-50`

### Storage Locations
```
SVA/
├── backend/
│   ├── chat_storage/          # Chat conversations
│   ├── generated_reports/     # PDF/PPT reports
│   ├── uploaded_videos/       # Processed videos
│   └── models/               # Downloaded AI models
└── frontend/
    └── src-tauri/target/     # Built application
```

## 🧪 Testing

### Run Backend Tests
```bash
cd backend
source ../venv/bin/activate

# Test individual components
python test_whisper.py          # Speech recognition
python vision_ai_test.py        # Object detection  
python test_transcription.py    # Text extraction
python comprehensive_test.py    # Full pipeline

# Test report generation
python -c "from mcp_servers.report_server import ReportGenerationServer; server = ReportGenerationServer(); print('✅ Report server ready')"
```

### Frontend Testing
```bash
cd frontend
npm test                        # Run React tests
npm run tauri build --debug    # Test build process
```

## 🐛 Troubleshooting

### Common Issues

**1. "Model download failed"**
```bash
# Clear model cache and retry
rm -rf ~/.cache/huggingface/
python test_whisper.py
```

**2. "FFmpeg not found"**
```bash
# Verify FFmpeg installation
ffmpeg -version

# On Ubuntu/Debian
sudo apt install ffmpeg

# On macOS
brew install ffmpeg
```

**3. "Permission denied on model files"**
```bash
# Fix model directory permissions
chmod -R 755 backend/models/
```

**4. "Tauri build fails"**
```bash
# Update Rust and Tauri
rustup update
npm install -g @tauri-apps/cli@latest
```

**5. "Reports not generating"**
```bash
# Check if in virtual environment
source venv/bin/activate
pip install reportlab python-pptx
```

### Performance Optimization

**For better performance:**
- Use shorter videos (< 2 minutes)
- Close other applications during analysis
- Ensure sufficient RAM (8GB minimum)
- Use SSD storage for faster model loading

### Debug Mode
```bash
# Enable detailed logging
export RUST_LOG=debug
export PYTHONPATH=$PYTHONPATH:/home/user/SVA/backend

# Run with debug output
npm run tauri dev
```

## 📁 Project Structure

```
SVA/
├── README.md                    # This file
├── requirements.txt            # Python dependencies
├── PROJECT_OVERVIEW.md         # Detailed project documentation
├── backend/                    # Python AI backend
│   ├── mcp_servers/           # MCP protocol servers
│   ├── grpc_server_fixed_new.py  # Main gRPC server
│   ├── simple_video_analyzer.py  # Core analysis engine
│   └── test_*.py              # Test scripts
├── frontend/                   # Tauri + React frontend
│   ├── src/                   # React components
│   ├── src-tauri/            # Tauri configuration
│   └── package.json          # Node.js dependencies
└── scripts/                   # Setup and utility scripts
    └── setup.ps1             # Windows setup script
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test thoroughly
4. Commit: `git commit -am 'Add new feature'`
5. Push: `git push origin feature-name`
6. Create a Pull Request

## 📝 License

This project is open source. See LICENSE file for details.

## 🆘 Support

- **Issues**: Report bugs on GitHub Issues
- **Documentation**: Check PROJECT_OVERVIEW.md for technical details
- **Performance**: See troubleshooting section above

---

**Built with ❤️ using HuggingFace Transformers, Tauri, and React**

