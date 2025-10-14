# SVA Backend Structure

## Core Modules

### 1. Video Processing (`video_processor.py`)
- Frame extraction
- Audio extraction  
- Video metadata parsing

### 2. AI Models (`models/`)
- `whisper_transcriber.py` - Speech-to-text
- `clip_analyzer.py` - Visual understanding
- `object_detector.py` - YOLO object detection

### 3. MCP Servers (`mcp_servers/`)
- `transcription_server.py`
- `vision_server.py` 
- `generation_server.py`
- `router_server.py`

### 4. API Layer (`api/`)
- `grpc_server.py` - Main gRPC service
- `handlers/` - Request handlers
- `schemas/` - Data models

### 5. Database (`database/`)
- `models.py` - SQLAlchemy models
- `crud.py` - Database operations
- `connection.py` - DB setup

### 6. Utils (`utils/`)
- `config.py` - Configuration management
- `logger.py` - Logging setup
- `helpers.py` - Common utilities