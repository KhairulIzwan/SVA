"""
Transcription MCP Server - Core component for SVA project
Implements Model Context Protocol for speech-to-text processing
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional, List
import whisper
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TranscriptionMCPServer:
    """MCP Server for speech-to-text processing with enhanced Malay support"""
    
    def __init__(self, model_size: str = "small"):
        self.server_name = "transcription"
        self.model_size = model_size
        self.model = None
        self.capabilities = [
            "transcribe_video",
            "extract_timestamps", 
            "detect_language",
            "analyze_audio_quality",
            "batch_transcribe"
        ]
        self.supported_languages = ["ms", "id", "en", "auto"]
        
    async def initialize(self):
        """Initialize Whisper model"""
        logger.info(f"🤖 Loading Whisper {self.model_size} model...")
        try:
            self.model = whisper.load_model(self.model_size)
            logger.info("✅ Transcription MCP server ready")
            return {"status": "success", "model": self.model_size}
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")
            return {"status": "error", "error": str(e)}
    
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Main MCP request handler"""
        action = request.get("action")
        request_id = request.get("request_id", f"req_{datetime.now().isoformat()}")
        
        logger.info(f"📝 Processing request {request_id}: {action}")
        
        try:
            if action == "transcribe_video":
                return await self._transcribe_video(request)
            elif action == "extract_timestamps":
                return await self._extract_timestamps(request)
            elif action == "detect_language":
                return await self._detect_language(request)
            elif action == "analyze_audio_quality":
                return await self._analyze_audio_quality(request)
            elif action == "get_capabilities":
                return await self._get_capabilities()
            else:
                return {
                    "request_id": request_id,
                    "status": "error",
                    "error": f"Unknown action: {action}",
                    "available_actions": self.capabilities
                }
        except Exception as e:
            logger.error(f"❌ Request processing failed: {e}")
            return {
                "request_id": request_id,
                "status": "error", 
                "error": str(e)
            }
    
    async def _transcribe_video(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced video transcription with Malay language support"""
        video_path = request.get("video_path")
        options = request.get("options", {})
        request_id = request.get("request_id", "unknown")
        
        if not video_path or not Path(video_path).exists():
            return {
                "request_id": request_id,
                "status": "error",
                "error": f"Video file not found: {video_path}"
            }
        
        try:
            # Enhanced transcription options
            transcribe_options = {
                "language": options.get("language"),  # None for auto-detect
                "task": options.get("task", "transcribe"),
                "word_timestamps": options.get("word_timestamps", True),
                "verbose": False,
                "fp16": False,
                "initial_prompt": "This is a clear speech recording in Malay or Indonesian language."
            }
            
            logger.info(f"🎤 Transcribing {video_path} with options: {transcribe_options}")
            
            result = self.model.transcribe(video_path, **transcribe_options)
            
            # Calculate confidence score
            confidence = self._calculate_confidence(result.get("text", ""), result.get("language", ""))
            
            # Prepare structured response
            response = {
                "request_id": request_id,
                "status": "success",
                "data": {
                    "text": result["text"].strip(),
                    "language": result["language"],
                    "confidence_score": confidence,
                    "duration": result.get("duration", 0),
                    "word_count": len(result["text"].split()),
                    "segments": result.get("segments", []),
                    "model_used": self.model_size,
                    "transcription_options": transcribe_options,
                    "timestamp": datetime.now().isoformat()
                }
            }
            
            logger.info(f"✅ Transcription completed: {result['language']} ({confidence:.2f} confidence)")
            return response
            
        except Exception as e:
            logger.error(f"❌ Transcription failed: {e}")
            return {
                "request_id": request_id,
                "status": "error",
                "error": str(e)
            }
    
    async def _extract_timestamps(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Extract detailed timestamp information from transcription"""
        video_path = request.get("video_path")
        request_id = request.get("request_id", "unknown")
        
        try:
            result = self.model.transcribe(
                video_path,
                word_timestamps=True,
                verbose=False
            )
            
            # Extract word-level timestamps
            word_timestamps = []
            for segment in result.get("segments", []):
                for word in segment.get("words", []):
                    word_timestamps.append({
                        "word": word.get("word", "").strip(),
                        "start": word.get("start", 0),
                        "end": word.get("end", 0),
                        "probability": word.get("probability", 0)
                    })
            
            return {
                "request_id": request_id,
                "status": "success",
                "data": {
                    "segments": result.get("segments", []),
                    "word_timestamps": word_timestamps,
                    "total_duration": result.get("duration", 0)
                }
            }
            
        except Exception as e:
            return {
                "request_id": request_id,
                "status": "error",
                "error": str(e)
            }
    
    async def _detect_language(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Detect language from audio with confidence scores"""
        video_path = request.get("video_path")
        request_id = request.get("request_id", "unknown")
        
        try:
            # Use a small sample for language detection
            audio = whisper.load_audio(video_path)
            audio = whisper.pad_or_trim(audio)
            
            # Make log-Mel spectrogram and move to the same device as the model
            mel = whisper.log_mel_spectrogram(audio).to(self.model.device)
            
            # Detect the spoken language
            _, probs = self.model.detect_language(mel)
            
            # Get top 3 language predictions
            top_languages = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
            
            return {
                "request_id": request_id,
                "status": "success",
                "data": {
                    "detected_language": max(probs, key=probs.get),
                    "confidence": max(probs.values()),
                    "top_predictions": [
                        {"language": lang, "confidence": conf}
                        for lang, conf in top_languages
                    ]
                }
            }
            
        except Exception as e:
            return {
                "request_id": request_id,
                "status": "error",
                "error": str(e)
            }
    
    async def _analyze_audio_quality(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze audio quality of the video file"""
        video_path = request.get("video_path")
        request_id = request.get("request_id", "unknown")
        
        try:
            import cv2
            
            # Basic video analysis
            cap = cv2.VideoCapture(str(video_path))
            
            if not cap.isOpened():
                return {
                    "request_id": request_id,
                    "status": "error",
                    "error": "Cannot open video file"
                }
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            duration = frame_count / fps if fps > 0 else 0
            
            cap.release()
            
            # Try to get audio stream info
            audio_info = {"has_audio": "unknown", "sample_rate": "unknown"}
            try:
                import subprocess
                cmd = ['ffprobe', '-v', 'quiet', '-show_streams', '-select_streams', 'a', str(video_path)]
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0 and result.stdout:
                    audio_info["has_audio"] = True
                    if 'sample_rate' in result.stdout:
                        audio_info["sample_rate"] = "detected"
                else:
                    audio_info["has_audio"] = False
                    
            except (subprocess.TimeoutExpired, FileNotFoundError):
                audio_info["has_audio"] = "ffprobe_unavailable"
            
            return {
                "request_id": request_id,
                "status": "success",
                "data": {
                    "video_duration": duration,
                    "fps": fps,
                    "frame_count": frame_count,
                    "audio_info": audio_info,
                    "file_size": Path(video_path).stat().st_size,
                    "recommended_model": "small" if duration > 30 else "base"
                }
            }
            
        except Exception as e:
            return {
                "request_id": request_id,
                "status": "error",
                "error": str(e)
            }
    
    async def _get_capabilities(self) -> Dict[str, Any]:
        """Return server capabilities"""
        return {
            "status": "success",
            "data": {
                "server_name": self.server_name,
                "model_size": self.model_size,
                "capabilities": self.capabilities,
                "supported_languages": self.supported_languages,
                "model_loaded": self.model is not None
            }
        }
    
    def _calculate_confidence(self, text: str, language: str) -> float:
        """Calculate confidence score for transcription quality"""
        if not text or not text.strip():
            return 0.0
        
        score = 0.5
        
        # Penalize very short text
        if len(text) < 10:
            score -= 0.3
        
        # Calculate word uniqueness ratio
        words = text.lower().split()
        unique_words = set(words)
        if len(words) > 0:
            repetition_ratio = len(unique_words) / len(words)
            score += repetition_ratio * 0.3
        
        # Bonus for reasonable text length
        if 50 <= len(text) <= 1000:
            score += 0.2
        
        # Language-specific adjustments
        if language in ['ms', 'id']:  # Malay/Indonesian
            score += 0.1
        elif language == 'en':
            score += 0.05
        
        # Check for common Malay words
        malay_words = ['kemerdekan', 'rakyat', 'nasi', 'malaysia', 'dengan', 'untuk', 'dalam']
        if any(word in text.lower() for word in malay_words):
            score += 0.1
        
        return max(0.0, min(1.0, score))

# Test functions
async def test_transcription_mcp_server():
    """Test the enhanced transcription MCP server"""
    print("🧪 Testing Enhanced Transcription MCP Server")
    print("=" * 50)
    
    # Initialize server
    server = TranscriptionMCPServer("small")
    init_result = await server.initialize()
    print(f"Initialization: {init_result}")
    
    if init_result["status"] != "success":
        print("❌ Server initialization failed!")
        return
    
    # Test video path
    video_path = "data/videos/test_video.mp4"
    
    # Test 1: Get capabilities
    print("\n🔍 Test 1: Get Capabilities")
    request = {"action": "get_capabilities"}
    result = await server.process_request(request)
    print(f"Capabilities: {json.dumps(result, indent=2)}")
    
    # Test 2: Analyze audio quality
    print("\n🔊 Test 2: Analyze Audio Quality")
    request = {
        "action": "analyze_audio_quality",
        "video_path": video_path
    }
    result = await server.process_request(request)
    print(f"Audio Quality: {json.dumps(result, indent=2)}")
    
    # Test 3: Detect language
    print("\n🌍 Test 3: Detect Language")
    request = {
        "action": "detect_language",
        "video_path": video_path
    }
    result = await server.process_request(request)
    print(f"Language Detection: {json.dumps(result, indent=2)}")
    
    # Test 4: Transcribe video (auto-detect)
    print("\n📝 Test 4: Transcribe Video (Auto-detect)")
    request = {
        "action": "transcribe_video",
        "video_path": video_path,
        "options": {
            "language": None,  # Auto-detect
            "word_timestamps": True
        }
    }
    result = await server.process_request(request)
    if result["status"] == "success":
        data = result["data"]
        print(f"Language: {data['language']}")
        print(f"Confidence: {data['confidence_score']:.2f}")
        print(f"Text: {data['text'][:200]}...")
        print(f"Duration: {data['duration']:.2f}s")
        print(f"Word count: {data['word_count']}")
    else:
        print(f"Error: {result['error']}")
    
    # Test 5: Transcribe with forced Malay
    print("\n📝 Test 5: Transcribe Video (Force Malay)")
    request = {
        "action": "transcribe_video",
        "video_path": video_path,
        "options": {
            "language": "ms",  # Force Malay
            "word_timestamps": True
        }
    }
    result = await server.process_request(request)
    if result["status"] == "success":
        data = result["data"]
        print(f"Language: {data['language']}")
        print(f"Confidence: {data['confidence_score']:.2f}")
        print(f"Text: {data['text'][:200]}...")
    else:
        print(f"Error: {result['error']}")
    
    print("\n✅ MCP Server testing completed!")

if __name__ == "__main__":
    asyncio.run(test_transcription_mcp_server())