#!/usr/bin/env python3
"""
Comprehensive SVA Video Analysis Test
Integrates all AI processing components for complete video analysis
"""

import os
import sys
import json
import time
import argparse
import tempfile
from pathlib import Path
import uuid
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ComprehensiveVideoAnalyzer:
    def __init__(self, session_id=None):
        self.session_id = session_id or str(uuid.uuid4())
        self.backend_path = Path(__file__).parent
        self.results = {
            "session_id": self.session_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "analysis_stages": {},
            "performance_metrics": {},
            "overall_success": False,
            "errors": []
        }
        
    def log_stage(self, stage_name, success, data=None, error=None, duration=None):
        """Log analysis stage results"""
        self.results["analysis_stages"][stage_name] = {
            "success": success,
            "duration_seconds": duration,
            "data": data,
            "error": error,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        if error:
            self.results["errors"].append({
                "stage": stage_name,
                "error": error
            })
    
    def validate_video_file(self, video_path):
        """Validate video file before processing"""
        stage_start = time.time()
        
        try:
            import cv2
            
            if not Path(video_path).exists():
                raise FileNotFoundError(f"Video file not found: {video_path}")
            
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                raise ValueError(f"Cannot open video file: {video_path}")
            
            # Get video properties
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0
            
            cap.release()
            
            video_info = {
                "path": str(video_path),
                "resolution": f"{width}x{height}",
                "fps": fps,
                "duration_seconds": duration,
                "frame_count": frame_count,
                "file_size_mb": Path(video_path).stat().st_size / (1024 * 1024)
            }
            
            stage_duration = time.time() - stage_start
            self.log_stage("video_validation", True, video_info, duration=stage_duration)
            
            logger.info(f"✅ Video validation successful: {video_info['resolution']}, {duration:.1f}s")
            return video_info
            
        except Exception as e:
            stage_duration = time.time() - stage_start
            self.log_stage("video_validation", False, error=str(e), duration=stage_duration)
            logger.error(f"❌ Video validation failed: {e}")
            raise
    
    def extract_audio(self, video_path):
        """Extract audio from video for transcription"""
        stage_start = time.time()
        
        try:
            import subprocess
            
            # Create temporary audio file
            temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_audio.close()
            
            # Extract audio using ffmpeg
            cmd = [
                'ffmpeg', '-i', str(video_path),
                '-ac', '1',  # mono
                '-ar', '16000',  # 16kHz sample rate
                '-y',  # overwrite output
                temp_audio.name
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise RuntimeError(f"FFmpeg failed: {result.stderr}")
            
            audio_info = {
                "temp_audio_path": temp_audio.name,
                "sample_rate": 16000,
                "channels": 1
            }
            
            stage_duration = time.time() - stage_start
            self.log_stage("audio_extraction", True, audio_info, duration=stage_duration)
            
            logger.info(f"✅ Audio extracted successfully")
            return audio_info
            
        except Exception as e:
            stage_duration = time.time() - stage_start
            self.log_stage("audio_extraction", False, error=str(e), duration=stage_duration)
            logger.error(f"❌ Audio extraction failed: {e}")
            raise
    
    def transcribe_audio(self, audio_info):
        """Transcribe audio using Whisper"""
        stage_start = time.time()
        
        try:
            # Check if we have MCP transcription server
            try:
                sys.path.append(str(self.backend_path / "mcp_servers"))
                from transcription_server import TranscriptionServer
                
                server = TranscriptionServer()
                transcription_result = server.transcribe(audio_info["temp_audio_path"])
                
                # Clean up temp file
                os.unlink(audio_info["temp_audio_path"])
                
                stage_duration = time.time() - stage_start
                self.log_stage("transcription", True, transcription_result, duration=stage_duration)
                
                logger.info(f"✅ Transcription completed via MCP server")
                return transcription_result
                
            except ImportError:
                # Fallback to direct Whisper usage
                import whisper
                
                model = whisper.load_model("base")
                result = model.transcribe(audio_info["temp_audio_path"])
                
                # Clean up temp file
                os.unlink(audio_info["temp_audio_path"])
                
                transcription_result = {
                    "text": result["text"],
                    "language": result.get("language", "unknown"),
                    "segments": result.get("segments", []),
                    "method": "whisper_direct"
                }
                
                stage_duration = time.time() - stage_start
                self.log_stage("transcription", True, transcription_result, duration=stage_duration)
                
                logger.info(f"✅ Transcription completed via direct Whisper")
                return transcription_result
                
        except Exception as e:
            # Clean up temp file if it exists
            if 'audio_info' in locals() and Path(audio_info["temp_audio_path"]).exists():
                os.unlink(audio_info["temp_audio_path"])
                
            stage_duration = time.time() - stage_start
            self.log_stage("transcription", False, error=str(e), duration=stage_duration)
            logger.error(f"❌ Transcription failed: {e}")
            raise
    
    def analyze_video_content(self, video_path):
        """Analyze video content using vision AI"""
        stage_start = time.time()
        
        try:
            # Extract key frames for analysis
            import cv2
            import numpy as np
            
            cap = cv2.VideoCapture(str(video_path))
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Extract frames at regular intervals
            frame_interval = max(1, frame_count // 10)  # Extract ~10 frames
            frames_data = []
            
            for i in range(0, frame_count, frame_interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                
                if ret:
                    # Basic frame analysis
                    frame_info = {
                        "frame_number": i,
                        "timestamp": i / cap.get(cv2.CAP_PROP_FPS),
                        "brightness": np.mean(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)),
                        "has_motion": True  # Simplified - could implement motion detection
                    }
                    frames_data.append(frame_info)
            
            cap.release()
            
            # Try to use MCP vision server if available
            try:
                sys.path.append(str(self.backend_path / "mcp_servers"))
                from vision_server import VisionServer
                
                server = VisionServer()
                vision_result = server.analyze_video(str(video_path))
                
                vision_analysis = {
                    "method": "mcp_vision_server",
                    "frames_analyzed": len(frames_data),
                    "vision_result": vision_result,
                    "frame_data": frames_data[:5]  # Include first 5 frames
                }
                
            except ImportError:
                # Fallback to basic analysis
                vision_analysis = {
                    "method": "basic_opencv",
                    "frames_analyzed": len(frames_data),
                    "frame_data": frames_data,
                    "average_brightness": np.mean([f["brightness"] for f in frames_data]),
                    "scene_changes": len(frames_data)  # Simplified
                }
            
            stage_duration = time.time() - stage_start
            self.log_stage("vision_analysis", True, vision_analysis, duration=stage_duration)
            
            logger.info(f"✅ Vision analysis completed: {len(frames_data)} frames analyzed")
            return vision_analysis
            
        except Exception as e:
            stage_duration = time.time() - stage_start
            self.log_stage("vision_analysis", False, error=str(e), duration=stage_duration)
            logger.error(f"❌ Vision analysis failed: {e}")
            raise
    
    def generate_summary_report(self, video_info, transcription, vision_analysis):
        """Generate comprehensive analysis summary"""
        stage_start = time.time()
        
        try:
            # Create comprehensive summary
            summary = {
                "video_metadata": video_info,
                "content_analysis": {
                    "duration_minutes": video_info["duration_seconds"] / 60,
                    "speech_detected": len(transcription.get("text", "")) > 10,
                    "language": transcription.get("language", "unknown"),
                    "word_count": len(transcription.get("text", "").split()),
                    "scene_complexity": len(vision_analysis.get("frame_data", [])),
                    "visual_quality": vision_analysis.get("average_brightness", 0)
                },
                "key_insights": [],
                "recommendations": []
            }
            
            # Add insights based on analysis
            if summary["content_analysis"]["speech_detected"]:
                summary["key_insights"].append("Speech content detected and transcribed")
            
            if summary["content_analysis"]["duration_minutes"] > 5:
                summary["recommendations"].append("Consider breaking into shorter segments for analysis")
            
            if summary["content_analysis"]["word_count"] > 100:
                summary["key_insights"].append("Rich speech content available for NLP analysis")
            
            # Try to use MCP generation server for enhanced summary
            try:
                sys.path.append(str(self.backend_path / "mcp_servers"))
                from generation_server import GenerationServer
                
                server = GenerationServer()
                enhanced_summary = server.generate_summary({
                    "transcription": transcription["text"],
                    "video_info": video_info,
                    "vision_data": vision_analysis
                })
                
                summary["enhanced_summary"] = enhanced_summary
                summary["generation_method"] = "mcp_generation_server"
                
            except ImportError:
                # Basic summary generation
                summary["basic_summary"] = f"Video analysis of {video_info['duration_seconds']:.1f}s content with {summary['content_analysis']['word_count']} words transcribed."
                summary["generation_method"] = "basic_template"
            
            stage_duration = time.time() - stage_start
            self.log_stage("summary_generation", True, summary, duration=stage_duration)
            
            logger.info(f"✅ Summary report generated")
            return summary
            
        except Exception as e:
            stage_duration = time.time() - stage_start
            self.log_stage("summary_generation", False, error=str(e), duration=stage_duration)
            logger.error(f"❌ Summary generation failed: {e}")
            raise
    
    def save_results(self, final_summary):
        """Save complete analysis results"""
        try:
            # Create reports directory
            reports_dir = self.backend_path / "reports"
            reports_dir.mkdir(exist_ok=True)
            
            # Update final results
            self.results["final_summary"] = final_summary
            self.results["overall_success"] = len(self.results["errors"]) == 0
            
            # Calculate total processing time
            total_duration = sum(
                stage.get("duration_seconds", 0) 
                for stage in self.results["analysis_stages"].values()
            )
            self.results["performance_metrics"]["total_processing_time"] = total_duration
            
            # Save detailed JSON report
            json_path = reports_dir / f"analysis_{self.session_id}.json"
            with open(json_path, 'w') as f:
                json.dump(self.results, f, indent=2)
            
            logger.info(f"✅ Results saved to: {json_path}")
            return json_path
            
        except Exception as e:
            logger.error(f"❌ Failed to save results: {e}")
            raise
    
    def analyze_video(self, video_path):
        """Run complete video analysis pipeline"""
        logger.info(f"🚀 Starting comprehensive analysis: {video_path}")
        logger.info(f"Session ID: {self.session_id}")
        
        try:
            # Stage 1: Validate video
            video_info = self.validate_video_file(video_path)
            
            # Stage 2: Extract audio
            audio_info = self.extract_audio(video_path)
            
            # Stage 3: Transcribe audio
            transcription = self.transcribe_audio(audio_info)
            
            # Stage 4: Analyze video content
            vision_analysis = self.analyze_video_content(video_path)
            
            # Stage 5: Generate summary
            final_summary = self.generate_summary_report(video_info, transcription, vision_analysis)
            
            # Stage 6: Save results
            results_path = self.save_results(final_summary)
            
            # Print summary to stdout for E2E testing
            print(json.dumps({
                "success": True,
                "session_id": self.session_id,
                "summary": final_summary,
                "processing_time": self.results["performance_metrics"]["total_processing_time"],
                "results_path": str(results_path)
            }, indent=2))
            
            logger.info(f"🎉 Analysis completed successfully!")
            return True
            
        except Exception as e:
            self.results["overall_success"] = False
            self.results["final_error"] = str(e)
            
            # Print error to stdout for E2E testing
            print(json.dumps({
                "success": False,
                "session_id": self.session_id,
                "error": str(e),
                "partial_results": self.results
            }, indent=2))
            
            logger.error(f"💥 Analysis failed: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description="Comprehensive SVA Video Analysis")
    parser.add_argument("--video-path", required=True, help="Path to video file to analyze")
    parser.add_argument("--session-id", help="Custom session ID for this analysis")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create analyzer and run analysis
    analyzer = ComprehensiveVideoAnalyzer(session_id=args.session_id)
    success = analyzer.analyze_video(args.video_path)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()