"""
Mock MCP servers for testing
"""

# Mock transcription server
class TranscriptionServer:
    def transcribe(self, audio_path):
        return {
            "text": "This is a mock transcription for testing purposes.",
            "language": "english",
            "segments": [
                {"start": 0.0, "end": 5.0, "text": "This is a mock transcription"},
                {"start": 5.0, "end": 10.0, "text": "for testing purposes."}
            ],
            "method": "mock_transcription_server"
        }

# Mock vision server  
class VisionServer:
    def analyze_video(self, video_path):
        return {
            "objects_detected": ["person", "table", "background"],
            "scene_description": "A person in an indoor setting",
            "confidence_scores": {"person": 0.95, "table": 0.87},
            "method": "mock_vision_server"
        }

# Mock generation server
class GenerationServer:
    def generate_summary(self, analysis_data):
        return {
            "summary": "Mock AI-generated summary of the video content",
            "key_points": ["Main topic discussed", "Visual elements present"],
            "recommendations": ["Consider improving lighting", "Add captions"],
            "method": "mock_generation_server"
        }