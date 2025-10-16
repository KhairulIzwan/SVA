"""Mock vision server for testing"""

class VisionServer:
    def analyze_video(self, video_path):
        return {
            "objects_detected": ["person", "table", "background"],
            "scene_description": "A person in an indoor setting",
            "confidence_scores": {"person": 0.95, "table": 0.87},
            "method": "mock_vision_server"
        }