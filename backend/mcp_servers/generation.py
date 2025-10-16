"""Mock generation server for testing"""

class GenerationServer:
    def generate_summary(self, analysis_data):
        return {
            "summary": "Mock AI-generated summary of the video content",
            "key_points": ["Main topic discussed", "Visual elements present"],
            "recommendations": ["Consider improving lighting", "Add captions"],
            "method": "mock_generation_server"
        }