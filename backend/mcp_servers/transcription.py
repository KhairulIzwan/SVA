"""Mock transcription server for testing"""

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