"""
Test Whisper transcription on our video files
"""

import whisper
import os
from pathlib import Path

def test_whisper_transcription():
    """Test Whisper transcription with our video files"""
    print("🎤 Testing Whisper Transcription...")
    
    # Load Whisper model (start with smallest for speed)
    print("Loading Whisper model...")
    model = whisper.load_model("base")  # ~142MB, good balance of speed/accuracy
    print("✅ Model loaded successfully!")
    
    # Find video files
    video_dir = Path("../data/videos")
    video_files = [f for f in video_dir.glob("*.mp4") if f.is_file()]
    
    if not video_files:
        print("❌ No video files found in data/videos/")
        return
    
    # Test with first video
    test_video = video_files[0]
    print(f"📹 Transcribing: {test_video.name}")
    
    try:
        # Transcribe
        result = model.transcribe(str(test_video))
        
        # Display results
        print("\n" + "="*50)
        print("🎯 TRANSCRIPTION RESULTS")
        print("="*50)
        print(f"Video: {test_video.name}")
        print(f"Text: {result['text']}")
        
        # Show segments with timestamps
        if 'segments' in result:
            print("\n📝 DETAILED SEGMENTS:")
            for segment in result['segments'][:5]:  # Show first 5
                start = segment['start']
                end = segment['end']
                text = segment['text']
                print(f"[{start:.1f}s - {end:.1f}s]: {text}")
        
        # Save transcript
        transcript_file = f"transcript_{test_video.stem}.txt"
        with open(transcript_file, "w", encoding="utf-8") as f:
            f.write(f"TRANSCRIPT: {test_video.name}\n")
            f.write("="*50 + "\n")
            f.write(f"Full Text: {result['text']}\n\n")
            f.write("Detailed Segments:\n")
            if 'segments' in result:
                for segment in result['segments']:
                    start = segment['start']
                    end = segment['end']
                    text = segment['text']
                    f.write(f"[{start:.1f}s - {end:.1f}s]: {text}\n")
        
        print(f"\n✅ Transcript saved to: {transcript_file}")
        return True
        
    except Exception as e:
        print(f"❌ Error during transcription: {e}")
        return False

if __name__ == "__main__":
    test_whisper_transcription()