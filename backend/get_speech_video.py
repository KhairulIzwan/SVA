"""
Download a sample video with guaranteed speech content
"""

import requests
import os
from pathlib import Path

def download_sample_video_with_speech():
    """Download a small sample video with clear speech"""
    
    # Known working sample videos with speech
    samples = [
        {
            "name": "sample_speech_test.mp4",
            "url": "https://www.learningcontainer.com/wp-content/uploads/2020/05/sample-mp4-file.mp4",
            "description": "Small test video with audio"
        }
    ]
    
    print("📥 DOWNLOADING SAMPLE VIDEO WITH SPEECH")
    print("="*50)
    
    video_dir = Path("../data/videos")
    video_dir.mkdir(exist_ok=True)
    
    for sample in samples:
        output_path = video_dir / sample["name"]
        
        if output_path.exists():
            print(f"✅ {sample['name']} already exists")
            continue
        
        try:
            print(f"⬇️  Downloading {sample['name']}...")
            response = requests.get(sample["url"], stream=True, timeout=30)
            response.raise_for_status()
            
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"✅ Downloaded: {sample['name']}")
            print(f"📁 Saved to: {output_path}")
            
        except Exception as e:
            print(f"❌ Error downloading {sample['name']}: {e}")

def suggest_manual_options():
    """Suggest manual ways to get test videos"""
    
    print(f"\n🎤 MANUAL OPTIONS FOR SPEECH VIDEO:")
    print("="*50)
    
    print("""
OPTION A: RECORD YOURSELF (RECOMMENDED - 2 MINUTES)
1. Use your phone camera
2. Record yourself saying:
   "Hello, this is a test for the Smart Video Assistant. 
    The system can analyze video content and transcribe speech.
    I'm demonstrating the AI capabilities including object detection,
    speech recognition, and natural language processing.
    This video will help test the transcription accuracy."
3. Save as MP4
4. Transfer to computer
5. Copy to: data/videos/my_speech_test.mp4

OPTION B: DOWNLOAD FROM YOUTUBE
1. Go to youtube.com
2. Find a short educational video (30-60 seconds)
3. Use online converter to download as MP4
4. Save to data/videos/ folder

OPTION C: SAMPLE VIDEO SITES
1. Visit: https://sample-videos.com/
2. Download "MP4 Video" samples
3. Choose one with clear speech
4. Save to data/videos/ folder

OPTION D: QUICK TEST WITH SCREEN RECORDING
1. Open voice recorder on computer
2. Record yourself speaking the test script
3. Use OBS or built-in screen recorder
4. Record with audio for 30-60 seconds
""")

if __name__ == "__main__":
    print("🎯 GET VIDEO WITH SPEECH FOR SVA TESTING")
    print("="*50)
    
    # Try to download sample
    download_sample_video_with_speech()
    
    # Show manual options
    suggest_manual_options()
    
    print(f"\n⏰ QUICK ACTION:")
    print("1. Pick the easiest option for you")
    print("2. Get a 30-60 second video with clear speech") 
    print("3. Save as MP4 in data/videos/ folder")
    print("4. Come back and we'll test Whisper transcription!")
    
    print(f"\n🎯 GOAL: Test the core AI transcription feature!")