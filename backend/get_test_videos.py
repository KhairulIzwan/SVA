"""
Smart video downloader for getting proper test videos with speech
"""

import requests
import subprocess
from pathlib import Path
import os

def download_sample_video():
    """Download a sample video with clear speech"""
    
    # Sample videos with known audio content
    sample_videos = [
        {
            "name": "sample_speech_demo.mp4",
            "url": "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4",
            "description": "Animation with clear dialogue",
            "note": "Large file - may take time to download"
        }
    ]
    
    print("📥 DOWNLOADING SAMPLE VIDEO WITH AUDIO")
    print("="*50)
    
    # For now, let's create instructions for manual download
    print("""
🎯 RECOMMENDED SAMPLE VIDEOS TO DOWNLOAD:

1. SHORT EDUCATIONAL VIDEOS (Recommended):
   - TED-Ed clips: https://www.youtube.com/c/TEDEd
   - Khan Academy: https://www.youtube.com/c/khanacademy
   - Crash Course: https://www.youtube.com/c/crashcourse
   
2. SAMPLE SITES:
   - Sample Videos: https://sample-videos.com/
   - Pexels: https://www.pexels.com/videos/
   - Pixabay: https://pixabay.com/videos/

3. CREATE YOUR OWN (BEST OPTION):
   - Record 60-second explanation using phone
   - Use clear speech, minimal background noise
   - Save as MP4 format

📱 QUICK RECORDING GUIDE:
1. Open phone camera/recording app
2. Record yourself explaining any topic for 1 minute
3. Transfer to computer
4. Copy to: data/videos/ folder
5. Rename to: my_test_speech.mp4

🎬 SAMPLE SCRIPT TO RECORD:
"Hello, this is a test video for the Smart Video Assistant. Today I'm going to explain how artificial intelligence can analyze video content. The system can transcribe speech, identify objects, and answer questions about what it sees and hears. This technology is useful for analyzing training videos, meeting recordings, and educational content. Thank you for watching this demonstration."
""")

def create_test_audio_video():
    """Create a simple test video with text-to-speech audio (if possible)"""
    print("\n🔊 ALTERNATIVE: CREATE TEST VIDEO WITH SYNTHETIC SPEECH")
    print("="*50)
    
    script_content = """
# TEST VIDEO CREATION SCRIPT

If you have access to text-to-speech software:

1. Use Windows Narrator or online TTS service
2. Convert this text to audio:

"Welcome to the Smart Video Assistant test. This system analyzes videos using artificial intelligence. It can transcribe speech, identify objects, and answer questions about content. The AI uses computer vision for visual analysis and speech recognition for audio processing. This demonstration shows the capability to process both audio and visual information simultaneously."

3. Create a simple video with this audio
4. Save as MP4 in data/videos/ folder

OR SIMPLER OPTION:
Record yourself reading the sample script using your phone!
"""
    
    with open("create_test_video_guide.txt", "w", encoding="utf-8") as f:
        f.write(script_content)
    
    print("✅ Guide created: create_test_video_guide.txt")

def check_if_videos_have_actual_audio():
    """Better check for audio content in existing videos"""
    print("\n🔍 CHECKING ACTUAL AUDIO CONTENT")
    print("="*50)
    
    video_dir = Path("../data/videos")
    video_files = [f for f in video_dir.glob("*.mp4") if f.is_file()]
    
    for video_file in video_files:
        print(f"\n📹 Analyzing: {video_file.name}")
        
        # Try to load with OpenCV and check properties
        import cv2
        cap = cv2.VideoCapture(str(video_file))
        
        if cap.isOpened():
            # Check if OpenCV can detect audio info
            # Note: OpenCV primarily handles video, not audio
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = frame_count / fps if fps > 0 else 0
            
            print(f"   Duration: {duration:.1f} seconds")
            print(f"   Video: ✅ (OpenCV can read)")
            
            # For audio, we need different approach
            print(f"   Audio: ❓ (Unknown - need FFmpeg or manual check)")
            
            # Suggest testing approach
            print(f"   Recommendation: Try playing {video_file.name} manually")
            print(f"   - If you hear sound: Video has audio ✅")
            print(f"   - If silent: Video has no audio ❌")
        
        cap.release()

def main():
    """Main function to help get suitable test videos"""
    print("🎯 GET SUITABLE TEST VIDEOS FOR SVA")
    print("="*50)
    
    # Check current videos
    check_if_videos_have_actual_audio()
    
    # Provide download/creation options  
    download_sample_video()
    create_test_audio_video()
    
    print(f"\n🎯 NEXT STEPS:")
    print("1. Check if your current videos have audio by playing them")
    print("2. If no audio: Record yourself speaking for 1 minute")
    print("3. Save as MP4 in data/videos/ folder")
    print("4. Test again with enhanced_audio_test.py")
    
    print(f"\n💡 EASIEST SOLUTION:")
    print("🎤 Record yourself reading the sample script!")
    print("📱 Use your phone camera, speak clearly for 60 seconds")
    print("💾 Save as MP4 and copy to data/videos/")

if __name__ == "__main__":
    main()