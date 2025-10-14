"""
Get a working video file with speech - multiple approaches
"""

import subprocess
import sys
from pathlib import Path
import requests

def check_video_file(video_path):
    """Check if video file is playable"""
    import cv2
    
    print(f"🔍 Checking video file: {Path(video_path).name}")
    
    if not Path(video_path).exists():
        print("❌ File does not exist")
        return False
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("❌ OpenCV cannot open this video file")
        return False
    
    # Try to read first frame
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("❌ Cannot read frames from video")
        return False
    
    print("✅ Video file is readable by OpenCV")
    return True

def download_simple_test_video():
    """Download a simple, guaranteed working test video"""
    
    # Known working sample videos
    test_videos = [
        {
            "name": "simple_test.mp4",
            "url": "https://sample-videos.com/zip/10/mp4/SampleVideo_1280x720_1mb.mp4",
            "description": "Simple test video"
        }
    ]
    
    video_dir = Path("../data/videos")
    video_dir.mkdir(exist_ok=True)
    
    for video in test_videos:
        output_path = video_dir / video["name"]
        
        if output_path.exists():
            print(f"✅ {video['name']} already exists")
            if check_video_file(output_path):
                return str(output_path)
            else:
                print(f"🗑️  Removing corrupted file: {video['name']}")
                output_path.unlink()
        
        try:
            print(f"⬇️  Downloading {video['name']}...")
            response = requests.get(video["url"], stream=True, timeout=60)
            response.raise_for_status()
            
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            print(f"✅ Downloaded: {video['name']}")
            
            # Test the downloaded file
            if check_video_file(output_path):
                return str(output_path)
            else:
                print(f"❌ Downloaded file is not working")
                output_path.unlink()
                
        except Exception as e:
            print(f"❌ Error downloading {video['name']}: {e}")
    
    return None

def create_manual_instructions():
    """Create clear instructions for manual video creation"""
    
    instructions = """
# MANUAL VIDEO CREATION GUIDE

## OPTION 1: Record with Phone (RECOMMENDED)
1. Open your phone's camera app
2. Switch to video mode
3. Record yourself saying this script clearly:

"Hello, this is a test video for the Smart Video Assistant. 
I am demonstrating the speech recognition capabilities.
The system uses Whisper AI to convert speech to text.
This test will validate the transcription accuracy.
Thank you for watching this demonstration."

4. Keep recording for about 30-60 seconds
5. Save the video
6. Transfer to your computer
7. Copy to: data/videos/my_speech_test.mp4

## OPTION 2: Use Windows Voice Recorder + Screen Recording
1. Open Windows Voice Recorder
2. Record the script above
3. Play the recording while using screen recording software
4. Save as MP4

## OPTION 3: Download from Free Sites
1. Visit: https://pixabay.com/videos/
2. Search for "speech" or "talking"
3. Download a short video with clear speech
4. Convert to MP4 if needed
5. Save to data/videos/ folder

## VERIFICATION STEPS
After getting your video:
1. Double-click the MP4 file
2. Verify it plays in Windows Media Player or VLC
3. Ensure you can hear clear speech
4. File should be 30-120 seconds long
5. Test with our Whisper script

## FILE REQUIREMENTS
✅ Format: MP4
✅ Duration: 30-120 seconds  
✅ Audio: Clear human speech
✅ Size: Under 50MB
✅ Playable in standard media players
"""
    
    with open("manual_video_guide.txt", "w", encoding="utf-8") as f:
        f.write(instructions)
    
    print("✅ Manual guide created: manual_video_guide.txt")

def cleanup_bad_videos():
    """Remove potentially corrupted video files"""
    video_dir = Path("../data/videos")
    
    problem_files = [
        "youtube_speech_clip.mp4",  # The one that didn't work
        "sample_speech_test.mp4"    # The one we couldn't play
    ]
    
    for file_name in problem_files:
        file_path = video_dir / file_name
        if file_path.exists():
            print(f"🗑️  Removing problematic file: {file_name}")
            try:
                file_path.unlink()
                print(f"✅ Removed: {file_name}")
            except Exception as e:
                print(f"❌ Could not remove {file_name}: {e}")

def main():
    """Main function to get a working video"""
    print("🎯 GET WORKING VIDEO WITH SPEECH")
    print("="*50)
    
    # Clean up bad files first
    cleanup_bad_videos()
    
    # Try to download a simple working video
    working_video = download_simple_test_video()
    
    if working_video:
        print(f"\n✅ SUCCESS: Working video available")
        print(f"📁 File: {working_video}")
        print(f"🎤 Note: This may not have speech, but file format is good")
    else:
        print(f"\n⚠️  Could not download working sample")
    
    # Create manual instructions
    create_manual_instructions()
    
    print(f"\n🎯 RECOMMENDED NEXT STEPS:")
    print("1. 📱 Record yourself with phone (30-60 seconds)")
    print("2. 💾 Save as MP4 and copy to data/videos/")
    print("3. ✅ Verify video plays in media player")
    print("4. 🎤 Test with Whisper transcription")
    
    print(f"\n💡 EASIEST SOLUTION:")
    print("🎬 Record yourself reading the script from manual_video_guide.txt")
    print("📱 Use phone camera - it creates perfect MP4 files!")

if __name__ == "__main__":
    main()