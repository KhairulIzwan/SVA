"""
Download YouTube clip with speech using yt-dlp
"""

import subprocess
import sys
from pathlib import Path

def install_yt_dlp():
    """Install yt-dlp if not available"""
    try:
        import yt_dlp
        print("✅ yt-dlp already installed")
        return True
    except ImportError:
        print("📦 Installing yt-dlp...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'yt-dlp'])
            print("✅ yt-dlp installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install yt-dlp: {e}")
            return False

def download_youtube_clip():
    """Download a short YouTube clip with clear speech"""
    
    if not install_yt_dlp():
        return False
    
    try:
        from yt_dlp import YoutubeDL
        
        # Educational video with clear speech
        url = "https://www.youtube.com/watch?v=f2gbU9Q5Rh8"
        
        # Set output directory to our videos folder
        output_dir = Path("../data/videos")
        output_dir.mkdir(exist_ok=True)
        
        output_path = output_dir / "youtube_speech_clip.%(ext)s"
        
        ydl_opts = {
            'format': 'mp4[height<=720]/best[height<=720]',  # Limit quality for smaller file
            'download_sections': {'*': [(0.07, 11.21)]},     # Download specific segment
            'outtmpl': str(output_path),
            'quiet': False,  # Show download progress
        }
        
        print("🎬 Downloading YouTube clip with speech...")
        print(f"URL: {url}")
        print(f"Segment: 0.07s to 11.21s (~11 seconds)")
        print(f"Output: {output_path}")
        
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        
        # Check if file was created
        downloaded_files = list(output_dir.glob("youtube_speech_clip.*"))
        if downloaded_files:
            downloaded_file = downloaded_files[0]
            print(f"✅ Successfully downloaded: {downloaded_file.name}")
            print(f"📁 Location: {downloaded_file}")
            return str(downloaded_file)
        else:
            print("❌ Download completed but file not found")
            return None
            
    except Exception as e:
        print(f"❌ Error downloading YouTube clip: {e}")
        return None

def test_downloaded_clip(video_path):
    """Quick test of the downloaded clip"""
    if not video_path or not Path(video_path).exists():
        print("❌ No video file to test")
        return
    
    import cv2
    
    print(f"\n🔍 Testing downloaded clip: {Path(video_path).name}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("❌ Cannot open downloaded video")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    duration = frame_count / fps if fps > 0 else 0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    cap.release()
    
    print(f"✅ Video properties:")
    print(f"   Duration: {duration:.1f} seconds")
    print(f"   Resolution: {width}x{height}")
    print(f"   FPS: {fps:.1f}")
    print(f"   File size: {Path(video_path).stat().st_size / 1024 / 1024:.1f} MB")
    
    if 5 <= duration <= 20:  # Good duration for testing
        print("✅ Perfect duration for testing!")
    else:
        print("⚠️  Duration might be too short/long, but should work")

def main():
    """Download and test YouTube clip"""
    print("🎯 DOWNLOAD YOUTUBE CLIP WITH SPEECH")
    print("="*50)
    
    # Download the clip
    video_path = download_youtube_clip()
    
    if video_path:
        # Test the downloaded clip
        test_downloaded_clip(video_path)
        
        print(f"\n🎉 SUCCESS!")
        print(f"✅ Downloaded video with speech content")
        print(f"📁 Ready to test with Whisper transcription")
        print(f"\n🚀 Next step: Run enhanced_audio_test.py to test transcription!")
    else:
        print(f"\n❌ Download failed")
        print(f"💡 Alternative: Record yourself speaking for 30 seconds")
        print(f"📱 Use phone camera and transfer the MP4 file")

if __name__ == "__main__":
    main()