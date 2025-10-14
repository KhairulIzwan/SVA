"""
Simple transcription test that addresses Windows file access issues
"""

import whisper
import subprocess
import tempfile
import os
from pathlib import Path
import shutil

def test_whisper_simple():
    """Simple Whisper transcription test"""
    print("🎤 Simple Whisper Transcription Test")
    print("="*50)
    
    # Load model
    print("🎤 Loading Whisper base model...")
    try:
        model = whisper.load_model("base")
        print("✅ Model loaded successfully!")
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return
    
    # Find video file
    video_path = Path(__file__).parent.parent / "data" / "videos" / "test_video.mp4"
    print(f"🔍 Looking for video: {video_path}")
    
    if not video_path.exists():
        print(f"❌ Video file not found: {video_path}")
        return
    
    print(f"✅ Found video: {video_path.name} ({video_path.stat().st_size} bytes)")
    
    # Method 1: Try direct transcription
    print("\n📝 Method 1: Direct transcription")
    try:
        # Convert to string and use forward slashes
        video_str = str(video_path).replace('\\', '/')
        print(f"🎤 Transcribing: {video_str}")
        
        result = model.transcribe(video_str, fp16=False, verbose=False)
        
        if result and result.get("text"):
            text = result["text"].strip()
            language = result.get("language", "unknown")
            
            print(f"✅ Transcription successful!")
            print(f"🌍 Language: {language}")
            print(f"📝 Text ({len(text)} chars): {text[:200]}{'...' if len(text) > 200 else ''}")
            
            # Save result
            with open("transcription_result.txt", "w", encoding="utf-8") as f:
                f.write(f"Video: {video_path.name}\n")
                f.write(f"Language: {language}\n")
                f.write(f"Full transcript:\n{text}\n")
            
            print("✅ Result saved to transcription_result.txt")
            return True
            
        else:
            print("❌ No text found in transcription result")
            
    except Exception as e:
        print(f"❌ Direct transcription failed: {e}")
    
    # Method 2: Copy to temp directory first
    print("\n📝 Method 2: Copy to temp directory")
    try:
        temp_dir = tempfile.mkdtemp()
        temp_video = Path(temp_dir) / "temp_video.mp4"
        
        print(f"📁 Copying to temp: {temp_video}")
        shutil.copy2(video_path, temp_video)
        
        print(f"🎤 Transcribing from temp location...")
        result = model.transcribe(str(temp_video), fp16=False, verbose=False)
        
        if result and result.get("text"):
            text = result["text"].strip()
            language = result.get("language", "unknown")
            
            print(f"✅ Transcription successful!")
            print(f"🌍 Language: {language}")
            print(f"📝 Text: {text[:200]}{'...' if len(text) > 200 else ''}")
            
            # Cleanup
            os.unlink(temp_video)
            os.rmdir(temp_dir)
            
            return True
        else:
            print("❌ No text found")
            
    except Exception as e:
        print(f"❌ Temp method failed: {e}")
        # Cleanup on error
        try:
            if temp_video.exists():
                os.unlink(temp_video)
            if Path(temp_dir).exists():
                os.rmdir(temp_dir)
        except:
            pass
    
    # Method 3: Check if video has audio
    print("\n🔍 Method 3: Check video info")
    try:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            duration = frame_count / fps if fps > 0 else 0
            
            print(f"📊 Video info:")
            print(f"   - FPS: {fps}")
            print(f"   - Frames: {frame_count}")
            print(f"   - Duration: {duration:.2f}s")
            
            cap.release()
        else:
            print("❌ Cannot open video with OpenCV")
            
    except Exception as e:
        print(f"❌ Video analysis failed: {e}")
    
    print("\n🚨 TROUBLESHOOTING SUGGESTIONS:")
    print("1. Video might not have audio track")
    print("2. File path contains special characters")
    print("3. FFmpeg not properly configured")
    print("4. Try a different video file")
    
    return False

if __name__ == "__main__":
    success = test_whisper_simple()
    if success:
        print("\n🎉 Transcription test PASSED!")
    else:
        print("\n❌ Transcription test FAILED!")
    
    print("\nNext step: Check transcription_result.txt if successful")