#!/usr/bin/env python3
"""
Test video creation and download script for SVA testing
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
import argparse
import time

def create_sample_test_video():
    """Create a simple test video for validation"""
    test_videos_dir = Path(__file__).parent / "test_videos"
    test_videos_dir.mkdir(exist_ok=True)
    
    output_path = test_videos_dir / "sample_test.mp4"
    
    # Video parameters
    width, height = 640, 480
    fps = 30
    duration = 10  # seconds
    total_frames = fps * duration
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    print(f"Creating test video: {output_path}")
    print(f"Duration: {duration}s, FPS: {fps}, Resolution: {width}x{height}")
    
    # Create frames with moving text and shapes
    for frame_num in range(total_frames):
        # Create frame with gradient background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add gradient background
        for y in range(height):
            intensity = int(255 * y / height)
            frame[y, :] = [intensity//3, intensity//2, intensity]
        
        # Add moving circle
        circle_x = int(width * (0.2 + 0.6 * (frame_num / total_frames)))
        circle_y = height // 2
        cv2.circle(frame, (circle_x, circle_y), 30, (0, 255, 255), -1)
        
        # Add frame counter text
        time_text = f"Time: {frame_num/fps:.1f}s"
        cv2.putText(frame, time_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add sample speech text that changes
        speech_texts = [
            "Welcome to SVA testing",
            "This is a sample video",
            "Testing speech recognition",
            "Object detection validation",
            "AI analysis in progress"
        ]
        current_text = speech_texts[frame_num // (total_frames // len(speech_texts))]
        cv2.putText(frame, current_text, (10, height-30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add progress bar
        progress = frame_num / total_frames
        bar_width = int(width * 0.8)
        bar_start_x = (width - bar_width) // 2
        bar_y = height - 60
        cv2.rectangle(frame, (bar_start_x, bar_y), (bar_start_x + bar_width, bar_y + 10), (100, 100, 100), -1)
        cv2.rectangle(frame, (bar_start_x, bar_y), (bar_start_x + int(bar_width * progress), bar_y + 10), (0, 255, 0), -1)
        
        out.write(frame)
        
        # Progress indicator
        if frame_num % (total_frames // 10) == 0:
            print(f"  Progress: {progress*100:.0f}%")
    
    out.release()
    
    # Now add audio using ffmpeg
    if output_path.exists():
        print("Adding audio track...")
        temp_video = output_path.with_suffix('.temp.mp4')
        
        # Generate a simple audio tone using ffmpeg
        audio_cmd = [
            'ffmpeg', '-y',
            '-f', 'lavfi', '-i', f'sine=frequency=440:duration={duration}',  # Generate tone
            '-i', str(output_path),  # Input video
            '-c:v', 'copy',  # Copy video stream
            '-c:a', 'aac',   # Encode audio as AAC
            '-shortest',     # Match shortest stream
            str(temp_video)
        ]
        
        try:
            import subprocess
            result = subprocess.run(audio_cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                # Replace original with audio version
                output_path.unlink()
                temp_video.rename(output_path)
                print("✅ Audio track added successfully")
            else:
                print(f"⚠️  Audio addition failed: {result.stderr}")
                print("Video created without audio")
                if temp_video.exists():
                    temp_video.unlink()
        except Exception as e:
            print(f"⚠️  Audio addition failed: {e}")
            print("Video created without audio")
            if temp_video.exists():
                temp_video.unlink()
    
    if output_path.exists():
        file_size = output_path.stat().st_size / (1024 * 1024)  # MB
        print(f"✅ Test video created successfully!")
        print(f"   Path: {output_path}")
        print(f"   Size: {file_size:.2f} MB")
        return True
    else:
        print("❌ Failed to create test video")
        return False

def download_sample_videos():
    """Download sample videos for testing (if internet available)"""
    try:
        import yt_dlp
        
        test_videos_dir = Path(__file__).parent / "test_videos"
        test_videos_dir.mkdir(exist_ok=True)
        
        # Short, free-to-use test videos
        test_urls = [
            "https://sample-videos.com/zip/10/mp4/SampleVideo_1280x720_1mb.mp4",
            "https://www.learningcontainer.com/wp-content/uploads/2020/05/sample-mp4-file.mp4"
        ]
        
        for i, url in enumerate(test_urls):
            output_path = test_videos_dir / f"downloaded_test_{i+1}.mp4"
            
            if output_path.exists():
                print(f"Video already exists: {output_path}")
                continue
                
            print(f"Downloading test video from: {url}")
            
            ydl_opts = {
                'outtmpl': str(output_path),
                'format': 'best[height<=720]',
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                try:
                    ydl.download([url])
                    print(f"✅ Downloaded: {output_path}")
                except Exception as e:
                    print(f"❌ Failed to download {url}: {e}")
                    
    except ImportError:
        print("yt-dlp not available, skipping downloads")
        return False
    except Exception as e:
        print(f"Download failed: {e}")
        return False
    
    return True

def list_available_videos():
    """List all available test videos"""
    test_videos_dir = Path(__file__).parent / "test_videos"
    
    if not test_videos_dir.exists():
        print("No test videos directory found")
        return []
    
    video_files = list(test_videos_dir.glob("*.mp4")) + list(test_videos_dir.glob("*.avi"))
    
    if not video_files:
        print("No test videos found")
        return []
    
    print(f"Available test videos in {test_videos_dir}:")
    for video_file in video_files:
        file_size = video_file.stat().st_size / (1024 * 1024)  # MB
        print(f"  - {video_file.name} ({file_size:.2f} MB)")
    
    return video_files

def validate_video_file(video_path):
    """Validate that a video file can be opened and has basic properties"""
    if not Path(video_path).exists():
        print(f"❌ Video file not found: {video_path}")
        return False
    
    try:
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            print(f"❌ Cannot open video file: {video_path}")
            return False
        
        # Get video properties
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        
        cap.release()
        
        print(f"✅ Video validation successful:")
        print(f"   Resolution: {width}x{height}")
        print(f"   FPS: {fps:.2f}")
        print(f"   Duration: {duration:.2f}s")
        print(f"   Frame count: {frame_count}")
        
        # Basic validation checks
        if width == 0 or height == 0:
            print("❌ Invalid video dimensions")
            return False
        
        if frame_count == 0:
            print("❌ Video has no frames")
            return False
        
        if duration < 1:
            print("⚠️  Video is very short (< 1 second)")
        
        return True
        
    except Exception as e:
        print(f"❌ Video validation failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="SVA Test Video Management")
    parser.add_argument("--create-sample", action="store_true", help="Create a sample test video")
    parser.add_argument("--download", action="store_true", help="Download sample videos from internet")
    parser.add_argument("--list", action="store_true", help="List available test videos")
    parser.add_argument("--validate", type=str, help="Validate a specific video file")
    parser.add_argument("--all", action="store_true", help="Create sample and download videos")
    
    args = parser.parse_args()
    
    if not any([args.create_sample, args.download, args.list, args.validate, args.all]):
        parser.print_help()
        return
    
    success = True
    
    if args.create_sample or args.all:
        print("Creating sample test video...")
        if not create_sample_test_video():
            success = False
    
    if args.download or args.all:
        print("\nDownloading sample videos...")
        download_sample_videos()  # Don't fail on download errors
    
    if args.list or args.all:
        print("\nListing available videos...")
        list_available_videos()
    
    if args.validate:
        print(f"\nValidating video: {args.validate}")
        if not validate_video_file(args.validate):
            success = False
    
    if success:
        print("\n✅ Test video operations completed successfully")
    else:
        print("\n❌ Some operations failed")
        sys.exit(1)

if __name__ == "__main__":
    main()