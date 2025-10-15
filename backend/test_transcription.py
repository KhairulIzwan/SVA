"""
Simple transcription test that addresses Windows file access issues
"""

import whisper
import subprocess
import tempfile
import os
from pathlib import Path
import shutil

def test_whisper_improved():
    """Improved Whisper transcription test with multiple models and language options"""
    print("🎤 Enhanced Whisper Transcription Test")
    print("="*50)
    
    # Test multiple model sizes
    models_to_test = ["tiny", "base", "small"]
    results = {}
    
    for model_name in models_to_test:
        print(f"\n🤖 Testing {model_name} model...")
        try:
            model = whisper.load_model(model_name)
            print(f"✅ {model_name} model loaded successfully!")
            results[model_name] = model
        except Exception as e:
            print(f"❌ {model_name} model loading failed: {e}")
            continue
    
    # Find video file
    video_path = Path(__file__).parent.parent / "data" / "videos" / "test_video.mp4"
    print(f"🔍 Looking for video: {video_path}")
    
    if not video_path.exists():
        print(f"❌ Video file not found: {video_path}")
        return
    
    print(f"✅ Found video: {video_path.name} ({video_path.stat().st_size} bytes)")
    
    # Test different transcription approaches
    transcription_configs = [
        {
            "name": "Auto-detect language",
            "options": {"language": None, "task": "transcribe", "fp16": False}
        },
        {
            "name": "Force Malay (ms)",
            "options": {"language": "ms", "task": "transcribe", "fp16": False}
        },
        {
            "name": "Force English (en)",
            "options": {"language": "en", "task": "transcribe", "fp16": False}
        },
        {
            "name": "Force Indonesian (id)",
            "options": {"language": "id", "task": "transcribe", "fp16": False}
        }
    ]
    
    best_result = None
    best_confidence = 0
    
    for model_name, model in results.items():
        print(f"\n📝 Testing {model_name} model with different configurations...")
        
        for config in transcription_configs:
            print(f"\n🔧 {config['name']}...")
            try:
                # Convert to string and use forward slashes
                video_str = str(video_path).replace('\\', '/')
                
                # Add verbose and word_timestamps for better analysis
                options = config['options'].copy()
                options.update({
                    'verbose': False,
                    'word_timestamps': True,
                    'initial_prompt': "This is a clear speech recording in Malay or Indonesian language."
                })
                
                result = model.transcribe(video_str, **options)
                
                if result and result.get("text"):
                    text = result["text"].strip()
                    language = result.get("language", "unknown")
                    
                    # Calculate a simple confidence score based on text quality
                    confidence = calculate_transcription_confidence(text, language)
                    
                    print(f"✅ Success! Language: {language}, Confidence: {confidence:.2f}")
                    print(f"📝 Text: {text[:150]}{'...' if len(text) > 150 else ''}")
                    
                    # Track best result
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_result = {
                            'model': model_name,
                            'config': config['name'],
                            'result': result,
                            'confidence': confidence
                        }
                    
                    # Save each result
                    output_file = f"transcription_{model_name}_{config['name'].replace(' ', '_').lower()}.json"
                    save_detailed_result(result, video_path.name, model_name, config['name'], output_file)
                    
                else:
                    print("❌ No text found in transcription result")
                    
            except Exception as e:
                print(f"❌ Configuration failed: {e}")
    
    # Save and display best result
    if best_result:
        print(f"\n🏆 BEST RESULT:")
        print(f"Model: {best_result['model']}")
        print(f"Configuration: {best_result['config']}")
        print(f"Confidence: {best_result['confidence']:.2f}")
        print(f"Language: {best_result['result']['language']}")
        print(f"Text: {best_result['result']['text']}")
        
        # Save the best result as the main output
        with open("transcription_result.txt", "w", encoding="utf-8") as f:
            f.write(f"=== BEST TRANSCRIPTION RESULT ===\n")
            f.write(f"Video: {video_path.name}\n")
            f.write(f"Model: {best_result['model']}\n")
            f.write(f"Configuration: {best_result['config']}\n")
            f.write(f"Language: {best_result['result']['language']}\n")
            f.write(f"Confidence Score: {best_result['confidence']:.2f}\n")
            f.write(f"Duration: {best_result['result'].get('duration', 'unknown')}s\n")
            f.write(f"\nFull transcript:\n{best_result['result']['text']}\n")
            
            # Add segments if available
            if 'segments' in best_result['result']:
                f.write(f"\n=== TIMESTAMPED SEGMENTS ===\n")
                for segment in best_result['result']['segments']:
                    start = segment.get('start', 0)
                    end = segment.get('end', 0)
                    text = segment.get('text', '')
                    f.write(f"[{start:.2f}s - {end:.2f}s]: {text}\n")
        
        print("✅ Best result saved to transcription_result.txt")
        return True
    else:
        print("❌ No successful transcriptions found")
        return False

def calculate_transcription_confidence(text: str, language: str) -> float:
    """Calculate a simple confidence score for transcription quality"""
    if not text or not text.strip():
        return 0.0
    
    # Base score
    score = 0.5
    
    # Penalize very short text
    if len(text) < 10:
        score -= 0.3
    
    # Penalize repeated words (sign of poor transcription)
    words = text.lower().split()
    unique_words = set(words)
    if len(words) > 0:
        repetition_ratio = len(unique_words) / len(words)
        score += repetition_ratio * 0.3
    
    # Bonus for reasonable text length
    if 50 <= len(text) <= 1000:
        score += 0.2
    
    # Language-specific adjustments
    if language in ['ms', 'id']:  # Malay/Indonesian
        score += 0.1
    elif language == 'en':  # English
        score += 0.05
    
    # Penalize gibberish patterns
    if any(word in text.lower() for word in ['københverdekal', 'naasje', 'ryrti']):
        score -= 0.4
    
    return max(0.0, min(1.0, score))

def save_detailed_result(result, video_name, model_name, config_name, filename):
    """Save detailed transcription result to JSON file"""
    import json
    
    detailed_result = {
        'video': video_name,
        'model': model_name,
        'configuration': config_name,
        'language': result.get('language'),
        'text': result.get('text'),
        'duration': result.get('duration'),
        'segments': result.get('segments', []),
        'confidence_score': calculate_transcription_confidence(result.get('text', ''), result.get('language', ''))
    }
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(detailed_result, f, indent=2, ensure_ascii=False)
        print(f"💾 Detailed result saved to {filename}")
    except Exception as e:
        print(f"❌ Failed to save {filename}: {e}")

def analyze_audio_quality():
    """Analyze the audio quality of the video file"""
    print(f"\n🔊 Audio Quality Analysis")
    print("-" * 30)
    
    video_path = Path(__file__).parent.parent / "data" / "videos" / "test_video.mp4"
    
    try:
        import cv2
        cap = cv2.VideoCapture(str(video_path))
        
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            duration = frame_count / fps if fps > 0 else 0
            
            print(f"📊 Video properties:")
            print(f"   - Duration: {duration:.2f}s")
            print(f"   - FPS: {fps}")
            print(f"   - Total frames: {frame_count}")
            
            cap.release()
            
            # Try to extract audio info using FFmpeg
            try:
                import subprocess
                cmd = ['ffprobe', '-v', 'quiet', '-show_streams', '-select_streams', 'a', str(video_path)]
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0 and result.stdout:
                    print(f"✅ Audio stream detected")
                    if 'sample_rate' in result.stdout:
                        print("✅ Audio appears to have proper sample rate")
                else:
                    print("⚠️  No audio stream detected or FFmpeg not available")
                    
            except Exception as e:
                print(f"⚠️  Could not analyze audio: {e}")
            
        else:
            print("❌ Cannot open video file")
            
    except Exception as e:
        print(f"❌ Video analysis failed: {e}")

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
    # Run audio quality analysis first
    analyze_audio_quality()
    
    # Run improved transcription test
    success = test_whisper_improved()
    
    if success:
        print("\n🎉 Enhanced transcription test PASSED!")
        print("📋 Check transcription_result.txt for the best result")
        print("💾 Check individual JSON files for detailed analysis")
    else:
        print("\n❌ Enhanced transcription test FAILED!")
        print("🔧 Try the troubleshooting suggestions above")
    
    print("\n📈 Next steps:")
    print("1. Review the best transcription result")
    print("2. If still inaccurate, try with 'medium' or 'large' Whisper model")
    print("3. Consider audio preprocessing if quality is poor")
    print("4. Proceed to build MCP server architecture")