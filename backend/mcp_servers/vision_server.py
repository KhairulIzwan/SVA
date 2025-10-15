"""
Vision MCP Server - Computer Vision component for SVA project
Implements Model Context Protocol for visual analysis, object detection, and text extraction
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import cv2
import numpy as np
import base64
from pathlib import Path

# Computer vision imports
try:
    from ultralytics import YOLO
except ImportError:
    print("⚠️ Ultralytics YOLO not installed. Object detection will be limited.")

try:
    import easyocr
except ImportError:
    print("⚠️ EasyOCR not installed. Text extraction will be limited.")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VisionMCPServer:
    """MCP Server for computer vision tasks including object detection and OCR"""
    
    def __init__(self):
        self.server_name = "vision"
        self.capabilities = [
            "detect_objects",
            "extract_text",
            "analyze_scene",
            "detect_charts_graphs",
            "analyze_document_structure",
            "get_frame_info", 
            "process_video_frames",
            "get_capabilities"
        ]
        self.yolo_model = None
        self.ocr_reader = None
        
    async def initialize(self):
        """Initialize vision models and dependencies"""
        logger.info("👁️ Initializing Vision MCP Server...")
        
        try:
            # Initialize YOLO for object detection
            logger.info("Loading YOLO model...")
            self.yolo_model = YOLO('yolov8n.pt')  # nano model for speed
            logger.info("✅ YOLO model loaded")
            
            # Initialize EasyOCR for text extraction
            logger.info("Loading EasyOCR...")
            self.ocr_reader = easyocr.Reader(['en', 'ms'])  # English and Malay
            logger.info("✅ EasyOCR initialized")
            
            return {
                "status": "success",
                "models_loaded": ["YOLOv8n", "EasyOCR"],
                "languages": ["en", "ms"]
            }
            
        except Exception as e:
            logger.error(f"❌ Vision server initialization failed: {e}")
            return {"status": "error", "error": str(e)}
    
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Main vision request handler"""
        action = request.get("action")
        request_id = request.get("request_id", f"req_{datetime.now().isoformat()}")
        
        logger.info(f"👁️ Processing vision request {request_id}: {action}")
        
        try:
            if action == "detect_objects":
                return await self._detect_objects(request)
            elif action == "extract_text":
                return await self._extract_text(request)
            elif action == "analyze_scene":
                return await self._analyze_scene(request)
            elif action == "detect_charts_graphs":
                return await self._detect_charts_graphs(request)
            elif action == "analyze_document_structure":
                return await self._analyze_document_structure(request)
            elif action == "get_frame_info":
                return await self._get_frame_info(request)
            elif action == "process_video_frames":
                return await self._process_video_frames(request)
            elif action == "get_capabilities":
                return await self._get_capabilities()
            else:
                return {
                    "request_id": request_id,
                    "status": "error",
                    "error": f"Unknown action: {action}",
                    "available_actions": self.capabilities
                }
                
        except Exception as e:
            logger.error(f"❌ Vision request processing failed: {e}")
            return {
                "request_id": request_id,
                "status": "error",
                "error": str(e)
            }
    
    async def _detect_objects(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Detect objects in frame using YOLO"""
        request_id = request.get("request_id", "unknown")
        frame_data = request.get("frame_data")
        confidence_threshold = request.get("confidence_threshold", 0.5)
        
        try:
            # Decode frame from base64 if needed
            if isinstance(frame_data, str):
                frame_bytes = base64.b64decode(frame_data)
                frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
                frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
            else:
                frame = frame_data
            
            # Run YOLO detection
            results = self.yolo_model(frame, conf=confidence_threshold)
            
            objects_detected = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        # Get confidence and class
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        class_name = self.yolo_model.names[class_id]
                        
                        objects_detected.append({
                            "class_name": class_name,
                            "class_id": class_id,
                            "confidence": confidence,
                            "bbox": [int(x1), int(y1), int(x2-x1), int(y2-y1)],  # x, y, w, h
                            "center": [int((x1+x2)/2), int((y1+y2)/2)]
                        })
            
            return {
                "request_id": request_id,
                "status": "success",
                "data": {
                    "objects_detected": objects_detected,
                    "object_count": len(objects_detected),
                    "confidence_threshold": confidence_threshold,
                    "frame_analyzed": True
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Object detection failed: {e}")
            return {
                "request_id": request_id,
                "status": "error",
                "error": str(e)
            }
    
    async def _extract_text(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Extract text from frame using OCR"""
        request_id = request.get("request_id", "unknown")
        frame_data = request.get("frame_data")
        languages = request.get("languages", ["en", "ms"])
        
        try:
            # Decode frame
            if isinstance(frame_data, str):
                frame_bytes = base64.b64decode(frame_data)
                frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
                frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
            else:
                frame = frame_data
            
            # Run OCR
            ocr_results = self.ocr_reader.readtext(frame)
            
            text_found = []
            for detection in ocr_results:
                bbox, text, confidence = detection
                
                # Convert bbox to standard format
                bbox_coords = np.array(bbox)
                x_coords = bbox_coords[:, 0]
                y_coords = bbox_coords[:, 1]
                
                x_min, x_max = int(np.min(x_coords)), int(np.max(x_coords))
                y_min, y_max = int(np.min(y_coords)), int(np.max(y_coords))
                
                text_found.append({
                    "text": text,
                    "confidence": confidence,
                    "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],  # x, y, w, h
                    "center": [(x_min + x_max) // 2, (y_min + y_max) // 2]
                })
            
            # Extract just the text for easy reading
            extracted_text = " ".join([item["text"] for item in text_found if item["confidence"] > 0.5])
            
            return {
                "request_id": request_id,
                "status": "success",
                "data": {
                    "text_found": text_found,
                    "extracted_text": extracted_text,
                    "text_count": len(text_found),
                    "languages_used": languages,
                    "frame_analyzed": True
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Text extraction failed: {e}")
            return {
                "request_id": request_id,
                "status": "error",
                "error": str(e)
            }
    
    async def _analyze_scene(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive scene analysis combining objects and text"""
        request_id = request.get("request_id", "unknown")
        frame_data = request.get("frame_data")
        
        try:
            # Run both object detection and text extraction
            objects_result = await self._detect_objects(request)
            text_result = await self._extract_text(request)
            
            if objects_result["status"] != "success" or text_result["status"] != "success":
                return {
                    "request_id": request_id,
                    "status": "error",
                    "error": "Failed to analyze scene components"
                }
            
            # Combine results
            scene_analysis = {
                "objects": objects_result["data"]["objects_detected"],
                "text_elements": text_result["data"]["text_found"],
                "summary": {
                    "object_count": len(objects_result["data"]["objects_detected"]),
                    "text_count": len(text_result["data"]["text_found"]),
                    "dominant_objects": [],
                    "key_text": text_result["data"]["extracted_text"]
                }
            }
            
            # Identify dominant objects (most confident)
            objects = objects_result["data"]["objects_detected"]
            if objects:
                objects_by_confidence = sorted(objects, key=lambda x: x["confidence"], reverse=True)
                scene_analysis["summary"]["dominant_objects"] = [
                    obj["class_name"] for obj in objects_by_confidence[:3]
                ]
            
            return {
                "request_id": request_id,
                "status": "success",
                "data": scene_analysis
            }
            
        except Exception as e:
            logger.error(f"❌ Scene analysis failed: {e}")
            return {
                "request_id": request_id,
                "status": "error",
                "error": str(e)
            }
    
    async def _detect_charts_graphs(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced detection for charts, graphs, and data visualizations"""
        request_id = request.get("request_id", "unknown")
        frame_data = request.get("frame_data")
        
        try:
            # Decode frame
            if isinstance(frame_data, str):
                frame_bytes = base64.b64decode(frame_data)
                frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
                frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
            else:
                frame = frame_data
            
            charts_detected = []
            
            # Convert to different color spaces for analysis
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Method 1: Detect rectangular chart areas (typical chart boundaries)
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            chart_candidates = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 1000:  # Minimum chart size
                    # Check if contour is roughly rectangular
                    approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
                    if len(approx) >= 4:
                        x, y, w, h = cv2.boundingRect(contour)
                        aspect_ratio = w / h
                        
                        # Charts typically have certain aspect ratios
                        if 0.5 <= aspect_ratio <= 3.0 and w > 100 and h > 100:
                            chart_candidates.append({
                                "type": "rectangular_chart",
                                "bbox": [x, y, w, h],
                                "area": area,
                                "aspect_ratio": aspect_ratio,
                                "confidence": 0.7
                            })
            
            # Method 2: Look for grid patterns (typical in charts)
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
            
            horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
            vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel)
            
            # Combine lines to detect grid patterns
            grid_pattern = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)
            
            # Find grid regions
            grid_contours, _ = cv2.findContours(grid_pattern, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in grid_contours:
                area = cv2.contourArea(contour)
                if area > 2000:
                    x, y, w, h = cv2.boundingRect(contour)
                    charts_detected.append({
                        "type": "grid_chart",
                        "bbox": [x, y, w, h],
                        "area": area,
                        "confidence": 0.8,
                        "features": ["grid_lines", "structured_layout"]
                    })
            
            # Method 3: Detect circular shapes (potential pie charts)
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 100, param1=50, param2=30, minRadius=30, maxRadius=200)
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                for (x, y, r) in circles:
                    charts_detected.append({
                        "type": "circular_chart",
                        "bbox": [x-r, y-r, 2*r, 2*r],
                        "center": [x, y],
                        "radius": r,
                        "confidence": 0.6,
                        "features": ["circular_shape"]
                    })
            
            # Merge overlapping detections
            final_charts = self._merge_chart_detections(charts_detected + chart_candidates)
            
            return {
                "request_id": request_id,
                "status": "success",
                "data": {
                    "charts_detected": final_charts,
                    "chart_count": len(final_charts),
                    "detection_methods": ["contour_analysis", "grid_detection", "circular_detection"],
                    "frame_analyzed": True
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Chart detection failed: {e}")
            return {
                "request_id": request_id,
                "status": "error",
                "error": str(e)
            }
    
    def _merge_chart_detections(self, detections: List[Dict]) -> List[Dict]:
        """Merge overlapping chart detections"""
        if not detections:
            return []
        
        merged = []
        used = set()
        
        for i, detection in enumerate(detections):
            if i in used:
                continue
                
            x1, y1, w1, h1 = detection["bbox"]
            overlapping = [detection]
            
            for j, other in enumerate(detections[i+1:], i+1):
                if j in used:
                    continue
                    
                x2, y2, w2, h2 = other["bbox"]
                
                # Calculate intersection over union (IoU)
                intersection_area = max(0, min(x1+w1, x2+w2) - max(x1, x2)) * max(0, min(y1+h1, y2+h2) - max(y1, y2))
                union_area = w1*h1 + w2*h2 - intersection_area
                
                if union_area > 0:
                    iou = intersection_area / union_area
                    if iou > 0.3:  # 30% overlap threshold
                        overlapping.append(other)
                        used.add(j)
            
            # Merge overlapping detections
            if len(overlapping) == 1:
                merged.append(overlapping[0])
            else:
                # Create merged detection
                all_x = [d["bbox"][0] for d in overlapping]
                all_y = [d["bbox"][1] for d in overlapping]
                all_x2 = [d["bbox"][0] + d["bbox"][2] for d in overlapping]
                all_y2 = [d["bbox"][1] + d["bbox"][3] for d in overlapping]
                
                merged_bbox = [
                    min(all_x),
                    min(all_y),
                    max(all_x2) - min(all_x),
                    max(all_y2) - min(all_y)
                ]
                
                avg_confidence = sum(d["confidence"] for d in overlapping) / len(overlapping)
                
                merged_detection = {
                    "type": "merged_chart",
                    "bbox": merged_bbox,
                    "confidence": avg_confidence,
                    "merged_from": [d["type"] for d in overlapping],
                    "detection_count": len(overlapping)
                }
                
                merged.append(merged_detection)
        
        return merged
    
    async def _analyze_document_structure(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze document structure including headers, sections, tables"""
        request_id = request.get("request_id", "unknown")
        frame_data = request.get("frame_data")
        
        try:
            # Decode frame
            if isinstance(frame_data, str):
                frame_bytes = base64.b64decode(frame_data)
                frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
                frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
            else:
                frame = frame_data
            
            # Extract all text with positions
            text_detections = self.ocr_reader.readtext(frame)
            
            document_elements = {
                "headers": [],
                "structure_confidence": 0.0,
                "text_elements_found": len(text_detections)
            }
            
            if not text_detections:
                return {
                    "request_id": request_id,
                    "status": "success",
                    "data": document_elements
                }
            
            # Analyze text elements
            text_elements = []
            for detection in text_detections:
                bbox, text, conf = detection
                if conf > 0.5:
                    # Calculate text position and size
                    points = np.array(bbox)
                    x_coords = points[:, 0]
                    y_coords = points[:, 1]
                    
                    text_x = int(np.min(x_coords))
                    text_y = int(np.min(y_coords))
                    text_w = int(np.max(x_coords) - np.min(x_coords))
                    text_h = int(np.max(y_coords) - np.min(y_coords))
                    
                    text_elements.append({
                        "text": text,
                        "confidence": conf,
                        "bbox": [text_x, text_y, text_w, text_h],
                        "center_y": text_y + text_h // 2,
                        "font_size_estimate": text_h,
                        "is_uppercase": text.isupper(),
                        "is_title_case": text.istitle()
                    })
            
            # Identify headers (larger text, often uppercase or title case)
            if text_elements:
                avg_font_size = sum(elem["font_size_estimate"] for elem in text_elements) / len(text_elements)
                
                for elem in text_elements:
                    font_ratio = elem["font_size_estimate"] / avg_font_size
                    
                    if (font_ratio > 1.3 or elem["is_uppercase"] or elem["is_title_case"]) and len(elem["text"]) > 3:
                        header_level = 1 if font_ratio > 1.8 else 2 if font_ratio > 1.4 else 3
                        
                        document_elements["headers"].append({
                            "text": elem["text"],
                            "level": header_level,
                            "position": elem["bbox"],
                            "confidence": elem["confidence"],
                            "font_size_ratio": font_ratio
                        })
            
            # Calculate structure confidence
            structure_indicators = 1 if document_elements["headers"] else 0
            document_elements["structure_confidence"] = min(structure_indicators, 1.0)
            
            return {
                "request_id": request_id,
                "status": "success",
                "data": document_elements
            }
            
        except Exception as e:
            logger.error(f"❌ Document structure analysis failed: {e}")
            return {
                "request_id": request_id,
                "status": "error",
                "error": str(e)
            }
    
    async def _get_frame_info(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Get basic information about a frame"""
        request_id = request.get("request_id", "unknown")
        frame_data = request.get("frame_data")
        
        try:
            if isinstance(frame_data, str):
                frame_bytes = base64.b64decode(frame_data)
                frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
                frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
            else:
                frame = frame_data
            
            height, width = frame.shape[:2]
            channels = frame.shape[2] if len(frame.shape) > 2 else 1
            
            return {
                "request_id": request_id,
                "status": "success",
                "data": {
                    "width": width,
                    "height": height,
                    "channels": channels,
                    "total_pixels": width * height,
                    "aspect_ratio": width / height
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Frame info extraction failed: {e}")
            return {
                "request_id": request_id,
                "status": "error",
                "error": str(e)
            }
    
    async def _process_video_frames(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process multiple frames from a video"""
        request_id = request.get("request_id", "unknown")
        video_path = request.get("video_path")
        max_frames = request.get("max_frames", 5)
        frame_skip = request.get("frame_skip", 10)
        
        try:
            if not Path(video_path).exists():
                return {
                    "request_id": request_id,
                    "status": "error",
                    "error": f"Video file not found: {video_path}"
                }
            
            # Open video
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            processed_frames = []
            frame_number = 0
            processed_count = 0
            
            while processed_count < max_frames and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_number % frame_skip == 0:
                    # Process this frame
                    timestamp = frame_number / fps
                    
                    # Run scene analysis on frame
                    frame_request = {
                        "request_id": f"{request_id}_frame_{frame_number}",
                        "frame_data": frame,
                        "action": "analyze_scene"
                    }
                    
                    scene_result = await self._analyze_scene(frame_request)
                    
                    if scene_result["status"] == "success":
                        processed_frames.append({
                            "frame_number": frame_number,
                            "timestamp": timestamp,
                            "analysis": scene_result["data"]
                        })
                        processed_count += 1
                
                frame_number += 1
            
            cap.release()
            
            return {
                "request_id": request_id,
                "status": "success",
                "data": {
                    "video_info": {
                        "total_frames": frame_count,
                        "fps": fps,
                        "duration": frame_count / fps if fps > 0 else 0
                    },
                    "processed_frames": processed_frames,
                    "frames_analyzed": len(processed_frames)
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Video frame processing failed: {e}")
            return {
                "request_id": request_id,
                "status": "error",
                "error": str(e)
            }
    
    async def _get_capabilities(self) -> Dict[str, Any]:
        """Return server capabilities"""
        return {
            "status": "success",
            "data": {
                "server_name": self.server_name,
                "capabilities": self.capabilities,
                "models_loaded": bool(self.yolo_model and self.ocr_reader),
                "supported_languages": ["en", "ms"]
            }
        }

# Test functions
async def test_vision_mcp_server():
    """Test the vision MCP server with enhanced features"""
    print("🧪 Testing Enhanced Vision MCP Server")
    print("=" * 50)
    
    # Initialize server
    server = VisionMCPServer()
    init_result = await server.initialize()
    print(f"Initialization: {json.dumps(init_result, indent=2)}")
    
    if init_result["status"] != "success":
        print("❌ Server initialization failed!")
        return
    
    # Test 1: Get capabilities
    print("\n📋 Test 1: Get Capabilities")
    result = await server.process_request({"action": "get_capabilities"})
    print(f"Capabilities: {json.dumps(result, indent=2)}")
    
    # Test 2: Process test video frames
    print("\n🎥 Test 2: Process Video Frames")
    test_video = "backend/test_video.mp4"
    if Path(test_video).exists():
        request = {
            "action": "process_video_frames",
            "video_path": test_video,
            "max_frames": 3,
            "frame_skip": 15
        }
        result = await server.process_request(request)
        if result["status"] == "success":
            print(f"✅ Processed {result['data']['frames_analyzed']} frames")
            for frame_data in result['data']['processed_frames']:
                frame_num = frame_data['frame_number']
                timestamp = frame_data['timestamp']
                objects = len(frame_data['analysis']['objects'])
                texts = len(frame_data['analysis']['text_elements'])
                print(f"  Frame {frame_num} ({timestamp:.1f}s): {objects} objects, {texts} texts")
        else:
            print(f"❌ Video processing failed: {result['error']}")
    else:
        print(f"❌ Test video not found at {test_video}")
    
    print("\n✅ Enhanced Vision MCP Server testing completed!")

if __name__ == "__main__":
    asyncio.run(test_vision_mcp_server())