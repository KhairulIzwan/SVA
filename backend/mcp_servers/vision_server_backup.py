"""
Vision MCP Server - Computer Vision component for SVA project
Implements Model Context Protocol for visual analysis, object d        except Exception as e:
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
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
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
            # Detect horizontal and vertical lines
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
            
            # Method 3: Color analysis for pie charts and bar charts
            # Look for distinct color regions that might indicate charts
            # Convert to HSV for better color analysis
            
            # Detect circular shapes (potential pie charts)
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 100, param1=50, param2=30, minRadius=30, maxRadius=200)
            
            if circles is not None:
                circles = np.round(circles[0, :]).astype("int")
                for (x, y, r) in circles:
                    # Check if this circular region has varied colors (typical of pie charts)
                    mask = np.zeros(gray.shape, dtype="uint8")
                    cv2.circle(mask, (x, y), r, 255, -1)
                    
                    # Analyze color variance in the circular region
                    roi = cv2.bitwise_and(frame, frame, mask=mask)
                    mean_color = cv2.mean(roi, mask=mask)
                    
                    charts_detected.append({
                        "type": "circular_chart",
                        "bbox": [x-r, y-r, 2*r, 2*r],
                        "center": [x, y],
                        "radius": r,
                        "confidence": 0.6,
                        "features": ["circular_shape", "color_variance"]
                    })
            
            # Merge overlapping detections and remove duplicates
            final_charts = self._merge_chart_detections(charts_detected + chart_candidates)
            
            # Enhanced text analysis for chart elements
            chart_elements = []
            for chart in final_charts:
                x, y, w, h = chart["bbox"]
                chart_roi = frame[y:y+h, x:x+w]
                
                # Extract text from chart region
                try:
                    chart_text = self.ocr_reader.readtext(chart_roi)
                    if chart_text:
                        text_elements = []
                        for detection in chart_text:
                            bbox_rel, text, conf = detection
                            if conf > 0.3:  # Lower threshold for chart text
                                text_elements.append({
                                    "text": text,
                                    "confidence": conf,
                                    "position": "within_chart"
                                })
                        
                        if text_elements:
                            chart["text_elements"] = text_elements
                            chart["has_labels"] = True
                            
                            # Try to identify chart type from text
                            chart_text_combined = " ".join([elem["text"] for elem in text_elements]).lower()
                            if any(word in chart_text_combined for word in ["percentage", "%", "total"]):
                                chart["likely_type"] = "pie_chart"
                            elif any(word in chart_text_combined for word in ["year", "month", "time", "date"]):
                                chart["likely_type"] = "time_series"
                            elif any(word in chart_text_combined for word in ["category", "group", "type"]):
                                chart["likely_type"] = "bar_chart"
                            
                except Exception as e:
                    logger.warning(f"Text extraction from chart failed: {e}")
            
            return {
                "request_id": request_id,
                "status": "success",
                "data": {
                    "charts_detected": final_charts,
                    "chart_count": len(final_charts),
                    "detection_methods": ["contour_analysis", "grid_detection", "circular_detection", "text_analysis"],
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
            
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            
            # Extract all text with positions
            text_detections = self.ocr_reader.readtext(frame)
            
            document_elements = {
                "headers": [],
                "sections": [],
                "tables": [],
                "lists": [],
                "structure_confidence": 0.0
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
            
            # Sort text elements by vertical position
            text_elements.sort(key=lambda x: x["center_y"])
            
            # Identify headers (larger text, often uppercase or title case)
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
            
            # Detect table structures using line detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Detect horizontal and vertical lines
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
            
            horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
            vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel)
            
            # Find table regions
            table_mask = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)
            contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 2000:  # Minimum table size
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Count text elements within table area
                    texts_in_table = []
                    for elem in text_elements:
                        ex, ey, ew, eh = elem["bbox"]
                        if (x <= ex <= x + w and y <= ey <= y + h):
                            texts_in_table.append(elem["text"])
                    
                    if len(texts_in_table) >= 4:  # Minimum content for table
                        document_elements["tables"].append({
                            "bbox": [x, y, w, h],
                            "cell_count_estimate": len(texts_in_table),
                            "content_preview": texts_in_table[:6],  # First 6 items
                            "confidence": 0.7
                        })
            
            # Detect list structures (aligned text elements)
            potential_lists = []
            for i, elem in enumerate(text_elements[:-2]):
                if elem["text"].strip().startswith(("•", "-", "*", "1.", "2.", "3.")):
                    list_items = [elem]
                    
                    # Look for similarly aligned items
                    for j in range(i+1, min(i+6, len(text_elements))):
                        next_elem = text_elements[j]
                        x_diff = abs(elem["bbox"][0] - next_elem["bbox"][0])
                        
                        if x_diff < 20:  # Similar x alignment
                            if next_elem["text"].strip().startswith(("•", "-", "*")) or next_elem["text"][0].isdigit():
                                list_items.append(next_elem)
                    
                    if len(list_items) >= 2:
                        potential_lists.append({
                            "items": [item["text"] for item in list_items],
                            "position": list_items[0]["bbox"],
                            "item_count": len(list_items),
                            "list_type": "bulleted" if list_items[0]["text"].strip().startswith(("•", "-", "*")) else "numbered"
                        })
            
            document_elements["lists"] = potential_lists
            
            # Calculate overall structure confidence
            structure_indicators = 0
            if document_elements["headers"]:
                structure_indicators += 1
            if document_elements["tables"]:
                structure_indicators += 1
            if document_elements["lists"]:
                structure_indicators += 1
            
            document_elements["structure_confidence"] = min(structure_indicators / 3.0, 1.0)
            
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
            }n, and scene understanding
"""

import asyncio
import json
import logging
import numpy as np
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime
import cv2
from PIL import Image, ImageDraw, ImageFont

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VisionMCPServer:
    """MCP Server for computer vision processing"""
    
    def __init__(self):
        self.server_name = "vision"
        self.models = {}
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
        self.supported_formats = [".mp4", ".avi", ".mov", ".mkv", ".jpg", ".png", ".jpeg"]
        
    async def initialize(self):
        """Initialize computer vision models"""
        logger.info("🤖 Initializing Vision MCP Server...")
        
        try:
            # Initialize YOLO for object detection
            await self._load_yolo_model()
            
            # Initialize other vision models
            await self._load_ocr_model()
            
            logger.info("✅ Vision MCP server ready")
            return {"status": "success", "models_loaded": list(self.models.keys())}
            
        except Exception as e:
            logger.error(f"❌ Vision server initialization failed: {e}")
            return {"status": "error", "error": str(e)}
    
    async def _load_yolo_model(self):
        """Load YOLO model for object detection"""
        try:
            from ultralytics import YOLO
            logger.info("📥 Loading YOLO model...")
            
            # Use YOLOv8 nano for speed (can upgrade to small/medium later)
            self.models["yolo"] = YOLO('yolov8n.pt')
            logger.info("✅ YOLO model loaded successfully")
            
        except ImportError:
            logger.warning("⚠️ YOLO not available - installing ultralytics...")
            import subprocess
            subprocess.run(["pip", "install", "ultralytics"], check=True)
            from ultralytics import YOLO
            self.models["yolo"] = YOLO('yolov8n.pt')
            logger.info("✅ YOLO model loaded after installation")
            
        except Exception as e:
            logger.error(f"❌ YOLO loading failed: {e}")
            self.models["yolo"] = None
    
    async def _load_ocr_model(self):
        """Load OCR model for text extraction"""
        try:
            import easyocr
            logger.info("📥 Loading OCR model...")
            
            # Initialize EasyOCR with English and Malay support
            self.models["ocr"] = easyocr.Reader(['en', 'ms'])
            logger.info("✅ OCR model loaded successfully")
            
        except ImportError:
            logger.warning("⚠️ EasyOCR not available - will use basic OCR")
            self.models["ocr"] = None
            
        except Exception as e:
            logger.error(f"❌ OCR loading failed: {e}")
            self.models["ocr"] = None
    
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Main MCP request handler"""
        action = request.get("action")
        request_id = request.get("request_id", f"req_{datetime.now().isoformat()}")
        
        logger.info(f"👁️ Processing vision request {request_id}: {action}")
        
        try:
            if action == "analyze_video_frames":
                return await self._analyze_video_frames(request)
            elif action == "detect_objects":
                return await self._detect_objects(request)
            elif action == "extract_text_from_frames":
                return await self._extract_text_from_frames(request)
            elif action == "analyze_scene":
                return await self._analyze_scene(request)
            elif action == "detect_charts_graphs":
                return await self._detect_charts_graphs(request)
            elif action == "analyze_document_structure":
                return await self._analyze_document_structure(request)
            elif action == "detect_graphs_charts":
                return await self._detect_graphs_charts(request)
            elif action == "extract_keyframes":
                return await self._extract_keyframes(request)
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
    
    async def _analyze_video_frames(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive video frame analysis"""
        video_path = request.get("video_path")
        options = request.get("options", {})
        request_id = request.get("request_id", "unknown")
        
        if not video_path or not Path(video_path).exists():
            return {
                "request_id": request_id,
                "status": "error",
                "error": f"Video file not found: {video_path}"
            }
        
        try:
            # Extract frames for analysis
            frames = await self._extract_sample_frames(video_path, options.get("sample_rate", 5))
            
            analysis_results = {
                "total_frames_analyzed": len(frames),
                "objects_detected": [],
                "text_found": [],
                "scene_analysis": [],
                "visual_summary": {}
            }
            
            # Analyze each frame
            for i, (timestamp, frame) in enumerate(frames):
                frame_analysis = await self._analyze_single_frame(frame, timestamp)
                
                # Aggregate results
                if frame_analysis.get("objects"):
                    analysis_results["objects_detected"].extend(frame_analysis["objects"])
                
                if frame_analysis.get("text"):
                    analysis_results["text_found"].extend(frame_analysis["text"])
                
                if frame_analysis.get("scene"):
                    analysis_results["scene_analysis"].append(frame_analysis["scene"])
            
            # Generate summary
            analysis_results["visual_summary"] = self._generate_visual_summary(analysis_results)
            
            return {
                "request_id": request_id,
                "status": "success",
                "data": analysis_results
            }
            
        except Exception as e:
            return {
                "request_id": request_id,
                "status": "error",
                "error": str(e)
            }
    
    async def _extract_sample_frames(self, video_path: str, sample_rate: int = 5) -> List[tuple]:
        """Extract sample frames from video for analysis"""
        frames = []
        
        try:
            cap = cv2.VideoCapture(str(video_path))
            
            if not cap.isOpened():
                raise ValueError("Cannot open video file")
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Sample frames every N seconds
            frame_interval = int(fps * sample_rate) if fps > 0 else 30
            
            for frame_num in range(0, total_frames, frame_interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                ret, frame = cap.read()
                
                if ret:
                    timestamp = frame_num / fps if fps > 0 else frame_num
                    frames.append((timestamp, frame))
                
                # Limit to prevent too many frames
                if len(frames) >= 20:
                    break
            
            cap.release()
            logger.info(f"📊 Extracted {len(frames)} sample frames")
            
        except Exception as e:
            logger.error(f"❌ Frame extraction failed: {e}")
        
        return frames
    
    async def _analyze_single_frame(self, frame: np.ndarray, timestamp: float) -> Dict[str, Any]:
        """Analyze a single video frame"""
        analysis = {
            "timestamp": timestamp,
            "objects": [],
            "text": [],
            "scene": {}
        }
        
        try:
            # Object detection
            if self.models.get("yolo"):
                objects = await self._detect_objects_in_frame(frame)
                analysis["objects"] = objects
            
            # Text extraction
            if self.models.get("ocr"):
                text = await self._extract_text_from_frame(frame)
                analysis["text"] = text
            
            # Basic scene analysis
            scene_info = await self._analyze_frame_scene(frame)
            analysis["scene"] = scene_info
            
        except Exception as e:
            logger.error(f"❌ Frame analysis failed: {e}")
        
        return analysis
    
    async def _detect_objects_in_frame(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Detect objects in a single frame using YOLO"""
        objects = []
        
        try:
            if not self.models.get("yolo"):
                return objects
            
            # Run YOLO detection
            results = self.models["yolo"](frame)
            
            # Process results
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Extract object information
                        obj_info = {
                            "class_name": result.names[int(box.cls)],
                            "confidence": float(box.conf),
                            "bbox": box.xyxy[0].tolist(),  # [x1, y1, x2, y2]
                            "area": float((box.xyxy[0][2] - box.xyxy[0][0]) * (box.xyxy[0][3] - box.xyxy[0][1]))
                        }
                        
                        # Only include high-confidence detections
                        if obj_info["confidence"] > 0.5:
                            objects.append(obj_info)
            
            logger.info(f"🔍 Detected {len(objects)} objects")
            
        except Exception as e:
            logger.error(f"❌ Object detection failed: {e}")
        
        return objects
    
    async def _extract_text_from_frame(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Extract text from frame using OCR"""
        text_results = []
        
        try:
            if not self.models.get("ocr"):
                return text_results
            
            # Convert BGR to RGB for OCR
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run OCR
            ocr_results = self.models["ocr"].readtext(rgb_frame)
            
            # Process results
            for (bbox, text, confidence) in ocr_results:
                if confidence > 0.5 and len(text.strip()) > 2:
                    text_info = {
                        "text": text.strip(),
                        "confidence": confidence,
                        "bbox": bbox,
                        "area": self._calculate_bbox_area(bbox)
                    }
                    text_results.append(text_info)
            
            logger.info(f"📝 Extracted {len(text_results)} text elements")
            
        except Exception as e:
            logger.error(f"❌ Text extraction failed: {e}")
        
        return text_results
    
    async def _analyze_frame_scene(self, frame: np.ndarray) -> Dict[str, Any]:
        """Basic scene analysis of frame"""
        scene_info = {}
        
        try:
            # Basic color analysis
            mean_color = np.mean(frame, axis=(0, 1))
            scene_info["dominant_colors"] = {
                "blue": int(mean_color[0]),
                "green": int(mean_color[1]), 
                "red": int(mean_color[2])
            }
            
            # Brightness analysis
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            scene_info["brightness"] = float(brightness)
            scene_info["brightness_level"] = "bright" if brightness > 128 else "dark"
            
            # Edge detection for complexity
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            scene_info["complexity"] = float(edge_density)
            scene_info["complexity_level"] = "complex" if edge_density > 0.1 else "simple"
            
            # Frame dimensions
            scene_info["dimensions"] = {
                "height": frame.shape[0],
                "width": frame.shape[1],
                "channels": frame.shape[2]
            }
            
        except Exception as e:
            logger.error(f"❌ Scene analysis failed: {e}")
        
        return scene_info
    
    async def _detect_objects(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Standalone object detection for images/video frames"""
        video_path = request.get("video_path")
        image_path = request.get("image_path")
        request_id = request.get("request_id", "unknown")
        
        try:
            if video_path:
                # Extract first frame for object detection
                cap = cv2.VideoCapture(str(video_path))
                ret, frame = cap.read()
                cap.release()
                
                if not ret:
                    raise ValueError("Cannot read frame from video")
                
            elif image_path:
                frame = cv2.imread(str(image_path))
                if frame is None:
                    raise ValueError("Cannot read image file")
            else:
                raise ValueError("No video_path or image_path provided")
            
            objects = await self._detect_objects_in_frame(frame)
            
            return {
                "request_id": request_id,
                "status": "success",
                "data": {
                    "objects_detected": objects,
                    "total_objects": len(objects),
                    "high_confidence_objects": [obj for obj in objects if obj["confidence"] > 0.7]
                }
            }
            
        except Exception as e:
            return {
                "request_id": request_id,
                "status": "error",
                "error": str(e)
            }
    
    async def _extract_text_from_frames(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Extract text from video frames"""
        video_path = request.get("video_path")
        request_id = request.get("request_id", "unknown")
        
        try:
            frames = await self._extract_sample_frames(video_path, sample_rate=10)
            all_text = []
            
            for timestamp, frame in frames:
                text_results = await self._extract_text_from_frame(frame)
                for text_info in text_results:
                    text_info["timestamp"] = timestamp
                    all_text.append(text_info)
            
            # Deduplicate similar text
            unique_text = self._deduplicate_text(all_text)
            
            return {
                "request_id": request_id,
                "status": "success",
                "data": {
                    "text_elements": unique_text,
                    "total_text_found": len(unique_text),
                    "frames_analyzed": len(frames)
                }
            }
            
        except Exception as e:
            return {
                "request_id": request_id,
                "status": "error",
                "error": str(e)
            }
    
    async def _detect_graphs_charts(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Detect graphs and charts in video frames"""
        video_path = request.get("video_path")
        request_id = request.get("request_id", "unknown")
        
        try:
            frames = await self._extract_sample_frames(video_path, sample_rate=5)
            chart_detections = []
            
            for timestamp, frame in frames:
                # Look for chart-like patterns
                chart_info = await self._detect_chart_patterns(frame, timestamp)
                if chart_info:
                    chart_detections.append(chart_info)
            
            return {
                "request_id": request_id,
                "status": "success",
                "data": {
                    "charts_detected": chart_detections,
                    "total_charts": len(chart_detections),
                    "frames_analyzed": len(frames)
                }
            }
            
        except Exception as e:
            return {
                "request_id": request_id,
                "status": "error",
                "error": str(e)
            }
    
    async def _detect_chart_patterns(self, frame: np.ndarray, timestamp: float) -> Optional[Dict[str, Any]]:
        """Detect chart/graph patterns in frame"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Look for straight lines (axes)
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=50, maxLineGap=10)
            
            if lines is not None and len(lines) > 5:
                # Check for grid-like patterns
                horizontal_lines = []
                vertical_lines = []
                
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                    
                    if abs(angle) < 10:  # Nearly horizontal
                        horizontal_lines.append(line)
                    elif abs(abs(angle) - 90) < 10:  # Nearly vertical
                        vertical_lines.append(line)
                
                if len(horizontal_lines) >= 2 and len(vertical_lines) >= 2:
                    return {
                        "timestamp": timestamp,
                        "type": "chart_detected",
                        "confidence": min(1.0, (len(horizontal_lines) + len(vertical_lines)) / 20),
                        "horizontal_lines": len(horizontal_lines),
                        "vertical_lines": len(vertical_lines),
                        "total_lines": len(lines)
                    }
        
        except Exception as e:
            logger.error(f"❌ Chart detection failed: {e}")
        
        return None
    
    async def _extract_keyframes(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key frames from video based on visual changes"""
        video_path = request.get("video_path")
        request_id = request.get("request_id", "unknown")
        threshold = request.get("threshold", 0.3)
        
        try:
            cap = cv2.VideoCapture(str(video_path))
            keyframes = []
            prev_frame = None
            frame_count = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                
                if prev_frame is not None:
                    # Calculate frame difference
                    diff = cv2.absdiff(frame, prev_frame)
                    diff_score = np.mean(diff) / 255.0
                    
                    if diff_score > threshold:
                        keyframes.append({
                            "timestamp": timestamp,
                            "frame_number": frame_count,
                            "change_score": diff_score
                        })
                
                prev_frame = frame.copy()
                frame_count += 1
                
                # Limit processing for performance
                if frame_count > 1000:
                    break
            
            cap.release()
            
            return {
                "request_id": request_id,
                "status": "success",
                "data": {
                    "keyframes": keyframes,
                    "total_keyframes": len(keyframes),
                    "frames_processed": frame_count
                }
            }
            
        except Exception as e:
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
                "supported_formats": self.supported_formats,
                "models_loaded": list(self.models.keys()),
                "model_status": {name: model is not None for name, model in self.models.items()}
            }
        }
    
    def _generate_visual_summary(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a summary of visual analysis"""
        summary = {}
        
        # Object summary
        objects = analysis_results.get("objects_detected", [])
        if objects:
            object_counts = {}
            for obj in objects:
                class_name = obj["class_name"]
                object_counts[class_name] = object_counts.get(class_name, 0) + 1
            
            summary["most_common_objects"] = sorted(object_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            summary["total_unique_objects"] = len(object_counts)
        
        # Text summary
        text_elements = analysis_results.get("text_found", [])
        if text_elements:
            all_text = " ".join([elem["text"] for elem in text_elements])
            summary["text_preview"] = all_text[:200] + "..." if len(all_text) > 200 else all_text
            summary["total_text_elements"] = len(text_elements)
        
        # Scene summary
        scenes = analysis_results.get("scene_analysis", [])
        if scenes:
            avg_brightness = np.mean([scene.get("brightness", 0) for scene in scenes])
            summary["average_brightness"] = float(avg_brightness)
            summary["scene_complexity"] = "high" if np.mean([scene.get("complexity", 0) for scene in scenes]) > 0.1 else "low"
        
        return summary
    
    def _calculate_bbox_area(self, bbox) -> float:
        """Calculate area of bounding box"""
        try:
            if isinstance(bbox, list) and len(bbox) == 4:
                # [x1, y1, x2, y2] format
                return abs((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
            else:
                # OCR format [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
                points = np.array(bbox)
                return cv2.contourArea(points)
        except:
            return 0.0
    
    def _deduplicate_text(self, text_list: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate text elements"""
        seen_text = set()
        unique_text = []
        
        for text_info in text_list:
            text = text_info["text"].lower().strip()
            if text not in seen_text and len(text) > 2:
                seen_text.add(text)
                unique_text.append(text_info)
        
        return unique_text

# Test functions
async def test_vision_mcp_server():
    """Test the vision MCP server"""
    print("🧪 Testing Vision MCP Server")
    print("=" * 50)
    
    # Initialize server
    server = VisionMCPServer()
    init_result = await server.initialize()
    print(f"Initialization: {init_result}")
    
    if init_result["status"] != "success":
        print("⚠️  Server initialization had issues, but continuing with available models...")
    
    # Test video path
    video_path = "data/videos/test_video.mp4"
    
    if not Path(video_path).exists():
        print(f"❌ Test video not found: {video_path}")
        return
    
    # Test 1: Get capabilities
    print("\n🔍 Test 1: Get Capabilities")
    request = {"action": "get_capabilities"}
    result = await server.process_request(request)
    print(f"Capabilities: {json.dumps(result, indent=2)}")
    
    # Test 2: Extract keyframes
    print("\n🎞️ Test 2: Extract Keyframes")
    request = {
        "action": "extract_keyframes",
        "video_path": video_path,
        "threshold": 0.2
    }
    result = await server.process_request(request)
    if result["status"] == "success":
        keyframes = result["data"]["keyframes"]
        print(f"Found {len(keyframes)} keyframes")
        for kf in keyframes[:3]:  # Show first 3
            print(f"  - {kf['timestamp']:.2f}s (change: {kf['change_score']:.3f})")
    else:
        print(f"Error: {result['error']}")
    
    # Test 3: Detect objects (if YOLO available)
    print("\n🔍 Test 3: Object Detection")
    request = {
        "action": "detect_objects",
        "video_path": video_path
    }
    result = await server.process_request(request)
    if result["status"] == "success":
        objects = result["data"]["objects_detected"]
        print(f"Detected {len(objects)} objects")
        for obj in objects[:5]:  # Show first 5
            print(f"  - {obj['class_name']}: {obj['confidence']:.2f}")
    else:
        print(f"Object detection: {result['error']}")
    
    # Test 4: Extract text (if OCR available)
    print("\n📝 Test 4: Text Extraction")
    request = {
        "action": "extract_text_from_frames",
        "video_path": video_path
    }
    result = await server.process_request(request)
    if result["status"] == "success":
        text_elements = result["data"]["text_elements"]
        print(f"Found {len(text_elements)} text elements")
        for text in text_elements[:3]:  # Show first 3
            print(f"  - '{text['text']}' (confidence: {text['confidence']:.2f})")
    else:
        print(f"Text extraction: {result['error']}")
    
    # Test 5: Full video analysis
    print("\n📊 Test 5: Full Video Analysis")
    request = {
        "action": "analyze_video_frames",
        "video_path": video_path,
        "options": {"sample_rate": 10}
    }
    result = await server.process_request(request)
    if result["status"] == "success":
        data = result["data"]
        print(f"Analyzed {data['total_frames_analyzed']} frames")
        print(f"Objects detected: {len(data['objects_detected'])}")
        print(f"Text elements: {len(data['text_found'])}")
        print(f"Visual summary: {data['visual_summary']}")
    else:
        print(f"Full analysis error: {result['error']}")
    
    print("\n✅ Vision MCP Server testing completed!")

if __name__ == "__main__":
    asyncio.run(test_vision_mcp_server())