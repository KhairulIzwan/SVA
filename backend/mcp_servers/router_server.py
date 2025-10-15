"""
Router MCP Server - Central coordination hub for SVA project
Implements Model Context Protocol for routing queries between specialized agents
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum
import re

# Import our specialized servers
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

try:
    from transcription_server import TranscriptionMCPServer
    from vision_server import VisionMCPServer
except ImportError:
    TranscriptionMCPServer = None
    VisionMCPServer = None

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QueryType(Enum):
    """Types of user queries"""
    TRANSCRIPTION = "transcription"
    VISION_ANALYSIS = "vision_analysis"
    OBJECT_DETECTION = "object_detection"
    TEXT_EXTRACTION = "text_extraction"
    MULTI_MODAL = "multi_modal"
    REPORT_GENERATION = "report_generation"
    UNKNOWN = "unknown"

class RouterMCPServer:
    """MCP Server for routing queries to appropriate specialized agents"""
    
    def __init__(self):
        self.server_name = "router"
        self.transcription_server = None
        self.vision_server = None
        self.capabilities = [
            "route_query",
            "analyze_user_intent",
            "coordinate_multi_agent",
            "generate_comprehensive_report",
            "get_server_status",
            "human_in_the_loop"
        ]
        
        # Query patterns for intent recognition
        self.query_patterns = {
            QueryType.TRANSCRIPTION: [
                r"transcribe",
                r"speech.*text",
                r"what.*said",
                r"convert.*audio",
                r"subtitle",
                r"captions"
            ],
            QueryType.OBJECT_DETECTION: [
                r"what.*objects",
                r"detect.*objects",
                r"what.*see",
                r"identify.*things",
                r"what.*shown",
                r"items.*video"
            ],
            QueryType.TEXT_EXTRACTION: [
                r"text.*video",
                r"read.*text",
                r"extract.*text",
                r"words.*screen",
                r"writing.*video"
            ],
            QueryType.VISION_ANALYSIS: [
                r"analyze.*video",
                r"describe.*scene",
                r"what.*happening",
                r"visual.*analysis",
                r"scene.*description"
            ],
            QueryType.REPORT_GENERATION: [
                r"generate.*report",
                r"create.*pdf",
                r"make.*powerpoint",
                r"summary.*report",
                r"export.*document"
            ]
        }
        
        # Example queries from PROJECT_OVERVIEW.md
        self.example_queries = [
            "Transcribe the video.",
            "Create a PowerPoint with the key points discussed in the video.",
            "What objects are shown in the video?",
            "Are there any graphs in the video? If yes, describe them.",
            "Summarize our discussion so far and generate a PDF."
        ]
        
    async def initialize(self):
        """Initialize router and all specialized servers"""
        logger.info("🚀 Initializing Router MCP Server...")
        
        initialization_results = {
            "router": {"status": "success"},
            "transcription": {"status": "not_available"},
            "vision": {"status": "not_available"}
        }
        
        try:
            # Initialize Transcription Server
            if TranscriptionMCPServer:
                logger.info("📝 Initializing Transcription Server...")
                self.transcription_server = TranscriptionMCPServer("small")
                transcription_result = await self.transcription_server.initialize()
                initialization_results["transcription"] = transcription_result
            
            # Initialize Vision Server
            if VisionMCPServer:
                logger.info("👁️ Initializing Vision Server...")
                self.vision_server = VisionMCPServer()
                vision_result = await self.vision_server.initialize()
                initialization_results["vision"] = vision_result
            
            logger.info("✅ Router MCP server ready")
            return {
                "status": "success",
                "servers_initialized": initialization_results
            }
            
        except Exception as e:
            logger.error(f"❌ Router initialization failed: {e}")
            return {
                "status": "partial", 
                "error": str(e),
                "servers_initialized": initialization_results
            }
    
    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Main router request handler"""
        action = request.get("action")
        request_id = request.get("request_id", f"req_{datetime.now().isoformat()}")
        
        logger.info(f"🎯 Router processing request {request_id}: {action}")
        
        try:
            if action == "route_query":
                return await self._route_user_query(request)
            elif action == "analyze_user_intent":
                return await self._analyze_user_intent(request)
            elif action == "coordinate_multi_agent":
                return await self._coordinate_multi_agent(request)
            elif action == "generate_comprehensive_report":
                return await self._generate_comprehensive_report(request)
            elif action == "get_server_status":
                return await self._get_server_status()
            elif action == "human_in_the_loop":
                return await self._human_in_the_loop(request)
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
            logger.error(f"❌ Router request processing failed: {e}")
            return {
                "request_id": request_id,
                "status": "error",
                "error": str(e)
            }
    
    async def _route_user_query(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Route user query to appropriate specialized server(s)"""
        user_query = request.get("query", "")
        video_path = request.get("video_path")
        request_id = request.get("request_id", "unknown")
        
        logger.info(f"🔍 Analyzing query: '{user_query}'")
        
        # Analyze intent
        intent_analysis = await self._analyze_intent(user_query)
        query_type = intent_analysis["primary_intent"]
        confidence = intent_analysis["confidence"]
        
        logger.info(f"🎯 Detected intent: {query_type.value} (confidence: {confidence:.2f})")
        
        # Route based on intent
        if query_type == QueryType.TRANSCRIPTION:
            return await self._handle_transcription_query(request, intent_analysis)
        
        elif query_type == QueryType.OBJECT_DETECTION:
            return await self._handle_object_detection_query(request, intent_analysis)
        
        elif query_type == QueryType.TEXT_EXTRACTION:
            return await self._handle_text_extraction_query(request, intent_analysis)
        
        elif query_type == QueryType.VISION_ANALYSIS:
            return await self._handle_vision_analysis_query(request, intent_analysis)
        
        elif query_type == QueryType.REPORT_GENERATION:
            return await self._handle_report_generation_query(request, intent_analysis)
        
        elif query_type == QueryType.MULTI_MODAL:
            return await self._handle_multi_modal_query(request, intent_analysis)
        
        else:
            # Unknown query - trigger human-in-the-loop
            return await self._trigger_clarification(request, intent_analysis)
    
    async def _analyze_intent(self, query: str) -> Dict[str, Any]:
        """Analyze user intent from natural language query"""
        query_lower = query.lower()
        intent_scores = {}
        
        # Calculate scores for each intent type
        for intent_type, patterns in self.query_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    score += 1
            
            # Normalize score
            intent_scores[intent_type] = score / len(patterns) if patterns else 0
        
        # Find primary intent
        primary_intent = max(intent_scores, key=intent_scores.get)
        confidence = intent_scores[primary_intent]
        
        # Check for multi-modal queries
        high_scoring_intents = [intent for intent, score in intent_scores.items() if score > 0.3]
        is_multi_modal = len(high_scoring_intents) > 1
        
        return {
            "primary_intent": primary_intent,
            "confidence": confidence,
            "is_multi_modal": is_multi_modal,
            "intent_scores": intent_scores,
            "suggested_agents": high_scoring_intents
        }
    
    async def _handle_transcription_query(self, request: Dict[str, Any], intent_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Handle transcription-related queries"""
        if not self.transcription_server:
            return self._create_error_response(request, "Transcription server not available")
        
        video_path = request.get("video_path")
        options = request.get("options", {})
        
        # Enhance options based on query analysis
        if "malay" in request.get("query", "").lower():
            options["language"] = "ms"
        
        transcription_request = {
            "action": "transcribe_video",
            "video_path": video_path,
            "options": options,
            "request_id": request.get("request_id")
        }
        
        result = await self.transcription_server.process_request(transcription_request)
        
        # Add routing information
        result["routing_info"] = {
            "routed_to": "transcription_server",
            "intent_analysis": intent_analysis,
            "processing_time": datetime.now().isoformat()
        }
        
        return result
    
    async def _handle_object_detection_query(self, request: Dict[str, Any], intent_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Handle object detection queries"""
        if not self.vision_server:
            return self._create_error_response(request, "Vision server not available")
        
        vision_request = {
            "action": "detect_objects",
            "video_path": request.get("video_path"),
            "request_id": request.get("request_id")
        }
        
        result = await self.vision_server.process_request(vision_request)
        
        # Add human-readable interpretation
        if result["status"] == "success":
            objects = result["data"]["objects_detected"]
            interpretation = self._interpret_object_detection(objects, request.get("query", ""))
            result["interpretation"] = interpretation
        
        result["routing_info"] = {
            "routed_to": "vision_server",
            "intent_analysis": intent_analysis
        }
        
        return result
    
    async def _handle_multi_modal_query(self, request: Dict[str, Any], intent_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Handle queries requiring multiple agents"""
        request_id = request.get("request_id", "unknown")
        video_path = request.get("video_path")
        
        results = {
            "request_id": request_id,
            "status": "success",
            "query_type": "multi_modal",
            "agent_results": {},
            "combined_analysis": {}
        }
        
        # Run transcription if needed
        if QueryType.TRANSCRIPTION in intent_analysis["suggested_agents"] and self.transcription_server:
            logger.info("🎤 Running transcription analysis...")
            transcription_result = await self.transcription_server.process_request({
                "action": "transcribe_video",
                "video_path": video_path,
                "options": {"language": None}
            })
            results["agent_results"]["transcription"] = transcription_result
        
        # Run vision analysis if needed
        vision_agents = [QueryType.OBJECT_DETECTION, QueryType.VISION_ANALYSIS, QueryType.TEXT_EXTRACTION]
        if any(agent in intent_analysis["suggested_agents"] for agent in vision_agents) and self.vision_server:
            logger.info("👁️ Running vision analysis...")
            vision_result = await self.vision_server.process_request({
                "action": "analyze_video_frames",
                "video_path": video_path,
                "options": {"sample_rate": 10}
            })
            results["agent_results"]["vision"] = vision_result
        
        # Combine results
        results["combined_analysis"] = self._combine_multi_modal_results(results["agent_results"], request.get("query", ""))
        results["routing_info"] = {
            "routed_to": "multi_agent_coordination",
            "intent_analysis": intent_analysis
        }
        
        return results
    
    async def _trigger_clarification(self, request: Dict[str, Any], intent_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Trigger human-in-the-loop clarification"""
        query = request.get("query", "")
        
        clarification_options = []
        
        # Suggest based on example queries
        if "transcribe" in query.lower():
            clarification_options.append("Did you mean transcribe the audio to text?")
        
        if "create" in query.lower() and ("report" in query.lower() or "powerpoint" in query.lower()):
            clarification_options.append("Would you like me to create a document with the video analysis?")
        
        if "objects" in query.lower() or "show" in query.lower():
            clarification_options.append("Are you asking about objects visible in the video?")
        
        if not clarification_options:
            clarification_options = [
                "Would you like me to transcribe the video?",
                "Should I analyze the visual content of the video?",
                "Do you want a comprehensive report combining audio and visual analysis?"
            ]
        
        return {
            "request_id": request.get("request_id", "unknown"),
            "status": "needs_clarification",
            "query_type": "human_in_the_loop",
            "original_query": query,
            "clarification_needed": True,
            "suggested_options": clarification_options,
            "intent_analysis": intent_analysis,
            "available_actions": [
                "Transcribe the video content",
                "Analyze visual objects and scenes", 
                "Extract text from video frames",
                "Generate comprehensive report"
            ]
        }
    
    def _interpret_object_detection(self, objects: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """Provide human-readable interpretation of object detection results"""
        if not objects:
            return {"summary": "No objects were detected in the video."}
        
        # Count objects by type
        object_counts = {}
        for obj in objects:
            class_name = obj["class_name"]
            object_counts[class_name] = object_counts.get(class_name, 0) + 1
        
        # Generate summary
        total_objects = len(objects)
        unique_types = len(object_counts)
        
        # Most common objects
        most_common = sorted(object_counts.items(), key=lambda x: x[1], reverse=True)
        
        summary_parts = [
            f"Found {total_objects} objects of {unique_types} different types."
        ]
        
        if most_common:
            top_objects = most_common[:3]
            object_list = ", ".join([f"{count} {name}{'s' if count > 1 else ''}" for name, count in top_objects])
            summary_parts.append(f"Most common objects: {object_list}.")
        
        return {
            "summary": " ".join(summary_parts),
            "object_counts": object_counts,
            "total_objects": total_objects,
            "unique_types": unique_types
        }
    
    def _combine_multi_modal_results(self, agent_results: Dict[str, Any], query: str) -> Dict[str, Any]:
        """Combine results from multiple agents"""
        combined = {
            "summary": "",
            "key_findings": [],
            "confidence_score": 0.0
        }
        
        findings = []
        confidence_scores = []
        
        # Process transcription results
        if "transcription" in agent_results:
            trans_result = agent_results["transcription"]
            if trans_result.get("status") == "success":
                data = trans_result["data"]
                findings.append(f"Speech content: {data['text'][:100]}...")
                findings.append(f"Language detected: {data['language']}")
                confidence_scores.append(data.get("confidence_score", 0))
        
        # Process vision results
        if "vision" in agent_results:
            vision_result = agent_results["vision"]
            if vision_result.get("status") == "success":
                data = vision_result["data"]
                if data.get("objects_detected"):
                    object_summary = data["visual_summary"].get("most_common_objects", [])
                    if object_summary:
                        objects_text = ", ".join([f"{count} {name}" for name, count in object_summary[:3]])
                        findings.append(f"Objects detected: {objects_text}")
                
                if data.get("text_found"):
                    text_preview = data["visual_summary"].get("text_preview", "")
                    if text_preview:
                        findings.append(f"Text in video: {text_preview[:50]}...")
        
        # Generate summary
        if findings:
            combined["summary"] = "Multi-modal analysis complete. " + " ".join(findings)
            combined["key_findings"] = findings
            combined["confidence_score"] = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.5
        else:
            combined["summary"] = "Analysis completed but no significant content detected."
        
        return combined
    
    async def _get_server_status(self) -> Dict[str, Any]:
        """Get status of all managed servers"""
        status = {
            "router": {"status": "active", "capabilities": self.capabilities},
            "transcription": {"status": "not_available"},
            "vision": {"status": "not_available"}
        }
        
        if self.transcription_server:
            try:
                caps = await self.transcription_server.process_request({"action": "get_capabilities"})
                status["transcription"] = {"status": "active", "details": caps}
            except Exception as e:
                status["transcription"] = {"status": "error", "error": str(e)}
        
        if self.vision_server:
            try:
                caps = await self.vision_server.process_request({"action": "get_capabilities"})
                status["vision"] = {"status": "active", "details": caps}
            except Exception as e:
                status["vision"] = {"status": "error", "error": str(e)}
        
        return {
            "status": "success",
            "data": status
        }
    
    def _create_error_response(self, request: Dict[str, Any], error_message: str) -> Dict[str, Any]:
        """Create standardized error response"""
        return {
            "request_id": request.get("request_id", "unknown"),
            "status": "error",
            "error": error_message,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _get_capabilities(self) -> Dict[str, Any]:
        """Return router capabilities"""
        return {
            "status": "success",
            "data": {
                "server_name": self.server_name,
                "capabilities": self.capabilities,
                "managed_servers": ["transcription", "vision"],
                "supported_query_types": [qt.value for qt in QueryType],
                "example_queries": self.example_queries
            }
        }

# Test functions
async def test_router_mcp_server():
    """Test the router MCP server with example queries"""
    print("🧪 Testing Router MCP Server")
    print("=" * 50)
    
    # Initialize router
    router = RouterMCPServer()
    init_result = await router.initialize()
    print(f"Initialization: {json.dumps(init_result, indent=2)}")
    
    if init_result["status"] == "partial":
        print("⚠️  Router partially initialized, continuing with available servers...")
    
    # Test video path
    video_path = "data/videos/test_video.mp4"
    
    # Test 1: Get capabilities
    print("\n🔍 Test 1: Get Capabilities")
    result = await router.process_request({"action": "get_capabilities"})
    print(f"Capabilities: {json.dumps(result, indent=2)}")
    
    # Test 2: Server status
    print("\n📊 Test 2: Server Status")
    result = await router.process_request({"action": "get_server_status"})
    print(f"Server Status: {json.dumps(result, indent=2)}")
    
    # Test 3: Example queries from PROJECT_OVERVIEW.md
    example_queries = [
        "Transcribe the video.",
        "What objects are shown in the video?",
        "Are there any graphs in the video? If yes, describe them."
    ]
    
    for i, query in enumerate(example_queries):
        print(f"\n🎯 Test {i+4}: '{query}'")
        
        request = {
            "action": "route_query",
            "query": query,
            "video_path": video_path,
            "request_id": f"test_{i+1}"
        }
        
        result = await router.process_request(request)
        
        if result["status"] == "success":
            if "data" in result:
                print(f"✅ Success: {result['data'].get('text', 'Analysis completed')[:100]}...")
            elif "combined_analysis" in result:
                print(f"✅ Multi-modal: {result['combined_analysis']['summary'][:100]}...")
            else:
                print(f"✅ Success: {result}")
        elif result["status"] == "needs_clarification":
            print(f"❓ Clarification needed: {result['suggested_options'][0]}")
        else:
            print(f"❌ Error: {result['error']}")
    
    print("\n✅ Router MCP Server testing completed!")

if __name__ == "__main__":
    asyncio.run(test_router_mcp_server())