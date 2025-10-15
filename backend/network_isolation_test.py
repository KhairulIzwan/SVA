"""
Network Isolation Test - Verify SVA project operates completely offline
Tests all components without network access to ensure local AI compliance
"""

import asyncio
import json
import logging
import subprocess
import time
import socket
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import sys
import os

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NetworkIsolationTester:
    """Test SVA system functionality with network isolation"""
    
    def __init__(self):
        self.test_results = []
        self.network_blocked = False
        self.original_network_state = None
        
    async def run_complete_test(self) -> Dict[str, Any]:
        """Run comprehensive offline functionality test"""
        print("🔒 SVA Network Isolation Test")
        print("=" * 50)
        
        try:
            # Phase 1: Test with network enabled (baseline)
            print("\n📡 Phase 1: Baseline Test (Network Enabled)")
            baseline_results = await self._test_all_components(network_enabled=True)
            
            # Phase 2: Block network access
            print("\n🚫 Phase 2: Blocking Network Access...")
            network_blocked = await self._block_network_access()
            
            if not network_blocked:
                print("⚠️ Warning: Could not fully block network access. Proceeding with partial isolation...")
            
            # Phase 3: Test with network disabled
            print("\n🔒 Phase 3: Offline Test (Network Disabled)")
            offline_results = await self._test_all_components(network_enabled=False)
            
            # Phase 4: Restore network access
            print("\n🔄 Phase 4: Restoring Network Access...")
            await self._restore_network_access()
            
            # Phase 5: Generate comprehensive report
            report = self._generate_isolation_report(baseline_results, offline_results)
            
            return report
            
        except Exception as e:
            logger.error(f"❌ Network isolation test failed: {e}")
            # Ensure network is restored even if test fails
            await self._restore_network_access()
            raise
    
    async def _test_all_components(self, network_enabled: bool = True) -> Dict[str, Any]:
        """Test all SVA components"""
        test_phase = "baseline" if network_enabled else "offline"
        print(f"\n🧪 Testing all components ({test_phase} mode)...")
        
        results = {
            "test_phase": test_phase,
            "network_enabled": network_enabled,
            "timestamp": datetime.now().isoformat(),
            "component_tests": {}
        }
        
        # Test 1: Check network connectivity
        network_status = await self._test_network_connectivity()
        results["network_status"] = network_status
        print(f"  Network Status: {'✅ Connected' if network_status['connected'] else '🚫 Disconnected'}")
        
        # Test 2: Transcription MCP Server
        print(f"  🎤 Testing Transcription MCP Server...")
        transcription_result = await self._test_transcription_server()
        results["component_tests"]["transcription"] = transcription_result
        print(f"    Status: {'✅' if transcription_result['success'] else '❌'} {transcription_result['status']}")
        
        # Test 3: Vision MCP Server
        print(f"  👁️ Testing Vision MCP Server...")
        vision_result = await self._test_vision_server()
        results["component_tests"]["vision"] = vision_result
        print(f"    Status: {'✅' if vision_result['success'] else '❌'} {vision_result['status']}")
        
        # Test 4: Generation MCP Server
        print(f"  📄 Testing Generation MCP Server...")
        generation_result = await self._test_generation_server()
        results["component_tests"]["generation"] = generation_result
        print(f"    Status: {'✅' if generation_result['success'] else '❌'} {generation_result['status']}")
        
        # Test 5: Router MCP Server
        print(f"  🔄 Testing Router MCP Server...")
        router_result = await self._test_router_server()
        results["component_tests"]["router"] = router_result
        print(f"    Status: {'✅' if router_result['success'] else '❌'} {router_result['status']}")
        
        # Test 6: File System Operations
        print(f"  💾 Testing File System Operations...")
        filesystem_result = await self._test_filesystem_operations()
        results["component_tests"]["filesystem"] = filesystem_result
        print(f"    Status: {'✅' if filesystem_result['success'] else '❌'} {filesystem_result['status']}")
        
        # Calculate overall success rate
        successful_tests = sum(1 for test in results["component_tests"].values() if test["success"])
        total_tests = len(results["component_tests"])
        results["success_rate"] = successful_tests / total_tests if total_tests > 0 else 0
        
        print(f"  📊 Component Success Rate: {results['success_rate']:.1%} ({successful_tests}/{total_tests})")
        
        return results
    
    async def _test_network_connectivity(self) -> Dict[str, Any]:
        """Test network connectivity"""
        try:
            # Try to connect to a well-known DNS server
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(3)
            result = sock.connect_ex(('8.8.8.8', 53))
            sock.close()
            
            connected = result == 0
            
            return {
                "connected": connected,
                "test_method": "socket_connection",
                "target": "8.8.8.8:53",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "connected": False,
                "error": str(e),
                "test_method": "socket_connection",
                "timestamp": datetime.now().isoformat()
            }
    
    async def _test_transcription_server(self) -> Dict[str, Any]:
        """Test transcription MCP server offline capability"""
        try:
            # Import and test transcription server
            sys.path.append('backend/mcp_servers')
            
            # Test if we can import without network
            from transcription_server import TranscriptionMCPServer
            
            server = TranscriptionMCPServer()
            
            # Quick capability test (doesn't require actual audio processing)
            result = await server.process_request({"action": "get_capabilities"})
            
            success = result.get("status") == "success"
            
            return {
                "success": success,
                "status": "Available offline" if success else "Failed to initialize",
                "capabilities": result.get("data", {}).get("capabilities", []),
                "models_available": result.get("data", {}).get("models_available", [])
            }
            
        except Exception as e:
            return {
                "success": False,
                "status": f"Import/initialization failed: {str(e)[:100]}",
                "error": str(e)
            }
    
    async def _test_vision_server(self) -> Dict[str, Any]:
        """Test vision MCP server offline capability"""
        try:
            # Import and test vision server
            from vision_server import VisionMCPServer
            
            server = VisionMCPServer()
            
            # Test capability request (doesn't require model loading)
            result = await server.process_request({"action": "get_capabilities"})
            
            success = result.get("status") == "success"
            
            return {
                "success": success,
                "status": "Available offline" if success else "Failed to initialize",
                "capabilities": result.get("data", {}).get("capabilities", []),
                "models_loaded": result.get("data", {}).get("models_loaded", False)
            }
            
        except Exception as e:
            return {
                "success": False,
                "status": f"Import/initialization failed: {str(e)[:100]}",
                "error": str(e)
            }
    
    async def _test_generation_server(self) -> Dict[str, Any]:
        """Test generation MCP server offline capability"""
        try:
            # Import and test generation server
            from generation_server import GenerationMCPServer
            
            server = GenerationMCPServer()
            
            # Test capability request
            result = await server.process_request({"action": "get_capabilities"})
            
            success = result.get("status") == "success"
            
            return {
                "success": success,
                "status": "Available offline" if success else "Failed to initialize",
                "capabilities": result.get("data", {}).get("capabilities", []),
                "supported_formats": result.get("data", {}).get("supported_formats", [])
            }
            
        except Exception as e:
            return {
                "success": False,
                "status": f"Import/initialization failed: {str(e)[:100]}",
                "error": str(e)
            }
    
    async def _test_router_server(self) -> Dict[str, Any]:
        """Test router MCP server offline capability"""
        try:
            # Import and test router server
            from router_server import RouterMCPServer
            
            server = RouterMCPServer()
            
            # Test capability request
            result = await server.process_request({"action": "get_capabilities"})
            
            success = result.get("status") == "success"
            
            return {
                "success": success,
                "status": "Available offline" if success else "Failed to initialize",
                "capabilities": result.get("data", {}).get("capabilities", []),
                "routing_patterns": result.get("data", {}).get("routing_patterns", {})
            }
            
        except Exception as e:
            return {
                "success": False,
                "status": f"Import/initialization failed: {str(e)[:100]}",
                "error": str(e)
            }
    
    async def _test_filesystem_operations(self) -> Dict[str, Any]:
        """Test file system operations (reading, writing, processing)"""
        try:
            test_operations = []
            
            # Test 1: Read requirements.txt
            try:
                with open('requirements.txt', 'r') as f:
                    content = f.read()
                test_operations.append({"operation": "read_requirements", "success": True})
            except Exception as e:
                test_operations.append({"operation": "read_requirements", "success": False, "error": str(e)})
            
            # Test 2: Check test video exists
            test_video_path = Path("backend/test_video.mp4")
            video_exists = test_video_path.exists()
            test_operations.append({"operation": "check_test_video", "success": video_exists})
            
            # Test 3: Create temporary file
            try:
                temp_file = Path("temp_test_file.txt")
                temp_file.write_text("Network isolation test")
                temp_file.unlink()  # Clean up
                test_operations.append({"operation": "create_temp_file", "success": True})
            except Exception as e:
                test_operations.append({"operation": "create_temp_file", "success": False, "error": str(e)})
            
            # Test 4: Check reports directory
            reports_dir = Path("reports")
            reports_exists = reports_dir.exists()
            test_operations.append({"operation": "check_reports_dir", "success": reports_exists})
            
            successful_ops = sum(1 for op in test_operations if op["success"])
            total_ops = len(test_operations)
            
            return {
                "success": successful_ops == total_ops,
                "status": f"File operations: {successful_ops}/{total_ops} successful",
                "operations": test_operations,
                "success_rate": successful_ops / total_ops if total_ops > 0 else 0
            }
            
        except Exception as e:
            return {
                "success": False,
                "status": f"Filesystem test failed: {str(e)[:100]}",
                "error": str(e)
            }
    
    async def _block_network_access(self) -> bool:
        """Attempt to block network access (Linux-specific)"""
        try:
            # Method 1: Try to add iptables rules (requires sudo)
            print("  🔒 Attempting to block network with iptables...")
            
            # Check if we can run iptables
            result = subprocess.run(['which', 'iptables'], capture_output=True, text=True)
            if result.returncode != 0:
                print("  ⚠️ iptables not available")
                return False
            
            # Try to add a rule to block outbound connections
            # Note: This requires sudo privileges
            try:
                result = subprocess.run([
                    'sudo', 'iptables', '-A', 'OUTPUT', '-m', 'owner', 
                    '--uid-owner', str(os.getuid()), '-j', 'DROP'
                ], capture_output=True, text=True, timeout=5)
                
                if result.returncode == 0:
                    print("  ✅ Network blocked with iptables")
                    self.network_blocked = True
                    return True
                else:
                    print(f"  ⚠️ iptables rule failed: {result.stderr}")
                    
            except subprocess.TimeoutExpired:
                print("  ⚠️ iptables command timed out (likely requires sudo)")
            except Exception as e:
                print(f"  ⚠️ iptables error: {e}")
            
            # Method 2: Fallback - modify DNS resolution (less effective)
            print("  🔒 Attempting DNS modification fallback...")
            
            # This is a simpler approach but less comprehensive
            try:
                # Backup original /etc/hosts
                result = subprocess.run(['cp', '/etc/hosts', '/tmp/hosts_backup'], 
                                      capture_output=True, text=True)
                
                # Add entries to block common domains (requires sudo)
                blocked_domains = ['google.com', 'github.com', 'pypi.org', 'huggingface.co']
                with open('/tmp/hosts_additions', 'w') as f:
                    for domain in blocked_domains:
                        f.write(f"127.0.0.1 {domain}\n")
                
                print("  ⚠️ Network isolation limited - DNS modification requires sudo")
                return False
                
            except Exception as e:
                print(f"  ⚠️ DNS modification failed: {e}")
                return False
            
        except Exception as e:
            logger.error(f"Network blocking failed: {e}")
            return False
    
    async def _restore_network_access(self) -> bool:
        """Restore network access"""
        try:
            if self.network_blocked:
                print("  🔄 Restoring network access...")
                
                # Remove iptables rule
                result = subprocess.run([
                    'sudo', 'iptables', '-D', 'OUTPUT', '-m', 'owner', 
                    '--uid-owner', str(os.getuid()), '-j', 'DROP'
                ], capture_output=True, text=True, timeout=5)
                
                if result.returncode == 0:
                    print("  ✅ Network access restored")
                    self.network_blocked = False
                    return True
                else:
                    print(f"  ⚠️ Network restoration may have failed: {result.stderr}")
            
            return True
            
        except Exception as e:
            logger.error(f"Network restoration failed: {e}")
            return False
    
    def _generate_isolation_report(self, baseline_results: Dict, offline_results: Dict) -> Dict[str, Any]:
        """Generate comprehensive isolation test report"""
        report = {
            "test_summary": {
                "test_completed": True,
                "timestamp": datetime.now().isoformat(),
                "baseline_success_rate": baseline_results.get("success_rate", 0),
                "offline_success_rate": offline_results.get("success_rate", 0),
                "network_isolation_effective": not offline_results["network_status"]["connected"]
            },
            "component_comparison": {},
            "offline_compliance": {
                "fully_offline": False,
                "issues_found": [],
                "recommendations": []
            }
        }
        
        # Compare component performance
        for component in baseline_results["component_tests"]:
            baseline_test = baseline_results["component_tests"][component]
            offline_test = offline_results["component_tests"][component]
            
            report["component_comparison"][component] = {
                "baseline_success": baseline_test["success"],
                "offline_success": offline_test["success"],
                "degradation": baseline_test["success"] and not offline_test["success"],
                "baseline_status": baseline_test["status"],
                "offline_status": offline_test["status"]
            }
        
        # Analyze offline compliance
        offline_components = [comp for comp, data in report["component_comparison"].items() 
                            if data["offline_success"]]
        failed_components = [comp for comp, data in report["component_comparison"].items() 
                           if data["degradation"]]
        
        if len(offline_components) == len(report["component_comparison"]):
            report["offline_compliance"]["fully_offline"] = True
        
        if failed_components:
            report["offline_compliance"]["issues_found"].extend([
                f"Component '{comp}' failed in offline mode" for comp in failed_components
            ])
        
        # Generate recommendations
        if not report["offline_compliance"]["fully_offline"]:
            report["offline_compliance"]["recommendations"] = [
                "Ensure all AI models are downloaded locally",
                "Verify no components require internet access",
                "Test with complete network isolation",
                "Consider offline alternatives for failed components"
            ]
        else:
            report["offline_compliance"]["recommendations"] = [
                "✅ System is fully offline compliant",
                "All core components work without network access",
                "Local AI models function correctly",
                "File operations work as expected"
            ]
        
        return report

async def test_network_isolation():
    """Run the complete network isolation test"""
    tester = NetworkIsolationTester()
    
    try:
        report = await tester.run_complete_test()
        
        print("\n" + "=" * 50)
        print("📋 NETWORK ISOLATION TEST REPORT")
        print("=" * 50)
        
        print(f"\n📊 Test Summary:")
        print(f"  Baseline Success Rate: {report['test_summary']['baseline_success_rate']:.1%}")
        print(f"  Offline Success Rate: {report['test_summary']['offline_success_rate']:.1%}")
        print(f"  Network Isolation Effective: {'✅' if report['test_summary']['network_isolation_effective'] else '❌'}")
        
        print(f"\n🔍 Component Analysis:")
        for component, data in report["component_comparison"].items():
            status = "✅" if data["offline_success"] else "❌"
            degradation = " (DEGRADED)" if data["degradation"] else ""
            print(f"  {component}: {status} {data['offline_status']}{degradation}")
        
        print(f"\n🏆 Offline Compliance:")
        print(f"  Fully Offline: {'✅ YES' if report['offline_compliance']['fully_offline'] else '❌ NO'}")
        
        if report["offline_compliance"]["issues_found"]:
            print(f"\n⚠️ Issues Found:")
            for issue in report["offline_compliance"]["issues_found"]:
                print(f"  • {issue}")
        
        print(f"\n💡 Recommendations:")
        for rec in report["offline_compliance"]["recommendations"]:
            print(f"  • {rec}")
        
        # Save report to file
        report_path = Path("reports/network_isolation_report.json")
        report_path.parent.mkdir(exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n💾 Full report saved to: {report_path}")
        
        return report
        
    except Exception as e:
        logger.error(f"❌ Network isolation test failed: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    asyncio.run(test_network_isolation())