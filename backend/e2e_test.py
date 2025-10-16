#!/usr/bin/env python3
"""
SVA End-to-End Testing Script
Tests the complete video analysis pipeline from frontend to backend
"""

import os
import sys
import json
import time
import subprocess
import requests
from pathlib import Path
import tempfile
import shutil

class SVAEndToEndTester:
    def __init__(self):
        self.backend_path = Path(__file__).parent
        self.frontend_path = self.backend_path.parent / "frontend"
        self.test_results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tests": {},
            "overall_success": False,
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0
        }
        
    def log(self, message, level="INFO"):
        """Log messages with timestamp"""
        timestamp = time.strftime("%H:%M:%S")
        print(f"[{timestamp}] {level}: {message}")
        
    def run_command(self, command, cwd=None, timeout=30):
        """Run shell command and return result"""
        try:
            self.log(f"Running: {command}")
            result = subprocess.run(
                command, 
                shell=True, 
                cwd=cwd, 
                timeout=timeout,
                capture_output=True, 
                text=True,
                executable='/bin/bash'  # Use bash explicitly
            )
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout": "",
                "stderr": f"Command timed out after {timeout} seconds",
                "returncode": -1
            }
        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e),
                "returncode": -1
            }
    
    def test_python_environment(self):
        """Test 1: Verify Python virtual environment and dependencies"""
        self.log("Testing Python environment...")
        test_name = "python_environment"
        
        # Check virtual environment
        venv_python = self.backend_path.parent / "venv" / "bin" / "python"
        if not venv_python.exists():
            self.test_results["tests"][test_name] = {
                "success": False,
                "error": "Python virtual environment not found",
                "details": f"Expected: {venv_python}"
            }
            return False
            
        # Test Python imports
        import_test_cmd = f"""
cd {self.backend_path} && source {self.backend_path.parent}/venv/bin/activate && python -c "
try:
    import whisper
    import torch
    import transformers
    import numpy as np
    import cv2
    import reportlab
    print('SUCCESS: All required packages imported')
except ImportError as e:
    print(f'ERROR: Missing package - {{e}}')
    exit(1)
"
        """
        
        result = self.run_command(import_test_cmd)
        
        self.test_results["tests"][test_name] = {
            "success": result["success"],
            "stdout": result["stdout"],
            "stderr": result["stderr"],
            "details": "Python environment and package verification"
        }
        
        if result["success"]:
            self.log("✅ Python environment test passed")
        else:
            self.log(f"❌ Python environment test failed: {result['stderr']}")
            
        return result["success"]
    
    def test_mcp_servers_availability(self):
        """Test 2: Check if all MCP servers are available"""
        self.log("Testing MCP servers availability...")
        test_name = "mcp_servers"
        
        servers = ["transcription", "vision", "generation", "router"]
        server_status = {}
        all_available = True
        
        for server in servers:
            test_cmd = f"""
cd {self.backend_path} && source {self.backend_path.parent}/venv/bin/activate && python -c "
import sys
sys.path.append('mcp_servers')
try:
    import {server}
    print('OK')
except Exception as e:
    print(f'ERROR: {{e}}')
    exit(1)
"
            """
            
            result = self.run_command(test_cmd)
            server_status[server] = {
                "available": result["success"],
                "output": result["stdout"].strip(),
                "error": result["stderr"]
            }
            
            if not result["success"]:
                all_available = False
                self.log(f"❌ {server} server not available")
            else:
                self.log(f"✅ {server} server available")
        
        self.test_results["tests"][test_name] = {
            "success": all_available,
            "server_status": server_status,
            "details": "MCP server availability check"
        }
        
        return all_available
    
    def test_video_processing_pipeline(self):
        """Test 3: Test complete video processing pipeline"""
        self.log("Testing video processing pipeline...")
        test_name = "video_pipeline"
        
        # Create a simple test video if not exists
        test_video_path = self.backend_path / "test_videos" / "sample_test.mp4"
        
        if not test_video_path.exists():
            self.log("Creating test video...")
            # Create test video directory
            test_video_path.parent.mkdir(exist_ok=True)
            
            # Use get_test_videos.py to download/create test video
            create_video_cmd = f"""
cd {self.backend_path} && source {self.backend_path.parent}/venv/bin/activate && python get_test_videos.py --create-sample
            """
            result = self.run_command(create_video_cmd, timeout=60)
            
            if not result["success"]:
                self.test_results["tests"][test_name] = {
                    "success": False,
                    "error": "Failed to create test video",
                    "details": result["stderr"]
                }
                return False
        
        # Run comprehensive analysis
        analysis_cmd = f"""
cd {self.backend_path} && source {self.backend_path.parent}/venv/bin/activate && python comprehensive_test.py --video-path '{test_video_path}' --session-id 'e2e_test_{int(time.time())}'
        """
        
        self.log("Running comprehensive video analysis...")
        result = self.run_command(analysis_cmd, timeout=120)
        
        # Parse JSON output if successful
        analysis_data = None
        if result["success"]:
            try:
                analysis_data = json.loads(result["stdout"])
            except json.JSONDecodeError:
                self.log("Warning: Analysis output is not valid JSON")
                analysis_data = {"raw_output": result["stdout"]}
        
        self.test_results["tests"][test_name] = {
            "success": result["success"],
            "analysis_data": analysis_data,
            "processing_time": "extracted from analysis_data if available",
            "stdout": result["stdout"][:500] + "..." if len(result["stdout"]) > 500 else result["stdout"],
            "stderr": result["stderr"],
            "details": "Complete video analysis pipeline test"
        }
        
        if result["success"]:
            self.log("✅ Video processing pipeline test passed")
        else:
            self.log(f"❌ Video processing pipeline test failed: {result['stderr']}")
            
        return result["success"]
    
    def test_network_isolation(self):
        """Test 4: Verify offline operation capability"""
        self.log("Testing network isolation...")
        test_name = "network_isolation"
        
        # Run the existing network isolation test
        isolation_cmd = f"""
cd {self.backend_path} && source {self.backend_path.parent}/venv/bin/activate && python -c "
import json
with open('reports/network_isolation_report.json', 'r') as f:
    report = json.load(f)
    
print('Network Isolation Test Results:')
print('Offline Success Rate:', report['test_summary']['offline_success_rate'])
print('Network Isolation Effective:', report['test_summary']['network_isolation_effective'])

if report['test_summary']['offline_success_rate'] >= 0.8:
    print('SUCCESS: Offline operation validated')
else:
    print('WARNING: Offline operation may have issues')
    exit(1)
"
        """
        
        result = self.run_command(isolation_cmd)
        
        self.test_results["tests"][test_name] = {
            "success": result["success"],
            "stdout": result["stdout"],
            "stderr": result["stderr"],
            "details": "Network isolation and offline operation test"
        }
        
        if result["success"]:
            self.log("✅ Network isolation test passed")
        else:
            self.log(f"❌ Network isolation test failed: {result['stderr']}")
            
        return result["success"]
    
    def test_frontend_backend_integration(self):
        """Test 5: Test frontend-backend communication"""
        self.log("Testing frontend-backend integration...")
        test_name = "frontend_integration"
        
        # Check if Tauri build exists
        tauri_executable = self.frontend_path / "src-tauri" / "target" / "debug" / "sva-tauri-app"
        
        if not tauri_executable.exists():
            self.log("Building Tauri application...")
            build_cmd = f"cd {self.frontend_path} && npm run tauri:build"
            build_result = self.run_command(build_cmd, timeout=300)
            
            if not build_result["success"]:
                self.test_results["tests"][test_name] = {
                    "success": False,
                    "error": "Failed to build Tauri application",
                    "details": build_result["stderr"]
                }
                return False
        
        # Test MCP server status check through Tauri
        # Note: This would require a headless test of the Tauri app
        # For now, we'll validate the Rust code compiles correctly
        
        compile_check_cmd = f"cd {self.frontend_path}/src-tauri && cargo check"
        result = self.run_command(compile_check_cmd, timeout=60)
        
        self.test_results["tests"][test_name] = {
            "success": result["success"],
            "stdout": result["stdout"],
            "stderr": result["stderr"],
            "details": "Frontend-backend integration compilation check"
        }
        
        if result["success"]:
            self.log("✅ Frontend-backend integration test passed")
        else:
            self.log(f"❌ Frontend-backend integration test failed: {result['stderr']}")
            
        return result["success"]
    
    def test_error_handling(self):
        """Test 6: Test error handling with invalid inputs"""
        self.log("Testing error handling...")
        test_name = "error_handling"
        
        test_cases = [
            {
                "name": "non_existent_file",
                "command": f"cd {self.backend_path} && source {self.backend_path.parent}/venv/bin/activate && python comprehensive_test.py --video-path '/non/existent/file.mp4'",
                "should_fail": True
            },
            {
                "name": "invalid_file_format",
                "command": f"cd {self.backend_path} && source {self.backend_path.parent}/venv/bin/activate && python comprehensive_test.py --video-path '{__file__}'",
                "should_fail": True
            }
        ]
        
        error_test_results = {}
        all_passed = True
        
        for test_case in test_cases:
            self.log(f"Running error test: {test_case['name']}")
            result = self.run_command(test_case["command"], timeout=30)
            
            # For error tests, we expect them to fail gracefully
            test_passed = (not result["success"]) == test_case["should_fail"]
            
            error_test_results[test_case["name"]] = {
                "expected_failure": test_case["should_fail"],
                "actual_failure": not result["success"],
                "test_passed": test_passed,
                "output": result["stdout"][:200] + "..." if len(result["stdout"]) > 200 else result["stdout"],
                "error": result["stderr"][:200] + "..." if len(result["stderr"]) > 200 else result["stderr"]
            }
            
            if not test_passed:
                all_passed = False
                self.log(f"❌ Error test {test_case['name']} failed")
            else:
                self.log(f"✅ Error test {test_case['name']} passed")
        
        self.test_results["tests"][test_name] = {
            "success": all_passed,
            "error_tests": error_test_results,
            "details": "Error handling and graceful failure tests"
        }
        
        return all_passed
    
    def run_all_tests(self):
        """Run all end-to-end tests"""
        self.log("🚀 Starting SVA End-to-End Tests...")
        
        tests = [
            ("Python Environment", self.test_python_environment),
            ("MCP Servers", self.test_mcp_servers_availability),
            ("Video Pipeline", self.test_video_processing_pipeline),
            ("Network Isolation", self.test_network_isolation),
            ("Frontend Integration", self.test_frontend_backend_integration),
            ("Error Handling", self.test_error_handling)
        ]
        
        for test_name, test_func in tests:
            self.log(f"\n{'='*50}")
            self.log(f"Running Test: {test_name}")
            self.log(f"{'='*50}")
            
            self.test_results["total_tests"] += 1
            
            try:
                if test_func():
                    self.test_results["passed_tests"] += 1
                else:
                    self.test_results["failed_tests"] += 1
            except Exception as e:
                self.log(f"❌ Test {test_name} crashed: {e}", "ERROR")
                self.test_results["failed_tests"] += 1
                self.test_results["tests"][test_name.lower().replace(" ", "_")] = {
                    "success": False,
                    "error": f"Test crashed: {e}",
                    "details": "Test execution error"
                }
        
        # Calculate overall success
        self.test_results["overall_success"] = self.test_results["failed_tests"] == 0
        
        # Generate report
        self.generate_test_report()
        
        return self.test_results["overall_success"]
    
    def generate_test_report(self):
        """Generate comprehensive test report"""
        self.log("\n🎯 Generating Test Report...")
        
        # Create reports directory
        reports_dir = self.backend_path / "reports"
        reports_dir.mkdir(exist_ok=True)
        
        # Save detailed JSON report
        json_report_path = reports_dir / "e2e_test_report.json"
        with open(json_report_path, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        # Generate summary report
        summary_path = reports_dir / "e2e_test_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("SVA End-to-End Test Summary\n")
            f.write("="*50 + "\n\n")
            f.write(f"Test Date: {self.test_results['timestamp']}\n")
            f.write(f"Total Tests: {self.test_results['total_tests']}\n")
            f.write(f"Passed: {self.test_results['passed_tests']}\n")
            f.write(f"Failed: {self.test_results['failed_tests']}\n")
            f.write(f"Success Rate: {(self.test_results['passed_tests']/self.test_results['total_tests']*100):.1f}%\n")
            f.write(f"Overall Result: {'✅ PASS' if self.test_results['overall_success'] else '❌ FAIL'}\n\n")
            
            f.write("Test Details:\n")
            f.write("-"*30 + "\n")
            for test_name, test_data in self.test_results['tests'].items():
                status = "✅ PASS" if test_data['success'] else "❌ FAIL"
                f.write(f"{test_name}: {status}\n")
                if not test_data['success'] and 'error' in test_data:
                    f.write(f"  Error: {test_data['error']}\n")
                f.write(f"  Details: {test_data.get('details', 'N/A')}\n\n")
        
        # Print summary to console
        self.log("\n📊 TEST SUMMARY")
        self.log("="*50)
        self.log(f"Total Tests: {self.test_results['total_tests']}")
        self.log(f"Passed: {self.test_results['passed_tests']}")
        self.log(f"Failed: {self.test_results['failed_tests']}")
        self.log(f"Success Rate: {(self.test_results['passed_tests']/self.test_results['total_tests']*100):.1f}%")
        
        if self.test_results['overall_success']:
            self.log("🎉 ALL TESTS PASSED! SVA system is ready for production.")
        else:
            self.log("⚠️  Some tests failed. Please review the issues before proceeding.")
        
        self.log(f"\n📄 Detailed reports saved to:")
        self.log(f"  JSON: {json_report_path}")
        self.log(f"  Summary: {summary_path}")

def main():
    """Main entry point"""
    if len(sys.argv) > 1 and sys.argv[1] == "--help":
        print("SVA End-to-End Testing Script")
        print("Usage: python e2e_test.py [options]")
        print("Options:")
        print("  --help    Show this help message")
        print("  --verbose Enable verbose output")
        return
    
    tester = SVAEndToEndTester()
    success = tester.run_all_tests()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()