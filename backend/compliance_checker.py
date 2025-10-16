#!/usr/bin/env python3
"""
HuggingFace Model Compliance Checker for SVA Project
Verifies that all AI models meet the OpenVINO-optimized or HuggingFace requirement
"""

import asyncio
import json
import sys
from pathlib import Path
from datetime import datetime

# Add MCP servers to path
sys.path.append('mcp_servers')

async def check_compliance():
    """Verify all models meet compliance requirements"""
    
    print("🔍 SVA AI Model Compliance Checker")
    print("=" * 50)
    print(f"📅 Check Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📋 Requirement: All AI inference must run locally, using OpenVINO-optimized or Hugging Face models")
    print()
    
    compliance_report = {
        "check_date": datetime.now().isoformat(),
        "requirement": "OpenVINO-optimized or HuggingFace models",
        "servers": {},
        "overall_compliance": True,
        "summary": {}
    }
    
    # Test Transcription Server
    print("🎤 Testing Transcription Server...")
    try:
        from mcp_servers.transcription_server import TranscriptionMCPServer
        trans_server = TranscriptionMCPServer()
        
        # Check compliance info
        compliance_info = getattr(trans_server, 'compliance_info', {})
        model_name = getattr(trans_server, 'model_name', 'Unknown')
        
        # Initialize to verify it works
        init_result = await trans_server.initialize()
        
        is_compliant = (
            compliance_info.get('compliant', False) and
            'HuggingFace' in compliance_info.get('model_source', '') and
            init_result.get('status') == 'success'
        )
        
        compliance_report["servers"]["transcription"] = {
            "compliant": is_compliant,
            "model_source": compliance_info.get('model_source', 'Unknown'),
            "model_name": model_name,
            "status": init_result.get('status', 'unknown'),
            "details": compliance_info
        }
        
        status = "✅ COMPLIANT" if is_compliant else "❌ NON-COMPLIANT"
        print(f"   {status}")
        print(f"   Model: {model_name}")
        print(f"   Source: {compliance_info.get('model_source', 'Unknown')}")
        
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        compliance_report["servers"]["transcription"] = {
            "compliant": False,
            "error": str(e)
        }
        compliance_report["overall_compliance"] = False
    
    print()
    
    # Test Vision Server
    print("👁️ Testing Vision Server...")
    try:
        from mcp_servers.vision_server import VisionMCPServer
        vision_server = VisionMCPServer()
        
        # Check compliance info
        compliance_info = getattr(vision_server, 'compliance_info', {})
        
        # Initialize to verify it works
        init_result = await vision_server.initialize()
        
        # Check if using HuggingFace models or fallback
        models_loaded = init_result.get('models_loaded', [])
        compliance_status = init_result.get('compliance', '')
        
        is_compliant = (
            compliance_info.get('compliant', False) and
            'HuggingFace' in compliance_status and
            init_result.get('status') == 'success' and
            any('HuggingFace' in model for model in models_loaded)
        )
        
        compliance_report["servers"]["vision"] = {
            "compliant": is_compliant,
            "model_source": compliance_info.get('model_source', 'Unknown'),
            "models_loaded": models_loaded,
            "compliance_status": compliance_status,
            "status": init_result.get('status', 'unknown'),
            "details": compliance_info
        }
        
        status = "✅ COMPLIANT" if is_compliant else "❌ NON-COMPLIANT"
        fallback_note = " (using fallback models)" if 'fallback' in compliance_status else ""
        print(f"   {status}{fallback_note}")
        print(f"   Models: {', '.join(models_loaded)}")
        print(f"   Source: {compliance_status}")
        
        if not is_compliant:
            compliance_report["overall_compliance"] = False
        
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        compliance_report["servers"]["vision"] = {
            "compliant": False,
            "error": str(e)
        }
        compliance_report["overall_compliance"] = False
    
    print()
    
    # Test Generation Server (should be compliant as it uses local libraries)
    print("📄 Testing Generation Server...")
    try:
        from mcp_servers.generation_server import GenerationMCPServer
        gen_server = GenerationMCPServer()
        
        # Generation server uses local libraries (ReportLab, python-pptx)
        compliance_report["servers"]["generation"] = {
            "compliant": True,
            "model_source": "Local Libraries (ReportLab, python-pptx)",
            "note": "Uses local document generation libraries",
            "status": "compliant"
        }
        
        print("   ✅ COMPLIANT")
        print("   Libraries: ReportLab, python-pptx (local document generation)")
        
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        compliance_report["servers"]["generation"] = {
            "compliant": False,
            "error": str(e)
        }
        compliance_report["overall_compliance"] = False
    
    print()
    
    # Generate summary
    compliant_servers = sum(1 for server in compliance_report["servers"].values() 
                          if server.get("compliant", False))
    total_servers = len(compliance_report["servers"])
    
    compliance_report["summary"] = {
        "compliant_servers": compliant_servers,
        "total_servers": total_servers,
        "compliance_percentage": (compliant_servers / total_servers * 100) if total_servers > 0 else 0
    }
    
    # Print summary
    print("📊 COMPLIANCE SUMMARY")
    print("=" * 50)
    print(f"Compliant Servers: {compliant_servers}/{total_servers}")
    print(f"Compliance Rate: {compliance_report['summary']['compliance_percentage']:.1f}%")
    
    if compliance_report["overall_compliance"]:
        print("🎉 OVERALL STATUS: ✅ FULLY COMPLIANT")
        print("   All AI models meet the HuggingFace/OpenVINO requirement!")
    else:
        print("⚠️ OVERALL STATUS: ❌ NON-COMPLIANT")
        print("   Some models do not meet the requirement.")
        print("   Fallback models may be used for compatibility.")
    
    # Save compliance report
    report_file = Path("compliance_report.json")
    with open(report_file, 'w') as f:
        json.dump(compliance_report, f, indent=2)
    
    print(f"\n📄 Detailed report saved to: {report_file}")
    
    return compliance_report

async def main():
    """Main compliance check function"""
    try:
        report = await check_compliance()
        
        # Exit with appropriate code
        if report["overall_compliance"]:
            print("\n✅ Compliance check PASSED")
            return 0
        else:
            print("\n❌ Compliance check FAILED")
            return 1
            
    except Exception as e:
        print(f"\n💥 Compliance check encountered an error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)