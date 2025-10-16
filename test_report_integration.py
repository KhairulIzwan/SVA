#!/usr/bin/env python3
"""
Test script to verify the complete report generation integration
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

import grpc
import sva_pb2
import sva_pb2_grpc
from pathlib import Path

def test_report_generation():
    """Test the complete report generation workflow"""
    
    # Connect to the gRPC server
    try:
        channel = grpc.insecure_channel('localhost:50051')
        stub = sva_pb2_grpc.SVAServiceStub(channel)
        
        print("🔗 Connected to gRPC server")
        
        # Test report generation
        request = sva_pb2.GenerateReportRequest(
            chat_id="test_chat_123",
            video_filename="test_video.mp4",
            format_type="pdf"
        )
        
        print("📋 Generating PDF report...")
        response = stub.GenerateReport(request)
        
        if response.success:
            print(f"✅ PDF Report generated successfully!")
            print(f"   📄 Filename: {response.filename}")
            print(f"   📂 Filepath: {response.filepath}")
            print(f"   📊 Format: {response.format}")
            print(f"   💾 Size: {response.size} bytes")
            print(f"   💬 Message: {response.message}")
            
            # Check if file exists
            if os.path.exists(response.filepath):
                print(f"   ✅ File confirmed to exist on disk")
            else:
                print(f"   ❌ File not found on disk!")
                
        else:
            print(f"❌ Report generation failed: {response.message}")
            
        # Test PPT generation
        print("\n📋 Generating PPT report...")
        request.format_type = "ppt"
        response = stub.GenerateReport(request)
        
        if response.success:
            print(f"✅ PPT Report generated successfully!")
            print(f"   📄 Filename: {response.filename}")
            print(f"   📂 Filepath: {response.filepath}")
            print(f"   📊 Format: {response.format}")
            print(f"   💾 Size: {response.size} bytes")
            print(f"   💬 Message: {response.message}")
            
            # Check if file exists
            if os.path.exists(response.filepath):
                print(f"   ✅ File confirmed to exist on disk")
            else:
                print(f"   ❌ File not found on disk!")
                
        else:
            print(f"❌ PPT generation failed: {response.message}")
            
        # Test listing reports
        print("\n📋 Listing generated reports...")
        list_request = sva_pb2.ListReportsRequest()
        list_response = stub.ListReports(list_request)
        
        if list_response.reports:
            print(f"✅ Found {len(list_response.reports)} reports:")
            for report in list_response.reports:
                print(f"   📄 {report.filename} ({report.format}) - {report.size} bytes")
        else:
            print("❌ No reports found")
            
    except grpc.RpcError as e:
        print(f"❌ gRPC Error: {e.code()} - {e.details()}")
    except Exception as e:
        print(f"❌ Error: {str(e)}")
    finally:
        if 'channel' in locals():
            channel.close()

if __name__ == "__main__":
    print("🧪 Testing Report Generation Integration")
    print("="*50)
    test_report_generation()
    print("="*50)
    print("🏁 Test completed")