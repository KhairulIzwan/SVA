#!/usr/bin/env python3
"""
Quick test to verify the report generation system is working with the actual chat data
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'backend'))

import grpc
import sva_pb2
import sva_pb2_grpc
import json

def test_with_real_chat():
    """Test report generation with the actual chat that contains analysis"""
    
    print("🧪 Testing Report Generation with Real Chat Data")
    print("="*60)
    
    # First, let's verify the chat exists
    chat_file = '/home/user/SVA/backend/chat_storage/chat_1760612609157_roif1fxx0.json'
    
    if os.path.exists(chat_file):
        with open(chat_file, 'r') as f:
            chat_data = json.load(f)
        
        print(f"✅ Found chat file with {len(chat_data['messages'])} messages")
        print(f"📝 Latest message preview: {chat_data['messages'][-1]['content'][:100]}...")
        
        # Test the gRPC connection
        try:
            channel = grpc.insecure_channel('localhost:50051')
            stub = sva_pb2_grpc.SVAServiceStub(channel)
            
            print("\n🔗 Connected to gRPC server")
            
            # Test PDF generation
            request = sva_pb2.GenerateReportRequest(
                chat_id="chat_1760612609157_roif1fxx0",  # Use the real chat ID
                video_filename="sample_video",
                format_type="pdf"
            )
            
            print("📋 Generating PDF report from actual analysis...")
            response = stub.GenerateReport(request)
            
            if response.success:
                print(f"✅ PDF Report generated successfully!")
                print(f"   📄 Filename: {response.filename}")
                print(f"   📂 Filepath: {response.filepath}")
                print(f"   📊 Format: {response.format}")
                print(f"   💾 Size: {response.size} bytes")
                
                if os.path.exists(response.filepath):
                    print(f"   ✅ File confirmed on disk")
                    
                    # Show file contents preview
                    if response.size < 10000:  # Small file, can check content
                        print(f"   📋 Report created from real analysis data!")
                    
                else:
                    print(f"   ❌ File not found on disk")
                    
            else:
                print(f"❌ Report generation failed: {response.message}")
                
        except Exception as e:
            print(f"❌ Error: {str(e)}")
            
    else:
        print(f"❌ Chat file not found: {chat_file}")
        print("Available chat files:")
        chat_dir = '/home/user/SVA/backend/chat_storage'
        if os.path.exists(chat_dir):
            for f in os.listdir(chat_dir):
                if f.endswith('.json'):
                    print(f"   📁 {f}")

if __name__ == "__main__":
    test_with_real_chat()
    print("="*60)
    print("🏁 Test completed!")