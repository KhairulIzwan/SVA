#!/usr/bin/env python3
"""Test real SVA report generation with actual analysis data"""

from report_server import ReportGenerationServer
import json
import os

def test_real_reports():
    print('🎯 Testing REAL SVA Report Generation with Chat Data...')
    server = ReportGenerationServer()

    # Test with multiple chat files
    chat_files = [
        'chat_1760653834095_rkyzli905.json',  # Your current chat
        'chat_1760615431012_depbcl162.json'   # Previous test chat
    ]
    
    for chat_file in chat_files:
        if os.path.exists(f'../chat_storage/{chat_file}'):
            chat_id = chat_file.replace('.json', '')
            print(f'\n📊 Testing with chat: {chat_id}')
            
            # Debug: Check what data is extracted
            with open(f'../chat_storage/{chat_file}', 'r') as f:
                chat_data = json.load(f)
            
            analysis_data = server._extract_analysis_from_chat(chat_data)
            print(f'🔍 Extracted data:')
            print(f'  Video filename: {analysis_data.get("video_filename")}')
            print(f'  Spoken text count: {len(analysis_data.get("spoken_text", []))}')
            print(f'  Visual text count: {len(analysis_data.get("visual_text", []))}')
            print(f'  Objects count: {len(analysis_data.get("objects_detected", []))}')
            if analysis_data.get("spoken_text"):
                print(f'  First spoken text: {analysis_data["spoken_text"][0][:50]}...')
            
            # Test PDF generation using chat data
            print('� Generating PDF from real chat data...')
            pdf_result = server.generate_report_from_chat(
                chat_id=chat_id,
                video_filename='sample_video',
                format_type='pdf'
            )
            print(f'PDF Result: {pdf_result}')
        else:
            print(f'❌ Chat file not found: {chat_file}')

    print('\n✅ Real SVA Report Generation Test Complete!')
    
    # List generated files
    report_dir = '../generated_reports'
    if os.path.exists(report_dir):
        print(f'\n📁 Generated files in {report_dir}:')
        for file in os.listdir(report_dir):
            path = os.path.join(report_dir, file)
            size = os.path.getsize(path)
            print(f'  📄 {file} ({size:,} bytes)')

if __name__ == '__main__':
    test_real_reports()