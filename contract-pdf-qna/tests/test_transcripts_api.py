#!/usr/bin/env python3
"""
Test script to verify the /transcripts API endpoint
"""

import sys
import os
import certifi
import json

# Set SSL environment variables
os.environ['SSL_CERT_FILE'] = certifi.where()
os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
os.environ['AIOHTTP_CA_BUNDLE'] = certifi.where()

sys.path.insert(0, os.path.dirname(__file__))

from app import list_transcript_files_gcp, gcs_fs, GCP_BUCKET_NAME

def test_transcripts_api():
    """Test the transcript listing functionality"""
    print("=" * 70)
    print("Testing /transcripts API Endpoint Logic")
    print("=" * 70)
    print()
    
    # Check gcs_fs
    if not gcs_fs:
        print("❌ ERROR: gcs_fs is None!")
        print("   GCP Storage is not initialized")
        return False
    
    print(f"✅ gcs_fs is initialized: {type(gcs_fs)}")
    print(f"✅ Bucket: {GCP_BUCKET_NAME}")
    print()
    
    # Test the function that the API calls
    print("Calling list_transcript_files_gcp()...")
    print("-" * 70)
    
    try:
        all_transcripts = list_transcript_files_gcp()
        
        print(f"\n✅ SUCCESS: Found {len(all_transcripts)} transcripts")
        print()
        
        if len(all_transcripts) == 0:
            print("❌ ERROR: Empty result!")
            print("   This means the API will return an empty array")
            return False
        
        # Show sample data
        print("Sample Transcript Data (first 3):")
        print("-" * 70)
        for i, transcript in enumerate(all_transcripts[:3], 1):
            print(f"\n{i}. {transcript.get('fileName')}")
            print(f"   Size: {transcript.get('fileSize')} bytes")
            print(f"   Contract Type: {transcript.get('contractType')}")
            print(f"   Plan Type: {transcript.get('planType')}")
            print(f"   State: {transcript.get('state')}")
            print(f"   Upload Date: {transcript.get('uploadDate')}")
        
        # Simulate API response
        print()
        print("=" * 70)
        print("Simulated API Response:")
        print("=" * 70)
        
        limit = 50
        offset = 0
        paginated = all_transcripts[offset:offset + limit]
        
        response = {
            "transcripts": paginated,
            "totalCount": len(all_transcripts),
            "limit": limit,
            "offset": offset
        }
        
        print(f"totalCount: {response['totalCount']}")
        print(f"limit: {response['limit']}")
        print(f"offset: {response['offset']}")
        print(f"transcripts: [{len(response['transcripts'])} items]")
        print()
        
        # Check if metadata is present
        with_metadata = sum(1 for t in all_transcripts if t.get('contractType') and t.get('planType') and t.get('state'))
        print(f"Transcripts with full metadata: {with_metadata}/{len(all_transcripts)} ({with_metadata/len(all_transcripts)*100:.1f}%)")
        print()
        
        print("=" * 70)
        print("✅ TEST PASSED: API endpoint logic is working correctly!")
        print("=" * 70)
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_transcripts_api()
    sys.exit(0 if success else 1)

