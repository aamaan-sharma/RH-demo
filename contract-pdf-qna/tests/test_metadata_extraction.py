#!/usr/bin/env python3
"""
Test script to verify metadata extraction from transcript files
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from app import extract_transcript_metadata, list_transcript_files_gcp, gcs_fs, gcs_fs

def test_metadata_extraction():
    """Test metadata extraction on actual transcript files"""
    print("Testing metadata extraction from transcript files using fsspec...")
    print("=" * 60)
    
    # Check if fsspec is initialized
    if not gcs_fs:
        print("ERROR: GCS filesystem not initialized. Check your GCP authentication.")
        return
    
    # Test with actual files from GCP
    try:
        transcripts = list_transcript_files_gcp()
        
        print(f"\nFound {len(transcripts)} transcript files\n")
        print("Sample results (first 5 files):")
        print("-" * 60)
        
        for i, transcript in enumerate(transcripts[:5], 1):
            print(f"\n{i}. {transcript['fileName']}")
            print(f"   Contract Type: {transcript.get('contractType', 'Not found')}")
            print(f"   Plan Type: {transcript.get('planType', 'Not found')}")
            print(f"   State: {transcript.get('state', 'Not found')}")
            print(f"   Size: {transcript.get('fileSize', 0)} bytes")
        
        # Count how many have metadata extracted
        with_contract = sum(1 for t in transcripts if t.get('contractType'))
        with_plan = sum(1 for t in transcripts if t.get('planType'))
        with_state = sum(1 for t in transcripts if t.get('state'))
        
        print("\n" + "=" * 60)
        print("Summary:")
        print(f"  Total files: {len(transcripts)}")
        print(f"  With contractType: {with_contract} ({with_contract/len(transcripts)*100:.1f}%)")
        print(f"  With planType: {with_plan} ({with_plan/len(transcripts)*100:.1f}%)")
        print(f"  With state: {with_state} ({with_state/len(transcripts)*100:.1f}%)")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_metadata_extraction()

