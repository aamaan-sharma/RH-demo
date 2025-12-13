#!/usr/bin/env python3
"""
Test script to verify fsspec GCS filesystem initialization
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from app import gcs_fs, GCP_BUCKET_NAME, GCP_PROJECT_ID

def test_gcs_fs():
    print("Testing GCS filesystem initialization...")
    print("=" * 60)
    
    if gcs_fs is None:
        print("❌ ERROR: gcs_fs is None")
        print("   GCP Storage filesystem was not initialized")
        return False
    else:
        print(f"✓ gcs_fs is initialized: {type(gcs_fs)}")
        print(f"  Bucket: {GCP_BUCKET_NAME}")
        print(f"  Project: {GCP_PROJECT_ID}")
        
        # Try to list files
        try:
            bucket_path = f"gs://{GCP_BUCKET_NAME}/"
            print(f"\nTesting connection to: {bucket_path}")
            files = gcs_fs.ls(bucket_path, detail=False)
            print(f"✓ SUCCESS: Found {len(files)} files")
            print(f"\nFirst 5 files:")
            for f in files[:5]:
                print(f"  - {f}")
            return True
        except Exception as e:
            print(f"❌ ERROR accessing bucket: {e}")
            print(f"   Error type: {type(e).__name__}")
            import traceback
            traceback.print_exc()
            return False

if __name__ == "__main__":
    success = test_gcs_fs()
    sys.exit(0 if success else 1)

