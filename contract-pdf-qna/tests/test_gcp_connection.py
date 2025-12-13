#!/usr/bin/env python3
"""
Test script to verify GCP Storage connection
Run this before testing the APIs
"""

from google.cloud import storage
import sys

def test_gcp_connection():
    """Test GCP Storage connection"""
    try:
        project_id = "generative-ai-390411"
        bucket_name = "ahs-demo-transcripts"
        
        print("Testing GCP Storage connection...")
        print(f"Project: {project_id}")
        print(f"Bucket: {bucket_name}")
        print("-" * 50)
        
        # Initialize client
        client = storage.Client(project=project_id)
        bucket = client.bucket(bucket_name)
        
        # Test connection
        bucket.reload()
        print("✓ GCP Storage connection successful!")
        print()
        
        # List files
        print("Listing files in bucket...")
        blobs = list(bucket.list_blobs(max_results=10))
        
        if len(blobs) == 0:
            print("⚠ No files found in bucket")
            print("  Make sure transcripts are uploaded to the bucket")
        else:
            print(f"✓ Found {len(blobs)} file(s):")
            for blob in blobs:
                print(f"  - {blob.name} ({blob.size} bytes)")
                if blob.time_created:
                    print(f"    Uploaded: {blob.time_created}")
        
        print()
        print("✓ GCP Storage is ready to use!")
        return True
        
    except Exception as e:
        print(f"✗ GCP Storage connection failed!")
        print(f"  Error: {e}")
        print()
        print("To fix this:")
        print("  1. Run: gcloud auth application-default login")
        print("  2. Make sure you have access to the bucket")
        print("  3. Verify project ID and bucket name are correct")
        return False

if __name__ == "__main__":
    success = test_gcp_connection()
    sys.exit(0 if success else 1)

