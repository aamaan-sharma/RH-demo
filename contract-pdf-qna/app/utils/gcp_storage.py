"""GCP Storage utilities for transcript file operations.

Handles GCS bucket operations using fsspec/gcsfs with SSL certificate configuration.
"""
import os
import ssl
import json
import re
from typing import Dict, List, Optional, Tuple, Any
from app.config.settings import settings
from app.utils.milvus_utils import CLEAR_STATE_ALIASES

# GCP Storage availability flag
GCP_STORAGE_AVAILABLE = False
gcs_fs = None
certifi = None

# Try to import GCP Storage dependencies
try:
    import fsspec
    import gcsfs
    import certifi
    
    # Configure SSL certificates for macOS compatibility
    # CRITICAL: Set these BEFORE creating any filesystem objects
    cert_path = certifi.where()
    
    # Always set these (don't check if already set - ensure they're correct)
    os.environ['SSL_CERT_FILE'] = cert_path
    os.environ['REQUESTS_CA_BUNDLE'] = cert_path
    os.environ['AIOHTTP_CA_BUNDLE'] = cert_path
    
    # Create SSL context with certifi certificates
    ssl_context = ssl.create_default_context(cafile=cert_path)
    
    print(f"✓ SSL certificates configured: {cert_path}")
    
    GCP_STORAGE_AVAILABLE = True
except ImportError:
    print("Warning: fsspec or gcsfs not installed. GCP Storage features disabled.")
    print("Install with: pip install fsspec gcsfs")
    GCP_STORAGE_AVAILABLE = False
    fsspec = None
    gcsfs = None
    certifi = None
    ssl_context = None

# Cache for transcript metadata to avoid re-reading files
transcript_metadata_cache: Dict[str, Dict[str, Optional[str]]] = {}
# Bump this when metadata extraction logic changes to avoid serving stale cached None values.
TRANSCRIPT_METADATA_CACHE_VERSION = "v2"


class GCPStorageService:
    """Service for GCP Storage operations."""
    
    def __init__(self, bucket_name: str, project_id: str, service_account_path: Optional[str] = None):
        """Initialize GCP Storage service.
        
        Args:
            bucket_name: GCS bucket name
            project_id: GCP project ID
            service_account_path: Optional path to service account JSON file
        """
        self.bucket_name = bucket_name
        self.project_id = project_id
        self.service_account_path = service_account_path
        self.fs = None
        self._initialize_filesystem()
    
    def _initialize_filesystem(self) -> None:
        """Initialize fsspec filesystem for GCS."""
        global gcs_fs
        
        if not GCP_STORAGE_AVAILABLE:
            print("GCP Storage not available - dependencies missing")
            return
        
        try:
            # Ensure SSL certificates are set
            if certifi:
                cert_path = certifi.where()
                if 'SSL_CERT_FILE' not in os.environ:
                    os.environ['SSL_CERT_FILE'] = cert_path
                if 'REQUESTS_CA_BUNDLE' not in os.environ:
                    os.environ['REQUESTS_CA_BUNDLE'] = cert_path
                if 'AIOHTTP_CA_BUNDLE' not in os.environ:
                    os.environ['AIOHTTP_CA_BUNDLE'] = cert_path
            
            if self.service_account_path and os.path.exists(self.service_account_path):
                # Use explicit service account file
                self.fs = fsspec.filesystem('gcs', token=self.service_account_path, project=self.project_id)
                print(f"✓ GCP Storage initialized using fsspec with service account from: {self.service_account_path}")
            else:
                # Use Application Default Credentials (ADC)
                try:
                    from google.auth import default as google_auth_default
                    
                    cert_path = certifi.where() if certifi else None
                    
                    # Get ADC credentials explicitly
                    credentials, _ = google_auth_default()
                    
                    # Create filesystem with explicit ADC credentials
                    self.fs = fsspec.filesystem('gcs', token=credentials, project=self.project_id)
                    print(f"✓ GCP Storage filesystem created using fsspec")
                    print(f"  Bucket: {self.bucket_name}")
                    print(f"  Project: {self.project_id}")
                    if cert_path:
                        print(f"  SSL Certificates: {cert_path}")
                    print(f"  Using Application Default Credentials")
                    
                    # Optional: Test connection
                    try:
                        bucket_path = f"gs://{self.bucket_name}/"
                        test_files = self.fs.ls(bucket_path, detail=False)
                        print(f"  ✓ Connection test successful - Found {len(test_files)} files in bucket")
                    except Exception as test_error:
                        error_msg = str(test_error)
                        if "SSL" in error_msg or "certificate" in error_msg.lower():
                            print(f"  ⚠ SSL certificate issue detected (common on macOS)")
                        else:
                            print(f"  ⚠ Connection test failed: {test_error}")
                except Exception as e:
                    print(f"✗ GCP Storage filesystem creation failed: {e}")
                    self.fs = None
            
            gcs_fs = self.fs
        except Exception as e:
            print(f"✗ GCP Storage initialization failed: {e}")
            self.fs = None
            gcs_fs = None
    
    def list_transcript_files(
        self,
        limit: Optional[int] = None,
        offset: int = 0,
        search: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], int]:
        """List transcript files from GCP bucket with pagination and search support.
        
        Args:
            limit: Number of records per page (None = all)
            offset: Number of records to skip
            search: Search term to filter by file name
            
        Returns:
            Tuple of (transcripts_list, total_count)
        """
        if not self.fs:
            print(f"ERROR list_transcript_files: fs is None!")
            return ([], 0) if limit else []
        
        all_file_info: List[Dict[str, Any]] = []
        
        try:
            bucket_path = f"gs://{self.bucket_name}/"
            prefixes = ["transcripts/", ""]
            seen_files = set()
            
            for prefix in prefixes:
                try:
                    full_path = bucket_path + prefix if prefix else bucket_path
                    files = self.fs.ls(full_path, detail=True)
                    
                    for file_info in files:
                        if isinstance(file_info, str):
                            file_path = file_info
                            file_size = 0
                            time_created = None
                        else:
                            file_path = file_info.get('name', '')
                            file_size = file_info.get('size', 0)
                            time_created = file_info.get('timeCreated', None)
                        
                        if file_path.endswith('/'):
                            continue
                        
                        if not (file_path.endswith('.json') or file_path.endswith('.txt')):
                            continue
                        
                        file_name = file_path.split("/")[-1]
                        
                        if file_name in seen_files:
                            continue
                        seen_files.add(file_name)
                        
                        upload_date = None
                        if time_created:
                            if isinstance(time_created, str):
                                upload_date = time_created
                            else:
                                upload_date = time_created.isoformat() if hasattr(time_created, 'isoformat') else str(time_created)
                        
                        all_file_info.append({
                            "fileName": file_name,
                            "filePath": file_path,
                            "uploadDate": upload_date,
                            "fileSize": file_size if file_size else 0,
                            "timeCreated": time_created
                        })
                except Exception as e:
                    print(f"ERROR listing files with prefix '{prefix}': {e}")
                    continue
            
            # Sort by upload date (newest first)
            all_file_info.sort(key=lambda x: x.get("uploadDate", "") or "", reverse=True)
            total_files_from_gcs = len(all_file_info)
            
            # Apply search filter if provided
            if search and search.strip():
                search_term = search.strip().lower()
                matching_files = [
                    f for f in all_file_info
                    if search_term in f.get("fileName", "").lower()
                ]
                all_file_info = matching_files
            
            total_count = len(all_file_info)
            
            # If limit is None, return all (backward compatibility)
            if limit is None:
                transcripts = []
                for file_info in all_file_info:
                    transcript_metadata = self._extract_metadata_cached(file_info)
                    transcripts.append({
                        "fileName": file_info['fileName'],
                        "filePath": file_info['filePath'],
                        "uploadDate": file_info['uploadDate'],
                        "fileSize": file_info['fileSize'],
                        "metadata": {},
                        "contractType": transcript_metadata.get("contractType"),
                        "planType": transcript_metadata.get("planType"),
                        "state": transcript_metadata.get("state")
                    })
                return transcripts
            
            # Apply pagination
            paginated_file_info = all_file_info[offset:offset + limit]
            
            # Read file contents only for paginated subset
            transcripts = []
            for file_info in paginated_file_info:
                transcript_metadata = self._extract_metadata_cached(file_info)
                transcripts.append({
                    "fileName": file_info['fileName'],
                    "filePath": file_info['filePath'],
                    "uploadDate": file_info['uploadDate'],
                    "fileSize": file_info['fileSize'],
                    "metadata": {},
                    "contractType": transcript_metadata.get("contractType"),
                    "planType": transcript_metadata.get("planType"),
                    "state": transcript_metadata.get("state")
                })
            
            return (transcripts, total_count)
            
        except Exception as e:
            print(f"Error listing transcript files from GCP: {e}")
            import traceback
            traceback.print_exc()
            return ([], 0) if limit else []
    
    def read_transcript_file(self, file_name: str) -> Tuple[str, Dict[str, Any]]:
        """Read transcript file content from GCP bucket.
        
        Args:
            file_name: Name of the transcript file
            
        Returns:
            Tuple of (content, file_metadata_dict)
        """
        if not self.fs:
            raise Exception("GCP Storage not available")
        
        bucket_path = f"gs://{self.bucket_name}/"
        possible_paths = [
            f"{bucket_path}transcripts/{file_name}",
            f"{bucket_path}{file_name}",
        ]
        
        file_path = None
        for path in possible_paths:
            if self.fs.exists(path):
                file_path = path
                break
        
        if not file_path:
            raise FileNotFoundError(f"Transcript file not found: {file_name}")
        
        # Read file content
        with self.fs.open(file_path, 'r') as f:
            content = f.read()
        
        # Get file metadata
        file_info = self.fs.info(file_path)
        time_created = file_info.get('timeCreated', None)
        
        upload_date = None
        if time_created:
            if isinstance(time_created, str):
                upload_date = time_created
            else:
                upload_date = time_created.isoformat() if hasattr(time_created, 'isoformat') else str(time_created)
        
        file_metadata = {
            "fileName": file_name,
            "fileSize": file_info.get('size', 0),
            "uploadDate": upload_date,
            "metadata": {}
        }
        
        return content, file_metadata
    
    def _extract_metadata_cached(self, file_info: Dict[str, Any]) -> Dict[str, Optional[str]]:
        """Extract metadata from file with caching."""
        cache_key = f"{TRANSCRIPT_METADATA_CACHE_VERSION}_{file_info['filePath']}_{file_info.get('timeCreated')}"
        
        if cache_key in transcript_metadata_cache:
            return transcript_metadata_cache[cache_key]
        
        transcript_metadata = {
            "contractType": None,
            "planType": None,
            "state": None
        }
        
        try:
            file_size = file_info.get('fileSize', 0)
            if file_size and file_size < 50000:  # Only read files < 50KB for metadata extraction
                with self.fs.open(file_info['filePath'], 'r') as f:
                    content = f.read()
                transcript_metadata = extract_transcript_metadata(content, file_info['fileName'])
                transcript_metadata_cache[cache_key] = transcript_metadata
            elif file_size:
                print(f"Skipping metadata extraction for large file: {file_info['fileName']} ({file_size} bytes)")
        except Exception as e:
            print(f"Error reading transcript {file_info['fileName']} for metadata extraction: {e}")
        
        return transcript_metadata


def extract_transcript_metadata(transcript_content: str, file_name: str) -> Dict[str, Optional[str]]:
    """Extract contractType, planType, and state from transcript file content.
    
    Uses hybrid approach: JSON parsing -> Regex patterns -> LLM (if needed)
    
    Args:
        transcript_content: Content of the transcript file
        file_name: Name of the transcript file
        
    Returns:
        Dictionary with contractType, planType, and state
    """
    metadata = {
        "contractType": None,
        "planType": None,
        "state": None
    }
    
    try:
        # Method 1: Try parsing as JSON first (fastest)
        try:
            transcript_data = json.loads(transcript_content)
            if isinstance(transcript_data, dict):
                metadata_fields = transcript_data.get("metadata", {})
                if not metadata_fields:
                    metadata_fields = transcript_data
                
                metadata["contractType"] = (
                    metadata_fields.get("contractType") or 
                    metadata_fields.get("contract_type") or
                    metadata_fields.get("type")
                )
                
                metadata["planType"] = (
                    metadata_fields.get("planType") or
                    metadata_fields.get("plan_type") or
                    metadata_fields.get("selectedPlan") or
                    metadata_fields.get("selected_plan") or
                    metadata_fields.get("plan")
                )
                
                metadata["state"] = (
                    metadata_fields.get("state") or
                    metadata_fields.get("selectedState") or
                    metadata_fields.get("selected_state") or
                    metadata_fields.get("stateCode")
                )
                
                if all([metadata["contractType"], metadata["planType"], metadata["state"]]):
                    return metadata
        except json.JSONDecodeError:
            pass
        
        # Method 2: Regex-based text parsing
        content_upper = transcript_content.upper()
        
        # Extract contract type
        if re.search(r'\bRE\b', content_upper) or "REAL ESTATE" in content_upper:
            metadata["contractType"] = "RE"
        elif re.search(r'\bDTC\b', content_upper) or "DIRECT TO CONSUMER" in content_upper or "DIRECT-TO-CONSUMER" in content_upper:
            metadata["contractType"] = "DTC"
        
        # Extract plan type using regex patterns
        plan_patterns = {
            "ShieldComplete": [r"SHIELD\s*COMPLETE", r"SHIELDCOMPLETE", r"COMPLETE\s*PLAN"],
            "ShieldEssential": [r"SHIELD\s*ESSENTIAL", r"SHIELDESSENTIAL", r"ESSENTIAL\s*PLAN"],
            "ShieldPlus": [r"SHIELD\s*PLUS", r"SHIELDPLUS", r"PLUS\s*PLAN"],
            "ShieldSilver": [r"SHIELD\s*SILVER", r"SHIELDSILVER", r"SILVER\s*PLAN"],
            "ShieldGold": [r"SHIELD\s*GOLD", r"SHIELDGOLD", r"GOLD\s*PLAN"],
            "ShieldPlatinum": [r"SHIELD\s*PLATINUM", r"SHIELDPLATINUM", r"PLATINUM\s*PLAN"]
        }
        
        for plan, patterns in plan_patterns.items():
            for pattern in patterns:
                if re.search(pattern, content_upper):
                    metadata["planType"] = plan
                    break
            if metadata["planType"]:
                break
        
        # Extract state codes
        state_names = {
            "CA": ["California", "Calif"], "NY": ["New York"], "TX": ["Texas"],
            "FL": ["Florida"], "IL": ["Illinois"], "PA": ["Pennsylvania"],
            "OH": ["Ohio"], "GA": ["Georgia"], "NC": ["North Carolina"],
            "MI": ["Michigan"], "NJ": ["New Jersey"], "VA": ["Virginia"],
            "WA": ["Washington"], "AZ": ["Arizona"], "MA": ["Massachusetts"],
            "TN": ["Tennessee"], "IN": ["Indiana"], "MO": ["Missouri"],
            "MD": ["Maryland"], "WI": ["Wisconsin"], "NV": ["Nevada"],
            "UT": ["Utah"], "HI": ["Hawaii"], "AK": ["Alaska"],
            "AR": ["Arkansas"], "CO": ["Colorado"], "CT": ["Connecticut"],
            "DE": ["Delaware"], "ID": ["Idaho"], "IA": ["Iowa"],
            "KS": ["Kansas"], "KY": ["Kentucky"], "LA": ["Louisiana"],
            "ME": ["Maine"], "MN": ["Minnesota"], "MS": ["Mississippi"],
        }
        
        for state_code, names in state_names.items():
            if any(str(name).upper() in content_upper for name in names):
                metadata["state"] = CLEAR_STATE_ALIASES.get(state_code, state_code)
                break
        
        # If not found by name, try state code matching with context
        if not metadata["state"]:
            common_state_codes = ["CA", "NY", "TX", "FL", "IL", "PA", "OH", "GA", "NC", "MI", 
                                 "NJ", "VA", "WA", "AZ", "MA", "TN", "IN", "MO", "MD", "WI"]
            other_state_codes = ["AL", "AK", "AR", "CO", "CT", "DE", "HI", "ID", "IA", "KS",
                                "KY", "LA", "ME", "MN", "MS", "MT", "NE", "NV", "NH", "NM",
                                "ND", "OK", "OR", "RI", "SC", "SD", "UT", "VT", "WV", "WY", "DC"]
            
            all_state_codes = common_state_codes + other_state_codes
            
            for state_code in all_state_codes:
                pattern = r'\b' + state_code + r'\b'
                matches = list(re.finditer(pattern, content_upper))
                
                for match in matches:
                    start = max(0, match.start() - 15)
                    end = min(len(content_upper), match.end() + 15)
                    context = content_upper[start:end]
                    
                    positive_keywords = ["STATE", "PLAN", "CONTRACT", "COVERAGE", "POLICY", 
                                       "CALIFORNIA", "TEXAS", "FLORIDA", "NEW YORK", "ILLINOIS"]
                    negative_keywords = ["CALLING", "INFORMATION", "INSPECTION", "INSTALLATION"]
                    
                    has_positive = any(keyword in context for keyword in positive_keywords)
                    has_negative = any(keyword in context for keyword in negative_keywords)
                    
                    if has_positive or (not has_negative and len(context.strip()) < 30):
                        metadata["state"] = CLEAR_STATE_ALIASES.get(state_code, state_code)
                        break
                
                if metadata["state"]:
                    break
    
    except Exception as e:
        print(f"Error extracting metadata from transcript {file_name}: {e}")
    
    return metadata


# Legacy function wrappers for backward compatibility
def list_transcript_files_gcp(limit: Optional[int] = None, offset: int = 0, search: Optional[str] = None) -> Tuple[List[Dict[str, Any]], int]:
    """Legacy wrapper for list_transcript_files_gcp."""
    service = GCPStorageService(
        bucket_name=settings.GCP_BUCKET_NAME,
        project_id=settings.GCP_PROJECT_ID,
        service_account_path=settings.GCP_SERVICE_ACCOUNT_PATH
    )
    return service.list_transcript_files(limit=limit, offset=offset, search=search)


def read_transcript_file_gcp(file_name: str) -> Tuple[str, Dict[str, Any]]:
    """Legacy wrapper for read_transcript_file_gcp."""
    service = GCPStorageService(
        bucket_name=settings.GCP_BUCKET_NAME,
        project_id=settings.GCP_PROJECT_ID,
        service_account_path=settings.GCP_SERVICE_ACCOUNT_PATH
    )
    return service.read_transcript_file(file_name)
