import json
import os
import re
from datetime import datetime
from typing import Dict, List, Tuple

from ..extensions import gcs_fs, ssl_context

GCP_BUCKET_NAME = os.getenv("GCP_BUCKET_NAME", "ahs-demo-transcripts")
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID", "generative-ai-390411")

transcript_metadata_cache: Dict[str, Dict] = {}
TRANSCRIPT_METADATA_CACHE_VERSION = "v2"

CLEAR_STATE_ALIASES = {
    "AZ": "Arizona",
    "CA": "California",
    "GA": "Georgia",
    "MD": "Maryland",
    "MN": "Minnesota",
    "NV": "Nevada",
    "TX": "Texas",
    "UT": "Utah",
    "WI": "Wisconsin",
}

_PLACEHOLDER_CHUNK_VALUES = {
    "[]",
    "",
    "(No supporting excerpts found)",
}


def extract_transcript_metadata(transcript_content: str, file_name: str) -> Dict:
    """
    Extract contractType, planType, and state from transcript file content.
    """
    metadata = {
        "contractType": None,
        "planType": None,
        "state": None,
    }

    try:
        # Method 1: Try parsing as JSON first (fastest)
        try:
            transcript_data = json.loads(transcript_content)
            if isinstance(transcript_data, dict):
                metadata_fields = transcript_data.get("metadata", {}) or transcript_data

                metadata["contractType"] = (
                    metadata_fields.get("contractType")
                    or metadata_fields.get("contract_type")
                    or metadata_fields.get("type")
                )

                metadata["planType"] = (
                    metadata_fields.get("planType")
                    or metadata_fields.get("plan_type")
                    or metadata_fields.get("selectedPlan")
                    or metadata_fields.get("selected_plan")
                    or metadata_fields.get("plan")
                )

                metadata["state"] = (
                    metadata_fields.get("state")
                    or metadata_fields.get("selectedState")
                    or metadata_fields.get("selected_state")
                    or metadata_fields.get("stateCode")
                )

                if all([metadata["contractType"], metadata["planType"], metadata["state"]]):
                    return metadata
        except json.JSONDecodeError:
            pass

        # Method 2: Regex-based text parsing
        content_upper = transcript_content.upper()

        if re.search(r"\bRE\b", content_upper) or "REAL ESTATE" in content_upper:
            metadata["contractType"] = "RE"
        elif re.search(r"\bDTC\b", content_upper) or "DIRECT TO CONSUMER" in content_upper or "DIRECT-TO-CONSUMER" in content_upper:
            metadata["contractType"] = "DTC"

        plan_patterns = {
            "ShieldComplete": [r"SHIELD\s*COMPLETE", r"SHIELDCOMPLETE", r"COMPLETE\s*PLAN"],
            "ShieldEssential": [r"SHIELD\s*ESSENTIAL", r"SHIELDESSENTIAL", r"ESSENTIAL\s*PLAN"],
            "ShieldPlus": [r"SHIELD\s*PLUS", r"SHIELDPLUS", r"PLUS\s*PLAN"],
        }
        for plan_type, patterns in plan_patterns.items():
            for pattern in patterns:
                if re.search(pattern, content_upper):
                    metadata["planType"] = plan_type
                    break
            if metadata["planType"]:
                break

        if not metadata["state"]:
            state_names = {
                "CA": ["California", "Calif"],
                "TX": ["Texas", "Tex."],
                "FL": ["Florida", "Fla."],
                "NY": ["New York", "N.Y."],
                "IL": ["Illinois", "Ill."],
                "AZ": ["Arizona", "Ariz."],
                "NV": ["Nevada", "Nev."],
                "UT": ["Utah"],
                "WI": ["Wisconsin", "Wis."],
                "GA": ["Georgia", "Ga."],
                "MD": ["Maryland", "Md."],
                "MN": ["Minnesota", "Minn."],
                "KS": ["Kansas"],
                "KY": ["Kentucky"],
                "LA": ["Louisiana"],
                "ME": ["Maine"],
                "MS": ["Mississippi"],
            }
            for state_code, names in state_names.items():
                if any(str(name).upper() in content_upper for name in names):
                    metadata["state"] = CLEAR_STATE_ALIASES.get(state_code, state_code)
                    break

            if not metadata["state"]:
                common_state_codes = [
                    "CA",
                    "NY",
                    "TX",
                    "FL",
                    "IL",
                    "PA",
                    "OH",
                    "GA",
                    "NC",
                    "MI",
                    "NJ",
                    "VA",
                    "WA",
                    "AZ",
                    "MA",
                    "TN",
                    "IN",
                    "MO",
                    "MD",
                    "WI",
                ]
                other_state_codes = [
                    "AL",
                    "AK",
                    "AR",
                    "CO",
                    "CT",
                    "DE",
                    "HI",
                    "ID",
                    "IA",
                    "KS",
                    "KY",
                    "LA",
                    "ME",
                    "MN",
                    "MS",
                    "MT",
                    "NE",
                    "NV",
                    "NH",
                    "NM",
                    "ND",
                    "OK",
                    "OR",
                    "RI",
                    "SC",
                    "SD",
                    "UT",
                    "VT",
                    "WV",
                    "WY",
                    "DC",
                ]

                all_state_codes = common_state_codes + other_state_codes
                for state_code in all_state_codes:
                    pattern = r"\b" + state_code + r"\b"
                    matches = list(re.finditer(pattern, content_upper))
                    for match in matches:
                        start = max(0, match.start() - 15)
                        end = min(len(content_upper), match.end() + 15)
                        context = content_upper[start:end]
                        positive_keywords = ["STATE", "PLAN", "CONTRACT", "COVERAGE", "POLICY", "CALIFORNIA", "TEXAS", "FLORIDA", "NEW YORK", "ILLINOIS"]
                        negative_keywords = ["CALLING", "INFORMATION", "INSPECTION", "INSTALLATION"]
                        has_positive = any(keyword in context for keyword in positive_keywords)
                        has_negative = any(keyword in context for keyword in negative_keywords)
                        if has_positive or (not has_negative and len(context.strip()) < 30):
                            metadata["state"] = CLEAR_STATE_ALIASES.get(state_code, state_code)
                            break
                    if metadata["state"]:
                        break
    except Exception as exc:  # pragma: no cover - best-effort fallback
        print(f"Error extracting metadata from transcript {file_name}: {exc}")

    return metadata


def list_transcript_files_gcp(limit: int | None = None, offset: int = 0, search: str | None = None) -> Tuple[List[Dict], int]:
    """
    List transcript files from GCP bucket using fsspec with pagination and search support.
    Returns tuple: (paginated_transcripts, total_count)
    """
    all_file_info: List[Dict] = []
    try:
        import certifi

        cert_path = certifi.where()
        os.environ.setdefault("SSL_CERT_FILE", cert_path)
        os.environ.setdefault("REQUESTS_CA_BUNDLE", cert_path)
        os.environ.setdefault("AIOHTTP_CA_BUNDLE", cert_path)

        from .. import extensions

        fs = extensions.ensure_gcs_fs()
        if fs is None:
            print("ERROR list_transcript_files_gcp: gcs_fs is None!")
            return ([], 0) if limit else []

        bucket_path = f"gs://{GCP_BUCKET_NAME}/"
        prefixes = ["transcripts/", ""]
        seen_files = set()

        for prefix in prefixes:
            try:
                full_path = bucket_path + prefix if prefix else bucket_path
                files = fs.ls(full_path, detail=True)

                for file_info in files:
                    if isinstance(file_info, str):
                        file_path = file_info
                        file_size = 0
                        time_created = None
                    else:
                        file_path = file_info.get("name", "")
                        file_size = file_info.get("size", 0)
                        time_created = file_info.get("timeCreated", None)

                    if not file_path.lower().endswith(".json"):
                        continue
                    if file_path in seen_files:
                        continue
                    seen_files.add(file_path)

                    file_name = file_path.split("/")[-1]
                    all_file_info.append(
                        {
                            "file_path": file_path,
                            "file_name": file_name,
                            "file_size": file_size,
                            "time_created": time_created,
                        }
                    )
            except Exception as exc:
                print(f"WARNING list_transcript_files_gcp: could not list files from prefix {prefix}: {exc}")

        if search:
            search_lower = search.lower()
            all_file_info = [info for info in all_file_info if search_lower in info["file_name"].lower()]

        all_file_info.sort(key=lambda x: x.get("time_created") or x["file_name"], reverse=True)
        total_count = len(all_file_info)

        if limit is not None:
            paginated_info = all_file_info[offset : offset + limit]
        else:
            paginated_info = all_file_info

        transcripts = []
        for info in paginated_info:
            file_path = info["file_path"]
            file_name = info["file_name"]
            file_size = info["file_size"]
            time_created = info["time_created"]

            cache_key = f"{TRANSCRIPT_METADATA_CACHE_VERSION}:{file_name}"
            cached = transcript_metadata_cache.get(cache_key)

            if cached:
                metadata = cached
            else:
                try:
                    with fs.open(file_path, "r") as f:
                        content = f.read()
                    metadata = extract_transcript_metadata(content, file_name)
                    transcript_metadata_cache[cache_key] = metadata
                except Exception as exc:
                    print(f"WARNING list_transcript_files_gcp: failed to read {file_path}: {exc}")
                    metadata = {"contractType": None, "planType": None, "state": None}

            transcripts.append(
                {
                    "fileName": file_name,
                    "fileSize": file_size,
                    "contractType": metadata.get("contractType"),
                    "planType": metadata.get("planType"),
                    "state": metadata.get("state"),
                    "uploadDate": time_created if isinstance(time_created, str) else _format_time_created(time_created),
                }
            )

        return transcripts, total_count
    except Exception as exc:  # pragma: no cover - upstream storage failure
        print(f"ERROR list_transcript_files_gcp: {exc}")
        if limit is None:
            return []
        return ([], 0)


def _format_time_created(time_created):
    if not time_created:
        return None
    try:
        if isinstance(time_created, datetime):
            return time_created.isoformat()
        return str(time_created)
    except Exception:
        return None


__all__ = ["list_transcript_files_gcp", "extract_transcript_metadata"]
