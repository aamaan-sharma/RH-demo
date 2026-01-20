"""Chunk normalization utilities."""
import os
from typing import List, Dict, Any


_PLACEHOLDER_CHUNK_VALUES = {
    "[]",
    "",
    "(No supporting excerpts found)",
}


def normalize_chunks_with_names(chunks: List[Any]) -> List[Dict[str, Any]]:
    """
    Normalize chunks to ensure they have a 'name' field.
    If chunks are strings, convert them to objects.
    If chunks are objects without names, generate names from metadata.
    
    Args:
        chunks: List of chunks (strings or dicts)
        
    Returns:
        List of normalized chunk dictionaries with 'name' field
    """
    if not isinstance(chunks, list):
        return []
    
    normalized = []
    for idx, chunk in enumerate(chunks, start=1):
        if isinstance(chunk, str):
            # Convert string to object
            normalized.append({
                "content": chunk,
                "name": f"Clause {idx}"
            })
        elif isinstance(chunk, dict):
            # Ensure it has a name field
            if "name" not in chunk or not chunk.get("name"):
                # Generate name from metadata if available
                metadata = chunk.get("metadata", {}) or {}
                chunk["name"] = generate_chunk_name(metadata, idx)
            normalized.append(chunk)
        else:
            # Fallback for unknown types
            normalized.append({
                "content": str(chunk),
                "name": f"Clause {idx}"
            })
    
    return normalized


def generate_chunk_name(metadata: Dict[str, Any], index: int) -> str:
    """
    Generate a meaningful name for a chunk based on its metadata.
    Priority order:
    1. Extract filename from source path (most common case) - ONLY filename, not full path
    2. section + clause/page number
    3. heading/title + page number
    4. clause number
    5. Fallback to "Clause {index}"
    
    Args:
        metadata: Chunk metadata dictionary
        index: Chunk index
        
    Returns:
        Generated chunk name
    """
    if not isinstance(metadata, dict):
        return f"Clause {index}"
    
    # Try to extract meaningful identifiers
    source = metadata.get("source") or metadata.get("file") or metadata.get("document") or metadata.get("Source") or ""
    section = metadata.get("section") or metadata.get("Section") or ""
    clause = metadata.get("clause") or metadata.get("Clause") or metadata.get("clause_number") or ""
    page = metadata.get("page") or metadata.get("Page") or metadata.get("page_number") or ""
    heading = metadata.get("heading") or metadata.get("Heading") or metadata.get("title") or metadata.get("Title") or ""
    
    # Build name parts
    name_parts = []
    
    # Priority 1: Extract ONLY filename from source path (most common case)
    if source:
        source_str = str(source).strip()
        
        # Extract just the filename part (last component of path)
        normalized_path = source_str.replace("\\", "/")
        source_name = os.path.basename(normalized_path)
        
        # Double-check: if basename still contains separators, manually extract last part
        if "/" in source_name or "\\" in source_name:
            if "\\" in source_str:
                parts = [p.strip() for p in source_str.split("\\") if p.strip()]
            elif "/" in source_str:
                parts = [p.strip() for p in source_str.split("/") if p.strip()]
            else:
                parts = [source_str]
            source_name = parts[-1] if parts else source_str
        
        # Remove file extension for cleaner display
        if "." in source_name:
            source_name = source_name.rsplit(".", 1)[0]
        
        # Replace underscores and hyphens with spaces for better readability
        source_name = source_name.replace("_", " ").replace("-", " ")
        
        # Capitalize first letter of each word
        source_name = " ".join(word.capitalize() for word in source_name.split() if word.strip())
        
        name_parts.append(source_name)
    
    # Priority 2: Section + Clause
    if section and clause and not name_parts:
        name_parts.append(f"{section} - Clause {clause}")
    elif section and not name_parts:
        name_parts.append(section)
    
    # Priority 3: Heading/Title
    if heading and not name_parts:
        name_parts.append(heading)
    
    # Add page number if available
    if page:
        name_parts.append(f"Page {page}")
    
    # If we have any meaningful parts, join them
    if name_parts:
        return " · ".join(name_parts)
    
    # Fallback to clause number or index
    if clause:
        return f"Clause {clause}"
    
    return f"Clause {index}"


def get_placeholder_chunk_values() -> set:
    """Get set of placeholder chunk values."""
    return _PLACEHOLDER_CHUNK_VALUES
