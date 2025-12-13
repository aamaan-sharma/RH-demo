# API Documentation - Transcripts Service

## 📋 Table of Contents
- [Base Configuration](#base-configuration)
- [Authentication](#authentication)
- [Endpoints](#endpoints)
  - [List Transcripts with Search & Pagination](#1-list-transcripts-with-search--pagination)
  - [Process Transcript](#2-process-transcript)
- [Error Handling](#error-handling)
- [Frontend Integration Examples](#frontend-integration-examples)

---

## Base Configuration

### Base URL
```
http://localhost:8001
```

**Note:** For production, replace with your production backend URL.

---

## Authentication

All API endpoints require authentication using a JWT Bearer token.

### Header Format
```
Authorization: Bearer <jwt_token>
```

### Getting the Token
The JWT token is obtained from Google OAuth login flow and stored in browser's session storage as `idToken`.

**Example:**
```javascript
const token = sessionStorage.getItem("idToken");
```

---

## Endpoints

### 1. List Transcripts with Search & Pagination

**Endpoint:** `GET /transcripts`

Lists transcript files from GCP bucket with pagination and search functionality.

#### Request

**Method:** `GET`

**URL:** `/transcripts`

**Headers:**
```
Authorization: Bearer <jwt_token>
Content-Type: application/json
```

**Query Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `limit` | integer | No | `10` | Number of records per page (max recommended: 50) |
| `offset` | integer | No | `0` | Number of records to skip (for pagination) |
| `search` | string | No | - | Search term to filter transcripts by file name (case-insensitive partial match) |
| `q` | string | No | - | Alias for `search` parameter |

**Important Notes:**
- Search searches through **ALL files** in the GCS bucket (all 147 files)
- Search is **case-insensitive** and supports **partial matching**
- Pagination is applied **after** search filtering
- Default page size is **10 records** for optimal performance

#### Request Examples

**Get first page (default 10 records):**
```
GET /transcripts
```

**Get next page:**
```
GET /transcripts?limit=10&offset=10
```

**Search for specific transcript:**
```
GET /transcripts?search=transcript_001
```

**Search with pagination:**
```
GET /transcripts?search=california&limit=20&offset=0
```

**Using 'q' parameter (alias for search):**
```
GET /transcripts?q=transcript&limit=10
```

#### Response

**Status Code:** `200 OK`

**Response Body:**
```json
{
  "transcripts": [
    {
      "fileName": "transcript_001.json",
      "filePath": "gs://ahs-demo-transcripts/transcripts/transcript_001.json",
      "uploadDate": "2024-01-15T10:30:00.000000",
      "fileSize": 102400,
      "contractType": "RE",
      "planType": "ShieldComplete",
      "state": "CA",
      "metadata": {}
    }
  ],
  "totalCount": 147,
  "limit": 10,
  "offset": 0,
  "hasMore": true,
  "search": "transcript_001"
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `transcripts` | array | List of transcript objects (paginated) |
| `transcripts[].fileName` | string | Name of the transcript file |
| `transcripts[].filePath` | string | Full GCS path to the file |
| `transcripts[].uploadDate` | string | ISO 8601 formatted upload date |
| `transcripts[].fileSize` | integer | File size in bytes |
| `transcripts[].contractType` | string \| null | Contract type (e.g., "RE", "DTC") |
| `transcripts[].planType` | string \| null | Plan type (e.g., "ShieldComplete") |
| `transcripts[].state` | string \| null | State code (e.g., "CA") |
| `transcripts[].metadata` | object | Additional metadata (currently empty) |
| `totalCount` | integer | Total number of transcripts (after search filter if applied) |
| `limit` | integer | Number of records per page |
| `offset` | integer | Current offset |
| `hasMore` | boolean | `true` if more pages are available (`offset + limit < totalCount`) |
| `search` | string \| null | Search term used (null if no search) |

#### Error Responses

**401 Unauthorized - Token Missing:**
```json
{
  "message": "Token is missing"
}
```

**401 Unauthorized - Invalid Token:**
```json
{
  "message": "Token is invalid"
}
```

**500 Internal Server Error - GCP Not Available:**
```json
{
  "error": "GCP Storage not configured or unavailable"
}
```

**500 Internal Server Error - General Error:**
```json
{
  "error": "An error occurred while fetching transcripts",
  "details": "Error message details"
}
```

---

### 2. Process Transcript

**Endpoint:** `POST /transcripts/process`

Processes a transcript file: downloads it from GCP, extracts questions, and generates answers.

#### Request

**Method:** `POST`

**URL:** `/transcripts/process`

**Headers:**
```
Authorization: Bearer <jwt_token>
Content-Type: application/json
```

**Request Body:**
```json
{
  "transcriptFileName": "transcript_001.json",
  "contractType": "RE",
  "selectedPlan": "ShieldComplete",
  "selectedState": "CA",
  "gptModel": "Search",
  "extractQuestions": true,
  "questions": []
}
```

**Request Body Fields:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `transcriptFileName` | string | Yes | Name of the transcript file in GCS bucket |
| `contractType` | string | Yes* | Contract type: "RE" or "DTC" (*required if extractQuestions=true) |
| `selectedPlan` | string | Yes* | Plan name (e.g., "ShieldComplete") (*required if extractQuestions=true) |
| `selectedState` | string | Yes* | State code (e.g., "CA") (*required if extractQuestions=true) |
| `gptModel` | string | Yes | Model to use: "Search" or "Infer" |
| `extractQuestions` | boolean | No | Whether to extract questions from transcript (default: true) |
| `questions` | array | No | Pre-defined questions to process (if extractQuestions=false) |

#### Response

**Status Code:** `200 OK`

**Response Body:**
```json
{
  "transcriptId": "transcript_001",
  "transcriptMetadata": {
    "fileName": "transcript_001.json",
    "uploadDate": "2024-01-15T10:30:00.000000",
    "fileSize": 102400
  },
  "questions": [
    {
      "questionId": "q1",
      "question": "What is the coverage limit?",
      "answer": "The coverage limit is $500,000...",
      "confidence": 0.95,
      "latency": 2.3,
      "context": "Additional context"
    }
  ],
  "summary": {
    "totalQuestions": 5,
    "processedQuestions": 5,
    "averageConfidence": 0.92,
    "totalLatency": 11.5
  }
}
```

---

## Error Handling

### HTTP Status Codes

| Code | Description |
|------|-------------|
| `200` | Success |
| `400` | Bad Request - Invalid parameters |
| `401` | Unauthorized - Missing or invalid token |
| `403` | Forbidden - Token expired or insufficient permissions |
| `404` | Not Found - Resource not found |
| `500` | Internal Server Error - Server-side error |

### Error Response Format

```json
{
  "error": "Error message",
  "details": "Detailed error information (optional)"
}
```

---

## Frontend Integration Examples

### JavaScript/React Example

#### Fetch Transcripts with Pagination

```javascript
const fetchTranscripts = async (page = 1, limit = 10, searchTerm = '') => {
  const token = sessionStorage.getItem('idToken');
  const offset = (page - 1) * limit;
  
  const params = new URLSearchParams({
    limit: limit.toString(),
    offset: offset.toString(),
  });
  
  if (searchTerm) {
    params.append('search', searchTerm);
  }
  
  try {
    const response = await fetch(`http://localhost:8001/transcripts?${params}`, {
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json',
      },
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error fetching transcripts:', error);
    throw error;
  }
};

// Usage
const data = await fetchTranscripts(1, 10, 'transcript');
console.log(`Found ${data.totalCount} transcripts`);
console.log(`Showing ${data.transcripts.length} on this page`);
console.log(`Has more: ${data.hasMore}`);
```

#### Search Transcripts

```javascript
const searchTranscripts = async (searchTerm, limit = 10) => {
  const token = sessionStorage.getItem('idToken');
  
  const params = new URLSearchParams({
    search: searchTerm,
    limit: limit.toString(),
    offset: '0',
  });
  
  try {
    const response = await fetch(`http://localhost:8001/transcripts?${params}`, {
      method: 'GET',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json',
      },
    });
    
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error searching transcripts:', error);
    throw error;
  }
};

// Usage
const results = await searchTranscripts('california');
console.log(`Found ${results.totalCount} matching transcripts`);
```

#### Using Axios

```javascript
import axios from 'axios';

const API_BASE_URL = 'http://localhost:8001';

// Setup axios interceptor for authentication
axios.interceptors.request.use((config) => {
  const token = sessionStorage.getItem('idToken');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Fetch transcripts with pagination
const fetchTranscripts = async (page = 1, limit = 10, searchTerm = '') => {
  try {
    const params = {
      limit,
      offset: (page - 1) * limit,
    };
    
    if (searchTerm) {
      params.search = searchTerm;
    }
    
    const response = await axios.get(`${API_BASE_URL}/transcripts`, { params });
    return response.data;
  } catch (error) {
    console.error('Error fetching transcripts:', error);
    throw error;
  }
};

// Usage
const data = await fetchTranscripts(1, 10, 'transcript');
```

### React Hook Example

```javascript
import { useState, useEffect, useCallback } from 'react';
import axios from 'axios';

const API_BASE_URL = 'http://localhost:8001';

const useTranscripts = (searchTerm = '', page = 1, limit = 10) => {
  const [transcripts, setTranscripts] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [pagination, setPagination] = useState({
    totalCount: 0,
    limit: 10,
    offset: 0,
    hasMore: false,
  });

  const fetchTranscripts = useCallback(async () => {
    setLoading(true);
    setError(null);
    
    try {
      const token = sessionStorage.getItem('idToken');
      const params = {
        limit,
        offset: (page - 1) * limit,
      };
      
      if (searchTerm) {
        params.search = searchTerm;
      }
      
      const response = await axios.get(`${API_BASE_URL}/transcripts`, {
        params,
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });
      
      setTranscripts(response.data.transcripts);
      setPagination({
        totalCount: response.data.totalCount,
        limit: response.data.limit,
        offset: response.data.offset,
        hasMore: response.data.hasMore,
      });
    } catch (err) {
      setError(err.message);
      console.error('Error fetching transcripts:', err);
    } finally {
      setLoading(false);
    }
  }, [searchTerm, page, limit]);

  useEffect(() => {
    fetchTranscripts();
  }, [fetchTranscripts]);

  return {
    transcripts,
    loading,
    error,
    pagination,
    refetch: fetchTranscripts,
  };
};

// Usage in component
function TranscriptList() {
  const [searchTerm, setSearchTerm] = useState('');
  const [page, setPage] = useState(1);
  const { transcripts, loading, error, pagination, refetch } = useTranscripts(searchTerm, page, 10);

  if (loading) return <div>Loading...</div>;
  if (error) return <div>Error: {error}</div>;

  return (
    <div>
      <input
        type="text"
        value={searchTerm}
        onChange={(e) => {
          setSearchTerm(e.target.value);
          setPage(1); // Reset to first page on new search
        }}
        placeholder="Search transcripts..."
      />
      
      <div>
        {transcripts.map((transcript) => (
          <div key={transcript.fileName}>
            <h3>{transcript.fileName}</h3>
            <p>State: {transcript.state}</p>
            <p>Contract: {transcript.contractType}</p>
          </div>
        ))}
      </div>
      
      <div>
        <button
          disabled={page === 1}
          onClick={() => setPage(page - 1)}
        >
          Previous
        </button>
        <span>Page {page} of {Math.ceil(pagination.totalCount / pagination.limit)}</span>
        <button
          disabled={!pagination.hasMore}
          onClick={() => setPage(page + 1)}
        >
          Next
        </button>
      </div>
    </div>
  );
}
```

---

## Pagination Best Practices

### Calculating Total Pages
```javascript
const totalPages = Math.ceil(totalCount / limit);
```

### Checking if More Pages Exist
```javascript
const hasMore = offset + limit < totalCount;
// Or use the hasMore field from API response
```

### Building Pagination UI
```javascript
const getPaginationInfo = (totalCount, limit, offset) => {
  const currentPage = Math.floor(offset / limit) + 1;
  const totalPages = Math.ceil(totalCount / limit);
  const startRecord = offset + 1;
  const endRecord = Math.min(offset + limit, totalCount);
  
  return {
    currentPage,
    totalPages,
    startRecord,
    endRecord,
    showing: `${startRecord}-${endRecord} of ${totalCount}`,
  };
};
```

---

## Search Best Practices

### Debouncing Search Input
```javascript
import { useState, useEffect } from 'react';

function useDebounce(value, delay) {
  const [debouncedValue, setDebouncedValue] = useState(value);

  useEffect(() => {
    const handler = setTimeout(() => {
      setDebouncedValue(value);
    }, delay);

    return () => {
      clearTimeout(handler);
    };
  }, [value, delay]);

  return debouncedValue;
}

// Usage
const [searchTerm, setSearchTerm] = useState('');
const debouncedSearchTerm = useDebounce(searchTerm, 500); // Wait 500ms after user stops typing
```

### Search Features
- **Case-insensitive**: "TRANSCRIPT" = "transcript" = "Transcript"
- **Partial match**: "001" will match "transcript_001.json"
- **Searches all files**: Searches through all 147 files in GCS
- **Works with pagination**: Search filters first, then pagination applies

---

## TypeScript Types

```typescript
interface Transcript {
  fileName: string;
  filePath: string;
  uploadDate: string;
  fileSize: number;
  contractType: string | null;
  planType: string | null;
  state: string | null;
  metadata: Record<string, any>;
}

interface TranscriptsResponse {
  transcripts: Transcript[];
  totalCount: number;
  limit: number;
  offset: number;
  hasMore: boolean;
  search: string | null;
}

interface PaginationParams {
  limit?: number;
  offset?: number;
  search?: string;
  q?: string; // Alias for search
}
```

---

## Testing Checklist

- [ ] Authentication token is included in all requests
- [ ] Pagination works correctly (next/previous pages)
- [ ] Search functionality works (case-insensitive, partial match)
- [ ] Error handling for 401, 403, 500 errors
- [ ] Loading states are shown during API calls
- [ ] Empty states are handled (no results)
- [ ] `hasMore` flag is used to disable/enable "Next" button
- [ ] Search resets pagination to page 1
- [ ] Debouncing is implemented for search input

---

## Support

For issues or questions:
1. Check server logs for detailed error messages
2. Verify JWT token is valid and not expired
3. Ensure backend server is running on port 8001
4. Check network tab in browser DevTools for request/response details

---

## Version

**API Version:** 1.0  
**Last Updated:** December 2024

