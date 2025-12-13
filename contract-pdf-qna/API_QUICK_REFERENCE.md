# API Quick Reference Card

## Base URL
```
http://localhost:8001
```

## Authentication
```javascript
headers: {
  'Authorization': `Bearer ${sessionStorage.getItem('idToken')}`
}
```

---

## GET /transcripts

### Basic Request
```javascript
GET /transcripts?limit=10&offset=0
```

### With Search
```javascript
GET /transcripts?search=transcript_001&limit=10&offset=0
```

### Query Parameters
| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `limit` | number | `10` | Records per page |
| `offset` | number | `0` | Records to skip |
| `search` | string | - | Search term (case-insensitive) |
| `q` | string | - | Alias for `search` |

### Response
```json
{
  "transcripts": [...],
  "totalCount": 147,
  "limit": 10,
  "offset": 0,
  "hasMore": true,
  "search": "transcript_001"
}
```

---

## Quick Code Examples

### Fetch with Axios
```javascript
const response = await axios.get('/transcripts', {
  params: { limit: 10, offset: 0, search: 'transcript' },
  headers: { Authorization: `Bearer ${token}` }
});
```

### Fetch with Fetch API
```javascript
const response = await fetch('/transcripts?limit=10&search=transcript', {
  headers: { Authorization: `Bearer ${token}` }
});
const data = await response.json();
```

### Pagination Logic
```javascript
const page = 1;
const limit = 10;
const offset = (page - 1) * limit;
const hasMore = data.hasMore;
const totalPages = Math.ceil(data.totalCount / limit);
```

---

## Error Codes
- `401` - Token missing/invalid
- `500` - Server error

---

**Full Documentation:** See `API_DOCUMENTATION.md`

