# CTI-NLP System API Documentation

## Overview

The CTI-NLP System provides a RESTful API for analyzing cyber threat intelligence using Natural Language Processing and Machine Learning techniques.

**Base URL**: `http://localhost:8000`
**API Version**: 1.0.0

## Authentication

Currently, the API is open for development. In production, implement proper authentication.

## Endpoints

### Health Check

#### GET `/health`

Check the health status of the application and its dependencies.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-08-26T21:42:00.000Z",
  "services": {
    "database": "connected",
    "redis": "connected",
    "api": "running"
  }
}
```

### Text Analysis

#### POST `/analyze`

Analyze threat intelligence text and extract entities, classify threat type, and predict severity.

**Request Body:**
```json
{
  "text": "APT29 group launching phishing campaign targeting financial institutions"
}
```

**Response:**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "original_text": "APT29 group launching phishing campaign targeting financial institutions",
  "entities": {
    "ORG": ["APT29"],
    "PERSON": [],
    "LOC": []
  },
  "threat_type": "Phishing",
  "severity": "High",
  "timestamp": "2025-08-26T21:42:00.000Z"
}
```

### Threat Intelligence Management

#### GET `/threats`

Retrieve threat intelligence records with optional filtering.

**Query Parameters:**
- `limit` (int): Maximum number of records to return (default: 100)
- `source` (string): Filter by data source (Twitter, DarkWeb, MITRE, Manual, Feed)
- `threat_type` (string): Filter by threat category
- `severity` (string): Filter by severity level

**Example:**
```
GET /threats?limit=50&threat_type=Phishing&severity=High
```

**Response:**
```json
{
  "threats": [
    {
      "id": "550e8400-e29b-41d4-a716-446655440000",
      "source": "Manual",
      "original_text": "Phishing campaign targeting banks",
      "threat_type": "Phishing",
      "severity": "High",
      "created_at": "2025-08-26T21:42:00.000Z",
      "confidence_score": 0.85
    }
  ],
  "count": 1
}
```

#### GET `/threats/{threat_id}`

Get detailed information about a specific threat.

**Response:**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "source": "Manual",
  "original_text": "APT29 phishing campaign",
  "threat_type": "Phishing",
  "severity": "High",
  "created_at": "2025-08-26T21:42:00.000Z",
  "confidence_score": 0.85,
  "entities": [
    {
      "type": "ORG",
      "value": "APT29",
      "confidence": 0.9
    }
  ]
}
```

#### DELETE `/threats/{threat_id}`

Delete a threat intelligence record.

**Response:**
```json
{
  "message": "Threat deleted successfully"
}
```

### Analytics

#### GET `/analytics`

Get threat intelligence analytics and statistics.

**Query Parameters:**
- `days` (int): Number of days to include in statistics (default: 30)

**Response:**
```json
{
  "total_threats": 150,
  "threats_by_category": {
    "Phishing": 45,
    "Malware": 38,
    "Ransomware": 25,
    "APT": 20,
    "Other": 22
  },
  "threats_by_severity": {
    "Low": 30,
    "Medium": 60,
    "High": 45,
    "Critical": 15
  },
  "threats_by_source": {
    "Twitter": 50,
    "DarkWeb": 35,
    "MITRE": 30,
    "Manual": 25,
    "Feed": 10
  },
  "period_days": 30
}
```

### Batch Processing

#### POST `/upload_csv`

Upload and analyze a CSV file containing threat intelligence data.

**Request:**
- Content-Type: `multipart/form-data`
- File: CSV file with a `text` column

**Response:**
- Content-Type: `application/json`
- File: JSON file with analysis results

### Real-time Ingestion

#### GET `/feed`

Get the latest ingested threat intelligence feed.

**Query Parameters:**
- `limit` (int): Maximum number of entries to return (default: 20)

**Response:**
```json
[
  {
    "source": "Twitter",
    "text": "New malware variant discovered",
    "timestamp": "2025-08-26T21:42:00.000Z",
    "entities": {...},
    "threat_type": "Malware",
    "severity": "Medium"
  }
]
```

#### POST `/ingest_now`

Manually trigger the data ingestion process.

**Response:**
```json
{
  "status": "success",
  "message": "Ingestion completed successfully",
  "ingestion_summary": {
    "last_run": "2025-08-26T21:42:00.000Z",
    "summary": {
      "darkweb": 15,
      "twitter": 25,
      "mitre_attack": 10
    },
    "total_records": 50
  }
}
```

### Dashboard

#### GET `/dashboard`

Serve the web dashboard interface.

**Response:** HTML page with interactive dashboard

## Data Models

### Threat Intelligence Record

```json
{
  "id": "uuid",
  "source": "string (enum)",
  "original_text": "string",
  "processed_text": "string (optional)",
  "threat_type": "string (enum)",
  "severity": "string (enum)",
  "confidence_score": "number (0-1)",
  "source_url": "string (optional)",
  "source_id": "string (optional)",
  "author": "string (optional)",
  "created_at": "datetime",
  "updated_at": "datetime",
  "language": "string",
  "region": "string (optional)",
  "tags": ["array of strings"]
}
```

### Entity

```json
{
  "id": "uuid",
  "threat_intel_id": "uuid",
  "entity_type": "string",
  "entity_value": "string",
  "confidence_score": "number (0-1)",
  "start_char": "integer (optional)",
  "end_char": "integer (optional)",
  "created_at": "datetime"
}
```

## Error Handling

### HTTP Status Codes

- `200 OK`: Successful request
- `400 Bad Request`: Invalid request data
- `404 Not Found`: Resource not found
- `500 Internal Server Error`: Server error

### Error Response Format

```json
{
  "error": "Description of the error",
  "code": "ERROR_CODE",
  "timestamp": "2025-08-26T21:42:00.000Z"
}
```

## Rate Limiting

Default rate limits:
- 60 requests per minute per IP address
- 1000 requests per hour per IP address

Headers included in responses:
- `X-RateLimit-Limit`: Request limit per window
- `X-RateLimit-Remaining`: Requests remaining in current window
- `X-RateLimit-Reset`: Time when the window resets

## SDKs and Examples

### Python SDK Example

```python
import requests

# Initialize client
base_url = "http://localhost:8000"

# Analyze text
response = requests.post(
    f"{base_url}/analyze",
    json={"text": "Phishing attack targeting banks"}
)
result = response.json()
print(f"Threat Type: {result['threat_type']}")
print(f"Severity: {result['severity']}")

# Get threats
response = requests.get(f"{base_url}/threats?limit=10")
threats = response.json()
print(f"Found {threats['count']} threats")
```

### JavaScript/Node.js Example

```javascript
const axios = require('axios');

const baseURL = 'http://localhost:8000';

// Analyze text
async function analyzeText(text) {
  try {
    const response = await axios.post(`${baseURL}/analyze`, { text });
    console.log('Analysis result:', response.data);
    return response.data;
  } catch (error) {
    console.error('Error:', error.response.data);
  }
}

// Get analytics
async function getAnalytics() {
  try {
    const response = await axios.get(`${baseURL}/analytics?days=7`);
    console.log('Analytics:', response.data);
    return response.data;
  } catch (error) {
    console.error('Error:', error.response.data);
  }
}
```

### cURL Examples

```bash
# Analyze text
curl -X POST "http://localhost:8000/analyze" \
  -H "Content-Type: application/json" \
  -d '{"text": "APT group targeting government agencies"}'

# Get threats
curl "http://localhost:8000/threats?limit=5&threat_type=Phishing"

# Get analytics
curl "http://localhost:8000/analytics?days=30"

# Upload CSV
curl -X POST "http://localhost:8000/upload_csv" \
  -F "file=@threats.csv"
```

## Webhook Support (Future)

Future versions will support webhooks for real-time notifications:

```json
{
  "url": "https://your-app.com/webhook",
  "events": ["threat.created", "threat.updated", "high_severity_detected"],
  "secret": "webhook_secret"
}
```

## API Versioning

The API uses URL versioning. Future versions will be available at:
- `/v2/analyze`
- `/v2/threats`

Current version remains at the root level for backward compatibility.

---

*For more information, visit the interactive API documentation at `/docs` when the server is running.*
