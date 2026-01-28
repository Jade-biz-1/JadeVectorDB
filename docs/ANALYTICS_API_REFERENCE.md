# JadeVectorDB Analytics API Reference

**Version**: 1.0
**Last Updated**: January 28, 2026
**Feature**: Query Analytics (Phase 16, T16.15-T16.22)

---

## Overview

The Analytics API provides comprehensive query tracking, analysis, and insights for JadeVectorDB. Monitor search patterns, identify performance bottlenecks, and gain actionable insights to optimize your vector database deployment.

### Key Features

- **Real-time Query Logging**: Automatic capture of all search queries
- **Statistical Analysis**: Compute aggregated metrics by time bucket
- **Pattern Recognition**: Identify common query patterns and trends
- **Performance Monitoring**: Track latency percentiles and slow queries
- **Zero-Result Detection**: Find content gaps in your database
- **Trending Queries**: Discover emerging search patterns
- **User Feedback**: Collect and analyze user satisfaction
- **Data Export**: Export analytics data in CSV or JSON format

### Base URL

```
http://localhost:8080/v1/databases/{database_id}/analytics
```

All endpoints require authentication via Bearer token or API key.

---

## Endpoints

### 1. Get Query Statistics

Retrieve aggregated statistics for a time range.

**Endpoint**: `GET /v1/databases/{database_id}/analytics/stats`

**Query Parameters**:
- `start_time` (integer, required): Unix timestamp in milliseconds
- `end_time` (integer, required): Unix timestamp in milliseconds
- `granularity` (string, optional): Time bucket granularity
  - Values: `hourly`, `daily`, `weekly`, `monthly`
  - Default: `hourly`

**Example Request**:
```bash
curl -X GET "http://localhost:8080/v1/databases/db123/analytics/stats?start_time=1706400000000&end_time=1706486400000&granularity=hourly" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

**Response** (200 OK):
```json
{
  "statistics": [
    {
      "time_bucket": 1706400000000,
      "total_queries": 1523,
      "successful_queries": 1498,
      "failed_queries": 25,
      "zero_result_queries": 87,
      "avg_latency_ms": 45.3,
      "p50_latency_ms": 32.0,
      "p95_latency_ms": 98.5,
      "p99_latency_ms": 156.2,
      "unique_users": 234,
      "unique_sessions": 512,
      "vector_queries": 890,
      "hybrid_queries": 456,
      "bm25_queries": 102,
      "reranked_queries": 75
    }
  ]
}
```

**Use Cases**:
- Dashboard metrics visualization
- Performance trend analysis
- Capacity planning
- SLA monitoring

---

### 2. Get Recent Queries

Retrieve individual query logs with filtering.

**Endpoint**: `GET /v1/databases/{database_id}/analytics/queries`

**Query Parameters**:
- `limit` (integer, optional): Maximum results (default: 50, max: 1000)
- `offset` (integer, optional): Pagination offset (default: 0)
- `start_time` (integer, optional): Filter by start time
- `end_time` (integer, optional): Filter by end time

**Example Request**:
```bash
curl -X GET "http://localhost:8080/v1/databases/db123/analytics/queries?limit=10&offset=0" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

**Response** (200 OK):
```json
{
  "queries": [
    {
      "query_id": "q1706400123456",
      "database_id": "db123",
      "query_text": "machine learning algorithms",
      "query_type": "hybrid",
      "retrieval_time_ms": 28,
      "total_time_ms": 45,
      "num_results": 10,
      "avg_similarity_score": 0.87,
      "user_id": "user_456",
      "session_id": "session_789",
      "timestamp": 1706400123456,
      "has_error": false,
      "hybrid_alpha": 0.7,
      "fusion_method": "RRF"
    }
  ],
  "total_count": 1523,
  "limit": 10,
  "offset": 0
}
```

**Use Cases**:
- Query debugging
- User behavior analysis
- Audit logging
- Search quality assessment

---

### 3. Get Query Patterns

Identify common query patterns.

**Endpoint**: `GET /v1/databases/{database_id}/analytics/patterns`

**Query Parameters**:
- `min_count` (integer, optional): Minimum pattern frequency (default: 2)
- `limit` (integer, optional): Maximum results (default: 50)

**Example Request**:
```bash
curl -X GET "http://localhost:8080/v1/databases/db123/analytics/patterns?min_count=5&limit=20" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

**Response** (200 OK):
```json
{
  "patterns": [
    {
      "normalized_text": "machine learning",
      "count": 156,
      "avg_latency_ms": 42.5,
      "avg_results": 8.7,
      "first_seen": 1706300000000,
      "last_seen": 1706486400000
    },
    {
      "normalized_text": "python tutorial",
      "count": 98,
      "avg_latency_ms": 35.2,
      "avg_results": 12.3,
      "first_seen": 1706310000000,
      "last_seen": 1706486200000
    }
  ]
}
```

**Pattern Normalization**:
- Lowercase conversion
- Stop words removal
- Punctuation removal
- Duplicate whitespace removal

**Use Cases**:
- Content optimization
- Auto-suggest features
- Popular topics identification
- Query canonicalization

---

### 4. Get Analytics Insights

Get automated insights and recommendations.

**Endpoint**: `GET /v1/databases/{database_id}/analytics/insights`

**Query Parameters**:
- `start_time` (integer, required): Unix timestamp in milliseconds
- `end_time` (integer, required): Unix timestamp in milliseconds

**Example Request**:
```bash
curl -X GET "http://localhost:8080/v1/databases/db123/analytics/insights?start_time=1706400000000&end_time=1706486400000" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

**Response** (200 OK):
```json
{
  "summary": {
    "total_queries": 1523,
    "successful_queries": 1498,
    "failed_queries": 25,
    "success_rate": 0.984,
    "avg_latency_ms": 45.3,
    "p95_latency_ms": 98.5,
    "queries_per_second": 0.42,
    "peak_hour": 1706443200000,
    "peak_hour_queries": 256
  },
  "top_patterns": [
    {
      "normalized_text": "machine learning",
      "count": 156,
      "avg_latency_ms": 42.5
    }
  ],
  "slow_queries": [
    {
      "query_id": "q1706400999888",
      "query_text": "complex vector similarity search",
      "total_time_ms": 1250,
      "num_results": 100,
      "timestamp": 1706400999888
    }
  ],
  "zero_result_queries": [
    {
      "normalized_text": "quantum computing basics",
      "count": 12
    }
  ],
  "trending_queries": [
    {
      "normalized_text": "large language models",
      "current_count": 45,
      "previous_count": 8,
      "growth_rate": 462.5
    }
  ]
}
```

**Insight Categories**:
- **Performance**: Slow queries, latency trends
- **Content Gaps**: Zero-result patterns
- **Trends**: Growing query volumes
- **Quality**: Success rates, error patterns

**Use Cases**:
- Proactive optimization
- Content strategy
- Capacity planning
- Quality monitoring

---

### 5. Get Trending Queries

Discover queries with significant growth.

**Endpoint**: `GET /v1/databases/{database_id}/analytics/trending`

**Query Parameters**:
- `time_bucket` (string, optional): Comparison period
  - Values: `hourly`, `daily`, `weekly`
  - Default: `daily`
- `min_growth_rate` (number, optional): Minimum growth percentage (default: 50)

**Example Request**:
```bash
curl -X GET "http://localhost:8080/v1/databases/db123/analytics/trending?time_bucket=daily&min_growth_rate=100" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

**Response** (200 OK):
```json
{
  "trending": [
    {
      "normalized_text": "large language models",
      "current_count": 45,
      "previous_count": 8,
      "growth_rate": 462.5,
      "query_text": "large language models tutorial"
    },
    {
      "normalized_text": "vector databases",
      "current_count": 67,
      "previous_count": 12,
      "growth_rate": 458.3,
      "query_text": "what are vector databases"
    }
  ]
}
```

**Growth Calculation**:
```
growth_rate = ((current - previous) / previous) * 100
```

**Use Cases**:
- Emerging topics detection
- Content prioritization
- Marketing insights
- Competitive intelligence

---

### 6. Submit User Feedback

Collect user feedback on search results.

**Endpoint**: `POST /v1/databases/{database_id}/analytics/feedback`

**Request Body**:
```json
{
  "query_id": "q1706400123456",
  "user_id": "user_456",
  "session_id": "session_789",
  "rating": 4,
  "feedback_text": "Results were mostly relevant",
  "clicked_result_id": "doc_123",
  "clicked_result_rank": 2
}
```

**Field Descriptions**:
- `query_id` (string, required): Associated query ID
- `user_id` (string, optional): User identifier
- `session_id` (string, optional): Session identifier
- `rating` (integer, optional): 1-5 star rating
- `feedback_text` (string, optional): Free-form feedback
- `clicked_result_id` (string, optional): Clicked document ID
- `clicked_result_rank` (integer, optional): Position in results

**Example Request**:
```bash
curl -X POST "http://localhost:8080/v1/databases/db123/analytics/feedback" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query_id": "q1706400123456",
    "user_id": "user_456",
    "rating": 4,
    "clicked_result_id": "doc_123",
    "clicked_result_rank": 2
  }'
```

**Response** (200 OK):
```json
{
  "success": true,
  "message": "Feedback recorded successfully"
}
```

**Use Cases**:
- Result quality measurement
- Click-through rate tracking
- User satisfaction monitoring
- Relevance tuning

---

### 7. Export Analytics Data

Export query data in CSV or JSON format.

**Endpoint**: `GET /v1/databases/{database_id}/analytics/export`

**Query Parameters**:
- `format` (string, required): Export format (`csv` or `json`)
- `start_time` (integer, required): Unix timestamp in milliseconds
- `end_time` (integer, required): Unix timestamp in milliseconds

**Example Request (CSV)**:
```bash
curl -X GET "http://localhost:8080/v1/databases/db123/analytics/export?format=csv&start_time=1706400000000&end_time=1706486400000" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -o analytics_export.csv
```

**CSV Response** (200 OK):
```csv
Type,Data,Count,Metric,Value
Summary,Total Queries,1523,,
Summary,Success Rate,,%,98.4
Summary,Avg Latency,,ms,45.3
Pattern,machine learning,156,avg_latency_ms,42.5
Pattern,python tutorial,98,avg_latency_ms,35.2
Slow Query,complex similarity search,,total_time_ms,1250
Zero Results,quantum computing basics,12,,
```

**Example Request (JSON)**:
```bash
curl -X GET "http://localhost:8080/v1/databases/db123/analytics/export?format=json&start_time=1706400000000&end_time=1706486400000" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -o analytics_export.json
```

**JSON Response** (200 OK):
```json
{
  "export_time": 1706486400000,
  "time_range": {
    "start": 1706400000000,
    "end": 1706486400000
  },
  "data": {
    "summary": {
      "total_queries": 1523,
      "success_rate": 0.984,
      "avg_latency_ms": 45.3
    },
    "patterns": [...],
    "slow_queries": [...],
    "zero_result_queries": [...],
    "trending_queries": [...]
  }
}
```

**Use Cases**:
- Data warehouse integration
- Business intelligence
- Custom analytics
- Reporting

---

## Error Responses

All endpoints return standard HTTP error codes:

### 400 Bad Request
Invalid parameters or malformed request.

```json
{
  "error": "Invalid time range",
  "message": "start_time must be less than end_time"
}
```

### 401 Unauthorized
Missing or invalid authentication.

```json
{
  "error": "Unauthorized",
  "message": "Invalid or expired authentication token"
}
```

### 404 Not Found
Database not found.

```json
{
  "error": "Database not found",
  "message": "Database 'db123' does not exist"
}
```

### 500 Internal Server Error
Server-side error.

```json
{
  "error": "Internal server error",
  "message": "Failed to retrieve analytics data"
}
```

---

## Rate Limiting

Analytics API endpoints have the following rate limits:

- **Statistics/Insights**: 100 requests/minute
- **Query Logs**: 60 requests/minute
- **Feedback**: 1000 requests/minute
- **Export**: 10 requests/minute

Exceeded rate limits return `429 Too Many Requests`.

---

## Best Practices

### 1. Time Range Selection

- **Dashboard metrics**: Use 24-hour window with hourly granularity
- **Trend analysis**: Use 7-30 day window with daily granularity
- **Real-time monitoring**: Use 1-hour window with minute-level polling

### 2. Pagination

For large datasets, use pagination:

```bash
# First page
GET /analytics/queries?limit=100&offset=0

# Second page
GET /analytics/queries?limit=100&offset=100
```

### 3. Caching

Analytics data is computed on-demand. For frequently accessed metrics:

- Cache results client-side for 30-60 seconds
- Use background aggregation jobs for historical data
- Implement incremental updates for real-time displays

### 4. Performance Optimization

- Request only needed time ranges
- Use appropriate granularity (hourly vs daily)
- Limit result sets with `limit` parameter
- Schedule heavy queries during off-peak hours

### 5. Data Retention

- Query logs: Retained for 30 days (configurable)
- Aggregated statistics: Retained for 1 year
- User feedback: Retained indefinitely

---

## Example Integration

### Python Client

```python
import requests
from datetime import datetime, timedelta

class AnalyticsClient:
    def __init__(self, base_url, token):
        self.base_url = base_url
        self.headers = {"Authorization": f"Bearer {token}"}

    def get_statistics(self, database_id, hours=24, granularity="hourly"):
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(hours=hours)).timestamp() * 1000)

        url = f"{self.base_url}/databases/{database_id}/analytics/stats"
        params = {
            "start_time": start_time,
            "end_time": end_time,
            "granularity": granularity
        }

        response = requests.get(url, headers=self.headers, params=params)
        response.raise_for_status()
        return response.json()

    def get_insights(self, database_id):
        end_time = int(datetime.now().timestamp() * 1000)
        start_time = int((datetime.now() - timedelta(days=1)).timestamp() * 1000)

        url = f"{self.base_url}/databases/{database_id}/analytics/insights"
        params = {"start_time": start_time, "end_time": end_time}

        response = requests.get(url, headers=self.headers, params=params)
        response.raise_for_status()
        return response.json()

# Usage
client = AnalyticsClient("http://localhost:8080/v1", "your_token")
stats = client.get_statistics("db123", hours=24)
insights = client.get_insights("db123")
```

### JavaScript Client

```javascript
class AnalyticsClient {
  constructor(baseURL, token) {
    this.baseURL = baseURL;
    this.headers = {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json'
    };
  }

  async getStatistics(databaseId, hours = 24, granularity = 'hourly') {
    const endTime = Date.now();
    const startTime = endTime - (hours * 60 * 60 * 1000);

    const url = `${this.baseURL}/databases/${databaseId}/analytics/stats`;
    const params = new URLSearchParams({
      start_time: startTime,
      end_time: endTime,
      granularity
    });

    const response = await fetch(`${url}?${params}`, {
      headers: this.headers
    });

    if (!response.ok) throw new Error(`API error: ${response.status}`);
    return response.json();
  }

  async submitFeedback(databaseId, feedbackData) {
    const url = `${this.baseURL}/databases/${databaseId}/analytics/feedback`;

    const response = await fetch(url, {
      method: 'POST',
      headers: this.headers,
      body: JSON.stringify(feedbackData)
    });

    if (!response.ok) throw new Error(`API error: ${response.status}`);
    return response.json();
  }
}

// Usage
const client = new AnalyticsClient('http://localhost:8080/v1', 'your_token');
const stats = await client.getStatistics('db123', 24, 'hourly');
await client.submitFeedback('db123', {
  query_id: 'q123',
  rating: 5,
  feedback_text: 'Great results!'
});
```

---

## Changelog

### Version 1.0 (January 28, 2026)
- Initial release
- 7 analytics endpoints
- CSV/JSON export support
- User feedback collection
- Performance optimizations

---

## Support

For questions or issues:
- Documentation: https://jadevectordb.com/docs/analytics
- GitHub Issues: https://github.com/jadevectordb/jadevectordb/issues
- Email: support@jadevectordb.com

---

**End of Analytics API Reference**
