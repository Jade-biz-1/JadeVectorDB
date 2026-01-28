# JadeVectorDB Analytics Dashboard - User Guide

**Version**: 1.0
**Last Updated**: January 28, 2026
**Feature**: Query Analytics Dashboard (Phase 16, T16.21)

---

## Table of Contents

1. [Introduction](#introduction)
2. [Accessing the Dashboard](#accessing-the-dashboard)
3. [Dashboard Overview](#dashboard-overview)
4. [Key Metrics](#key-metrics)
5. [Time Range Selection](#time-range-selection)
6. [Tabs and Views](#tabs-and-views)
7. [Use Cases](#use-cases)
8. [Best Practices](#best-practices)
9. [Troubleshooting](#troubleshooting)

---

## Introduction

The Analytics Dashboard provides real-time visibility into your JadeVectorDB search operations. Monitor query performance, identify patterns, detect anomalies, and gain actionable insights to optimize your vector database deployment.

### Key Features

- **Real-time Metrics**: Live query counts, success rates, and latency
- **Interactive Charts**: Time-series and distribution visualizations
- **Pattern Analysis**: Identify common search patterns
- **Performance Monitoring**: Track slow queries and latency percentiles
- **Content Gap Detection**: Find queries returning zero results
- **Trending Queries**: Discover emerging search topics
- **Automated Insights**: AI-generated recommendations

---

## Accessing the Dashboard

### URL

Navigate to: `http://localhost:3000/analytics`

(Replace `localhost:3000` with your frontend deployment URL)

### Authentication

The dashboard requires authentication. You must be logged in with valid credentials.

**Required Permissions**:
- `analytics:read` - View analytics data
- `analytics:export` (optional) - Export analytics data

### Browser Compatibility

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

---

## Dashboard Overview

### Layout

The dashboard consists of four main sections:

```
┌─────────────────────────────────────────────────────────┐
│  Header (Database Selector | Time Range | Refresh)     │
├─────────────────────────────────────────────────────────┤
│  Tabs (Overview | Query Explorer | Patterns | Insights)│
├─────────────────────────────────────────────────────────┤
│                                                          │
│  Content Area (varies by tab)                          │
│                                                          │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Header Controls

#### Database Selector
- Dropdown menu listing all available databases
- Auto-selects first database on load
- Changes refresh all data for selected database

#### Time Range Picker
- **1h**: Last hour (minute-level granularity)
- **24h**: Last 24 hours (hourly granularity)
- **7d**: Last 7 days (daily granularity)
- **30d**: Last 30 days (daily granularity)

#### Refresh Button
- Manual refresh of all dashboard data
- Disabled during data loading
- Dashboard auto-refreshes every 30 seconds

#### Last Updated
- Displays timestamp of last successful data refresh
- Format: "Last updated: HH:MM:SS AM/PM"

---

## Key Metrics

### Metric Cards (Overview Tab)

Four prominent metric cards display at the top of the Overview tab:

#### 1. Total Queries
- **Description**: Total number of queries in selected time range
- **Color**: Purple gradient
- **Interpretation**:
  - Increasing trend: Growing usage
  - Sudden drop: Potential system issue
  - Spikes: Marketing campaigns or viral content

#### 2. Success Rate
- **Description**: Percentage of queries that completed successfully
- **Color**: Pink gradient
- **Calculation**: `(successful_queries / total_queries) * 100`
- **Target**: ≥ 95%
- **Interpretation**:
  - < 90%: Investigate error logs
  - 90-95%: Monitor closely
  - > 95%: Healthy

#### 3. Avg Latency
- **Description**: Average query execution time
- **Color**: Blue gradient
- **Format**: Milliseconds or seconds
- **Target**: < 100ms for most use cases
- **Interpretation**:
  - < 50ms: Excellent
  - 50-100ms: Good
  - 100-500ms: Acceptable
  - > 500ms: Needs optimization

#### 4. Queries Per Second (QPS)
- **Description**: Average query rate
- **Color**: Green gradient
- **Calculation**: `total_queries / time_range_seconds`
- **Interpretation**:
  - Use for capacity planning
  - Compare to provisioned throughput
  - Identify peak usage hours

---

## Time Range Selection

### Choosing the Right Range

| Use Case | Recommended Range | Granularity |
|----------|------------------|-------------|
| Real-time monitoring | 1h | Minute-level |
| Daily operations | 24h | Hourly |
| Weekly review | 7d | Daily |
| Monthly reporting | 30d | Daily |
| Trend analysis | 7d or 30d | Daily |

### Data Granularity

- **1h range**: Data points every ~5 minutes
- **24h range**: Data points every hour
- **7d range**: Data points every day
- **30d range**: Data points every day

---

## Tabs and Views

### 1. Overview Tab

The default view showing high-level metrics and trends.

#### Components:

**A. Key Metrics Cards**
- Four gradient cards (see [Key Metrics](#key-metrics))

**B. Queries Over Time Chart**
- **Type**: Line chart
- **Data**: Total, successful, and failed queries
- **X-axis**: Time (hour or day)
- **Y-axis**: Query count
- **Legend**:
  - Blue: Total queries
  - Green: Successful queries
  - Red: Failed queries
- **Interactions**:
  - Hover over points for exact values
  - Click legend to toggle series

**C. Latency Distribution Chart**
- **Type**: Bar chart
- **Data**: Avg, P95, P99 latencies
- **X-axis**: Time bucket
- **Y-axis**: Latency (ms)
- **Colors**:
  - Blue: Average latency
  - Orange: P95 latency (95th percentile)
  - Red: P99 latency (99th percentile)
- **Interpretation**:
  - Avg shows typical performance
  - P95 shows outliers
  - P99 shows worst-case scenarios

**D. Top Query Patterns**
- **Type**: Table (top 5)
- **Columns**:
  - Pattern: Normalized query text
  - Count: Number of occurrences
  - Avg Latency: Average execution time
- **Use**: Identify most common searches

**E. Slow Queries**
- **Type**: Table (top 5)
- **Columns**:
  - Query: Query text (truncated)
  - Latency: Total execution time
  - Time: Timestamp
- **Threshold**: Queries > 1000ms
- **Use**: Performance troubleshooting

---

### 2. Query Explorer Tab

Detailed view of individual queries with search and filtering.

#### Features:

**Recent Queries Table**
- **Columns**:
  - Timestamp: When query was executed
  - Query Text: Full or truncated query
  - Type: Vector, Hybrid, BM25, or Reranked
  - Results: Number of results returned
  - Latency: Total execution time
  - Status: Success or Failed badge

- **Pagination**: 50 queries per page

- **Query Type Badges**:
  - Blue: Vector search
  - Purple: Hybrid search
  - Green: BM25 search
  - Orange: Reranked results

- **Status Badges**:
  - Green: Success
  - Red: Failed

#### Use Cases:

- Debug specific queries
- Analyze user search behavior
- Verify query logging
- Audit search activity

---

### 3. Patterns Tab

Analysis of common query patterns and trends.

#### Components:

**A. Common Query Patterns Table**
- **Columns**:
  - Pattern: Normalized query text
  - Count: Total occurrences
  - Avg Latency: Average execution time
  - Avg Results: Average results returned
  - First Seen: First occurrence timestamp
  - Last Seen: Most recent occurrence

- **Sorting**: Default by Count (descending)

- **Normalization**: Queries are normalized by:
  - Lowercase conversion
  - Stop word removal
  - Punctuation removal
  - Whitespace normalization

**B. Trending Queries Table** (if data available)
- **Columns**:
  - Query Pattern: Normalized text
  - Current Count: Queries in current period
  - Previous Count: Queries in previous period
  - Growth Rate: Percentage increase

- **Growth Badge**: Green badge showing +X%

- **Threshold**: Minimum 50% growth to appear

#### Use Cases:

- Content optimization
- SEO/SEM strategy
- Auto-suggest features
- Query canonicalization

---

### 4. Insights Tab

Automated insights and recommendations.

#### Components:

**A. Automated Insights Panel**

Color-coded insight cards with recommendations:

- **Success (Green)**:
  - Query performance metrics
  - Healthy system indicators
  - Trending query alerts

- **Info (Blue)**:
  - Peak usage information
  - General statistics
  - Neutral observations

- **Warning (Yellow)**:
  - Performance alerts (slow queries)
  - Content gaps (zero results)
  - Capacity warnings

- **Error (Red)**:
  - Critical performance issues
  - High error rates
  - System failures

**Example Insights**:

```
✓ Query Performance
  1,523 queries processed with 98.4% success rate.
  Average latency: 45ms

ℹ Peak Usage
  Highest traffic at 2:00 PM with 256 queries

⚠ Performance Alert
  5 slow queries detected (> 1000ms).
  Consider optimizing indexes or query patterns.

✗ Content Gaps
  12 query patterns returned zero results.
  Consider adding relevant content.
```

**B. Zero-Result Queries Table** (if applicable)
- **Columns**:
  - Query Pattern: Normalized text
  - Occurrence Count: How often it returned zero results

- **Use**: Identify content gaps in your database

#### Use Cases:

- Proactive optimization
- Content strategy
- Quality monitoring
- Issue detection

---

## Use Cases

### 1. Real-Time Monitoring

**Scenario**: Monitor live system health

**Steps**:
1. Select 1h time range
2. Stay on Overview tab
3. Watch key metrics cards
4. Monitor Queries Over Time chart
5. Auto-refresh keeps data current

**Key Metrics**:
- Success rate should be > 95%
- Avg latency should be consistent
- No sudden drops in QPS

---

### 2. Performance Troubleshooting

**Scenario**: Investigate slow queries

**Steps**:
1. Go to Overview tab
2. Check Latency Distribution chart
3. Look for spikes in P95/P99
4. Review Slow Queries table
5. Click Query Explorer tab
6. Filter by timestamp of spike
7. Analyze specific slow queries

**Actions**:
- Optimize slow query patterns
- Add missing indexes
- Scale resources if needed

---

### 3. Content Optimization

**Scenario**: Identify content gaps

**Steps**:
1. Go to Insights tab
2. Review Zero-Result Queries section
3. Note common patterns with no results
4. Switch to Patterns tab
5. Compare zero-result patterns with successful ones

**Actions**:
- Add missing content
- Improve existing documents
- Update embeddings
- Adjust search parameters

---

### 4. Trend Analysis

**Scenario**: Discover emerging topics

**Steps**:
1. Select 7d or 30d time range
2. Go to Patterns tab
3. Review Trending Queries section
4. Note queries with high growth rates

**Actions**:
- Prioritize high-growth topics
- Create content for trending queries
- Adjust marketing strategy
- Plan capacity expansion

---

### 5. Capacity Planning

**Scenario**: Plan for growth

**Steps**:
1. Select 30d time range
2. Review QPS metric card
3. Analyze Queries Over Time chart
4. Identify peak usage patterns
5. Review Latency Distribution for degradation

**Metrics to Track**:
- Peak QPS
- Average QPS growth rate
- Latency trends under load
- Success rate correlation with QPS

**Actions**:
- Scale infrastructure
- Optimize indexes
- Implement caching
- Load balancing

---

## Best Practices

### 1. Daily Monitoring

**Morning Check** (5 minutes):
- Select 24h time range
- Review key metrics
- Check for red/yellow insights
- Verify success rate > 95%

**Weekly Review** (15 minutes):
- Select 7d time range
- Analyze trends
- Review top patterns
- Check slow queries
- Export data for reporting

### 2. Alert Thresholds

Set up external monitoring for:
- Success rate < 95%
- Avg latency > 100ms
- P95 latency > 200ms
- QPS > 80% of capacity

### 3. Data Export

Export analytics data regularly:
- Daily: For business intelligence
- Weekly: For trend analysis
- Monthly: For reporting

### 4. Query Optimization

Regularly review:
- Slow queries (> 100ms)
- Zero-result patterns
- High-frequency queries

Optimize:
- Add indexes for slow patterns
- Improve embeddings
- Adjust search parameters

### 5. Dashboard Customization

- Bookmark favorite time ranges
- Create multiple browser tabs for different databases
- Use browser zoom for better visibility
- Take screenshots for reports

---

## Troubleshooting

### Dashboard Not Loading

**Symptoms**: Blank page or loading spinner

**Causes & Solutions**:

1. **Authentication Issue**
   - Verify you're logged in
   - Check token expiration
   - Re-authenticate if needed

2. **Network Error**
   - Check backend server is running
   - Verify API endpoint URL
   - Check browser console for errors

3. **Browser Cache**
   - Clear browser cache
   - Hard refresh (Ctrl+Shift+R)
   - Try incognito mode

### No Data Displayed

**Symptoms**: "No queries found" or empty charts

**Causes & Solutions**:

1. **No Queries in Time Range**
   - Expand time range (try 7d or 30d)
   - Verify database has been queried
   - Check database selector

2. **Database Not Selected**
   - Select database from dropdown
   - Verify database exists

3. **Analytics Not Enabled**
   - Confirm analytics logging is enabled
   - Check backend configuration
   - Verify database permissions

### Charts Not Rendering

**Symptoms**: Missing or broken charts

**Causes & Solutions**:

1. **Browser Compatibility**
   - Update browser to latest version
   - Try different browser

2. **JavaScript Error**
   - Check browser console
   - Disable browser extensions
   - Clear cache

3. **Data Format Issue**
   - Check API response format
   - Verify backend version compatibility

### Slow Dashboard Performance

**Symptoms**: Laggy interactions, slow loading

**Solutions**:

1. **Reduce Time Range**
   - Use shorter time range (1h or 24h)
   - Reduce granularity

2. **Limit Data Points**
   - Use fewer databases
   - Clear browser cache

3. **Network Issues**
   - Check network latency
   - Use local deployment if possible

### Auto-Refresh Not Working

**Symptoms**: Data doesn't update automatically

**Solutions**:

1. **Manual Refresh**
   - Click Refresh button
   - Verify timestamp updates

2. **Browser Issue**
   - Close and reopen dashboard
   - Clear browser cache

3. **Backend Issue**
   - Check server logs
   - Verify analytics service running

---

## Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| R | Refresh dashboard |
| 1 | Switch to Overview tab |
| 2 | Switch to Query Explorer tab |
| 3 | Switch to Patterns tab |
| 4 | Switch to Insights tab |
| D | Toggle database selector |
| T | Cycle time range (1h → 24h → 7d → 30d) |

*(Note: Shortcuts may vary based on implementation)*

---

## Privacy and Data Handling

### Data Collection

The dashboard displays data from:
- Query logs (text, metadata, performance)
- User IDs (if provided)
- Session IDs (if provided)
- IP addresses (backend only, not displayed)

### Data Retention

- Query logs: 30 days (configurable)
- Aggregated statistics: 1 year
- User feedback: Indefinite

### Privacy Considerations

- Sensitive data in queries is logged
- Configure data masking if needed
- Review retention policies regularly
- Comply with data protection regulations (GDPR, CCPA)

---

## Feedback and Support

### Reporting Issues

If you encounter problems:

1. Check [Troubleshooting](#troubleshooting) section
2. Review browser console for errors
3. Check backend server logs
4. Submit issue with:
   - Browser and version
   - Steps to reproduce
   - Screenshots if applicable
   - Console error messages

### Feature Requests

Submit feature requests via:
- GitHub Issues
- Email: support@jadevectordb.com
- Community forum

---

## Appendix

### Glossary

- **P50/P95/P99**: Latency percentiles (50th, 95th, 99th)
- **QPS**: Queries Per Second
- **Normalized Query**: Query with stop words and punctuation removed
- **Zero-Result Query**: Query returning no results
- **Trending Query**: Query with significant growth in frequency
- **Granularity**: Time bucket size (hourly, daily, etc.)

### Related Documentation

- [Analytics API Reference](ANALYTICS_API_REFERENCE.md)
- [Metrics Interpretation Guide](ANALYTICS_METRICS_GUIDE.md)
- [Privacy Policy](ANALYTICS_PRIVACY_POLICY.md)
- [JadeVectorDB User Guide](UserGuide.md)

---

**End of Analytics Dashboard User Guide**

*For the latest updates and documentation, visit: https://jadevectordb.com/docs*
