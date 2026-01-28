# JadeVectorDB Analytics - Metrics Interpretation Guide

**Version**: 1.0
**Last Updated**: January 28, 2026
**Audience**: Administrators, DevOps Engineers, Data Analysts

---

## Table of Contents

1. [Introduction](#introduction)
2. [Core Metrics](#core-metrics)
3. [Latency Metrics](#latency-metrics)
4. [Query Type Metrics](#query-type-metrics)
5. [Success and Error Metrics](#success-and-error-metrics)
6. [User Engagement Metrics](#user-engagement-metrics)
7. [Performance Benchmarks](#performance-benchmarks)
8. [Alerting Thresholds](#alerting-thresholds)
9. [Troubleshooting Guide](#troubleshooting-guide)

---

## Introduction

This guide helps you understand and interpret the metrics provided by JadeVectorDB Analytics. Each metric is explained with:
- **Definition**: What the metric measures
- **Calculation**: How it's computed
- **Interpretation**: What values mean
- **Target Values**: Recommended thresholds
- **Action Items**: What to do when metrics are out of range

---

## Core Metrics

### Total Queries

**Definition**: Total number of search queries executed in the time range.

**Calculation**:
```
Total Queries = COUNT(all queries in time range)
```

**Interpretation**:
- Indicates overall system usage
- Shows adoption and engagement trends
- Useful for capacity planning

**Typical Values**:
| Scale | Queries/Day | Queries/Hour | QPS |
|-------|-------------|--------------|-----|
| Small | 100-1,000 | 4-42 | 0.001-0.01 |
| Medium | 1,000-10,000 | 42-417 | 0.01-0.1 |
| Large | 10,000-100,000 | 417-4,167 | 0.1-1.2 |
| Enterprise | 100,000+ | 4,167+ | 1.2+ |

**Trends to Watch**:
- ✅ Steady growth: Healthy adoption
- ✅ Predictable patterns: Normal usage
- ⚠️ Sudden spike: Marketing event or attack
- ⚠️ Sudden drop: System outage or client issue
- ⚠️ Gradual decline: User churn

**Action Items**:
- Declining: Investigate user experience issues
- Spiking: Verify capacity and check for abuse
- Flat: Consider growth initiatives

---

### Success Rate

**Definition**: Percentage of queries that completed without errors.

**Calculation**:
```
Success Rate = (Successful Queries / Total Queries) × 100%
```

**Interpretation**:
- Measures system reliability
- Indicates data quality
- Reflects user experience

**Target Values**:
| Rating | Success Rate | Status |
|--------|-------------|---------|
| Excellent | ≥ 99% | ✅ Healthy |
| Good | 95-99% | ✅ Acceptable |
| Fair | 90-95% | ⚠️ Investigate |
| Poor | < 90% | ❌ Critical |

**Common Causes of Low Success Rate**:

**80-90% Success Rate**:
- Invalid query formats
- Missing vector dimensions
- Malformed metadata filters
- Client-side errors

**60-80% Success Rate**:
- Database connectivity issues
- Timeout configuration problems
- Index corruption
- Resource exhaustion

**< 60% Success Rate**:
- Service outage
- Database unavailable
- Configuration errors
- Network failures

**Action Items**:
- < 95%: Review error logs
- < 90%: Emergency investigation
- Sudden drop: Check recent deployments

---

### Queries Per Second (QPS)

**Definition**: Average rate of query execution.

**Calculation**:
```
QPS = Total Queries / Time Range (seconds)
```

**Interpretation**:
- Measures system load
- Key capacity planning metric
- Indicates peak usage patterns

**Typical Values by Use Case**:
| Use Case | Expected QPS |
|----------|-------------|
| Internal tools | 0.01-0.1 |
| Small application | 0.1-1 |
| Medium application | 1-10 |
| Large application | 10-100 |
| High-traffic service | 100-1,000 |
| Global platform | 1,000+ |

**Capacity Planning**:
- **Current QPS**: Actual observed rate
- **Peak QPS**: Maximum observed rate
- **Provisioned QPS**: System capacity
- **Safety Margin**: 20-50% headroom recommended

**Formula**:
```
Required Capacity = Peak QPS × (1 + Safety Margin) × Growth Factor
```

**Example**:
- Peak QPS: 50
- Safety Margin: 30% (1.3)
- Growth Factor: 2.0 (100% growth expected)
- Required: 50 × 1.3 × 2.0 = 130 QPS capacity

**Action Items**:
- Approaching 70% capacity: Plan scale-up
- Above 80% capacity: Scale urgently
- Exceeding 100% capacity: Emergency scaling

---

## Latency Metrics

### Average Latency

**Definition**: Mean query execution time.

**Calculation**:
```
Average Latency = SUM(all query latencies) / COUNT(queries)
```

**Interpretation**:
- Shows typical user experience
- Useful for SLA compliance
- Sensitive to outliers

**Target Values**:
| Rating | Latency | User Experience |
|--------|---------|-----------------|
| Excellent | < 50ms | Instant |
| Good | 50-100ms | Very fast |
| Acceptable | 100-300ms | Fast |
| Slow | 300-1,000ms | Noticeable |
| Very Slow | > 1,000ms | Poor |

**Factors Affecting Latency**:
1. **Vector Dimension**: Higher dimensions = slower
2. **Index Type**: HNSW faster than IVF
3. **Result Count**: More results = slower
4. **Database Size**: Larger databases = slower (without proper indexing)
5. **Query Type**: Hybrid > Vector > BM25

**Action Items**:
- > 100ms: Review index configuration
- > 300ms: Optimize queries or scale resources
- > 1,000ms: Critical performance issue

---

### P50 (Median) Latency

**Definition**: 50th percentile latency - half of queries are faster.

**Calculation**:
```
Sort all latencies ascending
P50 = latency at position (count / 2)
```

**Interpretation**:
- Typical user experience
- Less affected by outliers than average
- Better metric than average for skewed distributions

**Comparison with Average**:
- **P50 ≈ Average**: Normal distribution
- **P50 < Average**: Positive skew (some very slow queries)
- **P50 > Average**: Negative skew (unusual, investigate)

**Example**:
```
Queries: 100
Latencies: [10, 12, 15, ..., 45, 2000, 3000]
Average: 85ms (skewed by outliers)
P50: 30ms (typical experience)
```

**Target**: P50 should be < 50ms for responsive applications

---

### P95 Latency

**Definition**: 95th percentile latency - 95% of queries are faster.

**Calculation**:
```
P95 = latency at position (count × 0.95)
```

**Interpretation**:
- Captures outliers but ignores extreme cases
- Important for SLA guarantees
- Reflects worst-case for most users

**Typical Ratios**:
| Ratio | Interpretation |
|-------|---------------|
| P95/P50 < 2 | Consistent performance |
| P95/P50 2-5 | Some variability |
| P95/P50 5-10 | High variability |
| P95/P50 > 10 | Performance issues |

**Example**:
```
P50 = 30ms
P95 = 90ms
Ratio = 3 (acceptable variability)
```

**Target**: P95 should be < 2× P50 for consistent performance

---

### P99 Latency

**Definition**: 99th percentile latency - 99% of queries are faster.

**Calculation**:
```
P99 = latency at position (count × 0.99)
```

**Interpretation**:
- Worst-case performance
- Includes network issues, GC pauses, etc.
- Critical for premium user experience

**Typical Values**:
| Service Tier | P99 Target |
|--------------|-----------|
| Premium | < 200ms |
| Standard | < 500ms |
| Basic | < 1,000ms |

**Action Items**:
- P99 > 500ms: Investigate slow queries
- P99 > 1,000ms: Critical performance issue
- P99 increasing: System degradation

---

### Latency Distribution Analysis

**Healthy Distribution**:
```
P50:  30ms
P95:  60ms (2× P50)
P99: 100ms (3.3× P50)
Avg:  35ms (close to P50)
```

**Problematic Distribution**:
```
P50:   30ms
P95:  200ms (6.7× P50)
P99: 1500ms (50× P50)
Avg:   80ms (2.7× P50)
```

**Interpretation**:
- Wide spread indicates inconsistent performance
- Investigate causes of high P99
- May need query optimization or infrastructure improvements

---

## Query Type Metrics

### Vector Queries

**Definition**: Queries using vector similarity search only.

**Typical Latency**: 20-50ms

**Factors**:
- Vector dimension
- Distance metric (cosine, euclidean, dot product)
- Top-K value
- Index type (HNSW, IVF)

**Optimization**:
- Use HNSW for speed
- Reduce dimension if possible
- Limit Top-K to needed results

---

### Hybrid Queries

**Definition**: Queries combining vector and BM25 keyword search.

**Typical Latency**: 30-80ms

**Components**:
- Vector search: 20-40ms
- BM25 search: 5-15ms
- Score fusion: 5-10ms
- Total: Sum of components

**Optimization**:
- Build BM25 index
- Use RRF fusion (faster than linear)
- Adjust candidate pool sizes

---

### BM25 Queries

**Definition**: Keyword-based search only.

**Typical Latency**: 10-30ms

**Factors**:
- Index size
- Query length
- Result count

**Optimization**:
- Ensure inverted index exists
- Use appropriate stop words
- Limit result set size

---

### Reranked Queries

**Definition**: Queries using cross-encoder re-ranking.

**Typical Latency**: 150-300ms (additional overhead)

**Components**:
- Initial search: 30-50ms
- Re-ranking: 100-250ms (depends on candidate count)

**Optimization**:
- Limit candidates to 50-100
- Use batch inference
- Consider GPU acceleration

---

## Success and Error Metrics

### Zero-Result Queries

**Definition**: Queries returning no results.

**Typical Rate**: 5-15% of queries

**Causes**:
1. **Content Gaps**: No relevant documents
2. **Poor Embeddings**: Low quality vectors
3. **Overly Strict Filters**: Metadata filters too restrictive
4. **Typos**: User input errors
5. **Out-of-Domain**: Queries outside training data

**Interpretation**:
| Rate | Status |
|------|--------|
| < 5% | ✅ Excellent |
| 5-10% | ✅ Good |
| 10-20% | ⚠️ Review |
| > 20% | ❌ Critical |

**Action Items**:
- Analyze common patterns
- Add missing content
- Improve embeddings
- Adjust similarity thresholds

---

### Failed Queries

**Definition**: Queries that threw an error.

**Common Error Types**:

1. **Validation Errors (40-60% of errors)**:
   - Invalid vector dimensions
   - Malformed JSON
   - Missing required fields

2. **Timeout Errors (20-30%)**:
   - Query too complex
   - System overload
   - Network issues

3. **Resource Errors (10-20%)**:
   - Out of memory
   - Connection pool exhausted
   - Disk space full

4. **System Errors (5-10%)**:
   - Database corruption
   - Index errors
   - Service failures

**Target**: < 1% error rate

**Action Items**:
- Review error logs
- Fix validation issues
- Optimize slow queries
- Scale resources if needed

---

## User Engagement Metrics

### Unique Users

**Definition**: Distinct users issuing queries.

**Calculation**:
```
Unique Users = COUNT(DISTINCT user_id)
```

**Interpretation**:
- Measures user adoption
- Useful for DAU/MAU metrics

**Typical Ratios**:
```
Queries per User = Total Queries / Unique Users
```

| Ratio | Interpretation |
|-------|---------------|
| 1-5 | Low engagement |
| 5-20 | Normal usage |
| 20-50 | High engagement |
| > 50 | Power users or automation |

---

### Unique Sessions

**Definition**: Distinct user sessions.

**Calculation**:
```
Unique Sessions = COUNT(DISTINCT session_id)
```

**Interpretation**:
- Measures visit frequency
- Indicates session duration

**Typical Ratios**:
```
Queries per Session = Total Queries / Unique Sessions
```

| Ratio | Interpretation |
|-------|---------------|
| 1-3 | Quick searches |
| 3-10 | Normal sessions |
| 10-30 | Extended sessions |
| > 30 | Research or data exploration |

---

## Performance Benchmarks

### By Database Size

| Vectors | Expected P50 | Expected P95 |
|---------|-------------|-------------|
| < 10K | < 20ms | < 40ms |
| 10K-100K | < 30ms | < 60ms |
| 100K-1M | < 50ms | < 100ms |
| 1M-10M | < 100ms | < 200ms |
| > 10M | < 200ms | < 500ms |

*Assumes HNSW index, 512-dimension vectors, cosine similarity*

---

### By Query Type

| Query Type | P50 Target | P95 Target |
|------------|-----------|-----------|
| Vector (HNSW) | < 30ms | < 60ms |
| Vector (IVF) | < 50ms | < 100ms |
| BM25 | < 20ms | < 40ms |
| Hybrid (RRF) | < 50ms | < 100ms |
| Hybrid + Rerank | < 200ms | < 400ms |

---

### By Vector Dimension

| Dimension | P50 Target | Notes |
|-----------|-----------|-------|
| 128 | < 20ms | Fast, lower accuracy |
| 384 | < 30ms | Good balance |
| 768 | < 50ms | High accuracy |
| 1536 | < 100ms | SOTA models |
| 3072+ | < 200ms | Specialized use cases |

---

## Alerting Thresholds

### Critical Alerts (Immediate Action)

| Metric | Threshold | Severity |
|--------|-----------|----------|
| Success Rate | < 90% | 🔴 Critical |
| P99 Latency | > 1000ms | 🔴 Critical |
| Error Rate | > 10% | 🔴 Critical |
| QPS | > 100% capacity | 🔴 Critical |

### Warning Alerts (Monitor Closely)

| Metric | Threshold | Severity |
|--------|-----------|----------|
| Success Rate | < 95% | 🟡 Warning |
| P95 Latency | > 500ms | 🟡 Warning |
| Error Rate | > 5% | 🟡 Warning |
| QPS | > 80% capacity | 🟡 Warning |
| Zero Results | > 20% | 🟡 Warning |

### Informational Alerts

| Metric | Threshold | Severity |
|--------|-----------|----------|
| QPS | > 50% capacity | ℹ️ Info |
| P50 Latency | > 100ms | ℹ️ Info |
| Trending Queries | > 200% growth | ℹ️ Info |

---

## Troubleshooting Guide

### High Latency

**Symptoms**: P95 > 500ms

**Diagnosis Steps**:

1. **Check Slow Queries**:
   ```
   View Insights → Slow Queries table
   ```

2. **Analyze Query Types**:
   - Vector: Check index type
   - Hybrid: Verify BM25 index built
   - Reranking: Review candidate count

3. **Review System Resources**:
   - CPU utilization
   - Memory usage
   - Disk I/O
   - Network latency

**Solutions**:
- Optimize indexes
- Reduce result counts
- Scale resources
- Enable caching

---

### Low Success Rate

**Symptoms**: Success Rate < 95%

**Diagnosis Steps**:

1. **Review Error Types**:
   - Check error logs
   - Categorize errors
   - Identify patterns

2. **Check Recent Changes**:
   - Deployments
   - Configuration changes
   - Schema updates

3. **Verify Client Behavior**:
   - API usage patterns
   - Request validation
   - Retry logic

**Solutions**:
- Fix validation errors
- Update client libraries
- Improve error handling
- Rollback problematic changes

---

### High Zero-Result Rate

**Symptoms**: Zero Results > 20%

**Diagnosis Steps**:

1. **Analyze Patterns**:
   ```
   View Insights → Zero-Result Queries
   ```

2. **Check Content Coverage**:
   - Review query topics
   - Compare with indexed documents
   - Verify embedding quality

3. **Test Similarity Threshold**:
   - Lower threshold temporarily
   - Check if results appear

**Solutions**:
- Add missing content
- Re-embed documents
- Adjust similarity thresholds
- Improve embeddings

---

## Conclusion

Regular monitoring of these metrics helps maintain optimal JadeVectorDB performance. Use this guide to:

1. ✅ Set appropriate alert thresholds
2. ✅ Diagnose performance issues
3. ✅ Plan capacity and scaling
4. ✅ Optimize query performance
5. ✅ Improve user experience

For further assistance, consult:
- [Analytics API Reference](ANALYTICS_API_REFERENCE.md)
- [Analytics Dashboard Guide](ANALYTICS_DASHBOARD_GUIDE.md)
- [JadeVectorDB Documentation](https://jadevectordb.com/docs)

---

**End of Metrics Interpretation Guide**
