#!/usr/bin/env python3
"""
JadeVectorDB API Example — Analytics
=======================================

JadeVectorDB tracks detailed analytics for every database, giving you
visibility into search usage, performance, and trends. This data helps
you optimize your vector search pipeline and understand user behavior.

This example demonstrates:
  1. Get analytics statistics (query counts, latency, throughput)
  2. View recent query analytics
  3. Discover query patterns (common search terms)
  4. Get analytics insights (automated recommendations)
  5. Find trending queries (growing in popularity)
  6. Submit search quality feedback
  7. Export analytics data

APIs covered:
  - client.get_analytics_stats(database_id, start_time, end_time, granularity)
  - client.get_analytics_queries(database_id, start_time, end_time, limit, offset)
  - client.get_analytics_patterns(database_id, start_time, end_time, min_count, limit)
  - client.get_analytics_insights(database_id, start_time, end_time)
  - client.get_analytics_trending(database_id, start_time, end_time, min_growth, limit)
  - client.submit_analytics_feedback(database_id, query_id, user_id, rating,
                                     feedback_text, clicked_result_id, clicked_rank)
  - client.export_analytics(database_id, format, start_time, end_time)
"""

import os
import sys
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "cli", "python"))
from jadevectordb import JadeVectorDB, JadeVectorDBError

SERVER_URL = os.environ.get("JADEVECTORDB_URL", "http://localhost:8080")
API_KEY = os.environ.get("JADEVECTORDB_API_KEY", "")

client = JadeVectorDB(base_url=SERVER_URL, api_key=API_KEY)

# --- Setup: create a database and run some searches to generate analytics data ---
DIMENSION = 128
db_id = client.create_database(
    name="analytics-example",
    vector_dimension=DIMENSION,
    index_type="HNSW",
)
print(f"Created database: {db_id}")


def make_vector(seed: float) -> list:
    return [round(seed + i * 0.001, 6) for i in range(DIMENSION)]


# Store a few vectors and run searches to generate analytics events
for i in range(5):
    client.store_vector(db_id, f"vec-{i}", make_vector(0.1 * i), {"text": f"Document {i}"})
for seed in [0.05, 0.15, 0.25, 0.35]:
    try:
        client.search(db_id, make_vector(seed), top_k=3)
    except JadeVectorDBError:
        pass
print("Generated some search activity\n")

# ---------------------------------------------------------------------------
# 1. Analytics statistics
# ---------------------------------------------------------------------------
# Overview metrics: total queries, average latency, throughput, etc.
# Granularity options: "hourly", "daily", "weekly"

print("=== 1. Analytics Statistics (hourly) ===")
try:
    stats = client.get_analytics_stats(
        database_id=db_id,
        granularity="hourly",
        # start_time="2025-01-01T00:00:00Z",   # Optional time range
        # end_time="2025-12-31T23:59:59Z",
    )
    print(json.dumps(stats, indent=2)[:500])
except JadeVectorDBError as e:
    print(f"Analytics stats: {e}")

# ---------------------------------------------------------------------------
# 2. Query analytics
# ---------------------------------------------------------------------------
# Detailed log of individual search queries. Supports pagination.

print("\n=== 2. Recent Queries ===")
try:
    queries = client.get_analytics_queries(
        database_id=db_id,
        limit=5,
        offset=0,
    )
    query_list = queries.get("queries", []) if isinstance(queries, dict) else queries
    print(f"Recent queries: {len(query_list)}")
    for q in query_list[:5]:
        qid = q.get("query_id", q.get("id", "?"))
        latency = q.get("latency_ms", q.get("latency", "?"))
        results = q.get("result_count", q.get("results", "?"))
        print(f"  {qid:20s}  latency={latency}ms  results={results}")
except JadeVectorDBError as e:
    print(f"Query analytics: {e}")

# ---------------------------------------------------------------------------
# 3. Query patterns
# ---------------------------------------------------------------------------
# Identifies recurring search patterns — useful for understanding what
# users are searching for and optimizing your data accordingly.

print("\n=== 3. Query Patterns ===")
try:
    patterns = client.get_analytics_patterns(
        database_id=db_id,
        min_count=1,        # Minimum number of occurrences
        limit=10,
    )
    print(json.dumps(patterns, indent=2)[:500])
except JadeVectorDBError as e:
    print(f"Query patterns: {e}")

# ---------------------------------------------------------------------------
# 4. Analytics insights
# ---------------------------------------------------------------------------
# Automated recommendations based on your usage data, such as:
#   - "Consider increasing top_k — 80% of queries use top_k=1"
#   - "Average latency is 50ms — consider adding an HNSW index"

print("\n=== 4. Analytics Insights ===")
try:
    insights = client.get_analytics_insights(database_id=db_id)
    print(json.dumps(insights, indent=2)[:500])
except JadeVectorDBError as e:
    print(f"Insights: {e}")

# ---------------------------------------------------------------------------
# 5. Trending queries
# ---------------------------------------------------------------------------
# Queries that are growing in popularity over time. The min_growth
# parameter sets the minimum growth rate (0.5 = 50% increase).

print("\n=== 5. Trending Queries ===")
try:
    trending = client.get_analytics_trending(
        database_id=db_id,
        min_growth=0.1,     # 10% growth threshold
        limit=10,
    )
    print(json.dumps(trending, indent=2)[:500])
except JadeVectorDBError as e:
    print(f"Trending: {e}")

# ---------------------------------------------------------------------------
# 6. Submit search quality feedback
# ---------------------------------------------------------------------------
# Capture user feedback on search results. This data feeds back into
# analytics insights and can be used to fine-tune search relevance.

print("\n=== 6. Submit Feedback ===")
try:
    feedback = client.submit_analytics_feedback(
        database_id=db_id,
        query_id="example-query-001",    # ID of the search query
        user_id="user-123",              # Who gave the feedback
        rating=4,                         # 1-5 star rating
        feedback_text="Results were relevant but missing some key items",
        clicked_result_id="vec-2",        # Which result the user clicked
        clicked_rank=1,                   # Position of the clicked result
    )
    print(f"Feedback submitted: {json.dumps(feedback, indent=2)}")
except JadeVectorDBError as e:
    print(f"Submit feedback: {e}")

# ---------------------------------------------------------------------------
# 7. Export analytics data
# ---------------------------------------------------------------------------
# Download all analytics data for a database in JSON or CSV format.
# Useful for importing into BI tools (Tableau, Looker, etc.) or
# running custom analysis in Python/R.

print("\n=== 7. Export Analytics ===")
try:
    exported = client.export_analytics(
        database_id=db_id,
        format="json",               # "json" or "csv"
        # start_time="2025-01-01T00:00:00Z",
        # end_time="2025-12-31T23:59:59Z",
    )
    print(f"Exported data keys: {list(exported.keys()) if isinstance(exported, dict) else type(exported)}")
except JadeVectorDBError as e:
    print(f"Export: {e}")

# --- Cleanup ---
client.delete_database(database_id=db_id)
print(f"\nCleaned up database: {db_id}")
print("Analytics examples complete.")
