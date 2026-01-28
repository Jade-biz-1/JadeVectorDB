# JadeVectorDB Analytics - Privacy and Data Retention Policy

**Version**: 1.0
**Effective Date**: January 28, 2026
**Last Updated**: January 28, 2026

---

## Table of Contents

1. [Overview](#overview)
2. [Data Collection](#data-collection)
3. [Data Storage](#data-storage)
4. [Data Retention](#data-retention)
5. [Data Access](#data-access)
6. [Data Protection](#data-protection)
7. [User Rights](#user-rights)
8. [Compliance](#compliance)
9. [Configuration Options](#configuration-options)
10. [Contact Information](#contact-information)

---

## Overview

This document describes how JadeVectorDB Analytics collects, stores, and manages query data. It outlines data retention policies, privacy considerations, and compliance measures.

### Scope

This policy applies to:
- Query analytics data
- User interaction logs
- Performance metrics
- System diagnostics

This policy does NOT apply to:
- Vector embeddings (see main privacy policy)
- User authentication data (see authentication policy)
- Business data stored in JadeVectorDB

---

## Data Collection

### What We Collect

Analytics automatically collects the following data for each query:

#### 1. Query Metadata
- **Query ID**: Unique identifier (auto-generated)
- **Database ID**: Target database identifier
- **Query Type**: Vector, Hybrid, BM25, or Reranked
- **Timestamp**: Unix timestamp in milliseconds

#### 2. Query Content
- **Query Text**: The actual search query (if applicable)
  - Hybrid queries: Full text
  - BM25 queries: Full text
  - Vector-only queries: May be empty
- **Query Vector**: NOT stored (to minimize storage)

#### 3. Performance Metrics
- **Retrieval Time**: Time to retrieve results (ms)
- **Total Time**: End-to-end query time (ms)
- **Result Count**: Number of results returned
- **Average Similarity Score**: Mean similarity of results

#### 4. User Context (Optional)
- **User ID**: If provided by application
- **Session ID**: If provided by application
- **Client IP**: NOT logged by default (can be enabled)

#### 5. Additional Data
- **Hybrid Search Specific**:
  - Alpha value (fusion weight)
  - Fusion method (RRF, LINEAR)

- **Re-ranking Specific**:
  - Re-ranking time
  - Re-ranking model used

- **Error Information**:
  - Error flag (true/false)
  - Error message (if applicable)

### How We Collect

Data collection is:
- **Automatic**: No manual intervention required
- **Asynchronous**: Non-blocking (< 1ms overhead)
- **Reliable**: Batched writes with retry logic

### Data NOT Collected

We do NOT collect:
- Vector embeddings (too large, privacy concern)
- Full result sets (only counts and scores)
- Personal Identifying Information (PII) - unless explicitly provided by application
- Request headers or cookies
- Client IP addresses (unless explicitly enabled)

---

## Data Storage

### Storage Location

Analytics data is stored in:
- **SQLite Database**: `/var/lib/jadevectordb/analytics.db` (default)
  - Or path specified in configuration: `JADEVECTORDB_ANALYTICS_DB`

### Database Schema

Data is stored in the following tables:

#### 1. `query_log` Table
Stores individual query records.

**Columns**:
- query_id (TEXT PRIMARY KEY)
- database_id (TEXT)
- query_text (TEXT) - May contain sensitive data
- query_type (TEXT)
- retrieval_time_ms (INTEGER)
- total_time_ms (INTEGER)
- num_results (INTEGER)
- avg_similarity_score (REAL)
- user_id (TEXT) - Optional
- session_id (TEXT) - Optional
- timestamp (INTEGER)
- has_error (INTEGER)
- error_message (TEXT)
- hybrid_alpha (REAL)
- fusion_method (TEXT)
- reranking_time_ms (INTEGER)
- reranking_model (TEXT)

#### 2. `query_stats` Table
Stores aggregated statistics.

**Columns**:
- time_bucket (INTEGER)
- total_queries (INTEGER)
- successful_queries (INTEGER)
- failed_queries (INTEGER)
- avg_latency_ms (REAL)
- p50/p95/p99_latency_ms (REAL)
- unique_users (INTEGER)
- unique_sessions (INTEGER)

#### 3. `search_patterns` Table
Stores common query patterns.

**Columns**:
- normalized_text (TEXT) - Anonymized
- count (INTEGER)
- avg_latency_ms (REAL)
- first_seen (INTEGER)
- last_seen (INTEGER)

#### 4. `user_feedback` Table
Stores user feedback (if enabled).

**Columns**:
- query_id (TEXT)
- user_id (TEXT)
- rating (INTEGER)
- feedback_text (TEXT) - May contain sensitive data
- clicked_result_id (TEXT)
- clicked_result_rank (INTEGER)

### Storage Size

Typical storage requirements:

| Query Volume | Daily | Monthly | Yearly |
|--------------|-------|---------|--------|
| 1,000 queries/day | ~500 KB | ~15 MB | ~180 MB |
| 10,000 queries/day | ~5 MB | ~150 MB | ~1.8 GB |
| 100,000 queries/day | ~50 MB | ~1.5 GB | ~18 GB |

*Estimates assume average query text length of 50 characters*

---

## Data Retention

### Default Retention Periods

| Data Type | Retention Period | Reason |
|-----------|------------------|--------|
| **Query Logs** | 30 days | Balance detail vs. storage |
| **Aggregated Statistics** | 1 year | Historical trend analysis |
| **Search Patterns** | 1 year | Long-term pattern tracking |
| **User Feedback** | Indefinite | Quality improvement |
| **Error Logs** | 90 days | Troubleshooting |

### Configurable Retention

Retention periods can be customized in `jadevectordb.conf`:

```ini
[analytics]
# Query log retention (days)
query_log_retention_days=30

# Statistics retention (days)
stats_retention_days=365

# Pattern retention (days)
pattern_retention_days=365

# Feedback retention (days, -1 = indefinite)
feedback_retention_days=-1
```

### Data Deletion

Data is automatically deleted via:
- **Daily Cleanup Job**: Runs at 2 AM (configurable)
- **Manual Cleanup**: Via API or CLI

**Delete Command**:
```bash
jade-db analytics cleanup --database=db123 --older-than=30d
```

**Delete via API**:
```bash
curl -X POST "http://localhost:8080/v1/admin/analytics/cleanup" \
  -H "Authorization: Bearer TOKEN" \
  -d '{"database_id": "db123", "retention_days": 30}'
```

### Immediate Deletion

For compliance or privacy reasons, you can delete data immediately:

**Delete All Analytics for Database**:
```bash
jade-db analytics delete --database=db123 --confirm
```

**Delete Specific Queries**:
```bash
jade-db analytics delete --query-ids=q1,q2,q3 --confirm
```

**Delete User Data**:
```bash
jade-db analytics delete --user-id=user123 --confirm
```

---

## Data Access

### Who Has Access

Access to analytics data is controlled by role-based access control (RBAC):

#### 1. Administrator Role
- Full access to all analytics
- Can export data
- Can delete data
- Can modify retention policies

#### 2. Analyst Role
- Read-only access to aggregated statistics
- Can view dashboards
- Can export anonymized data
- Cannot access raw query logs

#### 3. Developer Role
- Read-only access to own database analytics
- Can view query patterns
- Limited export capabilities
- Cannot access other users' data

#### 4. User Role
- No access to analytics
- Can only see own queries (if user_id provided)

### Access Logging

All access to analytics data is logged:
- Who accessed the data
- When it was accessed
- What data was accessed
- Export actions

Audit logs retained for 1 year.

### API Access

Analytics API requires authentication:
- Bearer token (JWT)
- API key with `analytics:read` permission

**Example**:
```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
  "http://localhost:8080/v1/databases/db123/analytics/stats"
```

---

## Data Protection

### Encryption

#### At Rest
- **SQLite Database**: File-system level encryption recommended
- **Environment**: Deploy in encrypted volumes
- **Backups**: Encrypt backup files

**Enable Encryption** (example with LUKS):
```bash
# Encrypt analytics data directory
cryptsetup luksFormat /dev/sdb1
cryptsetup open /dev/sdb1 analytics_data
mkfs.ext4 /dev/mapper/analytics_data
mount /dev/mapper/analytics_data /var/lib/jadevectordb
```

#### In Transit
- **HTTPS/TLS**: All API communication encrypted
- **Minimum TLS Version**: 1.2
- **Recommended**: TLS 1.3

### Access Controls

- **Authentication**: Required for all analytics access
- **Authorization**: RBAC enforced
- **Rate Limiting**: Prevents abuse
- **IP Whitelisting**: Optional restriction

### Data Minimization

Best practices to minimize sensitive data:

1. **Anonymize User IDs**:
   ```python
   import hashlib
   user_id_hash = hashlib.sha256(real_user_id.encode()).hexdigest()[:16]
   ```

2. **Sanitize Query Text**:
   ```python
   # Remove PII before logging
   query_text = remove_pii(original_query)
   ```

3. **Disable Sensitive Fields**:
   ```ini
   [analytics]
   log_user_ids=false
   log_session_ids=false
   log_query_text=false  # Only log for debugging
   ```

4. **Use Hashing for Patterns**:
   ```python
   # Store hash instead of actual query
   pattern_hash = hash(normalized_query)
   ```

---

## User Rights

### Right to Access

Users can request access to their analytics data:

**Request Process**:
1. Submit request via email or API
2. Verify identity
3. Receive data export within 30 days

**Export Format**: JSON or CSV

**Data Included**:
- All queries with user's user_id
- Timestamps
- Query types
- Performance metrics

### Right to Deletion (Right to be Forgotten)

Users can request deletion of their data:

**Request Process**:
1. Submit deletion request
2. Verify identity
3. Data deleted within 30 days

**Deletion Scope**:
- Query logs with user's user_id
- User feedback records
- Anonymized aggregated data retained

**Confirmation**: Email confirmation sent upon completion

### Right to Rectification

Users can request correction of inaccurate data:

**Process**:
1. Identify incorrect data
2. Submit correction request
3. Manual review and update

### Right to Restriction

Users can request restriction of processing:

**Options**:
- Pause analytics logging for specific user
- Exclude user from aggregated statistics
- Disable feedback collection

**Configuration**:
```ini
[analytics]
excluded_user_ids=user123,user456
```

---

## Compliance

### GDPR Compliance

JadeVectorDB Analytics supports GDPR requirements:

| Requirement | Implementation |
|-------------|----------------|
| Lawful Basis | Legitimate interest (service improvement) |
| Consent | Optional: Explicit consent can be required |
| Data Minimization | Configurable fields, minimal by default |
| Purpose Limitation | Analytics only, not used for other purposes |
| Storage Limitation | Automatic deletion after retention period |
| Accuracy | User can request corrections |
| Integrity & Confidentiality | Encryption, access controls |
| Accountability | Audit logs, privacy policy |

**Data Processing Agreement**: Available upon request

### CCPA Compliance

California Consumer Privacy Act (CCPA) compliance:

| Right | Implementation |
|-------|----------------|
| Right to Know | Data export functionality |
| Right to Delete | Deletion API and commands |
| Right to Opt-Out | Disable analytics per user |
| Non-Discrimination | No service impact from opt-out |

### HIPAA Considerations

For healthcare applications handling PHI:

⚠️ **WARNING**: Query text may contain PHI (symptoms, conditions, etc.)

**Recommendations**:
1. **Do NOT log query text** if it may contain PHI
2. **Enable encryption at rest**
3. **Sign Business Associate Agreement (BAA)**
4. **Implement audit logging**
5. **Conduct regular risk assessments**

**Configuration**:
```ini
[analytics]
log_query_text=false  # Disable for HIPAA
encryption_enabled=true
audit_logging=true
```

### SOC 2 Compliance

For SOC 2 compliance:

| Control | Implementation |
|---------|----------------|
| Access Control | RBAC enforced |
| Encryption | TLS + optional at-rest |
| Monitoring | Comprehensive logging |
| Incident Response | Automated alerts |
| Change Management | Version control |

---

## Configuration Options

### Privacy-Focused Configuration

Minimal data collection:

```ini
[analytics]
# Disable user tracking
log_user_ids=false
log_session_ids=false
log_client_ips=false

# Disable query text (only log patterns)
log_query_text=false

# Short retention
query_log_retention_days=7
stats_retention_days=30

# Disable feedback
feedback_enabled=false
```

### Analytics-Focused Configuration

Maximum visibility:

```ini
[analytics]
# Enable all tracking
log_user_ids=true
log_session_ids=true
log_query_text=true

# Long retention
query_log_retention_days=90
stats_retention_days=730  # 2 years

# Enable feedback
feedback_enabled=true
```

### Compliance Mode

GDPR/CCPA compliant:

```ini
[analytics]
# Anonymized tracking
log_user_ids=true  # But hash them
hash_user_ids=true
log_session_ids=false
log_query_text=false  # Or sanitize

# Standard retention
query_log_retention_days=30
stats_retention_days=365

# Enable user rights
enable_data_export=true
enable_data_deletion=true
enable_opt_out=true
```

---

## Contact Information

### Data Privacy Officer

**Email**: privacy@jadevectordb.com
**Response Time**: Within 5 business days

### Data Subject Requests

For access, deletion, or correction requests:

**Email**: dsar@jadevectordb.com
**Portal**: https://jadevectordb.com/privacy/requests
**Response Time**: Within 30 days

### Security Incidents

Report security incidents:

**Email**: security@jadevectordb.com
**Emergency**: +1 (555) 123-4567
**Response Time**: Within 24 hours

### General Inquiries

**Email**: support@jadevectordb.com
**Documentation**: https://jadevectordb.com/docs/analytics
**Community**: https://community.jadevectordb.com

---

## Policy Updates

This policy may be updated periodically. Changes will be:

1. Documented in changelog below
2. Announced via email to administrators
3. Posted on website 30 days before effective date

### Changelog

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-28 | Initial release |

---

## Acknowledgment

By using JadeVectorDB Analytics, you acknowledge that you have read and understood this Privacy and Data Retention Policy.

For questions or concerns, please contact our Data Privacy Officer.

---

**End of Privacy and Data Retention Policy**

*Last Reviewed: January 28, 2026*
