# RBAC Administration Guide

**Last Updated**: December 17, 2025  
**Audience**: System Administrators  
**Version**: 1.0

---

## Table of Contents

1. [Overview](#overview)
2. [Initial Setup](#initial-setup)
3. [User Management](#user-management)
4. [Group Management](#group-management)
5. [Permission Management](#permission-management)
6. [Database Access Control](#database-access-control)
7. [Monitoring & Auditing](#monitoring--auditing)
8. [Troubleshooting](#troubleshooting)
9. [Best Practices](#best-practices)

---

## Overview

This guide provides comprehensive instructions for administrators to manage the JadeVectorDB RBAC (Role-Based Access Control) system. The RBAC system provides fine-grained access control for users, groups, and database resources.

### Key Responsibilities

As an administrator, you are responsible for:
- Creating and managing user accounts
- Assigning roles and permissions
- Managing groups and group memberships
- Controlling database access
- Monitoring security events
- Responding to security incidents

---

## Initial Setup

### Default Admin Account

JadeVectorDB comes with a default admin account:

```
Username: admin
Password: admin123
```

**⚠️ CRITICAL**: Change the default password immediately after first login!

### Changing Admin Password

```bash
# Using CLI
./jade-db.sh change-password \
  --username admin \
  --old-password admin123 \
  --new-password "YourSecurePassword123!"

# Using API
curl -X PUT https://api.example.com/v1/auth/password \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "old_password": "admin123",
    "new_password": "YourSecurePassword123!"
  }'
```

### Creating Additional Admins

```bash
# 1. Create user account
./jade-db.sh create-user \
  --username jane_admin \
  --email jane@example.com \
  --password "SecurePassword123!"

# 2. Assign admin role
./jade-db.sh assign-role \
  --user-id <user_id> \
  --role role_admin
```

---

## User Management

### Creating Users

**Web Interface**: Admin Panel → Users → Create User

**CLI**:
```bash
./jade-db.sh create-user \
  --username alice \
  --email alice@example.com \
  --password "SecurePassword123!" \
  --role role_user
```

**API**:
```bash
curl -X POST https://api.example.com/v1/users \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "alice",
    "email": "alice@example.com",
    "password": "SecurePassword123!",
    "roles": ["role_user"]
  }'
```

### Password Requirements

Default password policy:
- Minimum 10 characters
- At least one uppercase letter
- At least one lowercase letter
- At least one number
- At least one special character

**Configuring Password Policy**:
Edit `config/authentication.json`:
```json
{
  "require_strong_passwords": true,
  "min_password_length": 10,
  "password_expiry_days": 90
}
```

### Listing Users

```bash
# List all users
./jade-db.sh list-users

# List with filters
./jade-db.sh list-users --role role_user --active true

# API
curl -X GET "https://api.example.com/v1/users?limit=100&offset=0" \
  -H "Authorization: Bearer $ADMIN_TOKEN"
```

### Deactivating Users

Temporarily disable user access without deleting the account:

```bash
./jade-db.sh deactivate-user --user-id user_abc123

# API
curl -X PUT https://api.example.com/v1/users/user_abc123/status \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"is_active": false}'
```

### Unlocking Locked Accounts

After 5 failed login attempts, accounts are locked for 15 minutes.

**Manual Unlock**:
```bash
./jade-db.sh unlock-user --user-id user_abc123

# Or reset failed attempts count
./jade-db.sh reset-failed-logins --user-id user_abc123
```

### Resetting Passwords (Admin)

```bash
./jade-db.sh reset-password \
  --user-id user_abc123 \
  --new-password "NewSecurePassword123!"
```

**Note**: User will be prompted to change password on next login.

---

## Group Management

### Creating Groups

Groups simplify permission management by allowing you to grant permissions to multiple users at once.

```bash
# Create group
./jade-db.sh create-group \
  --name developers \
  --description "Development team members"

# API
curl -X POST https://api.example.com/v1/groups \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "developers",
    "description": "Development team members"
  }'
```

### Adding Users to Groups

```bash
./jade-db.sh add-to-group \
  --group-id group_xyz789 \
  --user-id user_abc123

# Add multiple users
./jade-db.sh add-to-group \
  --group-id group_xyz789 \
  --user-ids user_abc123,user_def456,user_ghi789
```

### Removing Users from Groups

```bash
./jade-db.sh remove-from-group \
  --group-id group_xyz789 \
  --user-id user_abc123
```

### Assigning Roles to Groups

All members of a group inherit the group's roles:

```bash
./jade-db.sh assign-group-role \
  --group-id group_xyz789 \
  --role role_user
```

### Listing Group Members

```bash
./jade-db.sh list-group-members --group-id group_xyz789

# API
curl -X GET https://api.example.com/v1/groups/group_xyz789/members \
  -H "Authorization: Bearer $ADMIN_TOKEN"
```

---

## Permission Management

### Understanding Permission Levels

**System-Level Permissions** (via Roles):
- `role_admin` - Full system access
- `role_user` - Standard user access
- `role_readonly` - Read-only access

**Database-Level Permissions**:
- `database:admin` - Full control over database
- `database:delete` - Can delete database
- `database:write` - Insert, update, delete vectors
- `database:read` - Query and read vectors

### Assigning Roles to Users

```bash
# Assign single role
./jade-db.sh assign-role \
  --user-id user_abc123 \
  --role role_user

# Assign multiple roles
./jade-db.sh assign-roles \
  --user-id user_abc123 \
  --roles role_user,role_admin
```

### Revoking Roles

```bash
./jade-db.sh revoke-role \
  --user-id user_abc123 \
  --role role_admin
```

### Viewing User Permissions

```bash
./jade-db.sh show-user-permissions --user-id user_abc123
```

Output:
```
User: alice (user_abc123)
Roles: role_user
Direct Permissions:
  - database:read on db_123
  - database:write on db_456
Group Permissions (from group 'developers'):
  - database:read on db_789
```

---

## Database Access Control

### Granting Database Access

**Grant to Individual User**:
```bash
./jade-db.sh grant-permission \
  --database-id db_xyz789 \
  --user-id user_abc123 \
  --permission database:read
```

**Grant to Group** (all members inherit):
```bash
./jade-db.sh grant-permission \
  --database-id db_xyz789 \
  --group-id group_developers \
  --permission database:write
```

**API**:
```bash
curl -X POST https://api.example.com/v1/databases/db_xyz789/permissions \
  -H "Authorization: Bearer $ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "principal_type": "user",
    "principal_id": "user_abc123",
    "permission": "database:read"
  }'
```

### Revoking Database Access

```bash
./jade-db.sh revoke-permission \
  --database-id db_xyz789 \
  --user-id user_abc123 \
  --permission database:read
```

### Listing Database Permissions

```bash
./jade-db.sh list-permissions --database-id db_xyz789

# Output:
# Database: vectors_db (db_xyz789)
# Permissions:
#   user:alice (user_abc123) - database:read
#   group:developers (group_xyz) - database:write
```

### Checking User Access

Verify if a user has specific permission:

```bash
./jade-db.sh check-permission \
  --database-id db_xyz789 \
  --user-id user_abc123 \
  --permission database:read
```

---

## Monitoring & Auditing

### Viewing Audit Logs

All security-relevant actions are logged:

```bash
# Recent activity
./jade-db.sh show-audit-logs --limit 100

# Filter by user
./jade-db.sh show-audit-logs --user-id user_abc123

# Filter by action
./jade-db.sh show-audit-logs --action login --limit 50

# Filter by time range
./jade-db.sh show-audit-logs \
  --start-time 2025-12-01 \
  --end-time 2025-12-17
```

### Logged Events

The system logs:
- ✅ User registration and login attempts
- ✅ Role assignments and revocations
- ✅ Permission grants and revokes
- ✅ API key creation and revocation
- ✅ Database creation and deletion
- ✅ Password changes
- ✅ Configuration changes
- ✅ Failed authentication attempts
- ✅ Account lockouts

### Monitoring Failed Logins

```bash
# Check failed login attempts
./jade-db.sh show-audit-logs --action login_failed --limit 100

# Check specific user
./jade-db.sh show-failed-logins --user-id user_abc123
```

### Exporting Audit Logs

```bash
# Export to CSV
./jade-db.sh export-audit-logs \
  --format csv \
  --output /path/to/audit_logs.csv \
  --start-time 2025-12-01 \
  --end-time 2025-12-17

# Export to JSON
./jade-db.sh export-audit-logs \
  --format json \
  --output /path/to/audit_logs.json
```

### Setting Up Alerts

Configure alerts for security events in `config/alerts.json`:

```json
{
  "alerts": [
    {
      "name": "Failed Login Attempts",
      "condition": "failed_logins > 5 in 5 minutes",
      "action": "email",
      "recipients": ["security@example.com"]
    },
    {
      "name": "Admin Role Assigned",
      "condition": "action = assign_role AND role = role_admin",
      "action": "slack",
      "webhook": "https://hooks.slack.com/..."
    }
  ]
}
```

---

## Troubleshooting

### Common Issues

#### User Cannot Login

**Symptoms**: 401 Unauthorized error

**Diagnosis**:
1. Check if account is active: `./jade-db.sh get-user --user-id user_abc123`
2. Check failed login attempts: `./jade-db.sh show-failed-logins --user-id user_abc123`
3. Verify credentials are correct

**Resolution**:
- If locked: `./jade-db.sh unlock-user --user-id user_abc123`
- If inactive: `./jade-db.sh activate-user --user-id user_abc123`
- Reset password if forgotten

#### User Cannot Access Database

**Symptoms**: 403 Forbidden when accessing database

**Diagnosis**:
```bash
# Check user's permissions
./jade-db.sh show-user-permissions --user-id user_abc123

# Check database permissions
./jade-db.sh list-permissions --database-id db_xyz789

# Test specific permission
./jade-db.sh check-permission \
  --database-id db_xyz789 \
  --user-id user_abc123 \
  --permission database:read
```

**Resolution**:
```bash
./jade-db.sh grant-permission \
  --database-id db_xyz789 \
  --user-id user_abc123 \
  --permission database:read
```

#### Audit Logs Not Recording

**Diagnosis**:
1. Check if audit logging is enabled in config
2. Verify database connectivity
3. Check disk space

**Resolution**:
- Enable in config: `"log_authentication_events": true`
- Restart service: `systemctl restart jadevectordb`

### Performance Issues

If permission checks are slow:

1. **Check indexes**:
```sql
-- Verify indexes exist on permission tables
SELECT name FROM sqlite_master 
WHERE type='index' 
AND tbl_name='database_permissions';
```

2. **Analyze query performance**:
```bash
./jade-db.sh analyze-permissions --user-id user_abc123
```

3. **Optimize group memberships**:
   - Avoid deeply nested groups
   - Limit users to 10-20 groups max

---

## Best Practices

### User Management

1. **Use Groups**: Manage permissions via groups, not individual users
2. **Regular Reviews**: Quarterly access reviews to remove unused accounts
3. **Naming Convention**: Use clear naming (firstname.lastname)
4. **Email Verification**: Enable email verification for new accounts
5. **Password Rotation**: Enforce 90-day password rotation

### Permission Management

1. **Least Privilege**: Grant minimum permissions needed
2. **Time-Limited Access**: Use temporary permissions when possible
3. **Document Permissions**: Maintain documentation of permission decisions
4. **Separation of Duties**: Different users for different tasks
5. **Regular Audits**: Review permissions quarterly

### Security

1. **Monitor Audit Logs**: Review daily for suspicious activity
2. **Enable MFA**: Enable multi-factor authentication for admins
3. **Network Security**: Use firewalls and IP whitelisting
4. **Backup Regularly**: Backup authentication database daily
5. **Incident Response**: Have a security incident response plan

### Automation

1. **Onboarding Scripts**: Automate user creation for new hires
2. **Offboarding Scripts**: Automate account deactivation
3. **Permission Reviews**: Automate quarterly permission reviews
4. **Audit Reports**: Automated weekly audit log summaries

---

## Example Workflows

### New Employee Onboarding

```bash
#!/bin/bash
# onboard_user.sh

USER_EMAIL=$1
USER_NAME=$2

# 1. Create user
USER_ID=$(./jade-db.sh create-user \
  --username "$USER_NAME" \
  --email "$USER_EMAIL" \
  --send-welcome-email \
  | grep -oP 'user_id: \K[^ ]+')

# 2. Add to appropriate group
./jade-db.sh add-to-group \
  --group-id group_developers \
  --user-id "$USER_ID"

# 3. Grant database access
./jade-db.sh grant-permission \
  --database-id db_dev \
  --user-id "$USER_ID" \
  --permission database:write

# 4. Log action
echo "Onboarded user: $USER_NAME ($USER_ID) at $(date)"
```

### Employee Offboarding

```bash
#!/bin/bash
# offboard_user.sh

USER_ID=$1

# 1. Deactivate account
./jade-db.sh deactivate-user --user-id "$USER_ID"

# 2. Revoke all API keys
./jade-db.sh revoke-all-api-keys --user-id "$USER_ID"

# 3. Remove from all groups
./jade-db.sh remove-from-all-groups --user-id "$USER_ID"

# 4. Archive user data
./jade-db.sh archive-user-data --user-id "$USER_ID" \
  --output "/backups/users/$USER_ID-$(date +%Y%m%d).tar.gz"

# 5. Log action
echo "Offboarded user: $USER_ID at $(date)"
```

### Quarterly Access Review

```bash
#!/bin/bash
# quarterly_review.sh

# 1. List all users with admin role
./jade-db.sh list-users --role role_admin > admin_users.txt

# 2. List inactive users (no login in 90 days)
./jade-db.sh list-users --inactive-days 90 > inactive_users.txt

# 3. Generate permission report
./jade-db.sh generate-permission-report \
  --output "permission_report_$(date +%Y%m%d).pdf"

# 4. Email report to security team
mail -s "Quarterly Access Review" \
  security@example.com < permission_report.pdf
```

---

## Emergency Procedures

### Compromised Account

If you suspect an account is compromised:

```bash
# 1. Immediately deactivate
./jade-db.sh deactivate-user --user-id user_abc123

# 2. Revoke all API keys
./jade-db.sh revoke-all-api-keys --user-id user_abc123

# 3. End all sessions
./jade-db.sh end-all-sessions --user-id user_abc123

# 4. Review recent activity
./jade-db.sh show-audit-logs --user-id user_abc123 --limit 1000

# 5. Reset password
./jade-db.sh reset-password --user-id user_abc123

# 6. Notify user and security team
```

### Mass Account Lockout

If authentication system is under attack:

```bash
# 1. Enable temporary IP-based rate limiting
./jade-db.sh enable-rate-limit --max-attempts 3 --window 1h

# 2. Review failed login patterns
./jade-db.sh show-audit-logs --action login_failed \
  --start-time "1 hour ago" \
  | awk '{print $NF}' | sort | uniq -c | sort -nr

# 3. Block suspicious IPs
./jade-db.sh block-ip --ip 192.168.1.100 --duration 24h

# 4. Notify security team
```

---

## Configuration Files

### authentication.json

```json
{
  "enabled": true,
  "token_expiry_seconds": 3600,
  "session_expiry_seconds": 86400,
  "max_failed_attempts": 5,
  "account_lockout_duration_seconds": 900,
  "require_strong_passwords": true,
  "min_password_length": 10,
  "enable_two_factor": false,
  "enable_api_keys": true,
  "password_hash_algorithm": "bcrypt",
  "bcrypt_work_factor": 12,
  "log_authentication_events": true
}
```

### rbac.json

```json
{
  "default_role": "role_user",
  "allow_self_registration": true,
  "require_email_verification": true,
  "password_expiry_days": 90,
  "session_inactivity_timeout_seconds": 1800,
  "enable_permission_caching": true,
  "cache_ttl_seconds": 300
}
```

---

## Support & Resources

- **Documentation**: https://docs.jadevectordb.com/rbac
- **Support Email**: support@jadevectordb.com
- **Security Issues**: security@jadevectordb.com
- **Community Forum**: https://community.jadevectordb.com

---

**Last Updated**: December 17, 2025  
**Version**: 1.0  
**Next Review**: March 17, 2026
