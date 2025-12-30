# JadeVectorDB Production Deployment Guide

This guide covers production deployment of JadeVectorDB with secure authentication, password management, and operational best practices.

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Environment Configuration](#environment-configuration)
4. [Production Admin Setup](#production-admin-setup)
5. [Password Security Policies](#password-security-policies)
6. [First-Time Deployment](#first-time-deployment)
7. [User Management](#user-management)
8. [Security Best Practices](#security-best-practices)
9. [Monitoring and Maintenance](#monitoring-and-maintenance)
10. [Troubleshooting](#troubleshooting)

---

## Overview

JadeVectorDB supports two runtime modes:

- **Development Mode**: Automatically creates default test users (admin/admin123, dev/dev123, test/test123)
- **Production Mode**: Requires explicit admin user creation via environment variables for security

This guide focuses on **Production Mode** deployment.

---

## Prerequisites

Before deploying JadeVectorDB in production:

- [ ] Linux server (Ubuntu 20.04+ recommended)
- [ ] C++20 compatible compiler (GCC 10+ or Clang 12+)
- [ ] CMake 3.20 or higher
- [ ] SQLite 3.35+
- [ ] Node.js 18+ and npm (for frontend)
- [ ] Sufficient disk space for database and vector storage
- [ ] Network configuration for port 8080 (backend) and 3000 (frontend)
- [ ] SSL/TLS certificates (recommended for production)

---

## Environment Configuration

### Runtime Environment Detection

JadeVectorDB determines production vs. development mode using:

```cpp
// Production mode is detected when:
// 1. JADEVECTORDB_ENV != "development"
// 2. RUNTIME_ENVIRONMENT != "development"
// 3. No dev-specific environment variables are set
```

### Required Environment Variables

#### Backend (Production Mode)

```bash
# CRITICAL: Admin password for production deployment
export JADEVECTORDB_ADMIN_PASSWORD="YourSecureAdminPassword123!"

# Optional: Explicit environment setting
export JADEVECTORDB_ENV="production"
# OR
export RUNTIME_ENVIRONMENT="production"

# Database configuration
export JADEVECTORDB_DB_PATH="/var/lib/jadevectordb/data"
export JADEVECTORDB_PORT="8080"

# JWT configuration
export JADEVECTORDB_JWT_SECRET="your-secure-jwt-secret-key-min-32-chars"
export JADEVECTORDB_JWT_EXPIRY="3600"  # 1 hour in seconds

# Optional: Enable audit logging
export JADEVECTORDB_AUDIT_LOG_PATH="/var/log/jadevectordb/audit.log"
```

#### Frontend

```bash
# API endpoint
export NEXT_PUBLIC_API_URL="http://localhost:8080"
# OR for production with domain
export NEXT_PUBLIC_API_URL="https://api.yourdomain.com"

# Optional: Node environment
export NODE_ENV="production"
```

---

## Production Admin Setup

### How Production Admin Creation Works

In production mode, JadeVectorDB will **NOT** create any default users automatically. Instead:

1. **Check for `JADEVECTORDB_ADMIN_PASSWORD` environment variable**
   - If set: Creates admin user with username "admin" and the provided password
   - If not set: Logs a warning and starts with no users (requires manual setup)

2. **Force Password Change**
   - Admin user created from environment variable has `must_change_password=true`
   - Admin MUST change password on first login for security

### Setting Admin Password

**Option 1: Environment Variable (Recommended)**

```bash
# Set strong admin password
export JADEVECTORDB_ADMIN_PASSWORD='Adm1n$ecureP@ssw0rd2025!'

# Start the backend
./jadevectordb_server
```

**Option 2: Command-line (One-time deployment)**

```bash
JADEVECTORDB_ADMIN_PASSWORD='Adm1n$ecureP@ssw0rd2025!' ./jadevectordb_server
```

**Option 3: Systemd Service File**

```ini
[Unit]
Description=JadeVectorDB Server
After=network.target

[Service]
Type=simple
User=jadevectordb
WorkingDirectory=/opt/jadevectordb
Environment="JADEVECTORDB_ADMIN_PASSWORD=Adm1n$ecureP@ssw0rd2025!"
Environment="JADEVECTORDB_ENV=production"
Environment="JADEVECTORDB_JWT_SECRET=your-jwt-secret-32-chars-minimum"
ExecStart=/opt/jadevectordb/bin/jadevectordb_server
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

**IMPORTANT SECURITY NOTES:**
- Use a strong, unique password (minimum 10 characters)
- Store the password securely (e.g., AWS Secrets Manager, HashiCorp Vault)
- Change the initial password immediately after first login
- Never commit passwords to version control

---

## Password Security Policies

### Password Requirements

All passwords in JadeVectorDB must meet these requirements:

- **Minimum Length**: 10 characters
- **Uppercase Letters**: At least one (A-Z)
- **Lowercase Letters**: At least one (a-z)
- **Digits**: At least one (0-9)
- **Special Characters**: At least one (e.g., !@#$%^&*)

**Example Valid Passwords:**
- `Adm1n$ecureP@ss`
- `MyP@ssw0rd123!`
- `SecureDB#2025Pass`

**Example Invalid Passwords:**
- `admin123` (no uppercase, no special char)
- `Admin@Pass` (no digit, too short)
- `password` (no complexity)

### Password Storage

- Passwords are hashed using **bcrypt** with a cost factor of 12
- Each password has a unique **salt**
- Password hashes are stored in SQLite database
- Original passwords are **never** stored

### Forced Password Changes

The system enforces password changes in these scenarios:

1. **Admin Bootstrapped from Environment**: `must_change_password=true`
2. **Admin Reset User Password**: `must_change_password=true`
3. **Security Policy Violation**: Future feature

When `must_change_password=true`:
- User can log in successfully
- Login response includes `"must_change_password": true`
- Frontend redirects to `/change-password` page
- User cannot access other features until password is changed

---

## First-Time Deployment

### Step-by-Step Production Setup

#### 1. Build the Backend

```bash
cd /path/to/JadeVectorDB/backend
mkdir -p build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

#### 2. Set Production Environment Variables

Create `/etc/jadevectordb/config.env`:

```bash
# Production environment
JADEVECTORDB_ENV=production
JADEVECTORDB_ADMIN_PASSWORD=ChangeThisSecurePassword123!

# Database
JADEVECTORDB_DB_PATH=/var/lib/jadevectordb/data
JADEVECTORDB_PORT=8080

# JWT
JADEVECTORDB_JWT_SECRET=your-very-secure-jwt-secret-key-min-32-characters
JADEVECTORDB_JWT_EXPIRY=3600
```

**Secure the config file:**
```bash
sudo chown root:jadevectordb /etc/jadevectordb/config.env
sudo chmod 640 /etc/jadevectordb/config.env
```

#### 3. Start the Backend

```bash
# Load environment variables
source /etc/jadevectordb/config.env

# Start server
cd /path/to/JadeVectorDB/backend/build
./jadevectordb_server
```

**Expected Output:**
```
[INFO] Environment: production
[INFO] Production mode: JADEVECTORDB_ADMIN_PASSWORD found
[INFO] Created admin user with must_change_password=true
[INFO] Server starting on port 8080...
```

#### 4. Build and Start the Frontend

```bash
cd /path/to/JadeVectorDB/frontend

# Install dependencies
npm install

# Build for production
npm run build

# Start production server
npm start
```

Frontend will be available at `http://localhost:3000`

#### 5. First Login and Password Change

1. Navigate to `http://localhost:3000/login`
2. Login with:
   - Username: `admin`
   - Password: `<value of JADEVECTORDB_ADMIN_PASSWORD>`
3. You will be automatically redirected to `/change-password`
4. Enter:
   - Current Password: `<initial password>`
   - New Password: `<your new secure password>`
   - Confirm New Password: `<same as new password>`
5. Click "Change Password"
6. You will be redirected to the dashboard

**IMPORTANT:** Delete or rotate the `JADEVECTORDB_ADMIN_PASSWORD` environment variable after first login.

---

## User Management

### Creating Additional Users

#### Via Frontend (Admin Interface)

1. Log in as admin
2. Navigate to **User Management** page
3. Fill in the "Add New User" form:
   - Username
   - Email
   - Password (meeting requirements)
   - Roles (comma-separated: admin, developer, user)
4. Click "Add User"

#### Via REST API

```bash
# Login to get JWT token
curl -X POST http://localhost:8080/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "username": "admin",
    "password": "YourAdminPassword123!"
  }'

# Response includes token
# {"token": "eyJ...", "user_id": "...", "username": "admin"}

# Create new user
curl -X POST http://localhost:8080/v1/admin/users \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <your-jwt-token>" \
  -d '{
    "username": "newuser",
    "email": "newuser@example.com",
    "password": "SecureP@ssw0rd123",
    "roles": ["developer", "user"]
  }'
```

### Admin Password Reset

Admins can reset any user's password. When reset, the user will be forced to change their password on next login.

#### Via Frontend

1. Navigate to **User Management** page
2. Find the user in the list
3. Click "Reset Password" button
4. Enter new temporary password
5. Click "Reset Password"
6. User will receive `must_change_password=true` on next login

#### Via REST API

```bash
curl -X POST http://localhost:8080/v1/admin/users/<user_id>/reset-password \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <admin-jwt-token>" \
  -d '{
    "new_password": "TemporaryP@ss123!"
  }'
```

### Self-Service Password Change

Users can change their own password at any time.

#### Via Frontend

1. Log in
2. Navigate to `/change-password` or click profile menu
3. Enter:
   - Current Password
   - New Password
   - Confirm New Password
4. Click "Change Password"

#### Via REST API

```bash
curl -X PUT http://localhost:8080/v1/users/<user_id>/password \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <user-jwt-token>" \
  -d '{
    "old_password": "CurrentP@ss123",
    "new_password": "NewSecureP@ss456!"
  }'
```

---

## Security Best Practices

### Environment Variables

- ✅ **DO**: Use secrets management systems (AWS Secrets Manager, Vault)
- ✅ **DO**: Rotate JWT secrets periodically
- ✅ **DO**: Use different passwords for each environment
- ❌ **DON'T**: Commit `.env` files to version control
- ❌ **DON'T**: Share admin passwords via email/chat

### Password Management

- ✅ **DO**: Enforce password changes for admin-reset passwords
- ✅ **DO**: Use strong, unique passwords
- ✅ **DO**: Implement password rotation policies
- ✅ **DO**: Monitor failed login attempts
- ❌ **DON'T**: Reuse passwords across systems
- ❌ **DON'T**: Share user credentials

### Network Security

- ✅ **DO**: Use HTTPS/TLS in production
- ✅ **DO**: Configure firewalls to restrict access
- ✅ **DO**: Use reverse proxy (nginx, Apache) for SSL termination
- ✅ **DO**: Enable CORS only for trusted domains
- ❌ **DON'T**: Expose backend directly to internet without proxy

### Database Security

- ✅ **DO**: Regular database backups
- ✅ **DO**: Encrypt data at rest
- ✅ **DO**: Set proper file permissions (640 for DB files)
- ✅ **DO**: Monitor database access logs
- ❌ **DON'T**: Run server as root user

### Audit Logging

- ✅ **DO**: Enable audit logging in production
- ✅ **DO**: Monitor authentication events
- ✅ **DO**: Review security audit logs regularly
- ✅ **DO**: Set up alerts for suspicious activity

---

## Monitoring and Maintenance

### Health Checks

```bash
# Basic health check
curl http://localhost:8080/health

# Database health check
curl http://localhost:8080/health/db

# System status
curl http://localhost:8080/status
```

### Audit Logs

View authentication and security events:

```bash
# List recent audit logs
curl http://localhost:8080/v1/security/audit/logs?limit=100 \
  -H "Authorization: Bearer <admin-token>"

# Get audit statistics
curl http://localhost:8080/v1/security/audit/stats \
  -H "Authorization: Bearer <admin-token>"
```

### Database Backups

```bash
#!/bin/bash
# backup-jadevectordb.sh

BACKUP_DIR="/var/backups/jadevectordb"
DB_PATH="/var/lib/jadevectordb/data"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

mkdir -p "$BACKUP_DIR"

# Backup SQLite database
sqlite3 "$DB_PATH/auth.db" ".backup '$BACKUP_DIR/auth_$TIMESTAMP.db'"

# Backup vector data
tar -czf "$BACKUP_DIR/vectors_$TIMESTAMP.tar.gz" "$DB_PATH/vectors/"

# Keep only last 30 days of backups
find "$BACKUP_DIR" -name "*.db" -mtime +30 -delete
find "$BACKUP_DIR" -name "*.tar.gz" -mtime +30 -delete

echo "Backup completed: $TIMESTAMP"
```

Add to crontab for daily backups:
```bash
0 2 * * * /usr/local/bin/backup-jadevectordb.sh >> /var/log/jadevectordb-backup.log 2>&1
```

---

## Troubleshooting

### Issue: No users created on startup

**Symptom:**
```
[INFO] Production mode: JADEVECTORDB_ADMIN_PASSWORD not set
[INFO] No default users created in production mode
```

**Solution:**
```bash
# Set the admin password environment variable
export JADEVECTORDB_ADMIN_PASSWORD='SecureP@ss123!'

# Restart the server
./jadevectordb_server
```

---

### Issue: "Password does not meet strength requirements"

**Symptom:**
```json
{
  "error": "Password does not meet strength requirements: minimum 10 characters, uppercase, lowercase, digit, and special character required"
}
```

**Solution:**
Ensure password has:
- ✅ Minimum 10 characters
- ✅ At least one uppercase (A-Z)
- ✅ At least one lowercase (a-z)
- ✅ At least one digit (0-9)
- ✅ At least one special character (!@#$%^&*)

---

### Issue: Stuck at password change page

**Symptom:**
User logs in but is always redirected to `/change-password`

**Cause:**
User has `must_change_password=true` flag set

**Solution:**
1. Complete the password change process
2. Ensure new password is different from old password
3. Ensure new password meets all requirements

**Admin Fix (if user forgot password):**
```bash
# Reset user password via API
curl -X POST http://localhost:8080/v1/admin/users/<user_id>/reset-password \
  -H "Authorization: Bearer <admin-token>" \
  -d '{"new_password": "NewTemp@ryPass123"}'
```

---

### Issue: JWT token expired

**Symptom:**
```json
{
  "error": "Token expired",
  "code": 401
}
```

**Solution:**
1. Log out and log back in
2. Adjust `JADEVECTORDB_JWT_EXPIRY` for longer sessions (not recommended for production)
3. Implement token refresh mechanism (future feature)

---

### Issue: Cannot connect to backend

**Symptom:**
Frontend shows "Failed to connect to API"

**Checklist:**
- [ ] Backend server is running (`ps aux | grep jadevectordb`)
- [ ] Port 8080 is accessible (`netstat -tuln | grep 8080`)
- [ ] Firewall allows connections
- [ ] `NEXT_PUBLIC_API_URL` is correct in frontend
- [ ] CORS is configured properly

---

## Additional Resources

- **Build Guide**: See `BUILD.md` for compilation instructions
- **API Documentation**: See `backend/docs/API.md` (if available)
- **Authentication Details**: See `AUTHENTICATION_PERSISTENCE_PLAN.md`
- **Testing Guide**: See `MANUAL_TESTING_GUIDE.md`

---

## Support and Contributions

For issues or questions:
- Open an issue on GitHub
- Review `CONTRIBUTING.md` for contribution guidelines
- Check existing documentation in the `docs/` directory

---

**Last Updated**: 2025-12-30
**Version**: 1.0.0
**Author**: JadeVectorDB Development Team
