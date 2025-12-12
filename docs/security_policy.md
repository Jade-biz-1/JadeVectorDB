# JadeVectorDB Security Policy

## Overview

This document outlines the security policies, configurations, and best practices for JadeVectorDB authentication and data protection.

## Password Policy

### Password Requirements

| Requirement | Default Value | Configuration |
|-------------|---------------|---------------|
| Minimum Length | 8 characters | `min_password_length` |
| Strong Password Required | Yes | `require_strong_passwords` |
| Hash Algorithm | bcrypt | `password_hash_algorithm` |
| BCrypt Work Factor | 12 | `bcrypt_work_factor` |

### Strong Password Requirements

When `require_strong_passwords` is enabled (default), passwords must contain:

- Minimum 10 characters (production recommendation)
- At least one uppercase letter (A-Z)
- At least one lowercase letter (a-z)
- At least one digit (0-9)
- At least one special character (!@#$%^&amp;*)

### Password Storage

Passwords are never stored in plain text. JadeVectorDB uses:

- **bcrypt** (default): Industry-standard adaptive hashing with configurable work factor
- **argon2**: Memory-hard algorithm for enhanced protection (optional)
- **scrypt**: Alternative memory-hard algorithm (optional)

## Token and Session Management

### Token Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| Token Expiry | 3600 seconds (1 hour) | `token_expiry_seconds` |
| Session Expiry | 86400 seconds (24 hours) | `session_expiry_seconds` |

### Token Lifecycle

1. **Generation**: Tokens are generated upon successful authentication
2. **Validation**: Each request validates token signature and expiry
3. **Refresh**: Tokens can be refreshed before expiration
4. **Revocation**: Logout invalidates the current token

### Session Management

- Sessions track user activity and can span multiple tokens
- Sessions expire after 24 hours of inactivity (configurable)
- Multiple concurrent sessions per user are supported
- Session metadata includes: user agent, IP address, created timestamp

## Account Protection

### Brute Force Protection

| Setting | Default | Description |
|---------|---------|-------------|
| Max Failed Attempts | 5 | `max_failed_attempts` |
| Lockout Duration | 900 seconds (15 min) | `account_lockout_duration_seconds` |

After 5 consecutive failed login attempts:
- Account is temporarily locked for 15 minutes
- User must wait or contact admin for unlock
- Event is logged for security audit

### Account Status

Users can be in the following states:
- **active**: Normal access
- **inactive**: Access denied, set by admin
- **locked**: Temporary lockout due to failed attempts

## API Key Security

### API Key Management

| Feature | Status |
|---------|--------|
| API Keys Enabled | Yes (default) |
| Multiple Keys per User | Supported |
| Key Revocation | Supported |
| Key Expiration | Configurable (optional) |

### API Key Best Practices

1. **Generate unique keys** for each application/integration
2. **Use descriptive names** to identify key purpose
3. **Revoke unused keys** promptly
4. **Rotate keys periodically** (recommended: every 90 days)
5. **Never commit keys** to version control

### API Key Storage

- API keys are hashed before storage
- Only the key prefix is stored for identification
- Full key is only shown once at generation time

## Audit Logging

### Events Logged

All security-relevant events are logged to the security audit log:

| Event Type | Description |
|------------|-------------|
| USER_LOGIN | Successful authentication |
| USER_LOGIN_FAILED | Failed authentication attempt |
| USER_LOGOUT | User session ended |
| USER_REGISTERED | New user created |
| USER_UPDATED | User profile modified |
| USER_DELETED | User account removed |
| PASSWORD_CHANGED | Password update |
| PASSWORD_RESET_REQUESTED | Reset token generated |
| API_KEY_CREATED | New API key generated |
| API_KEY_REVOKED | API key invalidated |
| ACCOUNT_LOCKED | Too many failed attempts |
| ACCOUNT_UNLOCKED | Admin unlocked account |

### Audit Retention

Audit logs should be retained based on compliance requirements:

| Environment | Recommended Retention |
|-------------|----------------------|
| Development | 7 days |
| Staging | 30 days |
| Production | 90+ days (or per compliance) |

## Data Encryption

### Encryption at Rest

JadeVectorDB uses AES-256-GCM for data encryption:

- **Algorithm**: AES-256-GCM (authenticated encryption)
- **Key Derivation**: PBKDF2 with SHA-256
- **IV Generation**: Random IV per encryption
- **Authentication**: GCM provides integrity verification

### Encryption Configuration

- Encryption can be enabled per-database
- Master keys should be stored securely (HSM recommended for production)
- Key rotation is supported via re-encryption

## Default Users (Development Only)

Default users are only created when `JADE_ENV` is set to `development`, `test`, or `local`:

| Username | Password | Roles | Purpose |
|----------|----------|-------|---------|
| admin | Admin@123456 | admin, developer, user | Full administrative access |
| dev | Developer@123 | developer, user | Development testing |
| test | Tester@123456 | tester, user | Automated testing |

**⚠️ WARNING**: These users are NOT created in production. You must create admin users manually with strong passwords.

## Production Recommendations

### Pre-Deployment Checklist

- [ ] Change all default configurations
- [ ] Set `JADE_ENV=production`
- [ ] Enable HTTPS/TLS
- [ ] Configure proper token expiry times
- [ ] Set up audit log retention
- [ ] Enable strong password requirements
- [ ] Configure rate limiting
- [ ] Set up monitoring and alerting

### Network Security

- Use TLS 1.2+ for all connections
- Enable CORS with strict origins
- Implement rate limiting
- Use network segmentation
- Enable firewall rules

### Key Rotation Schedule

| Key Type | Rotation Frequency |
|----------|-------------------|
| JWT Signing Keys | Every 90 days |
| API Keys | Every 90 days (recommended) |
| Encryption Keys | Annually (or on compromise) |
| TLS Certificates | Before expiration |

## Vulnerability Reporting

To report security vulnerabilities, please contact the security team directly. Do not create public issues for security matters.

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-12-12 | Initial security policy documentation |
