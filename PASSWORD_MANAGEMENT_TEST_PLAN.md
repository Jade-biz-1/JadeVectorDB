# Password Management Test Plan

This document provides a comprehensive testing plan for the password management features implemented in JadeVectorDB.

## Test Summary

### Features Implemented

1. **Production Admin Bootstrap**
   - Admin user creation via `JADEVECTORDB_ADMIN_PASSWORD` environment variable
   - Forced password change on first login (`must_change_password=true`)

2. **Password Change**
   - Self-service password change (PUT `/v1/users/{id}/password`)
   - Old password verification
   - New password strength validation
   - `must_change_password` flag cleared after successful change

3. **Admin Password Reset**
   - Admin can reset any user's password (POST `/v1/admin/users/{id}/reset-password`)
   - Forces user to change password on next login
   - Security audit logged

4. **Frontend Integration**
   - `/change-password` page for forced and voluntary password changes
   - Admin password reset UI in `/users` page
   - Automatic redirect on `must_change_password=true`
   - Password strength indicators

---

## Test Cases

### 1. Development Mode Tests

#### Test 1.1: Default Users Creation
**Objective**: Verify default users are created in development mode

**Steps**:
```bash
# Ensure JADEVECTORDB_ENV is not set or set to "development"
unset JADEVECTORDB_ADMIN_PASSWORD
unset JADEVECTORDB_ENV

# Start server
cd backend/build
./jadevectordb
```

**Expected Result**:
- Server logs show: `"Proceeding with seeding for environment: development"`
- Three users created: admin, dev, test
- Users have default passwords meeting 10+ character requirement

**Verification**:
```sql
sqlite3 data/system.db "SELECT user_id, username, must_change_password FROM users;"
```

Expected output:
```
admin|admin|0
dev|dev|0
test|test|0
```

---

#### Test 1.2: Login with Default Users
**Objective**: Verify authentication works with default users

**Steps**:
```bash
# Login as admin
curl -X POST http://localhost:8080/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "Admin@123456"}'
```

**Expected Result**:
```json
{
  "token": "eyJ...",
  "user_id": "...",
  "username": "admin",
  "must_change_password": false
}
```

---

### 2. Production Mode Tests

#### Test 2.1: Production Mode Without Admin Password
**Objective**: Verify no users created when `JADEVECTORDB_ADMIN_PASSWORD` is not set

**Steps**:
```bash
# Clear database
rm -f backend/build/data/system.db

# Set production environment
export JADEVECTORDB_ENV="production"
unset JADEVECTORDB_ADMIN_PASSWORD

# Start server
cd backend/build
./jadevectordb
```

**Expected Result**:
- Server logs show: `"Production mode: JADEVECTORDB_ADMIN_PASSWORD not set"`
- No users created
- Server starts successfully

**Verification**:
```sql
sqlite3 data/system.db "SELECT COUNT(*) FROM users;"
```
Expected output: `0`

---

#### Test 2.2: Production Admin Bootstrap
**Objective**: Verify admin user created from environment variable with forced password change

**Steps**:
```bash
# Clear database
rm -f backend/build/data/system.db

# Set production admin password
export JADEVECTORDB_ENV="production"
export JADEVECTORDB_ADMIN_PASSWORD="Adm1nProduction@2025!"

# Start server
cd backend/build
./jadevectordb
```

**Expected Result**:
- Server logs show: `"Production mode: JADEVECTORDB_ADMIN_PASSWORD found"`
- Server logs show: `"Created admin user with must_change_password=true"`
- Admin user created with `must_change_password=true`

**Verification**:
```sql
sqlite3 data/system.db "SELECT username, must_change_password FROM users WHERE username='admin';"
```
Expected output: `admin|1`

---

#### Test 2.3: First Login in Production (Forced Password Change)
**Objective**: Verify admin must change password on first login

**Steps**:
```bash
# Login as admin with bootstrap password
curl -X POST http://localhost:8080/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "Adm1nProduction@2025!"}'
```

**Expected Result**:
```json
{
  "token": "eyJ...",
  "user_id": "...",
  "username": "admin",
  "must_change_password": true,
  "message": "Login successful. You must change your password before continuing."
}
```

**Frontend Verification**:
- User should be redirected to `/change-password`
- Cannot access dashboard until password changed

---

### 3. Password Change Tests

#### Test 3.1: Self-Service Password Change
**Objective**: Verify user can change their own password

**Prerequisites**: User logged in with JWT token

**Steps**:
```bash
# Change password
curl -X PUT http://localhost:8080/v1/users/{user_id}/password \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer {jwt_token}" \
  -d '{
    "old_password": "Adm1nProduction@2025!",
    "new_password": "NewSecureP@ssw0rd2025"
  }'
```

**Expected Result**:
```json
{
  "success": true,
  "message": "Password changed successfully"
}
```

**Verification**:
```sql
sqlite3 data/system.db "SELECT must_change_password FROM users WHERE user_id='{user_id}';"
```
Expected output: `0` (flag cleared)

**Test Re-login**:
```bash
# Should work with new password
curl -X POST http://localhost:8080/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "NewSecureP@ssw0rd2025"}'
```

Expected: Login successful with `"must_change_password": false`

---

#### Test 3.2: Password Change with Incorrect Old Password
**Objective**: Verify old password is validated

**Steps**:
```bash
curl -X PUT http://localhost:8080/v1/users/{user_id}/password \
  -H "Authorization: Bearer {jwt_token}" \
  -d '{
    "old_password": "WrongPassword123!",
    "new_password": "NewSecureP@ssw0rd2025"
  }'
```

**Expected Result**:
```json
{
  "error": "Old password is incorrect"
}
```
HTTP Status: 401 Unauthorized

---

#### Test 3.3: Password Change with Weak New Password
**Objective**: Verify password strength requirements enforced

**Steps**:
```bash
curl -X PUT http://localhost:8080/v1/users/{user_id}/password \
  -H "Authorization: Bearer {jwt_token}" \
  -d '{
    "old_password": "CurrentP@ss123",
    "new_password": "weak"
  }'
```

**Expected Result**:
```json
{
  "error": "Password does not meet strength requirements: minimum 10 characters, uppercase, lowercase, digit, and special character required"
}
```
HTTP Status: 400 Bad Request

---

#### Test 3.4: Password Change with Same Password
**Objective**: Verify user cannot reuse the same password

**Steps**:
```bash
curl -X PUT http://localhost:8080/v1/users/{user_id}/password \
  -H "Authorization: Bearer {jwt_token}" \
  -d '{
    "old_password": "CurrentP@ss123",
    "new_password": "CurrentP@ss123"
  }'
```

**Expected Result**:
```json
{
  "error": "New password must be different from old password"
}
```
HTTP Status: 400 Bad Request

---

### 4. Admin Password Reset Tests

#### Test 4.1: Admin Resets User Password
**Objective**: Verify admin can reset any user's password

**Prerequisites**: Logged in as admin

**Steps**:
```bash
# Admin resets user password
curl -X POST http://localhost:8080/v1/admin/users/{user_id}/reset-password \
  -H "Authorization: Bearer {admin_jwt_token}" \
  -H "Content-Type: application/json" \
  -d '{
    "new_password": "TempResetP@ss123!"
  }'
```

**Expected Result**:
```json
{
  "success": true,
  "message": "Password reset successfully",
  "must_change_password": true
}
```

**Verification**:
```sql
sqlite3 data/system.db "SELECT must_change_password FROM users WHERE user_id='{user_id}';"
```
Expected output: `1` (flag set)

**User Next Login**:
```bash
curl -X POST http://localhost:8080/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "{username}", "password": "TempResetP@ss123!"}'
```

Expected: `"must_change_password": true`

---

#### Test 4.2: Non-Admin Cannot Reset Passwords
**Objective**: Verify authorization check for admin endpoint

**Prerequisites**: Logged in as non-admin user (e.g., "developer" role)

**Steps**:
```bash
curl -X POST http://localhost:8080/v1/admin/users/{user_id}/reset-password \
  -H "Authorization: Bearer {non_admin_token}" \
  -d '{"new_password": "NewP@ss123!"}'
```

**Expected Result**:
```json
{
  "error": "Insufficient permissions"
}
```
HTTP Status: 403 Forbidden

---

### 5. Frontend Tests

#### Test 5.1: Change Password Page
**Objective**: Verify change-password page works correctly

**Steps**:
1. Start frontend: `cd frontend && npm run dev`
2. Navigate to `http://localhost:3000/login`
3. Login with user having `must_change_password=true`
4. Should auto-redirect to `/change-password`

**Verification**:
- Page displays "Password Change Required" alert
- Form has three fields: Current Password, New Password, Confirm New Password
- Password strength indicator shows for new password
- Submit button changes to "Changing Password..." when loading

**Test Password Change**:
- Enter current password
- Enter new strong password (e.g., `MyNewP@ssword123!`)
- Confirm new password
- Click "Change Password"
- Should show success message and redirect to dashboard

---

#### Test 5.2: Admin Password Reset UI
**Objective**: Verify admin can reset passwords via frontend

**Steps**:
1. Login as admin
2. Navigate to `/users` page
3. Find a user in the table
4. Click "Reset Password" button

**Verification**:
- Modal opens with title "Reset Password"
- Shows username being reset
- Password field with strength validation
- Warning: "User will be required to change this password on their next login"

**Test Reset**:
- Enter new password (e.g., `AdminReset@123`)
- Click "Reset Password"
- Success message displays
- User's next login requires password change

---

### 6. Security Tests

#### Test 6.1: Password Hash Storage
**Objective**: Verify passwords are never stored in plaintext

**Steps**:
```sql
sqlite3 data/system.db "SELECT password_hash, salt FROM users LIMIT 1;"
```

**Expected Result**:
- `password_hash` is a bcrypt hash (starts with `$2b$` or similar)
- `salt` is a random string
- Neither field contains the actual password

---

#### Test 6.2: JWT Token Includes must_change_password
**Objective**: Verify JWT payload contains password change requirement

**Steps**:
```bash
# Login with user who must change password
TOKEN=$(curl -s -X POST http://localhost:8080/v1/auth/login \
  -d '{"username":"admin","password":"Adm1nProduction@2025!"}' | jq -r '.token')

# Decode JWT (using online decoder or jwt-cli)
echo $TOKEN | cut -d. -f2 | base64 -d 2>/dev/null || echo "JWT payload"
```

**Expected Result**:
JWT payload should include:
```json
{
  "user_id": "...",
  "username": "admin",
  "must_change_password": true
}
```

---

## Automated Test Scripts

### Development Mode Test Script

```bash
#!/bin/bash
# test-dev-mode.sh

set -e

echo "=== Testing Development Mode ==="

# Clean database
rm -f backend/build/data/system.db

# Start server
unset JADEVECTORDB_ENV
unset JADEVECTORDB_ADMIN_PASSWORD
cd backend/build
./jadevectordb > server.log 2>&1 &
SERVER_PID=$!
sleep 3

# Test health
curl -f http://localhost:8080/health

# Test login
RESPONSE=$(curl -s -X POST http://localhost:8080/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"Admin@123456"}')

echo "Login response: $RESPONSE"

if echo "$RESPONSE" | grep -q "token"; then
  echo "✓ Development mode login successful"
else
  echo "✗ Development mode login failed"
  kill $SERVER_PID
  exit 1
fi

# Cleanup
kill $SERVER_PID
echo "=== Development Mode Tests Passed ==="
```

### Production Mode Test Script

```bash
#!/bin/bash
# test-prod-mode.sh

set -e

echo "=== Testing Production Mode ==="

# Clean database
rm -f backend/build/data/system.db

# Start server with admin password
export JADEVECTORDB_ENV="production"
export JADEVECTORDB_ADMIN_PASSWORD="Adm1nProduction@2025!"
cd backend/build
./jadevectordb > server.log 2>&1 &
SERVER_PID=$!
sleep 3

# Test login
RESPONSE=$(curl -s -X POST http://localhost:8080/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"Adm1nProduction@2025!"}')

echo "Login response: $RESPONSE"

# Check must_change_password flag
if echo "$RESPONSE" | grep -q '"must_change_password":true'; then
  echo "✓ Production admin requires password change"
else
  echo "✗ must_change_password flag not set"
  kill $SERVER_PID
  exit 1
fi

# Extract token and user_id
TOKEN=$(echo "$RESPONSE" | jq -r '.token')
USER_ID=$(echo "$RESPONSE" | jq -r '.user_id')

# Test password change
CHANGE_RESPONSE=$(curl -s -X PUT http://localhost:8080/v1/users/$USER_ID/password \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "old_password": "Adm1nProduction@2025!",
    "new_password": "MyNewAdminP@ss2025!"
  }')

echo "Password change response: $CHANGE_RESPONSE"

if echo "$CHANGE_RESPONSE" | grep -q "success"; then
  echo "✓ Password change successful"
else
  echo "✗ Password change failed"
  kill $SERVER_PID
  exit 1
fi

# Test re-login with new password
RELOGIN=$(curl -s -X POST http://localhost:8080/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"MyNewAdminP@ss2025!"}')

if echo "$RELOGIN" | grep -q '"must_change_password":false'; then
  echo "✓ Re-login successful, must_change_password cleared"
else
  echo "✗ Re-login failed or flag not cleared"
  kill $SERVER_PID
  exit 1
fi

# Cleanup
kill $SERVER_PID
echo "=== Production Mode Tests Passed ==="
```

---

## Test Coverage Summary

| Feature | Backend API | Frontend UI | Test Coverage |
|---------|------------|-------------|---------------|
| Production Admin Bootstrap | ✅ | N/A | Manual |
| Forced Password Change | ✅ | ✅ | Manual |
| Self-Service Password Change | ✅ | ✅ | Manual |
| Admin Password Reset | ✅ | ✅ | Manual |
| Password Strength Validation | ✅ | ✅ | Manual |
| Old Password Verification | ✅ | ✅ | Manual |
| JWT with must_change_password | ✅ | ✅ | Manual |
| Frontend Redirect Logic | N/A | ✅ | Manual |
| Password Reset Modal | N/A | ✅ | Manual |

---

## Known Issues and Limitations

1. **Database Schema Migration**: Existing databases need manual `ALTER TABLE` to add `must_change_password` column
2. **Password History**: Not implemented (future feature)
3. **Password Expiry**: Not implemented (future feature)
4. **Account Lockout**: Existing but not integrated with password management
5. **Email Notifications**: Not implemented for password resets

---

## Next Steps

1. Run automated test scripts
2. Test with real frontend deployment
3. Verify in production-like environment
4. Add integration tests to CI/CD
5. Consider implementing password history
6. Add password expiry policy (optional)

---

**Last Updated**: 2025-12-30
**Author**: JadeVectorDB Development Team
