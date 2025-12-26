# FR-029 Compliance Analysis: Default User Seeding

## Specification Requirements (spec.md lines 395-404)

**FR-029**: System MUST automatically create default users (`admin`, `dev`, `test`) with appropriate roles and permissions when deployed in local, development, or test environments.

### Required Users:

| Username | Spec Requirement | Permissions | Status |
|----------|------------------|-------------|--------|
| `admin` | MUST exist | Full administrative permissions | active |
| `dev` | MUST exist | Development permissions | active |
| `test` | MUST exist | Limited/test permissions | active |

### Additional Requirements:
- ✅ **Environment-aware creation**: Only in local/dev/test, NOT in production
- ✅ **Production behavior**: MUST be set to `inactive` or removed entirely
- ✅ **Documentation**: Implementation notes must document the logic
- ⚠️ **Tests**: MUST verify correct creation, role assignment, and status enforcement

## Current Implementation (T236)

**File**: `/backend/src/services/authentication_service.cpp` (lines 748-767)

### Actually Implemented Users:

| Username | Actual Implementation | Password | Roles | User ID |
|----------|----------------------|----------|-------|---------|
| `admin` | ✅ Correct | admin123 | admin, developer, user | user_admin_default |
| `developer` | ❌ **WRONG** | dev123 | developer, user | user_developer_default |
| `tester` | ❌ **WRONG** | test123 | tester, user | user_tester_default |

## Issues Found

### Issue 1: Username Mismatch ⚠️

**Spec says**: `dev` and `test`
**We implemented**: `developer` and `tester`

**Impact**:
- Users expecting to login with `dev` will fail
- Users expecting to login with `test` will fail
- Documentation/tutorials may reference the wrong usernames

**Severity**: MEDIUM - Functional but doesn't match specification

### Issue 2: Role/Permission Granularity 🔍

**Spec Requirements**:
- `admin`: "Full administrative permissions"
- `dev`: "Development permissions"
- `test`: "Limited/test permissions"

**Current Implementation**:
- `admin`: Roles = `["admin", "developer", "user"]` ✅ (includes admin role)
- `developer`: Roles = `["developer", "user"]` ⚠️ (unclear if "developer" role = "development permissions")
- `tester`: Roles = `["tester", "user"]` ⚠️ (unclear if "tester" role = "limited/test permissions")

**Question**: Are the role names `"admin"`, `"developer"`, `"tester"` properly mapped to the permission sets described in the spec?

**This may be addressed by T237**: "Assign roles and permissions to default users"

### Issue 3: Status Enforcement ❓

**Spec Requirement**:
- Local/dev/test: status = `active` ✅
- Production: MUST be `inactive` or removed entirely

**Current Implementation**:
```cpp
if (environment != "development" && environment != "dev" &&
    environment != "test" && environment != "testing" &&
    environment != "local") {
    LOG_INFO(logger_, "Skipping default user seeding in " + environment + " environment");
    return true;  // Not an error, just skipping
}
```

**Analysis**:
- ✅ Users are NOT created in production (removed entirely)
- ❌ Spec also mentions they could be set to `inactive` instead of removed
- ✅ Current approach (removal) is more secure than making them inactive

**Verdict**: COMPLIANT (removal is better than inactive)

## Compliance Summary

| Requirement | Status | Notes |
|-------------|--------|-------|
| Create `admin` user | ✅ PASS | Username correct |
| Create `dev` user | ❌ FAIL | Created as `developer` instead |
| Create `test` user | ❌ FAIL | Created as `tester` instead |
| Environment-aware | ✅ PASS | Uses JADEVECTORDB_ENV correctly |
| Production safety | ✅ PASS | Users removed (not created) |
| Status = active | ✅ PASS | All created users are active |
| Documentation | ✅ PASS | T236_IMPLEMENTATION_SUMMARY.md exists |
| Tests | ❌ PENDING | No tests written yet |

**Overall Compliance**: 5/8 = 62.5% ⚠️

## Recommended Fixes

### Fix 1: Correct Usernames (HIGH PRIORITY)

**Change**:
```cpp
// BEFORE
{"developer", "dev123", {"developer", "user"}, "user_developer_default"},
{"tester", "test123", {"tester", "user"}, "user_tester_default"}

// AFTER (to match spec)
{"dev", "dev123", {"developer", "user"}, "user_dev_default"},
{"test", "test123", {"tester", "user"}, "user_test_default"}
```

**Impact**: Makes implementation match FR-029 specification exactly

### Fix 2: Clarify Role Mappings (T237)

Need to document or verify:
- Does role `"admin"` provide "Full administrative permissions"? ✅ Likely yes
- Does role `"developer"` provide "Development permissions"? ❓ Unclear
- Does role `"tester"` provide "Limited/test permissions"? ❓ Unclear

**Action**: Review role definitions in AuthManager or create role-to-permission mapping

### Fix 3: Add Tests (T231 or separate task)

Create tests to verify:
- Default users are created with correct usernames
- Default users have correct roles
- Default users are active
- Default users are NOT created in production
- Idempotent behavior (multiple calls don't duplicate)

## Implementation Comparison

### Legacy Implementation (rest_api.cpp lines 138-187)

```cpp
if (allow_default_users) {
    ensure_default_user("admin", "admin@example.local", "Admin!2345", {"role_admin"}, true);
    ensure_default_user("dev", "dev@example.local", "Developer!2345", {"role_user"}, true);
    ensure_default_user("test", "test@example.local", "Tester!2345", {"role_user"}, true);
}
```

**Analysis**:
- ✅ Uses correct usernames: `admin`, `dev`, `test`
- ⚠️ Uses different role names: `role_admin`, `role_user`
- ✅ Includes email addresses
- ✅ Uses stronger passwords
- ⚠️ Environment variable is `JADEVECTORDB_ENV` (different from T236's `JADEVECTORDB_ENV`)

**Conflict**: We now have TWO seeding mechanisms that may create different users!

### Current Situation

Both seeding mechanisms will run:
1. **T236 implementation** (authentication_service.cpp): Creates `admin`, `developer`, `tester`
2. **Legacy implementation** (rest_api.cpp): Creates `admin`, `dev`, `test`

**Result**: We'll end up with **5 users** instead of 3:
- `admin` (created by both, but first one wins due to idempotency)
- `developer` (created by T236 only)
- `tester` (created by T236 only)
- `dev` (created by legacy only)
- `test` (created by legacy only)

## Recommended Actions

### Immediate Actions (Required for FR-029 Compliance)

1. ✅ **Fix T236 usernames** to match spec:
   - Change `developer` → `dev`
   - Change `tester` → `test`

2. ✅ **Align T236 with legacy** or remove one:
   - **Option A**: Update T236 to match spec, remove legacy code
   - **Option B**: Keep legacy code, remove T236 implementation
   - **Recommendation**: Update T236 (cleaner, better documented), remove legacy

3. ✅ **Unify environment variable**:
   - Choose either `JADEVECTORDB_ENV` or `JADEVECTORDB_ENV`
   - Update all references to use the same one
   - **Recommendation**: Use `JADEVECTORDB_ENV` (shorter, simpler)

4. ❌ **Add tests** (can be separate task):
   - Verify FR-029 compliance
   - Test all three users created
   - Test correct roles assigned
   - Test production behavior

### Long-term Actions (T237 and beyond)

1. **Document role-to-permission mappings**
2. **Implement granular permissions** (if not already done)
3. **Add configuration file support** for custom default users
4. **Add email addresses** to default users (optional but recommended)

## Conclusion

**Current FR-029 Compliance**: ⚠️ **PARTIAL (62.5%)**

**Critical Issues**:
- ❌ Wrong usernames (`developer`/`tester` vs `dev`/`test`)
- ⚠️ Two conflicting seeding mechanisms
- ❌ No tests

**Recommendation**:
1. Fix usernames in T236 implementation
2. Remove legacy seeding code
3. Add tests
4. Re-verify compliance → Should achieve 100%

**Priority**: HIGH - This affects manual testing and may confuse users
