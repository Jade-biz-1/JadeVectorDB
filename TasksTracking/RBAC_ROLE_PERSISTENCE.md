# RBAC Role Persistence Implementation Plan

**Status:** Not Implemented
**Priority:** Medium
**Created:** 2025-12-22

## Current State

### What's Implemented
- ✅ RBAC authorization system with roles and permissions (AuthorizationService)
- ✅ SQLite tables created for RBAC:
  - `roles` - stores role definitions
  - `role_permissions` - stores role-permission mappings
  - `user_roles` - stores user-role assignments (WORKING)
  - `group_roles` - stores group-role assignments (WORKING)
- ✅ User-role assignment persistence works (assign_role_to_user, revoke_role_from_user, get_user_roles)
- ✅ Group-role assignment persistence works
- ✅ In-memory role management (create_role, update_role, delete_role, list_roles, get_role)

### What's NOT Implemented
- ❌ Role definitions (Role objects) are NOT persisted to SQLite
- ❌ Role creation/update/deletion does NOT save to database
- ❌ Roles are only stored in memory in AuthorizationService::roles_ map
- ❌ Default system roles are initialized in memory only via initialize_default_roles()
- ❌ Role permissions are NOT persisted (only the mapping table exists)
- ❌ Methods declared in SQLitePersistenceLayer header are NOT implemented:
  - `list_roles()` - declared but not implemented
  - `get_role()` - declared but not implemented

### Impact
On server restart, all role definitions are lost except:
- Default system roles (admin, user, readonly) are recreated via initialize_default_roles()
- User-role assignments persist (but roles themselves don't)
- This means custom roles created at runtime are lost on restart

## Implementation Tasks

### Backend Implementation

#### 1. Implement Role Persistence in SQLitePersistenceLayer
Location: `backend/src/services/sqlite_persistence_layer.cpp`

**Tasks:**
- [ ] Implement `create_role(const Role& role)` method
  - Insert role into `roles` table
  - Insert permissions into `role_permissions` table
  - Handle is_system_role flag
  - Return Result<std::string> with role_id

- [ ] Implement `update_role(const std::string& role_id, const Role& role)` method
  - Update role in `roles` table
  - Delete old permissions from `role_permissions`
  - Insert new permissions into `role_permissions`
  - Prevent updates to system roles

- [ ] Implement `delete_role(const std::string& role_id)` method
  - Check if role is system role (prevent deletion)
  - Delete from `role_permissions` (cascade)
  - Delete from `user_roles` (cascade)
  - Delete from `group_roles` (cascade)
  - Delete from `roles`

- [ ] Implement `list_roles()` method (already declared)
  - Query all roles from `roles` table
  - Load permissions for each role from `role_permissions`
  - Return Result<std::vector<Role>>

- [ ] Implement `get_role(const std::string& role_id)` method (already declared)
  - Query role from `roles` table
  - Load permissions from `role_permissions`
  - Return Result<Role>

#### 2. Integrate with AuthorizationService
Location: `backend/src/services/authorization_service.cpp`

**Tasks:**
- [ ] Add SQLitePersistenceLayer dependency to AuthorizationService
  - Add constructor parameter
  - Store as member variable

- [ ] Update `create_role()` to persist to database
  - Call persistence_layer_->create_role() after in-memory storage
  - Handle errors appropriately

- [ ] Update `update_role()` to persist to database
  - Call persistence_layer_->update_role()
  - Update in-memory cache on success

- [ ] Update `delete_role()` to persist to database
  - Call persistence_layer_->delete_role()
  - Remove from in-memory cache on success

- [ ] Update `initialize_default_roles()` to check database first
  - Query persistence_layer_->list_roles()
  - Only create default roles if database is empty
  - Load existing roles from database into memory

- [ ] Add role cache refresh method
  - Load all roles from database into memory
  - Call on service initialization

#### 3. Update Database Schema (if needed)
Location: `backend/src/services/sqlite_persistence_layer.cpp` (initialize method)

**Tasks:**
- [ ] Verify `roles` table has all needed columns:
  - role_id (PRIMARY KEY)
  - role_name
  - description
  - is_system_role (boolean/integer)
  - created_at
  - updated_at

- [ ] Verify `role_permissions` table structure:
  - role_id (FOREIGN KEY to roles.role_id)
  - permission_id
  - resource_type
  - action
  - scope
  - description

- [ ] Seed default system roles in database on first initialization
  - Check if roles table is empty
  - Insert admin, user, readonly roles
  - Insert their permissions

### Testing

#### 1. Unit Tests for SQLitePersistenceLayer
Location: `backend/tests/unit/test_sqlite_persistence_layer.cpp` (create if doesn't exist)

**Test Cases:**
- [ ] Test create_role()
  - Create role with permissions
  - Verify role stored in database
  - Verify permissions stored in role_permissions table
  - Test duplicate role_id handling

- [ ] Test update_role()
  - Create role, then update
  - Verify role fields updated
  - Verify permissions replaced correctly
  - Test updating system role (should fail)

- [ ] Test delete_role()
  - Create role, then delete
  - Verify role removed from database
  - Verify permissions removed (cascade)
  - Verify user_roles cleaned up
  - Test deleting system role (should fail)

- [ ] Test list_roles()
  - Insert multiple roles
  - Verify all roles returned with permissions

- [ ] Test get_role()
  - Create role
  - Retrieve by role_id
  - Verify all fields and permissions match
  - Test non-existent role_id

#### 2. Integration Tests for AuthorizationService
Location: `backend/tests/unit/test_authorization_service.cpp` (create if doesn't exist)

**Test Cases:**
- [ ] Test role persistence across service restarts
  - Create AuthorizationService instance
  - Create custom role
  - Destroy service instance
  - Create new AuthorizationService instance
  - Verify custom role still exists

- [ ] Test default roles initialization
  - Start with empty database
  - Initialize AuthorizationService
  - Verify admin, user, readonly roles exist
  - Restart service
  - Verify roles not duplicated

- [ ] Test role CRUD operations with persistence
  - Create role
  - Update role
  - Verify changes persisted
  - Delete role
  - Verify deleted from database

- [ ] Test user-role assignment with persisted roles
  - Create custom role and persist
  - Assign to user
  - Restart service
  - Verify user still has role
  - Verify role definition still exists

#### 3. End-to-End Tests
Location: `backend/tests/test_rbac_persistence_e2e.cpp` (create new)

**Test Cases:**
- [ ] Test complete RBAC lifecycle
  - Create database and user
  - Create custom role with specific permissions
  - Assign role to user
  - Authorize user action
  - Restart database service
  - Verify role persists
  - Verify user-role assignment persists
  - Verify authorization still works

## Implementation Order

1. **Phase 1: Persistence Layer** (Estimated: 4-6 hours)
   - Implement all SQLitePersistenceLayer methods
   - Write unit tests for persistence layer

2. **Phase 2: Service Integration** (Estimated: 3-4 hours)
   - Integrate persistence with AuthorizationService
   - Update role management methods
   - Update initialization logic

3. **Phase 3: Testing** (Estimated: 3-4 hours)
   - Write integration tests
   - Write end-to-end tests
   - Test restart scenarios

4. **Phase 4: Validation** (Estimated: 1-2 hours)
   - Manual testing
   - Verify all tests pass
   - Test with real server restarts

## Files to Modify

### Implementation
- `backend/src/services/sqlite_persistence_layer.h` (already has declarations)
- `backend/src/services/sqlite_persistence_layer.cpp` (add implementations)
- `backend/src/services/authorization_service.h` (add persistence layer dependency)
- `backend/src/services/authorization_service.cpp` (integrate persistence)

### Tests
- `backend/tests/unit/test_sqlite_persistence_layer.cpp` (create if doesn't exist)
- `backend/tests/unit/test_authorization_service.cpp` (create if doesn't exist)
- `backend/tests/test_rbac_persistence_e2e.cpp` (create new)

## Success Criteria

- [ ] All role CRUD operations persist to SQLite
- [ ] Custom roles survive server restarts
- [ ] Default system roles are seeded once and loaded from database
- [ ] All unit tests pass
- [ ] All integration tests pass
- [ ] Manual restart tests confirm persistence

## Notes

- This is a non-breaking change - user-role assignments already persist
- Default system roles can be seeded once during first database initialization
- Consider adding migration logic if schema changes are needed
- Ensure proper error handling for database operations
- Consider adding role version tracking for future migrations
