# Frontend RBAC Implementation Guide

**Document Version**: 1.0  
**Last Updated**: 2024-12-16  
**Related Epic**: T11-PERSISTENCE  
**Status**: PLANNING

---

## 📋 Executive Summary

This document outlines the frontend changes required to support the full RBAC (Role-Based Access Control) system and persistent storage implementation in JadeVectorDB.

### Current Frontend State

**Technology Stack**:
- Next.js 14
- React 18
- JavaScript (not TypeScript)
- Pages-based routing

**Existing Pages**:
- `login.js` - User login
- `register.js` - User registration
- `users.js` - Basic user management (create, list, update, delete)
- `api-keys.js` - API key management (basic)
- `databases.js` - Database management
- `auth.js` - Authentication utilities
- `dashboard.js` - Main dashboard

**Existing API Client**: `frontend/src/lib/api.js` with:
- `authApi` - Login, logout, register
- `usersApi` - User CRUD operations
- `apiKeysApi` - Basic API key management
- `databaseApi` - Database operations
- Other APIs (vector, search, monitoring, etc.)

**Existing UI Components**: `frontend/src/components/ui/`:
- `button.js`, `card.js`, `input.js`, `select.js`, `alert.js`

### What's Missing for Full RBAC

1. **Group Management** - No UI to create, manage, or view groups
2. **Role Assignment** - No UI to assign or revoke roles
3. **Permission Management** - No UI to grant or revoke database-level permissions
4. **Permission Visualization** - No visibility into user's effective permissions
5. **Enhanced API Keys** - Missing scopes, expiration, and status features
6. **Database Permissions** - Database list doesn't show user's access level
7. **Admin Features** - No admin-only UI sections

---

## 🎯 Implementation Overview

### Total Frontend Tasks: 43

| Category | Task Count | Description |
|----------|------------|-------------|
| Group Management UI | 4 | Pages, components, and API integration for groups |
| Role & Permission UI | 5 | Role assignment, permission matrix, API functions |
| Enhanced API Keys | 3 | Scopes, expiration, status indicators |
| Database Enhancements | 3 | Permission indicators, permission management page |
| User Management | 3 | Enhanced user list, profile page, activity timeline |
| Navigation & Layout | 3 | Updated menu, permission-based rendering, admin UI |
| Persistence Indicators | 3 | Save status, data freshness, error handling |
| State Management | 3 | Auth context, permission caching, optimistic updates |
| **Testing** | **16** | Unit tests, integration tests, E2E tests, accessibility |

**Total: 43 tasks**

### Timeline Impact

- **Original Timeline**: 5 weeks (backend only)
- **Updated Timeline**: 6-7 weeks (backend + frontend)
- **Frontend Development**: Weeks 1-3 (parallel with backend Phase 1+2)
- **Frontend Testing**: Weeks 4-5 (parallel with backend Phase 3)
- **Integration & E2E**: Weeks 5-7

---

## 🏗️ Detailed Implementation Plan

### Week 1-3: Core RBAC UI (Parallel with Backend Phase 1+2)

#### Sprint 1.6: Group Management (Days 1-6)
- **T11.21.1**: Create `pages/groups.js` - Group list, create, edit, delete
- **T11.21.2**: Create `pages/groups/[id].js` - Group detail, member management
- **T11.21.3**: Add Group API Functions to `lib/api.js`
- **T11.21.4**: Create group-related components (GroupCard, GroupMemberList, AddMemberModal)

**Deliverable**: Users can create groups, add members, and view group details

#### Sprint 1.6: Role & Permission UI (Days 7-12)
- **T11.22.1**: Create `pages/roles.js` - Role list, permission view
- **T11.22.2**: Enhance `pages/users.js` - Add role assignment UI
- **T11.22.3**: Create `pages/permissions.js` - Permission matrix view
- **T11.22.4**: Add Role API Functions to `lib/api.js`
- **T11.22.5**: Add Permission API Functions to `lib/api.js`

**Deliverable**: Users can assign roles and view permissions

#### Sprint 1.6: Enhanced Features (Days 13-18)
- **T11.23.1**: Enhance `pages/api-keys.js` - Scopes, expiration, status
- **T11.23.2**: Create API Key Scopes Component
- **T11.23.3**: Add API Key Status Indicators
- **T11.24.1**: Add permission indicators to database list
- **T11.24.2**: Create database permission management page
- **T11.25.1**: Enhance user list with groups and roles columns
- **T11.25.2**: Create user detail/profile page

**Deliverable**: Full-featured UI for all RBAC operations

### Week 4-5: Testing & Integration (Parallel with Backend Phase 3)

#### Sprint 1.7: Frontend Testing (Days 19-25)
- **Unit Tests** (T11.29.1 - T11.29.4): Test all new components
- **Integration Tests** (T11.30.1 - T11.30.4): Test workflows (groups, roles, permissions, API keys)
- **E2E Tests** (T11.31.1 - T11.31.4): Test complete RBAC setup, permission enforcement, persistence
- **Accessibility Tests** (T11.32.1 - T11.32.3): Keyboard navigation, screen readers, mobile responsiveness
- **Performance Tests** (T11.33.1 - T11.33.2): Large dataset rendering, permission check performance

**Deliverable**: 95%+ test coverage, all E2E workflows passing

### Week 6-7: Documentation & Polish

#### Sprint 3.3: Documentation & Final Testing (Days 26-42)
- **T11.34.1**: Create Frontend Developer Guide
- **T11.34.2**: Create UI/UX Style Guide
- **T11.34.3**: Update User Guide with screenshots
- Final integration testing
- Bug fixes and polish

**Deliverable**: Complete documentation and production-ready frontend

---

## 📊 New Frontend Components

### New Pages

1. **`pages/groups.js`**
   - List all groups
   - Create new group
   - Edit group details
   - Delete group
   - Filter and search groups

2. **`pages/groups/[id].js`**
   - Group details and metadata
   - Member list with add/remove
   - Inherited roles and permissions
   - Group activity log

3. **`pages/roles.js`**
   - List all roles (system + custom)
   - Show permissions per role
   - Show users assigned to each role
   - Filter and search roles

4. **`pages/permissions.js`**
   - Permission matrix (users × databases)
   - Color-coded permission levels
   - Filter by user, group, or database
   - Visual permission overview

5. **`pages/databases/[id]/permissions.js`**
   - List users/groups with access
   - Grant permission modal
   - Revoke permission button
   - Permission audit log

6. **`pages/users/[id].js`**
   - User profile and details
   - Group memberships
   - Assigned roles
   - Effective permissions (direct + inherited)
   - Owned databases
   - User's API keys
   - Activity timeline

### New Components

1. **Group Components**
   - `GroupCard.js` - Card display for group
   - `GroupMemberList.js` - List of group members
   - `AddMemberModal.js` - Modal to add members

2. **Role Components**
   - `RoleAssignmentModal.js` - Modal to assign roles
   - `RoleBadge.js` - Visual role indicator
   - `PermissionMatrix.js` - Permission visualization

3. **API Key Components**
   - `ApiKeyScopes.js` - Multi-select for scopes
   - `ApiKeyStatusBadge.js` - Status indicator
   - `ApiKeyExpirationPicker.js` - Date picker for expiration

4. **Database Components**
   - `DatabasePermissionBadge.js` - Permission level indicator
   - `GrantPermissionModal.js` - Modal to grant permissions

5. **Layout & Navigation**
   - Enhanced `Layout.js` with RBAC menu items
   - `AdminOnlySection.js` - Wrapper for admin UI
   - `PermissionGuard.js` - Component to check permissions

### New Hooks

1. **`useAuth()`** - Access auth context (user, roles, permissions)
2. **`usePermissions()`** - Check permissions for current user
3. **`useGroups()`** - Fetch and manage groups
4. **`useRoles()`** - Fetch and manage roles

### Enhanced API Client

**New API Modules in `lib/api.js`**:

```javascript
// Group Management
export const groupApi = {
  createGroup: async (name, description, ownerId) => {},
  listGroups: async (limit, offset) => {},
  getGroup: async (groupId) => {},
  updateGroup: async (groupId, data) => {},
  deleteGroup: async (groupId) => {},
  addMember: async (groupId, userId) => {},
  removeMember: async (groupId, userId) => {},
  getMembers: async (groupId) => {},
};

// Role Management
export const roleApi = {
  listRoles: async () => {},
  getUserRoles: async (userId) => {},
  assignRole: async (userId, roleId) => {},
  revokeRole: async (userId, roleId) => {},
  getGroupRoles: async (groupId) => {},
  assignGroupRole: async (groupId, roleId) => {},
};

// Permission Management
export const permissionApi = {
  listPermissions: async () => {},
  getUserPermissions: async (userId) => {},
  grantDatabasePermission: async (dbId, principalId, principalType, permissionId) => {},
  revokeDatabasePermission: async (dbId, principalId, principalType, permissionId) => {},
  getDatabasePermissions: async (databaseId) => {},
};
```

---

## 🎨 UI/UX Design Patterns

### Permission Indicators

**Color Coding**:
- 🟢 **Green** - Full access (owner/admin)
- 🔵 **Blue** - Write access
- 🟡 **Yellow** - Read access
- 🔴 **Red** - No access / Revoked

**Badge Examples**:
```
Owner | Admin | Read/Write | Read-Only | No Access
```

### Group Visualization

- **Group Card**: Name, description, member count, owner
- **Member List**: Avatar, name, role, "Remove" button
- **Add Member**: Searchable user dropdown with role selection

### Role Assignment

- **User Table**: Expandable row showing assigned roles
- **Assign Role Button**: Opens modal with role selector
- **Role Badge**: Color-coded badge showing role name

### Permission Matrix

```
         | Database A | Database B | Database C
---------|------------|------------|------------
User 1   | Admin      | Read       | None
User 2   | Read/Write | Read       | Read
Group A  | Read       | Read/Write | Admin
```

---

## 🔧 State Management Strategy

### Authentication Context

```javascript
// frontend/src/context/AuthContext.js
const AuthContext = createContext();

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [roles, setRoles] = useState([]);
  const [permissions, setPermissions] = useState([]);
  
  const login = async (username, password) => {
    // Login and fetch user, roles, permissions
  };
  
  const checkPermission = (permissionName) => {
    return permissions.includes(permissionName);
  };
  
  return (
    <AuthContext.Provider value={{ user, roles, permissions, login, checkPermission }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => useContext(AuthContext);
```

### Permission Checking Pattern

```javascript
// In components
const { checkPermission } = useAuth();

if (checkPermission('database:delete')) {
  return <Button onClick={handleDelete}>Delete Database</Button>;
}

// Or use PermissionGuard component
<PermissionGuard permission="database:delete">
  <Button onClick={handleDelete}>Delete Database</Button>
</PermissionGuard>
```

### Optimistic UI Updates

```javascript
const handleAddMember = async (groupId, userId) => {
  // Optimistically update UI
  setMembers([...members, { userId, name: '...' }]);
  
  try {
    await groupApi.addMember(groupId, userId);
    // Success - fetch fresh data
    fetchMembers();
  } catch (error) {
    // Rollback optimistic update
    setMembers(members.filter(m => m.userId !== userId));
    setError(error.message);
  }
};
```

---

## ✅ Testing Strategy

### Unit Tests (16 tasks)

**Component Tests**:
- Test rendering with various props
- Test user interactions (clicks, inputs)
- Test form validation
- Mock API calls

**Example** (T11.29.1):
```javascript
describe('GroupCard', () => {
  it('renders group name and description', () => {
    render(<GroupCard group={{ name: 'Test Group', description: 'Test' }} />);
    expect(screen.getByText('Test Group')).toBeInTheDocument();
  });
  
  it('calls onDelete when delete button clicked', () => {
    const onDelete = jest.fn();
    render(<GroupCard group={{ name: 'Test' }} onDelete={onDelete} />);
    fireEvent.click(screen.getByText('Delete'));
    expect(onDelete).toHaveBeenCalled();
  });
});
```

### Integration Tests (4 tasks)

**Workflow Tests**:
- Test multi-step workflows
- Test API integration
- Test state management

**Example** (T11.30.1):
```javascript
describe('Group Workflows', () => {
  it('creates group, adds members, and deletes group', async () => {
    render(<GroupsPage />);
    
    // Create group
    fireEvent.click(screen.getByText('Create Group'));
    fireEvent.change(screen.getByLabelText('Group Name'), { target: { value: 'Test Group' } });
    fireEvent.click(screen.getByText('Save'));
    await waitFor(() => expect(screen.getByText('Test Group')).toBeInTheDocument());
    
    // Add member
    fireEvent.click(screen.getByText('Add Member'));
    // ... rest of test
  });
});
```

### E2E Tests (4 tasks)

**Full System Tests** (using Cypress):
- Test complete user journeys
- Test across backend and frontend
- Test persistence verification

**Example** (T11.31.1):
```javascript
describe('Complete RBAC Setup', () => {
  it('admin creates group, assigns users, grants database permission', () => {
    cy.login('admin', 'password');
    
    // Create group
    cy.visit('/groups');
    cy.contains('Create Group').click();
    cy.get('input[name="name"]').type('Engineering');
    cy.contains('Save').click();
    cy.contains('Engineering').should('be.visible');
    
    // Add user to group
    cy.contains('Engineering').click();
    cy.contains('Add Member').click();
    cy.get('select[name="userId"]').select('john@example.com');
    cy.contains('Add').click();
    
    // Grant database permission
    cy.visit('/databases');
    cy.contains('my-database').click();
    cy.contains('Permissions').click();
    cy.contains('Grant Permission').click();
    cy.get('select[name="principalType"]').select('Group');
    cy.get('select[name="principalId"]').select('Engineering');
    cy.get('select[name="permission"]').select('Read/Write');
    cy.contains('Grant').click();
    
    // Verify as user
    cy.logout();
    cy.login('john', 'password');
    cy.visit('/databases');
    cy.contains('my-database').should('be.visible');
    cy.contains('Read/Write').should('be.visible');
  });
});
```

### Accessibility Tests (3 tasks)

- Keyboard navigation (Tab, Enter, Escape)
- Screen reader compatibility (ARIA labels)
- Mobile responsiveness (touch, small screens)

### Performance Tests (2 tasks)

- Large dataset rendering (10,000+ items)
- Permission check latency (<50ms)

---

## 📖 Frontend API Endpoints

### New Endpoints Required from Backend

| Method | Endpoint | Purpose |
|--------|----------|---------|
| **Groups** |
| POST | `/v1/groups` | Create group |
| GET | `/v1/groups` | List groups |
| GET | `/v1/groups/{id}` | Get group details |
| PUT | `/v1/groups/{id}` | Update group |
| DELETE | `/v1/groups/{id}` | Delete group |
| POST | `/v1/groups/{id}/members` | Add member |
| DELETE | `/v1/groups/{id}/members/{user_id}` | Remove member |
| GET | `/v1/groups/{id}/members` | List members |
| **Roles** |
| GET | `/v1/roles` | List all roles |
| POST | `/v1/users/{id}/roles` | Assign role to user |
| DELETE | `/v1/users/{id}/roles/{role_id}` | Revoke role from user |
| GET | `/v1/users/{id}/roles` | Get user's roles |
| POST | `/v1/groups/{id}/roles` | Assign role to group |
| **Permissions** |
| GET | `/v1/permissions` | List all permissions |
| GET | `/v1/users/{id}/permissions` | Get user's effective permissions |
| POST | `/v1/databases/{id}/permissions` | Grant database permission |
| DELETE | `/v1/databases/{id}/permissions/{perm_id}` | Revoke database permission |
| GET | `/v1/databases/{id}/permissions` | List database permissions |
| **API Keys** |
| POST | `/v1/api-keys` | Create API key (with scopes) |
| GET | `/v1/api-keys` | List user's API keys |
| DELETE | `/v1/api-keys/{id}` | Revoke API key |

---

## 🚀 Deployment Considerations

### Environment Variables

```bash
# Frontend environment
NEXT_PUBLIC_API_URL=http://localhost:8080/api
NEXT_PUBLIC_ENABLE_RBAC=true
NEXT_PUBLIC_SESSION_TIMEOUT=3600
```

### Build Process

```bash
cd frontend
npm install
npm run build
npm run start  # Production mode
```

### Docker Deployment

Update `docker-compose.yml` to ensure frontend can access backend:

```yaml
services:
  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_API_URL=http://backend:8080/api
    depends_on:
      - backend
```

---

## 📚 Documentation Updates Needed

1. **User Guide** - Add sections on:
   - Managing groups
   - Assigning roles
   - Granting database permissions
   - Creating and using API keys

2. **Administrator Guide** - Add sections on:
   - RBAC system overview
   - User and group management
   - Permission model explanation
   - Security best practices

3. **Frontend Developer Guide** - Create new document with:
   - Component architecture
   - API integration patterns
   - State management guide
   - Permission checking patterns

4. **UI/UX Style Guide** - Create new document with:
   - Component usage examples
   - Design patterns for RBAC UI
   - Accessibility guidelines
   - Responsive design guidelines

---

## ✅ Definition of Done - Frontend

### Phase 1+2 Frontend Complete When:
- ✅ All group management pages implemented and tested
- ✅ Role assignment UI functional
- ✅ Permission matrix view implemented
- ✅ Enhanced API key management with scopes and expiration
- ✅ Database permission indicators visible
- ✅ User profile page showing groups, roles, permissions
- ✅ Navigation menu updated with RBAC sections
- ✅ Permission-based UI rendering works
- ✅ 95%+ unit test coverage
- ✅ All integration tests pass
- ✅ All E2E tests pass
- ✅ Accessibility tests pass (keyboard, screen reader, mobile)
- ✅ Performance tests pass (large datasets, permission checks)
- ✅ Frontend documentation complete

---

## 🔗 Related Documents

- [Persistent Storage Implementation Plan](../TasksTracking/11-persistent-storage-implementation.md)
- [Architecture Documentation](./architecture.md)
- [Specification](../specs/002-check-if-we/spec.md)
- [RBAC CLI Commands Reference](../cli/RBAC_COMMANDS_REFERENCE.md)

---

**Document Version**: 1.0  
**Last Updated**: 2024-12-16  
**Author**: GitHub Copilot  
**Status**: APPROVED FOR IMPLEMENTATION
