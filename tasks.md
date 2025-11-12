# Task Breakdown: Default User Creation for Local Deploy

## Tasks

- Update spec.md to require default admin, dev, and test users for local/dev/test deploys (done)
- Update plan.md and README.md to reflect new requirements and rationale
- Implement backend logic for environment-aware default user creation
- Ensure default users are not created/enabled in production
- Assign correct roles and permissions to each default user
- Set status to `active` in local/dev/test, `inactive` or removed in production
- Add/modify tests to verify user creation, role assignment, and status enforcement

## Acceptance Criteria

- Default users are created only in local/dev/test environments
- Roles and permissions are correctly assigned
- Status is `active` in non-production, `inactive` or removed in production
- Documentation and tests are updated
