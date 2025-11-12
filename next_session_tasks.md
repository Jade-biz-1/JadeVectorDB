# Next Session Tasks for JadeVectorDB

## Completed Items

- [x] T202–T214 advanced feature track (archived in master task list; no further action)
- [x] T216–T218 cURL command generation stream (deliverables captured in `cli/curl_command_generation_summary.md`)

## Upcoming Focus

### Implementation

1. [ ] Implement authentication and user management handlers in `backend/src/api/rest/rest_api.cpp`, wiring `handle_register_request`, `handle_login_request`, `handle_logout_request`, `handle_forgot_password_request`, `handle_reset_password_request`, `handle_create_user_request`, `handle_list_users_request`, `handle_update_user_request`, `handle_delete_user_request`, and `handle_user_status_request` to `AuthenticationService`, `AuthManager`, and `SecurityAuditLogger`.
2. [ ] Finish API key management endpoints (`handle_list_api_keys_request`, `handle_create_api_key_request`, `handle_revoke_api_key_request`) using the new AuthManager helpers and emit audit events.
3. [ ] Provide concrete implementations for audit, alert, cluster, and performance routes (or explicit 501 responses) so that `handle_security_routes`, `handle_alert_routes`, `handle_cluster_routes`, and `handle_performance_routes` expose usable Crow handlers backed by `SecurityAuditLogger` and relevant services.
4. [ ] Replace placeholder database/vector/index route installers (e.g., `handle_create_database`, `handle_store_vector`) with live Crow route bindings that call into the corresponding services, eliminating pseudo-code blocks and enabling end-to-end API operation.

### Enhancements

1. [ ] Build shadcn-based authentication UI (login, register, forgot/reset password, API key management) that consumes the new backend endpoints and persists API keys securely (`frontend/src/pages/auth/*`, `frontend/src/lib/api.js`).
2. [ ] Refresh admin/search interfaces to surface enriched metadata (`tags`, `permissions`, timestamps) returned by the search API and prepare views for audit log/API key management summaries.
3. [ ] Update documentation (`docs/api_documentation.md`, `docs/search_functionality.md`, `README.md`) to describe the updated search response schema (`score`, nested `vector`) and the authentication lifecycle.

### Testing

1. [ ] Add backend unit and integration coverage for search serialization (with/without `includeVectorData`), authentication flows, and API key lifecycle within `backend/tests`.
2. [ ] Extend frontend Jest/Cypress suites to cover login/logout flows, API key revocation UX, and search result rendering toggles.
3. [ ] Introduce smoke/performance test scripts that exercise `/v1/databases/{id}/search` and authentication endpoints (consider reusable harness under `scripts/` or `property-tests/`).

## Notes & Dependencies

- Coordinate with security stakeholders on password hashing policy, audit retention windows, and API key rotation requirements before finalizing handlers.
- Ensure environment-specific default user seeding remains idempotent once the authentication routes are active.
- Mirror backend contract changes in `backend/src/api/rest/rest_api_simple.cpp` or formally deprecate the simple API to avoid drift.
- Remaining tutorial sub-tasks (T215.29–T215.30) stay tracked in `tutorial_pending_tasks.md`; revisit after the auth/UI work stabilizes.
