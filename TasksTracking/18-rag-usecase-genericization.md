# RAG_USECASE.md Genericization & User Management

**Phase**: Enhancement  
**Source**: User review comments (2026-04-14) + enhancement plan proposal  
**Added**: 2026-04-14  
**Completed**: 2026-04-14

---

## Background

RAG_USECASE.md was written for a maintenance-documentation use case with field engineers, device types, and equipment-specific language throughout. The user reviewed it and requested 7 categories of changes to make it fully generic, plus a new user management specification.

---

## Tasks

### T-RAG01: Generalize All Narrative Content
**Status**: ✅ COMPLETED  
**Description**: Replace all maintenance/field-engineer-specific language with generic equivalents throughout the document.  
- Title changed to "Document Q&A System"
- "Field Engineer" → "User" in all diagrams and narrative
- "maintenance_docs" → "rag_documents"
- CLI examples updated to HR/finance/IT

---

### T-RAG02: Generic Metadata Schema
**Status**: ✅ COMPLETED  
**Description**: Remove `device_type` and `device_category` from all metadata schemas. Replace with generic `category` field.  
- `ingest_documents.py` metadata updated
- All `device_type` references removed from schema sections

---

### T-RAG03: Generic Query Processing Code
**Status**: ✅ COMPLETED  
**Description**: Remove device-type filter from `RAGService.query()`. Replace with optional generic `category_filter`.  
- `query(device_filter=None)` → `query(category_filter=None)`
- Streamlit sidebar updated

---

### T-RAG04: Generic System & User Prompts
**Status**: ✅ COMPLETED  
**Description**: Replace maintenance-assistant system prompt with a generic assistant prompt.  
- System prompt: "helpful maintenance assistant for field engineers" → "helpful assistant"
- User prompt template updated

---

### T-RAG05: Correct Context Window Budget
**Status**: ✅ COMPLETED  
**Description**: Fix incorrect "4K tokens for Llama 3.2" statement.  
- Removed "4K context" claim
- Added practical 8K operating budget table
- `num_ctx: 8192`, `max_tokens: 1024` added to Ollama config

---

### T-RAG06: Generic Architecture Diagrams & Prompts
**Status**: ✅ COMPLETED  
**Description**: Update labels in ASCII architecture diagrams.  
- "Field Engineer" box → "User"
- Diagram prompt text blocks updated

---

### T-RAG07: Add User Management Specification
**Status**: ✅ COMPLETED  
**Description**: Added new "User Management" section to RAG_USECASE.md.  
- Data model, REST API endpoints table, frontend behavior, deployment bootstrap all documented
- Table of Contents updated

---

### T-RAG08: Update EnterpriseRAG Backend — User Management API
**Status**: ✅ COMPLETED  
**Description**: Implemented user management endpoints in the EnterpriseRAG FastAPI backend.  
- `users` table added to MetadataDB with all user fields
- `POST /api/auth/login` — returns JWT + `must_change_password`
- `POST /api/auth/change-password` — self-service, clears flag
- `GET /api/users`, `POST /api/users`, `DELETE /api/users/{id}`, `POST /api/users/{id}/reset-password` (admin-only)
- JWT via `python-jose[cryptography]` + bcrypt via `passlib`
- `_bootstrap_admin()` integrated into FastAPI `lifespan` startup
- `device_type` → `category` across all backend files

---

### T-RAG09: Update EnterpriseRAG Frontend — User Management UI
**Status**: ✅ COMPLETED  
**Description**: Updated the React/Vite EnterpriseRAG frontend.  
- `LoginPage` — no register link, redirects to `/change-password` if flag is set
- `ChangePasswordPage` — forced on first login, sign-out link when forced
- `UsersPage` — admin-only; Create User shows one-time password; Reset Password per row shows new password; Deactivate button
- `AuthContext` — JWT stored in localStorage, 401 interceptor auto-redirects to `/login`
- `RequireAuth` / `RequireAdmin` route guards
- Navbar shows username, Change Password link, Sign Out; Users nav link visible to admins only
- `device_type` → `category` in all components, API calls, and AnalyticsPage

---

## Completion Criteria — ALL MET

- RAG_USECASE.md contains zero maintenance/field-engineer/device-type references ✅
- All code blocks use generic metadata (no `device_type`) ✅
- Context window section correctly references 128K (llama3.2:3b) with practical 8K budget table ✅
- New User Management section covers all 4 sub-areas (data model, API, frontend, bootstrap) ✅
- EnterpriseRAG backend has working user management endpoints ✅
- EnterpriseRAG frontend enforces must_change_password flow ✅
