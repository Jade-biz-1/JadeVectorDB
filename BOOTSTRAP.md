# JadeVectorDB - Bootstrap Guide for New Sessions

## 🎯 Purpose
This document helps you (Claude) quickly get up to speed when starting a new session with this project.

---

## 🚨 ARCHITECTURE POLICY: NO DUPLICATE SERVICES

**CRITICAL RULE: Never create duplicate or parallel systems for the same functionality!**

### What This Means:
- **One authentication system only**: Use `AuthenticationService` (services/authentication_service.h)
- **NEVER create alternatives**: Do not create `AuthManager`, `AuthSystem`, `AuthService`, etc.
- **One implementation per feature**: If a service exists, extend it - don't create a new one
- **Check before creating**: Always search for existing implementations before writing new code

### Why This Matters:
- Having two authentication systems caused significant technical debt
- Wasted development time maintaining duplicate code
- Confusion about which system to use
- Difficult migration and cleanup work

### The AuthManager Lesson:
In December 2025, we discovered two parallel authentication systems:
1. `lib/auth.h` / `lib/auth.cpp` - `AuthManager` class (old, deprecated)
2. `services/authentication_service.h` / `services/authentication_service.cpp` - `AuthenticationService` class (current)

This duplication was **completely unwanted** and required extensive cleanup work to stub out AuthManager and consolidate on AuthenticationService.

**Action Required**: Before implementing ANY new service or major feature:
1. Search the codebase for existing implementations
2. If found, extend or refactor the existing code
3. If creating new, use clear, unique names and document in BOOTSTRAP.md
4. Never create "alternative" or "parallel" implementations

---

## ⚠️ CRITICAL: Build System

**NEVER use `make` or `cmake` directly! ALWAYS use the build script!**

### The ONLY Correct Way to Build:

```bash
# Must run from backend/ directory
cd backend
./build.sh
```

### Common Build Commands:

```bash
# Standard build (Release, with tests and benchmarks)
cd backend && ./build.sh

# Clean build (removes build directory first)
cd backend && ./build.sh --clean

# Debug build
cd backend && ./build.sh --type Debug --clean

# Production build (no tests/benchmarks, optimized)
cd backend && ./build.sh --no-tests --no-benchmarks

# Fast build (limit parallel jobs)
cd backend && ./build.sh --jobs 2
```

### Build Script Must-Knows:

1. **Always run from `backend/` directory**
2. **Script location**: `backend/build.sh`
3. **Output**: `backend/build/jadevectordb` (executable)
4. **First build**: ~12 minutes (fetches dependencies)
5. **Incremental**: ~1 minute
6. **All dependencies built from source** (no apt-get needed!)

### Known Build Issues:

⚠️ **Tests have compilation errors** - Use `--no-tests --no-benchmarks`:
```bash
cd backend && ./build.sh --no-tests --no-benchmarks
```

⚠️ **Runtime crash on startup**: Duplicate route handlers in `rest_api.cpp`
- Binary builds successfully
- But crashes on startup with "handler already exists for /v1/databases"
- Related to distributed system integration work

---

## 📚 Essential Documentation Files

**Read these in order when starting a session:**

1. **`BUILD.md`** (root) - **CRITICAL: Main build system guide - READ THIS for all build-related information**
2. **`docs/COMPLETE_BUILD_SYSTEM_SETUP.md`** - Complete build system overview
3. **`backend/BUILD_QUICK_REFERENCE.md`** - Quick command reference
4. **`RECOVERY_SUMMARY.md`** - Last session status (if exists)
5. **`TasksTracking/status-dashboard.md`** - **Current focus and recent work - CHECK THIS FIRST**
6. **`TasksTracking/README.md`** - **Task tracking system guide - ALWAYS update status when completing tasks**
7. **`TasksTracking/overview.md`** - **Overall project progress and completion status**

### Quick Reference:

| File | Purpose |
|------|---------|
| `BUILD.md` (root) | **Main build documentation - READ THIS FIRST for builds** |
| `docs/COMPLETE_BUILD_SYSTEM_SETUP.md` | Build system overview & features |
| `backend/BUILD_QUICK_REFERENCE.md` | Quick build commands |
| `backend/build.sh` | **THE BUILD SCRIPT** |
| `RECOVERY_SUMMARY.md` | Last session recovery summary |
| `DEVELOPER_GUIDE.md` | Developer onboarding |
| `TasksTracking/status-dashboard.md` | **Current focus and status - CHECK THIS** |
| `TasksTracking/overview.md` | Overall progress and milestones |
| `TasksTracking/06-current-auth-api.md` | **Current work: Auth & API tasks** |
| `TasksTracking/README.md` | Task tracking system organization |

---

## 📁 Project Structure

```
JadeVectorDB/
├── backend/                    # Main C++ codebase
│   ├── src/                   # Source code
│   │   ├── api/              # API layer (REST, gRPC)
│   │   ├── services/         # Core services
│   │   ├── models/           # Data models
│   │   └── utils/            # Utilities
│   ├── tests/                # Unit & integration tests
│   ├── build.sh              # ⚠️ THE BUILD SCRIPT - USE THIS!
│   ├── CMakeLists.txt        # CMake configuration
│   └── BUILD*.md             # Build documentation
├── frontend/                  # Next.js Web UI
│   ├── src/                  # React components
│   └── public/               # Static assets
├── cli/                       # CLI tools
│   ├── python/               # Python CLI
│   └── shell/                # Shell CLI
├── docs/                      # Documentation
│   ├── COMPLETE_BUILD_SYSTEM_SETUP.md
│   └── archive/              # Archived documentation
├── TasksTracking/             # **TASK TRACKING - SINGLE SOURCE OF TRUTH**
│   ├── README.md             # Task system guide
│   ├── status-dashboard.md   # Current focus (CHECK THIS FIRST!)
│   ├── overview.md           # Overall progress
│   ├── 01-setup-foundational.md
│   ├── 02-core-features.md
│   ├── 03-advanced-features.md
│   ├── 04-monitoring-polish.md
│   ├── 05-tutorial.md
│   ├── 06-current-auth-api.md  # Current work
│   ├── 07-backend-core.md
│   ├── 08-distributed-completion.md
│   └── 09-distributed-tasks.md
├── specs/                     # Specifications
├── BUILD.md                   # Main build guide (root)
├── RECOVERY_SUMMARY.md        # Last session status
└── BOOTSTRAP.md               # This file
```

---

## 🚀 Quick Start for New Sessions

### Step 1: Ask the User
```
"Should I load the bootstrap document to understand the project setup?"
```

### Step 2: Check Project Status
```bash
# Check uncommitted changes
git status

# Check recent commits
git log --oneline -5

# Check current branch
git branch
```

### Step 3: Read Recovery Summary (if exists)
```bash
cat RECOVERY_SUMMARY.md
```

### Step 4: Build the Project (Correct Way!)
```bash
cd backend && ./build.sh --no-tests --no-benchmarks
```

### Step 5: Ask What to Work On
```
"I've reviewed the project status. What would you like to work on today?"
```

---

## 🌐 API Configuration

**Important API Notes:**

- API endpoints may be versioned: Check if API is accessible as `/api/v1/...` instead of just `/api/...`
- Default port: 8080
- REST API uses Crow framework
- gRPC support is optional (disabled by default)

**Example API endpoints to verify:**
```bash
# May be /api/v1/databases or /v1/databases
curl http://localhost:8080/api/v1/health
curl http://localhost:8080/v1/health

# Always check actual routes in backend/src/api/rest/
```

---

## 🔐 Authentication & Security

**CRITICAL: Password Requirements**

The authentication system enforces the following password rules:
- **Minimum length**: 10 characters
- **Strong passwords required** in production mode

**Default User Credentials (Development/Test Only)**

When `JADE_ENV` is set to `development`, `dev`, `test`, `testing`, or `local`, the system automatically creates default users:

| Username | Password | Roles | User ID |
|----------|----------|-------|---------|
| `admin` | `Admin@123456` | admin, developer, user | user_admin_default |
| `dev` | `Developer@123` | developer, user | user_dev_default |
| `test` | `Tester@123456` | tester, user | user_test_default |

**IMPORTANT**: These default users are ONLY created in non-production environments and are NOT created in production mode.

**Authentication Service**

The project uses **AuthenticationService** (located in `backend/src/services/authentication_service.h`) as the single source of truth for authentication:
- User registration and login
- Session management
- API key management
- Password hashing and validation
- Default user seeding (dev/test only)

**Common Authentication Mistakes to Avoid:**
1. ❌ **DON'T** use passwords shorter than 10 characters
2. ❌ **DON'T** use simple passwords like "admin123" (too short!)
3. ❌ **DON'T** expect default users in production mode
4. ✅ **DO** use strong passwords that meet minimum requirements (10+ chars)
5. ✅ **DO** set `JADE_ENV=development` for local testing to enable default users
6. ✅ **DO** use AuthenticationService (NOT AuthManager) for all auth operations

---

## 📊 Current Project Status (as of Dec 11, 2025)

### Overall Progress: ~91.6% complete (283/309 tasks)

### Completion Status:
- ✅ **Core vector database**: 100% complete (US1-US4)
- ✅ **Advanced features**: 100% complete (US5-US9, embedding, lifecycle)
- ✅ **Monitoring & polish**: 100% complete
- ✅ **Build system**: 100% functional (self-contained, all deps from source)
- ✅ **Documentation**: Complete
- ✅ **CLI tools**: 100% complete (Python & shell with cURL generation)
- 🔄 **Authentication & API**: 60% complete (Phase 14 - current focus)
- 🔄 **Backend core fixes**: 60% complete (Phase 15)
- 🔄 **Distributed system**: ~53% complete (worker service completion in progress)
- 🔄 **Interactive tutorial**: 83% complete (core done, enhancements pending)
- ✅ **Frontend basic UI**: Complete (dashboard, databases, users, monitoring)

### Current Branch:
- `run-and-fix`

### Current Focus (Phase 14):
1. **Authentication handlers** - ✅ Complete (T219-T222)
2. **User management** - ✅ Complete (T220)
3. **API key management** - ✅ Complete (T221)
4. **Security audit routes** - ✅ Complete (T222)
5. **Backend tests** - ⏳ In progress (T231-T232, T237)
6. **Remaining API routes** - ⏳ Pending (T223-T225)

### Known Issues:
1. **Tests** - Have compilation errors (use `--no-tests --no-benchmarks`)
2. **distributed_worker_service.cpp** - Incomplete stubs (~40% complete, T259)
3. **Database ID mismatch** - IDs in list response don't match get endpoint
4. **AuthManager cleanup needed** - Old AuthManager code still exists, needs removal (use AuthenticationService instead)

### Recent Work (Last 7 Days):
- ✅ Completed all authentication endpoints (T219-T222)
- ✅ Implemented user management handlers (T220)
- ✅ Implemented API key management (T221)
- ✅ Implemented security audit routes (T222)
- ✅ Built shadcn-based authentication UI (T227)
- ✅ Environment-specific default user seeding (T236)
- ✅ Fixed ShardState enum comparison errors in distributed_worker_service.cpp
- ✅ Fixed authentication system - consolidated to AuthenticationService only
- ✅ Fixed default user passwords to meet 10-character minimum requirement
- ✅ Added list_users() and list_api_keys() methods to AuthenticationService
- ✅ Verified end-to-end authentication flow (login working)

---

## 🔧 Development Patterns

### Error Handling:
```cpp
// Return errors using Result<T> type
return tl::make_unexpected(ErrorHandler::create_error(
    ErrorCode::INTERNAL_ERROR,
    "Error message"
));
```

### Logging:
```cpp
// Initialize logger
logger_ = logging::LoggerManager::get_logger("ServiceName");

// Use logger
logger_->info("message");
logger_->error("error message");
```

### Type Conventions:
- Use `SearchResults` (plural) for result structs
- Use `vector_count` not `record_count`
- Use `primary` not `is_primary`

---

## ⚠️ MANDATORY: Documentation Update Protocol

**CRITICAL REQUIREMENT - MUST DO AFTER EVERY CHANGE:**

After **ANY** of the following activities, you **MUST** update documentation:
- ✅ Code implementation
- ✅ Defect fixing
- ✅ Feature completion
- ✅ Task completion
- ✅ Configuration changes
- ✅ Architecture decisions

**Required Updates (in order):**

1. **BOOTSTRAP.md** (this file)
   - Update "Current Project Status" section with new progress
   - Update "Known Issues" if bugs were fixed
   - Add new sections for significant features
   - Update "Last Updated" date at bottom
   - Add new password requirements, auth changes, etc.

2. **TasksTracking/** (Task tracking system)
   - Mark tasks as completed in appropriate file (e.g., `06-current-auth-api.md`)
   - Update `status-dashboard.md` with recent completions
   - Update `overview.md` with new completion percentages
   - Move completed items from "In Progress" to "Completed"

3. **README.md** (if needed)
   - Update feature list if new features added
   - Update installation steps if build process changed
   - Update API examples if endpoints changed
   - Update version number if applicable

**Example Workflow:**
```
1. Fix authentication bug
2. Update BOOTSTRAP.md: Remove from "Known Issues"
3. Update TasksTracking/06-current-auth-api.md: Mark T231 as completed
4. Update TasksTracking/status-dashboard.md: Add to recent completions
5. Update TasksTracking/overview.md: Increment completion count
6. Check README.md: Update authentication section if needed
```

**This is NOT optional - it's a critical part of completing any task!**

---

## ❌ Common Mistakes to Avoid

1. ❌ **DON'T** run `make` directly
2. ❌ **DON'T** run `cmake` directly (except through build.sh)
3. ❌ **DON'T** forget to run from `backend/` directory
4. ❌ **DON'T** build with tests (they have errors): use `--no-tests --no-benchmarks`
5. ❌ **DON'T** use `docker-compose` (obsolete) - use `docker compose` instead
6. ❌ **DON'T** create unnecessary summaries - ask user first if summary is needed
7. ❌ **DON'T** forget to update TasksTracking files when completing tasks
8. ❌ **DON'T** update tasks in old locations (specs/002-check-if-we/tasks.md is archived)
9. ❌ **DON'T** skip documentation updates after completing work (see Mandatory Documentation Update Protocol above)
10. ✅ **DO** use `cd backend && ./build.sh`
11. ✅ **DO** read `BUILD.md` (root) for ALL build-related information
12. ✅ **DO** use `TodoWrite` for complex tasks
13. ✅ **DO** check `RECOVERY_SUMMARY.md` when resuming work
14. ✅ **DO** verify API endpoint versions (/api/v1 vs /v1)
15. ✅ **DO** update BOOTSTRAP.md, TasksTracking, and README.md after EVERY change
16. ✅ **DO** update TasksTracking/status-dashboard.md when completing tasks
17. ✅ **DO** check TasksTracking/06-current-auth-api.md for current work

---

## 🔄 Session Workflow

1. **Greet user** and ask to load bootstrap
2. **Read** `TasksTracking/status-dashboard.md` to understand current focus
3. **Read** `RECOVERY_SUMMARY.md` if it exists
4. **Read** `BUILD.md` for build-related information (if needed)
5. **Check** git status and recent commits
6. **Ask** user what they want to work on
7. **Create** todo list with `TodoWrite` for complex tasks
8. **Build** using `cd backend && ./build.sh --no-tests --no-benchmarks`
9. **Test** changes (if tests are fixed)
10. **⚠️ MANDATORY: Update documentation** after completing work:
    - Update `BOOTSTRAP.md`: Status, known issues, new features
    - Update `TasksTracking/` files: Mark tasks complete, update progress
    - Update `README.md`: If features/API/build process changed
    - **This is REQUIRED, not optional!**
11. **Ask before creating summaries** - don't create unnecessary summaries unless requested
12. **Commit** when user asks

---

## 🐳 Docker (Alternative to Local Build)

```bash
# Build Docker image (from project root)
docker build -t jadevectordb:latest .

# Run
docker run -p 8080:8080 jadevectordb:latest

# With docker compose (note: docker compose, NOT docker-compose)
docker compose up -d

# Distributed cluster
docker compose -f docker-compose.distributed.yml up -d
```

**Note**: Docker uses the same `build.sh` internally!
**Important**: Use `docker compose` (with space), NOT `docker-compose` (with hyphen). The latter is obsolete.

---

## 📈 Build Performance

| Configuration | First Build | Incremental |
|--------------|-------------|-------------|
| Standard (no gRPC) | ~12 min | ~1 min |
| Minimal (no tests/benchmarks) | ~8 min | ~30 sec |
| With gRPC | ~40 min | ~1 min |

**Why first build is slow**: All dependencies (Eigen, FlatBuffers, Crow, Google Test, Benchmark, Arrow) are fetched and compiled from source!

---

## 🎯 Next Session Checklist

When starting a new session, verify:

- [ ] Did I ask user about loading bootstrap?
- [ ] Did I read `TasksTracking/status-dashboard.md` for current focus?
- [ ] Did I read `RECOVERY_SUMMARY.md`?
- [ ] Did I check git status?
- [ ] Did I read `BUILD.md` (root) for build-related information?
- [ ] Am I in the `backend/` directory for building?
- [ ] Am I using `./build.sh` (NOT make or cmake)?
- [ ] Am I using `--no-tests --no-benchmarks` to avoid test errors?
- [ ] Did I create a todo list for complex tasks?
- [ ] **⚠️ Did I update BOOTSTRAP.md after completing work?**
- [ ] **⚠️ Did I update TasksTracking files when completing tasks?**
- [ ] **⚠️ Did I check if README.md needs updates?**
- [ ] Did I avoid creating unnecessary summaries (ask user first)?
- [ ] Did I verify API endpoint versions (/api/v1 vs /v1)?
- [ ] Am I using `docker compose` (NOT `docker-compose`)?

---

## 🛠️ Troubleshooting

| Problem | Solution |
|---------|----------|
| Out of memory | `./build.sh --jobs 2` |
| Slow build | `./build.sh --no-tests --no-benchmarks` |
| Build fails | `./build.sh --clean` |
| Test compilation errors | Use `--no-tests --no-benchmarks` |
| Runtime crash | Known issue: duplicate route handlers |

---

## 📞 Key Reminders

1. **⚠️ MANDATORY**: Update BOOTSTRAP.md, TasksTracking, and README.md after EVERY change!
2. **Build System**: Self-contained, fetches all dependencies from source
3. **Build Script**: `cd backend && ./build.sh`
4. **Build Documentation**: Read `BUILD.md` (root) for ALL build information
5. **Tests**: Currently broken, use `--no-tests --no-benchmarks`
6. **Recovery**: Check `RECOVERY_SUMMARY.md` when resuming
7. **Current Focus**: Check `TasksTracking/status-dashboard.md` FIRST
8. **Task Tracking**: Update TasksTracking files when completing tasks (NOT old specs/002-check-if-we/tasks.md)
9. **Task System**: TasksTracking is the SINGLE SOURCE OF TRUTH for all tasks
10. **Todo List**: Use `TodoWrite` for complex, multi-step tasks
11. **Docker**: Use `docker compose` (NOT `docker-compose` - it's obsolete)
12. **API Versioning**: Check if endpoints are /api/v1 or /v1
13. **Summaries**: Don't create unnecessary summaries - ask user first
14. **Git Branch**: Currently on `run-and-fix`
15. **Project Progress**: ~91.6% complete (283/309 tasks)
16. **Authentication**: Use AuthenticationService (NOT AuthManager)
17. **Passwords**: Minimum 10 characters (see Authentication section)

---

## 📝 Dependencies (All Auto-Fetched)

The build system automatically fetches and compiles:
- Eigen 3.4.0 (linear algebra)
- FlatBuffers v23.5.26 (serialization)
- Crow v1.0+5 (REST API framework)
- Google Test v1.14.0 (testing)
- Google Benchmark v1.8.3 (benchmarks)
- Apache Arrow 14.0.0 (columnar format)
- gRPC v1.60.0 (optional, disabled by default)

**No manual installation needed!**

---

**Last Updated**: December 11, 2025
**Current Branch**: run-and-fix
**Build System Version**: Self-contained FetchContent-based

**Critical Reminders**:
- ⚠️ **MANDATORY**: Update BOOTSTRAP.md, TasksTracking, and README.md after EVERY change!
- Use `cd backend && ./build.sh`, NOT `make` or `cmake` directly!
- Read `BUILD.md` (root) for ALL build-related information
- Update `TasksTracking/` files when completing tasks (NOT old specs/002-check-if-we/tasks.md)
- Use `docker compose` (NOT `docker-compose`)
- Check API versioning (/api/v1 vs /v1)
- Don't create unnecessary summaries - ask user first!
- Use AuthenticationService (NOT AuthManager) - single source of truth for auth
- Passwords must be minimum 10 characters (see Authentication section)
