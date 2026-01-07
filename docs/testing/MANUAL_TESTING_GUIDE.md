# Manual Testing Guide - Persistent Vector Storage

**Date:** December 29, 2025
**Sprint:** 2.3 - Advanced Persistence Features (COMPLETE)
**Latest Features:** Admin Shutdown Endpoint + Vector Listing Endpoint
**Status:** Ready for Comprehensive Manual Testing

---

## 🚀 Getting Started - READ THIS FIRST!

Before you begin testing, you need to start both the backend server and frontend application, and understand the default login credentials.

### Prerequisites

Ensure you have:
- [x] JadeVectorDB backend compiled (`backend/build/jadevectordb` executable exists)
- [x] Node.js installed (for frontend)
- [x] Terminal access to the project directory

### Step 1: Start the Backend Server

**IMPORTANT**: The backend must be started with the development environment variable to enable default test users.

```bash
# Navigate to backend build directory
cd /home/deepak/Public/JadeVectorDB/backend/build

# Set environment to development (enables default users)
export JADEVECTORDB_ENV=development

# Start the backend server with logs to console
# Note: Application logs go to logs/jadevectordb.log by default
# To see logs in console, tail the log file in another terminal:
# tail -f logs/jadevectordb.log
./jadevectordb
```

**Expected Output:**
```
[INFO] JadeVectorDB Server Starting...
[INFO] Environment: development
[INFO] Creating default users for development environment
[INFO] Default user 'admin' created successfully
[INFO] Default user 'dev' created successfully
[INFO] Default user 'test' created successfully
[INFO] Crow/1.0 server is running at http://0.0.0.0:8080 using 24 threads
[INFO] Server started successfully
```

**Alternative**: Run in background
```bash
# Start backend in background
export JADEVECTORDB_ENV=development
cd /home/deepak/Public/JadeVectorDB/backend/build
./jadevectordb &

# Note the process ID for later shutdown
echo $! > jadevectordb.pid
```

**Health Check:**
```bash
# Verify backend is running
curl http://localhost:8080/health

# Expected response: {"status": "healthy"}
```

---

### Step 2: Start the Frontend Application

Open a **new terminal** and run:

```bash
# Navigate to frontend directory
cd /home/deepak/Public/JadeVectorDB/frontend

# Install dependencies (first time only)
npm install

# Start the development server
npm run dev
```

**Expected Output:**
```
> jadevectordb-frontend@0.1.0 dev
> next dev -p 3003

  ▲ Next.js 14.2.33
  - Local:        http://localhost:3003
  - Ready in 2.3s
```

**Access the UI:**
Open your browser and navigate to: **http://localhost:3003**

---

### Step 3: Default User Credentials

**⚠️ IMPORTANT**: Default users are **ONLY** created in development/test environments. They will **NOT** be available in production.

The following default users are automatically created when `JADEVECTORDB_ENV` is set to `development`, `dev`, `test`, `testing`, or `local`:

| Username | Password | Roles | User ID | Use Case |
|----------|----------|-------|---------|----------|
| **admin** | `admin123` | admin, developer, user | user_admin_default | Full system access |
| **dev** | `dev123` | developer, user | user_dev_default | Development tasks |
| **test** | `test123` | tester, user | user_test_default | Testing workflows |

**Note:** These are simplified development passwords. In production, use strong passwords with proper complexity requirements.

**Login Steps:**
1. Open http://localhost:3003
2. Click **"Login"** (if not redirected automatically)
3. Enter username: `admin`
4. Enter password: `admin123`
5. Click **"Sign In"**

**First-Time Setup (If Default Users Not Created):**

If you forgot to set `JADEVECTORDB_ENV=development`, default users won't exist. You have two options:

**Option 1: Restart with Environment Variable (Recommended)**
```bash
# Stop backend
pkill jadevectordb

# Restart with development environment
export JADEVECTORDB_ENV=development
cd /home/deepak/Public/JadeVectorDB/backend/build
./jadevectordb
```

**Option 2: Register New User**
1. On login page, click **"Register"** or **"Sign Up"**
2. Create account with strong password (10+ chars)
3. Login with new credentials

---

### Step 4: Verify System is Ready

Before starting tests, verify everything is working:

**✅ Backend Health Check:**
```bash
curl http://localhost:8080/health
# Expected: {"status": "healthy"}
```

**✅ Frontend Accessible:**
- Open http://localhost:3003 in browser
- Should see JadeVectorDB login page

**✅ Can Login:**
- Login with `admin` / `admin123`
- Should see dashboard after successful login

**✅ Check Logs:**
```bash
# Backend logs (in terminal where backend is running)
# Should see no ERROR messages

# Check log files (if configured)
tail -f /home/deepak/Public/JadeVectorDB/backend/build/logs/*.log
```

**✅ Test API Access:**
```bash
# Test authentication endpoint
curl -X POST http://localhost:8080/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"admin123"}'

# Should return: {"token":"<jwt-token>","user":{...}}
```

---

### Troubleshooting Startup Issues

**Issue: Backend won't start**
```bash
# Check if port 8080 is already in use
lsof -i :8080

# Kill existing process if needed
pkill jadevectordb

# Check for build errors
cd /home/deepak/Public/JadeVectorDB/backend
./build.sh --no-tests --no-benchmarks
```

**Issue: Default users not created**
```bash
# Verify environment variable is set
echo $JADEVECTORDB_ENV
# Should output: development

# If not set, restart backend with:
export JADEVECTORDB_ENV=development
./jadevectordb
```

**Issue: Frontend won't start**
```bash
# Check if port 3003 is in use
lsof -i :3003

# Reinstall dependencies
cd /home/deepak/Public/JadeVectorDB/frontend
rm -rf node_modules package-lock.json
npm install
npm run dev
```

**Issue: Cannot login**
- Check backend is running: `curl http://localhost:8080/health`
- Verify password meets requirements (10+ chars, uppercase, lowercase, digit, special)
- Check backend logs for authentication errors
- Try resetting: stop backend, delete `data/jadevectordb_auth.db`, restart with `JADEVECTORDB_ENV=development`

---

### Quick Commands Reference

```bash
# START BACKEND (Development Mode)
export JADEVECTORDB_ENV=development
cd /home/deepak/Public/JadeVectorDB/backend/build
./jadevectordb

# START FRONTEND
cd /home/deepak/Public/JadeVectorDB/frontend
npm run dev

# STOP BACKEND (Graceful)
# Option 1: Use the admin shutdown endpoint (recommended - NEW!)
# First, get admin token
TOKEN=$(curl -s -X POST http://localhost:8080/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"admin123"}' | jq -r '.token')
# Then shutdown
curl -X POST http://localhost:8080/admin/shutdown \
  -H "Authorization: Bearer $TOKEN"
# Option 2: Use the dashboard shutdown button (admin users only)
# Login to http://localhost:3003/dashboard and click "Shutdown Server" button
# Option 3: Send SIGTERM signal
pkill -SIGTERM jadevectordb
# Option 4: Press Ctrl+C in the backend terminal (may hang - avoid if possible)

# STOP FRONTEND
# Press Ctrl+C in terminal where npm is running

# CHECK RUNNING PROCESSES
ps aux | grep jadevectordb
ps aux | grep "next dev"

# VIEW LOGS
tail -f /home/deepak/Public/JadeVectorDB/backend/build/logs/*.log
```

---

## System Status

### Backend
- **URL:** http://localhost:8080
- **Status:** ✅ Running (Crow web server with 24 threads)
- **Health:** http://localhost:8080/health
- **Metrics:** http://localhost:8080/metrics

### Frontend
- **URL:** http://localhost:3003
- **Status:** ✅ Running (Next.js development server)
- **Framework:** Next.js 14.2.33

### Storage Configuration
- **Database Path:** `./data/jadevectordb.db` (SQLite metadata)
- **Vector Storage Path:** `./data/jadevectordb/databases/` (`.jvdb` files - not created yet)
- **Working Directory:** `/home/deepak/Public/JadeVectorDB/backend/build/`

---

## What We're Testing

**Sprint 2.1** implemented **persistent vector storage** with:
- Memory-mapped `.jvdb` files for vector data
- Cross-platform support (Unix mmap)
- LRU cache for efficient file descriptor management
- Automatic periodic flushing (5-minute default)
- Graceful shutdown with signal handlers

**Sprint 2.2** added **compaction and bulk operations**:
- File compaction to reclaim deleted space
- Bulk vector operations for efficiency
- Enhanced storage optimization

**Sprint 2.3** added **advanced persistence features** (100% complete, 18/18 tests passing):
- **Index Resize**: Automatic capacity growth at 75% utilization (bug fixed ✅)
- **Free List**: Space reuse for deleted vectors (reduces fragmentation 50%+)
- **Write-Ahead Log (WAL)**: Crash recovery with CRC32 checksums
- **Snapshot Manager**: Point-in-time backups with checksum verification
- **Persistence Statistics**: Thread-safe operation tracking
- **Data Integrity Verifier**: Index validation and repair functionality
- **Database Listing**: Enables background compaction automation

**Recent Enhancements** (December 26-29, 2025):
- **Admin Shutdown Endpoint**: Graceful server shutdown via authenticated API
- **Vector Listing Endpoint**: Enhanced vector management with listing capabilities
- **Frontend Improvements**: Better vector management UI with enhanced display

---

## Test Scenarios

### Test 1: Basic Database Creation
**Goal:** Verify that databases can be created and `.jvdb` files are generated

**Steps:**
1. Open frontend: http://localhost:3003
2. Navigate to **Databases** page
3. Click "Create Database"
4. Fill in details:
   - Name: `test_persistence`
   - Dimension: 512
   - Distance Metric: Cosine
   - Index Type: FLAT
5. Submit

**Expected Results:**
- ✅ Database created successfully
- ✅ File created: `./data/jadevectordb/databases/{uuid}/vectors.jvdb`
- ✅ Database appears in list

**Verification Commands:**
```bash
# Check if data directory was created
ls -la /home/deepak/Public/JadeVectorDB/backend/build/data/

# Look for .jvdb files
find /home/deepak/Public/JadeVectorDB/backend/build/data/ -name "*.jvdb" -ls
```

---

### Test 2: Store Vectors
**Goal:** Verify vectors can be stored and retrieved

**Steps:**
1. Navigate to the created database
2. Click "Add Vector" or "Store Vector"
3. Enter vector data:
   - Vector ID: `1`
   - Data: 512 random float values (or use UI generator)
   - Metadata: `{"label": "test", "category": "persistence"}`
4. Submit
5. Repeat for 5-10 more vectors

**Expected Results:**
- ✅ Vectors stored successfully
- ✅ `.jvdb` file size increases
- ✅ Vectors appear in database vector list
- ✅ Can retrieve individual vectors by ID

**Verification Commands:**
```bash
# Check file size growth
ls -lh $(find /home/deepak/Public/JadeVectorDB/backend/build/data/ -name "*.jvdb")

# Check backend logs for successful writes
# (Look in terminal where backend is running)
```

---

### Test 3: Restart Persistence (CRITICAL TEST)
**Goal:** Verify data survives server restart

**Steps:**
1. Note current vectors in database (count and IDs)
2. Stop backend server:
   ```bash
   # Find process ID
   ps aux | grep jadevectordb
   
   # Kill gracefully (allows flush)
   kill <PID>
   ```
3. Verify server stopped
4. Restart backend:
   ```bash
   cd /home/deepak/Public/JadeVectorDB/backend/build
   ./jadevectordb &
   ```
5. Refresh frontend
6. Navigate to database
7. Check if vectors are still present

**Expected Results:**
- ✅ Server restarts successfully
- ✅ Database still exists in list
- ✅ All vectors are present (same count and IDs)
- ✅ Vector data and metadata intact
- ✅ Can retrieve vectors with exact same values

**Verification:**
- Compare vector counts before/after restart
- Verify specific vector IDs exist
- Check metadata is preserved

---

### Test 4: Multiple Databases
**Goal:** Verify independent storage for multiple databases

**Steps:**
1. Create 3 databases:
   - `db_256` (dimension: 256)
   - `db_512` (dimension: 512)
   - `db_1024` (dimension: 1024)
2. Store 10 vectors in each database
3. Verify each database has its own `.jvdb` file
4. Restart server
5. Verify all 3 databases and their vectors persist

**Expected Results:**
- ✅ 3 separate `.jvdb` files created
- ✅ Each database maintains its own vectors
- ✅ No data leakage between databases
- ✅ All survive restart

**Verification Commands:**
```bash
# Count .jvdb files (should be 3)
find /home/deepak/Public/JadeVectorDB/backend/build/data/ -name "*.jvdb" | wc -l

# Check sizes of each file
find /home/deepak/Public/JadeVectorDB/backend/build/data/ -name "*.jvdb" -exec ls -lh {} \;
```

---

### Test 5: Update and Delete Operations
**Goal:** Verify updates and deletes persist

**Steps:**
1. Store vector with ID `100`
2. Note its data and metadata
3. Update vector `100` with new data
4. Restart server
5. Verify vector `100` has updated data
6. Delete vector `100`
7. Restart server
8. Verify vector `100` is gone

**Expected Results:**
- ✅ Updates persist across restart
- ✅ Deletes persist across restart
- ✅ File size adjusts appropriately

---

### Test 6: Large Dataset (Optional - Time Permitting)
**Goal:** Verify performance with larger datasets

**Steps:**
1. Create database: `large_test`
2. Store 1,000 vectors (use batch operation if available)
3. Note storage time and file size
4. Restart server
5. Verify all 1,000 vectors present

**Expected Results:**
- ✅ Can store 1,000 vectors
- ✅ File size ~2-4MB (depending on dimension)
- ✅ Restart loads quickly (lazy loading)
- ✅ All vectors accessible

---

### Test 7: Flush Operations
**Goal:** Verify manual and automatic flushing

**Steps:**
1. Store 50 vectors
2. Wait 6 minutes (auto-flush is every 5 minutes)
3. Check backend logs for "flush" messages
4. Alternatively, trigger manual flush via API if available

**Expected Results:**
- ✅ Auto-flush occurs after 5-6 minutes
- ✅ Log messages indicate successful flush
- ✅ Data survives ungraceful shutdown after flush

---

### Test 8: Database Deletion
**Goal:** Verify `.jvdb` files are deleted when database is deleted

**Steps:**
1. Create database: `temp_test`
2. Store 10 vectors
3. Note the `.jvdb` file path
4. Delete the database via UI
5. Check if `.jvdb` file is removed

**Expected Results:**
- ✅ Database deleted from list
- ✅ `.jvdb` file removed from filesystem
- ✅ Directory cleaned up

**Verification Commands:**
```bash
# Before deletion - note the path
find /home/deepak/Public/JadeVectorDB/backend/build/data/ -name "*.jvdb"

# After deletion - verify it's gone
find /home/deepak/Public/JadeVectorDB/backend/build/data/ -name "*.jvdb"
```

---

### Test 9: Index Resize (Sprint 2.3 - Critical Bug Fixed ✅)
**Goal:** Verify automatic index capacity growth preserves data integrity

**Steps:**
1. Create database: `resize_test` (dimension: 128)
2. Store vectors until reaching ~75% capacity (monitor backend logs)
3. Store additional vectors to trigger resize
4. Verify all vectors can be retrieved with correct data
5. Check backend logs for "Resizing index" messages

**Expected Results:**
- ✅ Index automatically resizes when 75% full
- ✅ Capacity doubles (e.g., 1024 → 2048)
- ✅ All vectors preserved with correct data (no corruption)
- ✅ Hash table properly rehashed
- ✅ No allocation failures

**Note:** This test validates the fix for a critical data corruption bug where retrieved vectors contained wrong data after resize.

---

### Test 10: Write-Ahead Log (WAL) - Crash Recovery
**Goal:** Verify WAL enables crash recovery

**Steps:**
1. Enable WAL for database (check if exposed in UI or use API)
2. Store 20 vectors
3. Check for WAL file: `find data/ -name "*.wal"`
4. Simulate crash (kill -9 backend process)
5. Restart backend
6. Verify all 20 vectors are present

**Expected Results:**
- ✅ WAL file created alongside `.jvdb` file
- ✅ After crash, WAL replays operations
- ✅ All vectors recovered successfully
- ✅ No data loss

**Verification Commands:**
```bash
# Check WAL files
find /home/deepak/Public/JadeVectorDB/backend/build/data/ -name "*.wal" -ls

# Simulate crash
pkill -9 jadevectordb
```

---

### Test 11: Snapshot Manager - Backup & Restore
**Goal:** Verify snapshot creation and restoration

**Steps:**
1. Create database: `snapshot_test`
2. Store 50 vectors
3. Create snapshot via API/UI (e.g., "snapshot_v1")
4. Store 20 more vectors (total: 70)
5. List snapshots - verify "snapshot_v1" exists
6. Restore from "snapshot_v1"
7. Verify database has 50 vectors (not 70)

**Expected Results:**
- ✅ Snapshot created with checksum
- ✅ Snapshot file separate from main `.jvdb`
- ✅ Restore returns database to exact snapshot state
- ✅ Later vectors (51-70) not present after restore

---

### Test 12: Persistence Statistics
**Goal:** Verify operation tracking and statistics

**Steps:**
1. Query statistics endpoint/API (e.g., `/api/statistics`)
2. Note initial counters
3. Store 10 vectors
4. Update 3 vectors
5. Delete 2 vectors
6. Query statistics again
7. Verify counters increased correctly

**Expected Results:**
- ✅ Statistics track: stores, updates, deletes, flushes, snapshots
- ✅ Counters increment atomically (thread-safe)
- ✅ Can reset statistics
- ✅ System-wide totals available

---

### Test 13: Data Integrity Verifier
**Goal:** Verify integrity checking and repair

**Steps:**
1. Create database with 100 vectors
2. Run integrity verification via API/UI
3. Check verification results (should be clean)
4. If corruption detection available, trigger verification
5. Check if repair functionality works

**Expected Results:**
- ✅ Verifier checks index consistency
- ✅ Validates free list integrity
- ✅ Can detect and report issues
- ✅ Repair functionality available for fixable issues

---

### Test 14: Free List - Space Reuse
**Goal:** Verify deleted space is reused efficiently

**Steps:**
1. Create database: `freelist_test`
2. Store 100 vectors
3. Note `.jvdb` file size
4. Delete 50 vectors
5. Store 50 new vectors
6. Check if file size grows minimally (should reuse deleted space)

**Expected Results:**
- ✅ File size grows minimally when storing after deletes
- ✅ Deleted space appears in free list
- ✅ Adjacent free blocks merged
- ✅ Reduced fragmentation

**Verification Commands:**
```bash
# Monitor file size
watch -n 1 'ls -lh $(find data/ -name "*.jvdb" | grep freelist_test)'
```

---

## Monitoring During Tests

### Backend Logs
Watch the backend terminal output for:
- Vector store operations: `store_vector`, `retrieve_vector`
- Flush operations: `flush completed`
- File operations: `opening`, `closing` database files
- LRU eviction: If testing many databases

### File System
Monitor the data directory:
```bash
# Watch for file changes in real-time
watch -n 2 'ls -lh $(find /home/deepak/Public/JadeVectorDB/backend/build/data/ -name "*.jvdb" 2>/dev/null)'

# Monitor directory size
watch -n 2 'du -sh /home/deepak/Public/JadeVectorDB/backend/build/data/'
```

### Memory Usage
```bash
# Monitor process memory
watch -n 2 'ps aux | grep jadevectordb | grep -v grep'
```

---

## Troubleshooting

### Issue: No .jvdb files created
**Possible Causes:**
- Persistence layer not properly integrated
- Using old InMemoryDatabasePersistence instead of PersistentDatabasePersistence
- File permission issues

**Solutions:**
- Check `main.cpp` for persistence initialization
- Verify storage directory permissions
- Check backend logs for errors

### Issue: Data not persisting after restart
**Possible Causes:**
- Data not flushed before shutdown
- Corrupted `.jvdb` file
- File path mismatch

**Solutions:**
- Use graceful shutdown (SIGTERM, not SIGKILL)
- Check file integrity (magic number 0x4A564442)
- Verify storage paths match configuration

### Issue: Server crashes with many databases
**Possible Causes:**
- Exceeding file descriptor limit
- LRU eviction not working
- Memory exhaustion

**Solutions:**
- Check `ulimit -n` (should be 1024+)
- Verify `max_open_files` configuration
- Monitor memory usage

---

## Success Criteria

### Must Pass (Core Persistence - Sprint 2.1):
- [ ] Test 1: Database creation with `.jvdb` file
- [ ] Test 2: Vector storage and retrieval
- [ ] Test 3: **Data persists after restart** (CRITICAL)
- [ ] Test 4: Multiple databases work independently
- [ ] Test 5: Updates and deletes persist
- [ ] Test 8: Database deletion removes `.jvdb` file

### Must Pass (Advanced Features - Sprint 2.3):
- [ ] Test 9: **Index resize preserves data** (CRITICAL - bug fixed)
- [ ] Test 10: WAL provides crash recovery
- [ ] Test 11: Snapshot backup and restore
- [ ] Test 12: Statistics tracking works
- [ ] Test 13: Integrity verifier detects issues
- [ ] Test 14: Free list reuses space efficiently
- [ ] Test 15: **Admin shutdown endpoint** (NEW - December 26, 2025)
- [ ] Test 16: **Vector listing endpoint** (NEW - December 29, 2025)

### Should Pass:
- [ ] Test 6: Can handle 1,000+ vectors
- [ ] Test 7: Auto-flush works

### Nice to Have:
- [ ] Performance: Sub-second retrieval for 10K vectors
- [ ] Memory: Stable memory usage over time
- [ ] No crashes under normal load
- [ ] Snapshot restore completes in <5 seconds for 10K vectors
- [ ] Integrity verification runs in <1 second for small databases

---

## Notes for Testing

### Authentication
If the UI requires authentication:
1. Register a new user or use existing credentials
2. Login before creating databases
3. Ensure API key is used for requests

### UI Navigation
Typical flow:
1. Login → Dashboard
2. Dashboard → Databases
3. Databases → Create Database
4. Database Detail → Store Vectors
5. Database Detail → Search/List Vectors

### Browser Console
Keep browser console open (F12) to:
- Monitor API requests
- Check for JavaScript errors
- View response data

---

## Documentation of Results

As you test, document:
- ✅ or ❌ for each test
- Screenshots of UI states
- File paths and sizes
- Any errors encountered
- Performance observations

**Example Log Entry:**
```
Test 3: Restart Persistence
- Stored 10 vectors in test_persistence
- File: ./data/jadevectordb/databases/abc-123/vectors.jvdb (52KB)
- Restarted server at 11:30
- Result: ✅ All 10 vectors present after restart
- Notes: Startup time was instant, retrieval was fast
```

---

### Test 15: Admin Shutdown Endpoint (NEW - December 26, 2025)
**Goal:** Verify graceful server shutdown via admin endpoint

**Steps:**
1. Login as admin user to get JWT token:
   ```bash
   TOKEN=$(curl -s -X POST http://localhost:8080/v1/auth/login \
     -H "Content-Type: application/json" \
     -d '{"username":"admin","password":"admin123"}' | jq -r '.token')
   ```
2. Call shutdown endpoint:
   ```bash
   curl -X POST http://localhost:8080/admin/shutdown \
     -H "Authorization: Bearer $TOKEN"
   ```
3. Verify server shuts down gracefully

**Expected Results:**
- ✅ Response: `{"status":"shutting_down","message":"Server shutdown initiated"}`
- ✅ Server stops accepting new connections
- ✅ In-flight requests complete
- ✅ Server process exits cleanly after ~500ms
- ✅ Shutdown attempt logged in audit logs

**Security Test:**
1. Try shutdown with non-admin user (e.g., `dev` / `dev123`)
2. Should return: `{"error":"Unauthorized: admin privileges required"}`
3. Try shutdown with invalid token
4. Should return 401 Unauthorized

**Frontend Test:**
1. Login to dashboard as admin user
2. Navigate to http://localhost:3003/dashboard
3. Verify "Shutdown Server" button is visible (red button, top right)
4. Click the shutdown button
5. Confirm in the dialog
6. Verify server shuts down
7. Login as non-admin user (e.g., `dev`)
8. Verify shutdown button is NOT visible

**Verification:**
```bash
# Check audit logs for shutdown attempt
grep -i shutdown /tmp/jadedb.log

# Expected log entries:
# [INFO] Shutdown authorized by user: admin
# [INFO] Shutdown initiated successfully
# [INFO] Executing shutdown callback...
```

---

### Test 16: Vector Listing Endpoint (NEW - December 29, 2025)
**Goal:** Verify the vector listing endpoint works correctly with pagination and filtering

**Steps:**
1. Create database: `list_test` (dimension: 128)
2. Store 25 vectors with varied metadata:
   ```bash
   # Example metadata variations:
   # vec_001-010: {"category": "A", "type": "test"}
   # vec_011-020: {"category": "B", "type": "test"}
   # vec_021-025: {"category": "A", "type": "prod"}
   ```
3. Test listing via UI:
   - Navigate to database detail page
   - Verify vector list displays all 25 vectors
   - Check pagination if more than default page size
   - Test sorting by ID, timestamp, etc.
4. Test listing via API:
   ```bash
   # Get admin token
   TOKEN=$(curl -s -X POST http://localhost:8080/v1/auth/login \
     -H "Content-Type: application/json" \
     -d '{"username":"admin","password":"admin123"}' | jq -r '.token')

   # List all vectors
   curl -X GET "http://localhost:8080/v1/databases/list_test/vectors" \
     -H "Authorization: Bearer $TOKEN"

   # List with pagination (if supported)
   curl -X GET "http://localhost:8080/v1/databases/list_test/vectors?limit=10&offset=0" \
     -H "Authorization: Bearer $TOKEN"

   # List with filters (if supported)
   curl -X GET "http://localhost:8080/v1/databases/list_test/vectors?category=A" \
     -H "Authorization: Bearer $TOKEN"
   ```

**Expected Results:**
- ✅ UI displays all vectors in database
- ✅ Vector IDs, metadata, and timestamps visible
- ✅ Pagination works correctly (if implemented)
- ✅ API returns complete vector list
- ✅ Response includes total count
- ✅ Filtering by metadata works (if implemented)
- ✅ Performance: listing 1,000 vectors completes in <1 second

**Frontend Enhancement Test:**
1. Navigate to database detail page
2. Verify enhanced vector management UI:
   - Vector list with proper formatting
   - Search/filter capabilities
   - Bulk operations (if available)
   - Vector details view
3. Test responsive design (resize browser window)
4. Verify no UI errors in console (F12)

**API Response Format:**
```json
{
  "database_id": "list_test",
  "total_count": 25,
  "vectors": [
    {
      "vector_id": "vec_001",
      "metadata": {"category": "A", "type": "test"},
      "timestamp": "2025-12-29T10:30:00Z"
    },
    ...
  ]
}
```

**Verification:**
```bash
# Verify count matches stored vectors
curl -s -X GET "http://localhost:8080/v1/databases/list_test/vectors" \
  -H "Authorization: Bearer $TOKEN" | jq '.total_count'
# Expected: 25
```

---

## After Testing

### If Tests Pass:
1. Document results in manual testing report
2. Verify automated test coverage matches manual tests
3. Update deployment readiness status
4. Plan next sprint or production deployment
5. All sprints (2.1, 2.2, 2.3) are 100% complete with 26/26 automated tests passing
   - Sprint 2.2: 8/8 tests passing ✅
   - Sprint 2.3: 18/18 tests passing ✅

### If Tests Fail:
1. Document specific failures with screenshots/logs
2. Check backend logs for error messages
3. Compare with automated test results:
   - Sprint 2.2: 8/8 tests passing
   - Sprint 2.3: 18/18 tests passing
   - Total: 26/26 tests passing
4. Review persistence implementation code
5. Create bug report with reproduction steps
6. Debug with gdb or add more logging
7. Re-run automated tests to validate fix

---

## CLI Testing (Optional)

The JadeVectorDB CLI tools provide command-line interfaces for managing databases, vectors, and users. Both Python and Shell CLIs are available.

### Prerequisites

**Get an API Token:**
```bash
# Login to get authentication token
TOKEN=$(curl -s -X POST http://localhost:8080/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin123"}' | python3 -c "import sys, json; print(json.load(sys.stdin)['token'])")

echo "Your token: $TOKEN"
```

---

### Python CLI Testing

**Installation:**
```bash
cd /home/deepak/Public/JadeVectorDB/cli/python
pip install -e .
```

#### User Management

**Add a new user:**
```bash
python3 -m jadevectordb.cli --url http://localhost:8080 --api-key "$TOKEN" \
  user-add john --role developer --password "Pass123!@#" --email "john@example.com"

# Output includes user_id - save it for other commands
# Example: "user_id": "f9ff8ff325a6fd26"
```

**List all users:**
```bash
python3 -m jadevectordb.cli --url http://localhost:8080 --api-key "$TOKEN" \
  user-list
```

**Show user details:**
```bash
# Use user_id from the add-user response
python3 -m jadevectordb.cli --url http://localhost:8080 --api-key "$TOKEN" \
  user-show f9ff8ff325a6fd26
```

**Deactivate a user:**
```bash
python3 -m jadevectordb.cli --url http://localhost:8080 --api-key "$TOKEN" \
  user-deactivate f9ff8ff325a6fd26
```

**Activate a user:**
```bash
python3 -m jadevectordb.cli --url http://localhost:8080 --api-key "$TOKEN" \
  user-activate f9ff8ff325a6fd26
```

**Delete a user:**
```bash
python3 -m jadevectordb.cli --url http://localhost:8080 --api-key "$TOKEN" \
  user-delete f9ff8ff325a6fd26
```

#### Database Operations

**Create a database:**
```bash
python3 -m jadevectordb.cli --url http://localhost:8080 --api-key "$TOKEN" \
  create-db --name my_db --dimension 128 --index-type hnsw
```

**List databases:**
```bash
python3 -m jadevectordb.cli --url http://localhost:8080 --api-key "$TOKEN" \
  list-dbs
```

**Get database info:**
```bash
python3 -m jadevectordb.cli --url http://localhost:8080 --api-key "$TOKEN" \
  get-db --database-id db_1766737450595945231
```

#### Vector Operations

**Store a vector:**
```bash
python3 -m jadevectordb.cli --url http://localhost:8080 --api-key "$TOKEN" \
  store --database-id db_1766737450595945231 \
  --vector-id vec_001 \
  --values '[0.1, 0.2, 0.3, 0.4, 0.5]' \
  --metadata '{"category": "test"}'
```

**Search vectors:**
```bash
python3 -m jadevectordb.cli --url http://localhost:8080 --api-key "$TOKEN" \
  search --database-id db_1766737450595945231 \
  --query-vector '[0.1, 0.2, 0.3, 0.4, 0.5]' \
  --top-k 5
```

#### Import/Export

**Export vectors:**
```bash
python3 -m jadevectordb.cli --url http://localhost:8080 --api-key "$TOKEN" \
  export --database-id db_1766737450595945231 \
  --file /tmp/vectors_export.json
```

**Import vectors:**
```bash
python3 -m jadevectordb.cli --url http://localhost:8080 --api-key "$TOKEN" \
  import --database-id db_1766737450595945231 \
  --file /tmp/vectors_export.json
```

---

### Shell CLI Testing

The Shell CLI is located at: `/home/deepak/Public/JadeVectorDB/cli/shell/scripts/jade-db.sh`

#### User Management

**Add a new user:**
```bash
cd /home/deepak/Public/JadeVectorDB/cli/shell
./scripts/jade-db.sh --url http://localhost:8080 --api-key "$TOKEN" \
  user-add john developer "Pass123!@#" "john@example.com"

# Save the user_id from the response
```

**List all users:**
```bash
./scripts/jade-db.sh --url http://localhost:8080 --api-key "$TOKEN" \
  user-list
```

**Show user details:**
```bash
# Use user_id from the add-user response
./scripts/jade-db.sh --url http://localhost:8080 --api-key "$TOKEN" \
  user-show f9ff8ff325a6fd26
```

**Deactivate a user:**
```bash
./scripts/jade-db.sh --url http://localhost:8080 --api-key "$TOKEN" \
  user-deactivate f9ff8ff325a6fd26
```

**Activate a user:**
```bash
./scripts/jade-db.sh --url http://localhost:8080 --api-key "$TOKEN" \
  user-activate f9ff8ff325a6fd26
```

**Delete a user:**
```bash
./scripts/jade-db.sh --url http://localhost:8080 --api-key "$TOKEN" \
  user-delete f9ff8ff325a6fd26
```

#### Database Operations

**Create a database:**
```bash
./scripts/jade-db.sh --url http://localhost:8080 --api-key "$TOKEN" \
  create-db my_db "Test database" 128 hnsw
```

**List databases:**
```bash
./scripts/jade-db.sh --url http://localhost:8080 --api-key "$TOKEN" \
  list-dbs
```

**Get database info:**
```bash
./scripts/jade-db.sh --url http://localhost:8080 --api-key "$TOKEN" \
  get-db db_1766737450595945231
```

---

### Automated CLI Test Suite

To run the comprehensive CLI test suite:

```bash
cd /home/deepak/Public/JadeVectorDB
python3 tests/run_cli_tests.py
```

This will test:
- Python CLI (database, vector, user operations)
- Shell CLI (database, vector, user operations)
- Persistence (data survives across operations)
- RBAC (role-based access control)
- Import/Export functionality

**Expected Results:**
- 36/36 tests should pass (as of December 29, 2025)
  - Python CLI: 7 tests (health, status, databases, vectors)
  - Shell CLI: 5 tests (health, status, databases)
  - Persistence: 3 tests (user and database persistence)
  - RBAC: 5 tests (role-based access control)
  - User Management (Python): 6 tests
  - Import/Export (Python): 2 tests
  - User Management (Shell): 6 tests
  - Import/Export (Shell): 2 tests
- Core functionality: Database creation, vector storage, search, user management
- All user management commands should work correctly

---

### CLI Tips

**Using Environment Variables:**
```bash
# Set these to avoid repeating them
export JADEVECTORDB_URL="http://localhost:8080"
export JADEVECTORDB_API_KEY="$TOKEN"

# Then commands become shorter
python3 -m jadevectordb.cli user-list
```

**Getting Help:**
```bash
# Python CLI help
python3 -m jadevectordb.cli --help
python3 -m jadevectordb.cli user-add --help

# Shell CLI help
./scripts/jade-db.sh --help
```

**Generate cURL Commands (Python CLI):**
```bash
# Use --curl-only to see the cURL command without executing
python3 -m jadevectordb.cli --curl-only --url http://localhost:8080 --api-key "$TOKEN" \
  user-list
```

---

## Quick Start Command Reference

```bash
# Start backend (Development Mode)
export JADEVECTORDB_ENV=development
cd /home/deepak/Public/JadeVectorDB/backend/build
./jadevectordb &

# Start frontend
cd /home/deepak/Public/JadeVectorDB/frontend
npm run dev

# Stop backend gracefully (NEW - Recommended Method)
TOKEN=$(curl -s -X POST http://localhost:8080/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"admin123"}' | jq -r '.token')
curl -X POST http://localhost:8080/admin/shutdown \
  -H "Authorization: Bearer $TOKEN"

# Alternative: Stop backend with SIGTERM
pkill -SIGTERM jadevectordb

# Monitor logs
tail -f /tmp/jadedb.log

# Check storage
find /home/deepak/Public/JadeVectorDB/backend/build/data/ -name "*.jvdb" -ls

# Test health
curl http://localhost:8080/health

# Test admin shutdown endpoint
TOKEN=$(curl -s -X POST http://localhost:8080/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"admin123"}' | jq -r '.token')
curl -X POST http://localhost:8080/admin/shutdown -H "Authorization: Bearer $TOKEN"
```

---

**Ready to test!** Open http://localhost:3003 in your browser and begin with Test 1.
