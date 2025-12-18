# Manual Testing Guide - Persistent Vector Storage

**Date:** December 18, 2025  
**Sprint:** 2.1 - Vector Data Persistence  
**Status:** Ready for Manual Testing

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

The Sprint 2.1 implementation added **persistent vector storage** with:
- Memory-mapped `.jvdb` files for vector data
- Cross-platform support (Unix mmap)
- LRU cache for efficient file descriptor management
- Automatic periodic flushing (5-minute default)
- Graceful shutdown with signal handlers
- Crash recovery capabilities

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

### Must Pass:
- [ ] Test 1: Database creation with `.jvdb` file
- [ ] Test 2: Vector storage and retrieval
- [ ] Test 3: **Data persists after restart** (CRITICAL)
- [ ] Test 4: Multiple databases work independently
- [ ] Test 5: Updates and deletes persist
- [ ] Test 8: Database deletion removes `.jvdb` file

### Should Pass:
- [ ] Test 6: Can handle 1,000+ vectors
- [ ] Test 7: Auto-flush works

### Nice to Have:
- [ ] Performance: Sub-second retrieval for 10K vectors
- [ ] Memory: Stable memory usage over time
- [ ] No crashes under normal load

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

## After Testing

### If Tests Pass:
1. Document results in Sprint 2.1 summary
2. Mark sprint as 100% complete
3. Prepare for merge to main branch
4. Plan next sprint (Sprint 2.2 - potential enhancements)

### If Tests Fail:
1. Document specific failures
2. Check logs for error messages
3. Review persistence implementation code
4. Fix issues and re-test
5. May need to debug with gdb or add more logging

---

## Quick Start Command Reference

```bash
# Start backend
cd /home/deepak/Public/JadeVectorDB/backend/build
./jadevectordb &

# Start frontend
cd /home/deepak/Public/JadeVectorDB/frontend
npm run dev

# Stop backend gracefully
pkill -SIGTERM jadevectordb

# Monitor logs
tail -f /home/deepak/Public/JadeVectorDB/backend/build/logs/*.log

# Check storage
find /home/deepak/Public/JadeVectorDB/backend/build/data/ -name "*.jvdb" -ls

# Test health
curl http://localhost:8080/health
```

---

**Ready to test!** Open http://localhost:3003 in your browser and begin with Test 1.
