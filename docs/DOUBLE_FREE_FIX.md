# Double-Free Error Fix - December 17, 2025

## Problem Description

During Docker deployment testing (T11.6.4), we encountered a critical memory corruption error:

```
(2025-12-17 08:31:19) [INFO    ] Exiting.
double free or corruption (fasttop)
```

This error occurred during application shutdown after receiving SIGTERM signal, causing unclean termination despite meeting the <30 second shutdown requirement (0.238s actual).

---

## Root Cause Analysis

### Investigation Process

1. **Initial Hypothesis**: Suspected destructor ordering issues in `main.cpp` where LoggerManager was shut down before the destructor tried to log
2. **Secondary Investigation**: Traced through shutdown sequence and found double-stop pattern in REST API service

### Confirmed Root Cause

The error was caused by **double-stopping the Crow web server**:

**Shutdown Sequence:**
1. `main()` calls `app->shutdown()` → calls `RestApiService::stop()`
2. `RestApiService::stop()` calls `api_impl_->stop_server()`
3. `api_impl_->stop_server()` calls `app_->stop()` ✅ (first stop - correct)
4. `RestApiService` destructor calls `stop()` again (defensive programming)
5. `RestApiImpl` destructor calls `app_->stop()` ❌ (second stop - **DOUBLE FREE**)

The Crow framework's internal state was being cleaned up twice, causing memory corruption.

---

## Solution Implementation

### Fix #1: Track Server State in RestApiImpl

**File**: `backend/src/api/rest/rest_api_impl.h`

Added a boolean flag to track server state:

```cpp
class RestApiImpl {
private:
    std::unique_ptr<crow::SimpleApp> app_;
    bool server_stopped_ = false;  // NEW: Track server state
    // ... other members
};
```

### Fix #2: Guard Multiple Stop Calls

**File**: `backend/src/api/rest/rest_api_impl.cpp`

#### In `stop_server()` method:
```cpp
void RestApiImpl::stop_server() {
    if (!server_stopped_ && app_) {
        std::cout << "Stopping REST API server..." << std::endl;
        app_->stop();
        server_stopped_ = true;  // Mark as stopped
    }
}
```

#### In destructor:
```cpp
RestApiImpl::~RestApiImpl() {
    std::cout << "RestApiImpl destructor called" << std::endl;
    
    // Only stop if not already stopped
    if (!server_stopped_ && app_) {
        std::cout << "Stopping server from destructor" << std::endl;
        app_->stop();
        server_stopped_ = true;
    }
    
    // Let unique_ptr handle app_ destruction automatically
}
```

### Fix #3: Clean Main Destructor

**File**: `backend/src/main.cpp`

Removed redundant logging after logger shutdown:

```cpp
~JadeVectorDBApp() {
    if (running_) {
        shutdown();
    }
    // Removed: LOG_INFO after LoggerManager::shutdown()
}
```

---

## Verification

### Build Process
```bash
cd backend
./build.sh --type Release --no-tests --no-benchmarks --jobs 4
# Build successful: 2m 14s
```

### Docker Build
```bash
docker build -t jadevectordb:test .
# Build successful: 231.5s
# Image size: 95.7MB
```

### Shutdown Test
```bash
docker compose up -d
docker stop jadevectordb
```

**Before Fix:**
```
(2025-12-17 08:31:19) [INFO    ] Exiting.
double free or corruption (fasttop)  ❌

Shutdown time: 0.238s
```

**After Fix:**
```
(2025-12-17 08:37:55) [INFO    ] Exiting.  ✅
(No error)

Shutdown time: 0.244s
```

---

## Technical Details

### Why This Pattern is Problematic

1. **RAII + Manual Cleanup**: Mixing manual `stop()` calls with automatic destructor cleanup creates opportunities for double-cleanup
2. **Defensive Programming Backfire**: Multiple defensive `if (running_) { stop(); }` guards don't help if state isn't properly tracked
3. **Framework Sensitivity**: Some frameworks (like Crow) have internal state that cannot safely be cleaned up twice

### Best Practices Applied

1. ✅ **Single Responsibility**: Server stop logic centralized in one method
2. ✅ **State Tracking**: Explicit `server_stopped_` flag prevents double-cleanup
3. ✅ **Idempotent Operations**: Multiple calls to `stop_server()` are now safe
4. ✅ **Clear Ownership**: `unique_ptr` handles memory, stop logic handles state

---

## Impact Assessment

### Before Fix
- ❌ Memory corruption on shutdown
- ❌ Unclean process termination
- ⚠️ Potential data loss risk
- ⚠️ Could cause issues in orchestrated environments (Kubernetes health checks fail)

### After Fix
- ✅ Clean shutdown with no errors
- ✅ Graceful termination (0.244s)
- ✅ Safe for production deployment
- ✅ Passes Docker health checks

---

## Related Tasks

- **T11.6.4**: Docker Deployment Optimization - This fix was discovered during Docker testing
- **T11.6.1**: Error Handling & Recovery - Graceful shutdown is part of error recovery strategy
- **Sprint 1.6**: Production Readiness - Clean shutdown is critical for production

---

## Lessons Learned

1. **Test Shutdown Paths**: Always test graceful shutdown scenarios, not just happy-path operations
2. **State Management**: Explicit state tracking is better than inferring state from object existence
3. **Framework Lifecycle**: Understand framework cleanup requirements (Crow requires single stop)
4. **Defensive vs Safe**: Defensive programming (multiple stop calls) can backfire without proper guards

---

## Files Modified

1. `backend/src/api/rest/rest_api_impl.h` - Added `server_stopped_` flag
2. `backend/src/api/rest/rest_api_impl.cpp` - Guarded `stop()` calls with state check
3. `backend/src/main.cpp` - Removed post-shutdown logging

---

## Testing Checklist

- [x] Build compiles successfully
- [x] Docker image builds (95.7MB)
- [x] Container starts and serves requests
- [x] Health check returns 200 OK
- [x] Graceful shutdown completes without errors
- [x] Shutdown time < 30 seconds (0.244s achieved)
- [x] No memory corruption errors
- [x] Non-root execution verified

---

**Fixed By**: GitHub Copilot (Claude Sonnet 4.5)  
**Date**: December 17, 2025  
**Sprint**: 1.6 (Production Readiness)  
**Task**: T11.6.4 (Docker Deployment Optimization)
