# Cross-Reference Updates Required - December 14, 2025

## 🎯 Focus: Files Being Moved + Required Updates

---

## 📁 FILES TO MOVE (2 files)

### 1. CLI_INFORMATION.md → `docs/CLI_INFORMATION.md`

**Cross-References to Update**: 1 file

| File | Line | Current Reference | New Reference |
|------|------|-------------------|---------------|
| `docs/SPRINT_STATUS.md` | 34 | `CLI_INFORMATION.md` | `CLI_INFORMATION.md` (just filename, path changes) |

**Action**: Verify if path update needed in SPRINT_STATUS.md

---

### 2. DOCKER_DEPLOYMENT.md → `docs/DOCKER_DEPLOYMENT.md`

**Cross-References to Update**: 3 files

| File | Line | Current Reference | New Reference |
|------|------|-------------------|---------------|
| `BOOTSTRAP.md` | 1018 | `[Deployment Guide](DOCKER_DEPLOYMENT.md)` | `[Deployment Guide](docs/DOCKER_DEPLOYMENT.md)` |
| `tutorials/README.md` | 189 | `[DOCKER_DEPLOYMENT.md](../DOCKER_DEPLOYMENT.md)` | `[DOCKER_DEPLOYMENT.md](../docs/DOCKER_DEPLOYMENT.md)` |
| `AUTOMATED_TEST_REPORT.md` | 82 | Lists `DOCKER_DEPLOYMENT.md` | Update to `docs/DOCKER_DEPLOYMENT.md` (but this file is being archived anyway) |

---

## 🗄️ FILES TO ARCHIVE (4 files)

### 3. AUTOMATED_TEST_REPORT.md → `docs/archive/AUTOMATED_TEST_REPORT_2025-12-13.md`

**Cross-References to Update**: 4 files

| File | Line | Current Reference | New Reference |
|------|------|-------------------|---------------|
| `README.md` | 45 | `` `AUTOMATED_TEST_REPORT.md` `` | `` `docs/archive/AUTOMATED_TEST_REPORT_2025-12-13.md` `` |
| `BOOTSTRAP.md` | 11 | `` `AUTOMATED_TEST_REPORT.md` `` | `` `docs/archive/AUTOMATED_TEST_REPORT_2025-12-13.md` `` |
| `BOOTSTRAP.md` | 29 | `` `AUTOMATED_TEST_REPORT.md` `` | `` `docs/archive/AUTOMATED_TEST_REPORT_2025-12-13.md` `` |
| `TasksTracking/status-dashboard.md` | 80 | `` `AUTOMATED_TEST_REPORT.md` `` | `` `docs/archive/AUTOMATED_TEST_REPORT_2025-12-13.md` `` |

---

### 4. DEC13.md → `docs/archive/DEC13_TASK_SUMMARY_2025-12-13.md`

**Cross-References to Update**: NONE

✅ **Safe to move - no external references**

---

### 5. DISTRIBUTED_SYSTEM_COMPLETE.md → `docs/archive/DISTRIBUTED_SYSTEM_COMPLETE_2025-12-13.md`

**Cross-References to Update**: 1 file (being archived)

| File | Line | Current Reference | New Reference |
|------|------|-------------------|---------------|
| `AUTOMATED_TEST_REPORT.md` | 75, 80, 160 | `DISTRIBUTED_SYSTEM_COMPLETE.md` | `docs/archive/DISTRIBUTED_SYSTEM_COMPLETE_2025-12-13.md` |

**Note**: AUTOMATED_TEST_REPORT.md is also being archived, so this is optional

---

### 6. DOCKER_CLEANUP_SUMMARY.md → `docs/archive/DOCKER_CLEANUP_SUMMARY.md`

**Cross-References to Update**: 2 files

| File | Lines | Current Reference | New Reference |
|------|-------|-------------------|---------------|
| `docs/COMPLETE_BUILD_SYSTEM_SETUP.md` | 26, 54, 125, 247, 355 | `DOCKER_CLEANUP_SUMMARY.md` | `docs/archive/DOCKER_CLEANUP_SUMMARY.md` |
| `docs/archive/summaries/COMPLETE_BUILD_SYSTEM_SETUP.md` | 26, 54, 125, 247, 355 | `DOCKER_CLEANUP_SUMMARY.md` | `docs/archive/DOCKER_CLEANUP_SUMMARY.md` |

---

## 🗑️ FILE TO DELETE (1 file)

### 7. CLAUDE.md

**Cross-References to Update**: 1 file (script)

| File | Line | Current Reference | Action |
|------|------|-------------------|--------|
| `.specify/scripts/bash/update-agent-context.sh` | 62 | `CLAUDE_FILE="$REPO_ROOT/CLAUDE.md"` | ⚠️ Check if script is active before deleting |

**Important**: Verify this script's status before proceeding with deletion

---

## ⚠️ CRITICAL: DEVELOPER_GUIDE.md (Already Deleted)

**This file was already deleted but still has references!**

**Cross-References to Update**: 5 files

| File | Line | Current Reference | New Reference |
|------|------|-------------------|---------------|
| `README.md` | 789 | `[Developer Guide](DEVELOPER_GUIDE.md)` | `[Developer Guide](BOOTSTRAP.md)` |
| `CONTRIBUTING.md` | 32 | `[Developer Guide](DEVELOPER_GUIDE.md)` | `[Developer Guide](BOOTSTRAP.md)` |
| `RECOVERY_SUMMARY.md` | 79 | `` `DEVELOPER_GUIDE.md` `` | `` `BOOTSTRAP.md` `` |
| `TasksTracking/04-monitoring-polish.md` | N/A | `DEVELOPER_GUIDE.md` | `BOOTSTRAP.md` |
| `tutorials/README.md` | 180 | `[DEVELOPER_GUIDE.md](../DEVELOPER_GUIDE.md)` | `[BOOTSTRAP.md](../BOOTSTRAP.md)` |

---

## 📊 Summary

| Action | Files | Cross-Refs to Update |
|--------|-------|---------------------|
| Move to docs/ | 2 | 4 references |
| Archive | 4 | 7 references (active) |
| Delete | 1 | 1 reference (script) |
| **Fix broken links** | 1 (deleted) | **5 references** |
| **TOTAL** | **8** | **17 references** |

---

## ✅ Execution Order

1. **FIX BROKEN LINKS FIRST** (DEVELOPER_GUIDE.md → BOOTSTRAP.md)
   - Update 5 files with broken links

2. **MOVE FILES** (CLI_INFORMATION.md, DOCKER_DEPLOYMENT.md)
   - Update 4 cross-references

3. **ARCHIVE FILES** (4 files)
   - Update 7 cross-references

4. **DELETE FILE** (CLAUDE.md)
   - Verify script status first
   - Update 1 reference if needed

