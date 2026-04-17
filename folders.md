# JadeVectorDB — Repository Folder Reference

**Last Updated**: 2026-04-17  
**Purpose**: Documents every top-level folder — what it contains, its status, and why it belongs (or doesn't) in the main repository.

---

## Core Application

| Folder | Contents | Status |
|--------|----------|--------|
| `backend/` | C++ server — vector storage, search, distributed system, auth | ✅ Active |
| `frontend/` | Next.js/React web dashboard | ✅ Active |
| `cli/` | CLI tools in Python, Shell, and JavaScript | ✅ Active |
| `EnterpriseRAG/` | Full RAG application (FastAPI backend + React frontend + Ollama) | ✅ Active |

---

## Documentation & Tracking

| Folder | Contents | Status |
|--------|----------|--------|
| `docs/` | 71 markdown files covering API reference, architecture, RBAC, deployment, analytics | ✅ Active |
| `TasksTracking/` | Sprint plans, phase completion records, status dashboards | ✅ Active |
| `specs/` | Design specifications and compliance docs | ✅ Active |
| `tutorials/` | CLI exercises (8) and interactive web tutorial | ✅ Active |
| `APIExamples/` | 15 numbered Python scripts demonstrating API usage | ✅ Active |

---

## Testing

| Folder | Contents | Status | Notes |
|--------|----------|--------|-------|
| `tests/` | Python integration and CLI test suites | ✅ Active | Primary test suite |
| `property-tests/` | C++ property-based testing framework | ✅ Keep | Tests mathematical invariants (vector norms, dimension consistency, concurrency, distributed system properties). No overlap with `tests/` — serves a distinct testing methodology. |
| `chaos-engineering/` | Resilience testing scripts (network partition, node failure, resource exhaustion) | ✅ Keep | Essential for production readiness validation. Includes Prometheus monitoring integration. Complements `k8s/` and `deployments/`. |

---

## Deployment & Infrastructure

| Folder | Contents | Status | Notes |
|--------|----------|--------|-------|
| `k8s/` | Raw Kubernetes manifests — standalone, cluster, monitoring | ✅ Keep | 4 files, 10KB README. Best for simple deployments and users who prefer direct manifest control. Complements `charts/` (different approach, not redundant). |
| `charts/` | Helm chart (`jadevectordb/`) — Chart.yaml, values.yaml, templates | ✅ Keep | Templated, parameterised K8s deployment. v0.1.0. Best for production deployments with per-environment config. Use `charts/` over `k8s/` when you need repeatable multi-environment deployments. |
| `deployments/` | Cloud IaC templates for AWS (CloudFormation), Azure (Bicep/ARM), GCP (Terraform), plus blue-green and multi-cloud scripts | ✅ Keep | 26 files. Well-organised by cloud provider. Blue-green deployment pattern is distinct from `k8s/` generic manifests. |
| `grafana/` | Grafana provisioning — datasources and pre-built dashboards (EnterpriseRAG + JadeVectorDB) | ✅ Active | Mounted by docker-compose.yml. Required for observability stack. |
| `config/` | Configuration templates | ✅ Keep | Minimal (1 file). Reference configuration for deployment. |

---

## Supporting Services & Examples

| Folder | Contents | Status | Notes |
|--------|----------|--------|-------|
| `python/` | Re-ranking microservice — cross-encoder model server with JSON IPC protocol | ✅ Keep | **Not** the same as `cli/python/`. This is a standalone Python service for Phase 1 re-ranking architecture. Production-quality with tests, docs, and deployment guide. Last modified Jan 28, 2026. |
| `rag-ui-examples/` | 4 complete RAG UI reference implementations: Gradio, Flask, FastAPI+React, Textual TUI | ✅ Keep | Educational reference implementations demonstrating different UI frameworks for RAG systems. Mock service included for offline testing. Last modified Apr 2, 2026. |
| `examples/` | CLI usage documentation only (python-examples.md, shell-examples.md, javascript-examples.md) | ⚠️ Review | Docs-only folder documenting `cli/`. Verify whether content is already covered by `cli/*/README.md` and `tutorials/`. If so, merge into tutorials and delete. |
| `scripts/` | 22 shell/Python automation scripts for build, test, security, performance, compliance | ✅ Keep | Consider organising into `scripts/security/`, `scripts/testing/`, `scripts/setup/` subdirectories for discoverability. |

---

## Should NOT Be in the Repository

| Folder | Contents | Action |
|--------|----------|--------|
| `data/` | `system.db` — 270MB SQLite runtime database from local dev environment | **Delete from git, add to .gitignore** |
| `backend/build/` | 626MB C++ compiled output | **Add `backend/build/` to .gitignore, remove from git** |
| `frontend/node_modules/` | 533MB npm packages | **Add to .gitignore** (likely already is, verify) |
| `frontend/.next/` | 154MB Next.js build cache | **Add to .gitignore** |
| `frontend/coverage/` | Test coverage reports | **Add to .gitignore** |
| `logs/` | Runtime application logs | **Add `logs/` to .gitignore, remove from git** |
| `.pytest_cache/` | pytest cache directory | **Add to .gitignore** |
| `**/.DS_Store` | macOS Finder metadata | **Add `**/.DS_Store` to .gitignore, run `git rm -r --cached`** |
| `**/__pycache__/` | Python bytecode cache | **Add to .gitignore** |

**Estimated space recoverable**: ~1.6 GB

---

## charts/ vs k8s/ — When to Use Which

Both folders deploy JadeVectorDB to Kubernetes but serve different users:

| | `k8s/` | `charts/` |
|-|--------|-----------|
| Format | Raw YAML manifests | Helm templates |
| Best for | Simple single-env deploys, learning K8s | Multi-environment, CI/CD pipelines |
| Customisation | Edit YAML directly | Override `values.yaml` |
| Prerequisites | `kubectl` | `kubectl` + Helm |
| Versioning | Manual | Via Helm release management |

---

## Folder Count Summary

| Category | Folders | Keep | Review | Remove |
|----------|---------|------|--------|--------|
| Core application | 4 | 4 | 0 | 0 |
| Docs & tracking | 5 | 5 | 0 | 0 |
| Testing | 3 | 3 | 0 | 0 |
| Deployment | 5 | 5 | 0 | 0 |
| Supporting / examples | 4 | 3 | 1 | 0 |
| Generated / runtime | 9 | 0 | 0 | 9 |
| **Total** | **30** | **20** | **1** | **9** |

---

## Recommended Next Actions

1. **Fix .gitignore** — add `data/`, `backend/build/`, `frontend/.next/`, `logs/`, `.pytest_cache/`, `**/.DS_Store`, `**/__pycache__/`
2. **Remove tracked generated files** — run `git rm -r --cached` for each generated path
3. **Review `examples/`** — check if content duplicates `cli/*/README.md`; if yes, merge into `tutorials/` and delete
4. **Organise `scripts/`** — group into `security/`, `testing/`, `setup/` subdirectories
5. **Add `.gitkeep`** — to `EnterpriseRAG/uploads/` so the empty directory is preserved
