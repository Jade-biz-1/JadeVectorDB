# Frontend Polish Tasks

**Phase**: Post-completion enhancements  
**Source**: Derived from frontend code review (frontendreport.md, now archived)  
**Added**: 2026-04-14

---

### T-FE01: Performance Optimization — Code Splitting & Lazy Loading
**[O] OPTIONAL**
**Dependencies**: None  
**Description**: Reduce first-load JS bundle size for heavy pages by lazy-loading large dependencies with `next/dynamic`.  
**Subtasks**:
- [ ] Lazy-load Recharts in `analytics.js` (currently 209 kB first load — largest page)
- [ ] Audit other pages for heavy imports that can be deferred
- [ ] Verify bundle size reduction with `next build` output
**Status**: [ ] PENDING  
**Priority**: LOW  
**Notes**: App is fully functional without this. Pure performance win — no user-visible behavior change.

---

### T-FE02: Accessibility — ARIA Labels & Keyboard Navigation
**[O] OPTIONAL**
**Dependencies**: None  
**Description**: Improve accessibility beyond what automated tests currently verify.  
**Subtasks**:
- [ ] Add focus trap to `Modal` component (currently closes on Escape but doesn't trap focus)
- [ ] Add `aria-sort` to sortable table columns
- [ ] Ensure form error messages are announced via `role="alert"` (FormField already has this for errors; audit for completeness)
- [ ] Verify keyboard navigation through pagination controls in `vectors.js`
- [ ] Add `aria-live` region for success/error alerts
**Status**: [ ] PENDING  
**Priority**: LOW  
**Notes**: App passes existing accessibility test suite. These are incremental improvements for screen reader and keyboard-only users.

---

### T-FE03: Unit Tests — Remaining 3 Pages
**[O] OPTIONAL**
**Dependencies**: None  
**Description**: Add unit tests for the three pages still missing dedicated test files (identified in `frontend/TESTING_IMPLEMENTATION_PLAN.md`). Overall pass rate is already 97% (714/736); these close the last gap in page coverage.  
**Subtasks**:
- [ ] `frontend/tests/unit/pages/databases-id.test.js` — dynamic route `databases/[id].js`
- [ ] `frontend/tests/unit/pages/tutorials.test.js` — interactive tutorials page
- [ ] `frontend/tests/unit/pages/quizzes.test.js` — quiz system page
**Status**: [ ] PENDING  
**Priority**: LOW  
**Notes**: All other pages have unit test files. These three are the last gaps. Not blocking production deployment.

---

### T-FE04: Automated Accessibility Tests (jest-axe)
**[O] OPTIONAL**
**Dependencies**: T-FE02  
**Description**: Add automated WCAG 2.1 Level AA checks using `jest-axe` across all 32 pages, as planned in Phase 5 of `frontend/TESTING_IMPLEMENTATION_PLAN.md`.  
**Subtasks**:
- [ ] Install `jest-axe` and configure in `jest.config.js`
- [ ] Add axe accessibility assertions to all 32 page unit tests
- [ ] Fix any violations found (expected: colour contrast, missing labels)
- [ ] Add to CI quality gate: fail on any new accessibility violation
**Status**: [ ] PENDING  
**Priority**: LOW  
**Notes**: Phase 5 of the 7-week testing plan. App is functional; this is a quality/compliance enhancement.

---

### T-FE05: Performance Benchmarks
**[O] OPTIONAL**
**Dependencies**: None  
**Description**: Establish render-time and bundle-size baselines as planned in Phase 6 of `frontend/TESTING_IMPLEMENTATION_PLAN.md`.  
**Subtasks**:
- [ ] Add `@next/bundle-analyzer` and document per-page bundle sizes
- [ ] Establish render-time benchmarks for the 5 heaviest pages
- [ ] Add budget assertions (fail CI if bundle exceeds threshold)
**Status**: [ ] PENDING  
**Priority**: LOW  
**Notes**: Phase 6 of the 7-week testing plan. Pairs well with T-FE01 (code splitting).
