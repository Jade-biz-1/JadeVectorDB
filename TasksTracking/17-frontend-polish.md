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
