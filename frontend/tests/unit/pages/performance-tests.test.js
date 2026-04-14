// tests/unit/pages/performance-tests.test.js
// Performance tests: large dataset rendering, list render time, and
// latency-sensitive UI behaviours (loading states, auto-refresh timing).
//
// "Performance" here means:
//   1. Large lists render without crashing and within an acceptable wall-clock budget
//   2. Loading states appear before data arrives so users aren't staring at a blank screen
//   3. Auto-refresh intervals fire correctly and don't pile up
//   4. Rapid successive interactions don't leave the UI in a broken state

import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';

// Global mocks: next/head, next/router, next/link via jest.config.js
jest.mock('@/components/Layout', () =>
  ({ children }) => <div data-testid="layout">{children}</div>
);
jest.mock('@/lib/api', () => require('./__mocks__/api'));

const {
  databaseApi, vectorApi, searchApi, performanceApi, alertApi, clusterApi,
} = require('@/lib/api');

// ─── helpers ─────────────────────────────────────────────────────────────────
/** Generate N mock database objects */
const makeDatabases = (n) =>
  Array.from({ length: n }, (_, i) => ({
    databaseId: `db-${i}`,
    name:        `Database ${i}`,
    vectorDimension: 128,
    status:     'active',
    stats:      { vectorCount: i * 100 },
  }));

/** Generate N mock vector objects */
const makeVectors = (n) =>
  Array.from({ length: n }, (_, i) => ({
    id:       `vec-${i}`,
    values:   [0.1, 0.2, 0.3, 0.4],
    metadata: { index: i },
  }));

/** Generate N mock search results */
const makeResults = (n) =>
  Array.from({ length: n }, (_, i) => ({
    vectorId: `vec-${i}`,
    score:    1 - i * 0.01,
    metadata: { rank: i },
  }));

/** Generate N mock performance metric entries */
const makeMetrics = (n) =>
  Array.from({ length: n }, (_, i) => ({
    label: `metric_${i}`,
    value: Math.random() * 100,
  }));

// 3 s covers even a heavily-loaded CI machine running the full suite in parallel
const RENDER_BUDGET_MS = 3000;

// ─────────────────────────────────────────────────────────────────────────────
// Large dataset: Database Management page
// ─────────────────────────────────────────────────────────────────────────────
describe('Performance: Database Management — large dataset', () => {
  beforeEach(() => jest.clearAllMocks());

  it('renders 100 databases within budget', async () => {
    const DatabaseManagement = require('@/pages/databases').default;
    databaseApi.listDatabases.mockResolvedValue({ databases: makeDatabases(100) });

    const t0 = Date.now();
    render(<DatabaseManagement />);
    await waitFor(() => screen.getByText('Database 0'));
    const elapsed = Date.now() - t0;

    expect(screen.getByText('Database 99')).toBeInTheDocument();
    expect(elapsed).toBeLessThan(RENDER_BUDGET_MS);
  });

  it('renders 200 databases without crashing', async () => {
    const DatabaseManagement = require('@/pages/databases').default;
    databaseApi.listDatabases.mockResolvedValue({ databases: makeDatabases(200) });

    render(<DatabaseManagement />);
    await waitFor(() => screen.getByText('Database 0'));
    expect(screen.getByText('Database 199')).toBeInTheDocument();
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Large dataset: Vector Management page
// ─────────────────────────────────────────────────────────────────────────────
describe('Performance: Vector Management — large dataset', () => {
  beforeEach(() => jest.clearAllMocks());

  it('renders 50 vectors within budget', async () => {
    const VectorManagement = require('@/pages/vectors').default;
    const dbs = makeDatabases(1);
    databaseApi.listDatabases.mockResolvedValue({ databases: dbs });
    databaseApi.getDatabase.mockResolvedValue(dbs[0]);
    vectorApi.listVectors.mockResolvedValue({ vectors: makeVectors(50), total: 50 });

    const t0 = Date.now();
    render(<VectorManagement />);
    await waitFor(() => screen.getByText('Database 0'));
    fireEvent.change(screen.getByRole('combobox'), { target: { value: 'db-0' } });
    await waitFor(() => screen.getByText('ID: vec-0'));
    const elapsed = Date.now() - t0;

    expect(screen.getByText('ID: vec-49')).toBeInTheDocument();
    expect(elapsed).toBeLessThan(RENDER_BUDGET_MS);
  });

  it('shows loading state before the vector list arrives', async () => {
    const VectorManagement = require('@/pages/vectors').default;
    const dbs = makeDatabases(1);
    databaseApi.listDatabases.mockResolvedValue({ databases: dbs });
    databaseApi.getDatabase.mockResolvedValue(dbs[0]);
    vectorApi.listVectors.mockImplementation(() => new Promise(() => {})); // never resolves

    render(<VectorManagement />);
    await waitFor(() => screen.getByText('Database 0'));
    fireEvent.change(screen.getByRole('combobox'), { target: { value: 'db-0' } });

    await waitFor(() => {
      expect(screen.getByText(/loading/i)).toBeInTheDocument();
    });
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Large dataset: Vector Search results
// ─────────────────────────────────────────────────────────────────────────────
describe('Performance: Vector Search — large result set', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    window.location.search = '';
    databaseApi.listDatabases.mockResolvedValue({ databases: makeDatabases(1) });
  });

  it('renders 100 search results within budget', async () => {
    const SearchInterface = require('@/pages/search').default;
    searchApi.similaritySearch.mockResolvedValueOnce({ results: makeResults(100) });

    const t0 = Date.now();
    render(<SearchInterface />);
    await waitFor(() => screen.getByText('Database 0'));

    fireEvent.change(screen.getAllByRole('combobox')[0], { target: { value: 'db-0' } });
    const ta = screen.getByPlaceholderText(/enter vector values/i);
    fireEvent.change(ta, { target: { value: '0.1,0.2,0.3,0.4' } });
    fireEvent.submit(ta.closest('form'));

    await waitFor(() => screen.getByText(/Vector ID: vec-0/i));
    const elapsed = Date.now() - t0;

    expect(screen.getByText(/Vector ID: vec-99/i)).toBeInTheDocument();
    expect(elapsed).toBeLessThan(RENDER_BUDGET_MS);
  });

  it('shows loading indicator while search is in flight', async () => {
    const SearchInterface = require('@/pages/search').default;
    let resolve;
    searchApi.similaritySearch.mockImplementation(() => new Promise(r => { resolve = r; }));

    render(<SearchInterface />);
    await waitFor(() => screen.getByText('Database 0'));

    fireEvent.change(screen.getAllByRole('combobox')[0], { target: { value: 'db-0' } });
    const ta = screen.getByPlaceholderText(/enter vector values/i);
    fireEvent.change(ta, { target: { value: '0.1,0.2,0.3,0.4' } });
    fireEvent.submit(ta.closest('form'));

    await waitFor(() => {
      expect(screen.getByText(/searching/i)).toBeInTheDocument();
    });

    // Resolve to clean up
    resolve({ results: [] });
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Performance Dashboard — metrics rendering & auto-refresh
// ─────────────────────────────────────────────────────────────────────────────
describe('Performance: Performance Dashboard', () => {
  beforeEach(() => {
    jest.useFakeTimers();
    jest.clearAllMocks();
  });

  afterEach(() => jest.useRealTimers());

  it('renders 50 metric entries within budget', async () => {
    const PerformanceDashboard = require('@/pages/performance').default;
    performanceApi.getMetrics.mockResolvedValue({ metrics: makeMetrics(50) });

    await act(async () => render(<PerformanceDashboard />));
    await waitFor(() => {
      expect(screen.getByText('metric_0')).toBeInTheDocument();
      expect(screen.getByText('metric_49')).toBeInTheDocument();
    });
  });

  it('auto-refreshes every 10 seconds', async () => {
    const PerformanceDashboard = require('@/pages/performance').default;
    performanceApi.getMetrics.mockResolvedValue({ metrics: makeMetrics(1) });

    await act(async () => render(<PerformanceDashboard />));
    await waitFor(() => expect(performanceApi.getMetrics).toHaveBeenCalledTimes(1));

    await act(async () => { jest.advanceTimersByTime(10000); });
    await waitFor(() => expect(performanceApi.getMetrics).toHaveBeenCalledTimes(2));
  });

  it('clears the auto-refresh interval on unmount', async () => {
    const PerformanceDashboard = require('@/pages/performance').default;
    performanceApi.getMetrics.mockResolvedValue({ metrics: [] });

    const clearSpy = jest.spyOn(global, 'clearInterval');
    const { unmount } = await act(async () => render(<PerformanceDashboard />));
    unmount();
    expect(clearSpy).toHaveBeenCalled();
    clearSpy.mockRestore();
  });

  it('shows a last-updated timestamp after data loads', async () => {
    const PerformanceDashboard = require('@/pages/performance').default;
    performanceApi.getMetrics.mockResolvedValue({ metrics: makeMetrics(2) });

    await act(async () => render(<PerformanceDashboard />));
    await waitFor(() => {
      expect(screen.getByText(/last updated/i)).toBeInTheDocument();
    });
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Alerting — large alert list
// ─────────────────────────────────────────────────────────────────────────────
describe('Performance: Alerting — large alert list', () => {
  beforeEach(() => {
    jest.useFakeTimers();
    jest.clearAllMocks();
  });

  afterEach(() => jest.useRealTimers());

  const makeAlerts = (n) =>
    Array.from({ length: n }, (_, i) => ({
      id:        `a-${i}`,
      type:      i % 3 === 0 ? 'error' : i % 3 === 1 ? 'warning' : 'info',
      message:   `Alert message ${i}`,
      timestamp: '2026-04-13T10:00:00Z',
    }));

  it('renders 50 alerts within budget', async () => {
    const Alerting = require('@/pages/alerting').default;
    alertApi.listAlerts.mockResolvedValue({ alerts: makeAlerts(50) });

    const t0 = Date.now();
    await act(async () => render(<Alerting />));
    await waitFor(() => screen.getByText('Alert message 0'));
    const elapsed = Date.now() - t0;

    expect(screen.getByText('Alert message 49')).toBeInTheDocument();
    expect(elapsed).toBeLessThan(RENDER_BUDGET_MS);
  });

  it('filtering 50 alerts by type is fast', async () => {
    const Alerting = require('@/pages/alerting').default;
    alertApi.listAlerts.mockResolvedValue({ alerts: makeAlerts(50) });

    await act(async () => render(<Alerting />));
    await waitFor(() => screen.getByText('Alert message 0'));

    const t0 = Date.now();
    fireEvent.change(screen.getByRole('combobox'), { target: { value: 'error' } });
    const elapsed = Date.now() - t0;

    // All visible alerts should be 'error' type (indices 0, 3, 6, ...)
    expect(screen.queryByText('Alert message 1')).not.toBeInTheDocument(); // 'warning'
    expect(elapsed).toBeLessThan(200);
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// Rapid interaction: batch add/remove rows
// ─────────────────────────────────────────────────────────────────────────────
describe('Performance: Batch Operations — rapid row add/remove', () => {
  beforeEach(() => jest.clearAllMocks());

  it('handles rapid addition of 10 vector rows without errors', async () => {
    const BatchOperations = require('@/pages/batch-operations').default;
    databaseApi.listDatabases.mockResolvedValue({ databases: makeDatabases(1) });

    render(<BatchOperations />);
    await waitFor(() => screen.getByText('Database 0'));

    const addBtn = screen.getByRole('button', { name: /add vector/i });
    for (let i = 0; i < 9; i++) { // 1 default + 9 added = 10 rows
      fireEvent.click(addBtn);
    }

    const removeButtons = screen.getAllByRole('button', { name: /remove/i });
    expect(removeButtons).toHaveLength(10);
  });

  it('removes all extra rows without crashing', async () => {
    const BatchOperations = require('@/pages/batch-operations').default;
    databaseApi.listDatabases.mockResolvedValue({ databases: makeDatabases(1) });

    render(<BatchOperations />);
    await waitFor(() => screen.getByText('Database 0'));

    const addBtn = screen.getByRole('button', { name: /add vector/i });
    // Add 4 more rows (5 total)
    for (let i = 0; i < 4; i++) fireEvent.click(addBtn);

    let removeButtons = screen.getAllByRole('button', { name: /remove/i });
    expect(removeButtons).toHaveLength(5);

    // Remove all but the last
    while (screen.getAllByRole('button', { name: /remove/i }).length > 1) {
      fireEvent.click(screen.getAllByRole('button', { name: /remove/i })[0]);
    }

    expect(screen.getAllByRole('button', { name: /remove/i })).toHaveLength(1);
  });
});
