import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';

// Global mocks: next/head, next/router, next/link via jest.config.js
jest.mock('@/components/Layout', () => ({ children }) => <div>{children}</div>);
jest.mock('@/lib/api', () => require('./__mocks__/api'));
const { performanceApi } = require('@/lib/api');

import PerformanceDashboard from '@/pages/performance';

const mockMetrics = [
  { label: 'Queries/sec', value: 1500 },
  { label: 'Avg Latency', value: '2ms' },
  { label: 'Memory Usage', value: '4.2 GB' },
];

describe('PerformanceDashboard page', () => {
  beforeEach(() => {
    jest.useFakeTimers();
    jest.clearAllMocks();
    performanceApi.getMetrics.mockResolvedValue({ metrics: mockMetrics });
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  it('renders the Performance Dashboard heading', async () => {
    await act(async () => render(<PerformanceDashboard />));
    expect(screen.getByRole('heading', { name: /performance dashboard/i })).toBeInTheDocument();
  });

  it('fetches and displays metrics on mount', async () => {
    await act(async () => render(<PerformanceDashboard />));
    await waitFor(() => {
      expect(screen.getByText('Queries/sec')).toBeInTheDocument();
      expect(screen.getByText('Avg Latency')).toBeInTheDocument();
    });
  });

  it('shows last updated time after fetch', async () => {
    await act(async () => render(<PerformanceDashboard />));
    await waitFor(() => {
      expect(screen.getByText(/last updated/i)).toBeInTheDocument();
    });
  });

  it('auto-refreshes every 10 seconds', async () => {
    await act(async () => render(<PerformanceDashboard />));
    await waitFor(() => expect(performanceApi.getMetrics).toHaveBeenCalledTimes(1));

    await act(async () => { jest.advanceTimersByTime(10000); });
    await waitFor(() => expect(performanceApi.getMetrics).toHaveBeenCalledTimes(2));
  });

  it('clears interval on unmount', async () => {
    const clearIntervalSpy = jest.spyOn(global, 'clearInterval');
    const { unmount } = await act(async () => render(<PerformanceDashboard />));
    unmount();
    expect(clearIntervalSpy).toHaveBeenCalled();
    clearIntervalSpy.mockRestore();
  });

  it('refresh button triggers refetch', async () => {
    await act(async () => render(<PerformanceDashboard />));
    await waitFor(() => expect(performanceApi.getMetrics).toHaveBeenCalledTimes(1));

    await act(async () => { fireEvent.click(screen.getByRole('button', { name: /refresh/i })); });
    await waitFor(() => expect(performanceApi.getMetrics).toHaveBeenCalledTimes(2));
  });

  it('handles API error gracefully', async () => {
    const consoleSpy = jest.spyOn(console, 'error').mockImplementation(() => {});
    performanceApi.getMetrics.mockRejectedValueOnce(new Error('Server error'));
    await act(async () => render(<PerformanceDashboard />));
    await waitFor(() => expect(consoleSpy).toHaveBeenCalled());
    consoleSpy.mockRestore();
  });
});
