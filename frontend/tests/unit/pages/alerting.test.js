import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';

// Global mocks: next/head, next/router, next/link via jest.config.js
jest.mock('@/components/Layout', () => ({ children }) => <div>{children}</div>);

// alertApi mock must be defined inside the factory to avoid hoisting issues
jest.mock('@/lib/api', () => ({
  alertApi: {
    listAlerts: jest.fn(),
    acknowledgeAlert: jest.fn(),
  },
}));

import Alerting from '@/pages/alerting';
const { alertApi } = require('@/lib/api');

const mockAlerts = [
  { id: 'a-1', type: 'error', message: 'Disk space critical', timestamp: '2026-04-13T10:00:00Z' },
  { id: 'a-2', type: 'warning', message: 'Memory high', timestamp: '2026-04-13T10:01:00Z' },
  { id: 'a-3', type: 'info', message: 'Backup completed', timestamp: '2026-04-13T10:02:00Z' },
];

describe('Alerting page', () => {
  beforeEach(() => {
    jest.useFakeTimers();
    jest.clearAllMocks();
    alertApi.listAlerts.mockResolvedValue({ alerts: mockAlerts });
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  it('renders the System Alerts heading', async () => {
    await act(async () => render(<Alerting />));
    // Both h1 and h2 contain "System Alerts"; target the h1 by level
    expect(screen.getByRole('heading', { level: 1, name: /system alerts/i })).toBeInTheDocument();
  });

  it('fetches and displays all alerts on mount', async () => {
    await act(async () => render(<Alerting />));
    await waitFor(() => {
      expect(screen.getByText(/disk space critical/i)).toBeInTheDocument();
      expect(screen.getByText(/memory high/i)).toBeInTheDocument();
      expect(screen.getByText(/backup completed/i)).toBeInTheDocument();
    });
  });

  it('shows alert count in heading', async () => {
    await act(async () => render(<Alerting />));
    await waitFor(() => {
      expect(screen.getByText(/\(3\)/)).toBeInTheDocument();
    });
  });

  it('filters alerts to only error type', async () => {
    await act(async () => render(<Alerting />));
    await waitFor(() => screen.getByText(/disk space critical/i));

    fireEvent.change(screen.getByRole('combobox'), { target: { value: 'error' } });

    expect(screen.getByText(/disk space critical/i)).toBeInTheDocument();
    expect(screen.queryByText(/memory high/i)).not.toBeInTheDocument();
    expect(screen.queryByText(/backup completed/i)).not.toBeInTheDocument();
  });

  it('shows all alerts when filter is reset to "all"', async () => {
    await act(async () => render(<Alerting />));
    await waitFor(() => screen.getByText(/disk space critical/i));

    fireEvent.change(screen.getByRole('combobox'), { target: { value: 'error' } });
    fireEvent.change(screen.getByRole('combobox'), { target: { value: 'all' } });

    expect(screen.getByText(/disk space critical/i)).toBeInTheDocument();
    expect(screen.getByText(/memory high/i)).toBeInTheDocument();
    expect(screen.getByText(/backup completed/i)).toBeInTheDocument();
  });

  it('auto-refreshes every 30 seconds', async () => {
    await act(async () => render(<Alerting />));
    await waitFor(() => expect(alertApi.listAlerts).toHaveBeenCalledTimes(1));

    await act(async () => { jest.advanceTimersByTime(30000); });
    await waitFor(() => expect(alertApi.listAlerts).toHaveBeenCalledTimes(2));
  });

  it('clears interval on unmount', async () => {
    const clearIntervalSpy = jest.spyOn(global, 'clearInterval');
    const { unmount } = await act(async () => render(<Alerting />));
    unmount();
    expect(clearIntervalSpy).toHaveBeenCalled();
    clearIntervalSpy.mockRestore();
  });

  it('handles empty alerts gracefully', async () => {
    alertApi.listAlerts.mockResolvedValueOnce({ alerts: [] });
    await act(async () => render(<Alerting />));
    await waitFor(() => {
      expect(screen.getByText(/\(0\)/)).toBeInTheDocument();
    });
  });
});
