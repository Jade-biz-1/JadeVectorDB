import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';

// Global mocks: next/head, next/router, next/link via jest.config.js
jest.mock('@/lib/api', () => require('./__mocks__/api'));
const { clusterApi } = require('@/lib/api');

import ClusterManagement from '@/pages/cluster';

const mockNodes = [
  { nodeId: 'node-1', host: 'localhost', port: 8080, status: 'active', role: 'leader' },
  { nodeId: 'node-2', host: 'host2', port: 8080, status: 'active', role: 'follower' },
];

describe('ClusterManagement page', () => {
  beforeEach(() => {
    jest.useFakeTimers();
    jest.clearAllMocks();
    clusterApi.listNodes.mockResolvedValue({ nodes: mockNodes });
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  it('renders the Cluster Management heading', async () => {
    await act(async () => render(<ClusterManagement />));
    expect(screen.getByRole('heading', { name: /cluster management/i })).toBeInTheDocument();
  });

  it('fetches and displays cluster node count on mount', async () => {
    await act(async () => render(<ClusterManagement />));
    await waitFor(() => {
      expect(screen.getByText(/2 nodes/i)).toBeInTheDocument();
    });
  });

  it('shows 0 nodes when none returned', async () => {
    clusterApi.listNodes.mockResolvedValueOnce({ nodes: [] });
    await act(async () => render(<ClusterManagement />));
    await waitFor(() => {
      expect(screen.getByText(/0 nodes/i)).toBeInTheDocument();
    });
  });

  it('auto-refreshes nodes every 15 seconds', async () => {
    await act(async () => render(<ClusterManagement />));
    await waitFor(() => expect(clusterApi.listNodes).toHaveBeenCalledTimes(1));

    await act(async () => { jest.advanceTimersByTime(15000); });
    await waitFor(() => expect(clusterApi.listNodes).toHaveBeenCalledTimes(2));
  });

  it('clears the interval on unmount', async () => {
    const clearIntervalSpy = jest.spyOn(global, 'clearInterval');
    const { unmount } = await act(async () => render(<ClusterManagement />));
    unmount();
    expect(clearIntervalSpy).toHaveBeenCalled();
    clearIntervalSpy.mockRestore();
  });

  it('refresh button triggers refetch', async () => {
    await act(async () => render(<ClusterManagement />));
    await waitFor(() => expect(clusterApi.listNodes).toHaveBeenCalledTimes(1));

    const refreshBtn = screen.getByRole('button', { name: /refresh/i });
    await act(async () => { fireEvent.click(refreshBtn); });

    await waitFor(() => expect(clusterApi.listNodes).toHaveBeenCalledTimes(2));
  });
});
