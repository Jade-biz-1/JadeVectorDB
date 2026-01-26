// frontend/tests/unit/pages/cluster-page.test.js
import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import ClusterManagement from '@/pages/cluster';

// Mock the API functions
jest.mock('@/lib/api', () => ({
  clusterApi: {
    listNodes: jest.fn(),
    getNodeStatus: jest.fn(),
  }
}));

import { clusterApi } from '@/lib/api';

// Mock localStorage
Object.defineProperty(window, 'localStorage', {
  value: {
    getItem: jest.fn(() => 'test-auth-token'),
  },
  writable: true,
});

// Mock next/router
jest.mock('next/router', () => ({
  useRouter: () => ({
    query: {},
    push: jest.fn(),
  })
}));

// Mock alert
beforeAll(() => {
  jest.spyOn(window, 'alert').mockImplementation(() => {});
});

describe('Cluster Management Page', () => {
  const mockNodes = [
    {
      id: 'node-1',
      role: 'primary',
      status: 'active',
      cpu: 8,
      memory: 32,
      storage: 500,
      network: 1000
    },
    {
      id: 'node-2',
      role: 'worker',
      status: 'active',
      cpu: 4,
      memory: 16,
      storage: 250,
      network: 1000
    },
    {
      id: 'node-3',
      role: 'worker',
      status: 'inactive',
      cpu: 4,
      memory: 16,
      storage: 250,
      network: 1000
    }
  ];

  const mockNodeDetails = {
    id: 'node-1',
    role: 'primary',
    status: 'active',
    cpu: 8,
    memory: 32,
    storage: 500,
    network: 1000,
    uptime: '10 days',
    lastHealthCheck: '2026-01-20T10:00:00Z',
    metrics: {
      cpuUsage: 45,
      memoryUsage: 60,
      diskUsage: 30
    }
  };

  beforeEach(() => {
    jest.clearAllMocks();
    jest.useFakeTimers();

    // Mock successful API responses
    clusterApi.listNodes.mockResolvedValue({ nodes: mockNodes });
    clusterApi.getNodeStatus.mockResolvedValue(mockNodeDetails);
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  describe('Rendering', () => {
    test('renders page title', async () => {
      render(<ClusterManagement />);
      expect(screen.getByText('Cluster Management')).toBeInTheDocument();
    });

    test('renders node status heading with count', async () => {
      render(<ClusterManagement />);

      await waitFor(() => {
        expect(screen.getByText(/Node Status \(3 nodes\)/)).toBeInTheDocument();
      });
    });

    test('renders refresh button', async () => {
      render(<ClusterManagement />);

      await waitFor(() => {
        expect(screen.getByRole('button', { name: /refresh/i })).toBeInTheDocument();
      });
    });

    test('renders table headers', async () => {
      render(<ClusterManagement />);

      expect(screen.getByText('Node ID')).toBeInTheDocument();
      expect(screen.getByText('Role')).toBeInTheDocument();
      expect(screen.getByText('Status')).toBeInTheDocument();
      expect(screen.getByText(/CPU/)).toBeInTheDocument();
      expect(screen.getByText(/Memory/)).toBeInTheDocument();
      expect(screen.getByText(/Storage/)).toBeInTheDocument();
      expect(screen.getByText(/Network/)).toBeInTheDocument();
      expect(screen.getByText('Actions')).toBeInTheDocument();
    });
  });

  describe('Data Fetching', () => {
    test('fetches nodes on mount', async () => {
      render(<ClusterManagement />);

      await waitFor(() => {
        expect(clusterApi.listNodes).toHaveBeenCalled();
      });
    });

    test('displays nodes in table', async () => {
      render(<ClusterManagement />);

      await waitFor(() => {
        expect(screen.getByText('node-1')).toBeInTheDocument();
        expect(screen.getByText('node-2')).toBeInTheDocument();
        expect(screen.getByText('node-3')).toBeInTheDocument();
      });
    });

    test('displays node roles', async () => {
      render(<ClusterManagement />);

      await waitFor(() => {
        expect(screen.getByText('primary')).toBeInTheDocument();
        const workerBadges = screen.getAllByText('worker');
        expect(workerBadges.length).toBe(2);
      });
    });

    test('displays node statuses', async () => {
      render(<ClusterManagement />);

      await waitFor(() => {
        const activeBadges = screen.getAllByText('active');
        expect(activeBadges.length).toBe(2);
        expect(screen.getByText('inactive')).toBeInTheDocument();
      });
    });

    test('displays node resources', async () => {
      render(<ClusterManagement />);

      await waitFor(() => {
        expect(screen.getByText('8')).toBeInTheDocument(); // CPU
        expect(screen.getByText('32')).toBeInTheDocument(); // Memory
        expect(screen.getByText('500')).toBeInTheDocument(); // Storage
      });
    });

    test('shows loading state', async () => {
      clusterApi.listNodes.mockImplementation(() => new Promise(() => {}));

      render(<ClusterManagement />);

      expect(screen.getByText('Loading nodes...')).toBeInTheDocument();
    });

    test('shows empty state when no nodes', async () => {
      clusterApi.listNodes.mockResolvedValue({ nodes: [] });

      render(<ClusterManagement />);

      await waitFor(() => {
        expect(screen.getByText('No nodes found.')).toBeInTheDocument();
      });
    });

    test('handles API error gracefully', async () => {
      clusterApi.listNodes.mockRejectedValue(new Error('Network error'));

      render(<ClusterManagement />);

      await waitFor(() => {
        expect(screen.getByText('No nodes found.')).toBeInTheDocument();
      });
    });
  });

  describe('Auto-refresh', () => {
    test('auto-refreshes every 15 seconds', async () => {
      render(<ClusterManagement />);

      await waitFor(() => {
        expect(screen.getByText('node-1')).toBeInTheDocument();
      });

      expect(clusterApi.listNodes).toHaveBeenCalledTimes(1);

      // Advance timer by 15 seconds
      jest.advanceTimersByTime(15000);

      await waitFor(() => {
        expect(clusterApi.listNodes).toHaveBeenCalledTimes(2);
      });
    });
  });

  describe('Refresh Functionality', () => {
    test('refresh button triggers data fetch', async () => {
      render(<ClusterManagement />);

      await waitFor(() => {
        expect(screen.getByText('node-1')).toBeInTheDocument();
      });

      jest.clearAllMocks();

      const refreshButton = screen.getByRole('button', { name: /refresh/i });
      fireEvent.click(refreshButton);

      await waitFor(() => {
        expect(clusterApi.listNodes).toHaveBeenCalled();
      });
    });

    test('shows loading state during refresh', async () => {
      render(<ClusterManagement />);

      await waitFor(() => {
        expect(screen.getByText('node-1')).toBeInTheDocument();
      });

      clusterApi.listNodes.mockImplementation(() => new Promise(() => {}));

      const refreshButton = screen.getByRole('button', { name: /refresh/i });
      fireEvent.click(refreshButton);

      expect(screen.getByRole('button', { name: /refreshing/i })).toBeInTheDocument();
    });

    test('disables refresh button during loading', async () => {
      render(<ClusterManagement />);

      await waitFor(() => {
        expect(screen.getByText('node-1')).toBeInTheDocument();
      });

      clusterApi.listNodes.mockImplementation(() => new Promise(() => {}));

      const refreshButton = screen.getByRole('button', { name: /refresh/i });
      fireEvent.click(refreshButton);

      expect(screen.getByRole('button', { name: /refreshing/i })).toBeDisabled();
    });
  });

  describe('Node Details', () => {
    test('view details button fetches node status', async () => {
      render(<ClusterManagement />);

      await waitFor(() => {
        expect(screen.getByText('node-1')).toBeInTheDocument();
      });

      const viewDetailsButtons = screen.getAllByRole('button', { name: /view details/i });
      fireEvent.click(viewDetailsButtons[0]);

      await waitFor(() => {
        expect(clusterApi.getNodeStatus).toHaveBeenCalledWith('node-1');
      });
    });

    test('shows node details panel after clicking view details', async () => {
      render(<ClusterManagement />);

      await waitFor(() => {
        expect(screen.getByText('node-1')).toBeInTheDocument();
      });

      const viewDetailsButtons = screen.getAllByRole('button', { name: /view details/i });
      fireEvent.click(viewDetailsButtons[0]);

      await waitFor(() => {
        expect(screen.getByText(/Node Details: node-1/)).toBeInTheDocument();
      });
    });

    test('displays node details as JSON', async () => {
      render(<ClusterManagement />);

      await waitFor(() => {
        expect(screen.getByText('node-1')).toBeInTheDocument();
      });

      const viewDetailsButtons = screen.getAllByRole('button', { name: /view details/i });
      fireEvent.click(viewDetailsButtons[0]);

      await waitFor(() => {
        // Check for some of the JSON content
        expect(screen.getByText(/10 days/)).toBeInTheDocument(); // uptime
      });
    });

    test('close button hides node details panel', async () => {
      render(<ClusterManagement />);

      await waitFor(() => {
        expect(screen.getByText('node-1')).toBeInTheDocument();
      });

      const viewDetailsButtons = screen.getAllByRole('button', { name: /view details/i });
      fireEvent.click(viewDetailsButtons[0]);

      await waitFor(() => {
        expect(screen.getByText(/Node Details: node-1/)).toBeInTheDocument();
      });

      const closeButton = screen.getByRole('button', { name: /close/i });
      fireEvent.click(closeButton);

      expect(screen.queryByText(/Node Details: node-1/)).not.toBeInTheDocument();
    });

    test('shows alert on node details error', async () => {
      clusterApi.getNodeStatus.mockRejectedValue(new Error('Node not found'));

      render(<ClusterManagement />);

      await waitFor(() => {
        expect(screen.getByText('node-1')).toBeInTheDocument();
      });

      const viewDetailsButtons = screen.getAllByRole('button', { name: /view details/i });
      fireEvent.click(viewDetailsButtons[0]);

      await waitFor(() => {
        expect(window.alert).toHaveBeenCalledWith(expect.stringContaining('Node not found'));
      });
    });
  });

  describe('Node Status Display', () => {
    test('displays default role when not provided', async () => {
      clusterApi.listNodes.mockResolvedValue({
        nodes: [{ id: 'node-no-role', status: 'active' }]
      });

      render(<ClusterManagement />);

      await waitFor(() => {
        expect(screen.getByText('worker')).toBeInTheDocument();
      });
    });

    test('displays unknown status when not provided', async () => {
      clusterApi.listNodes.mockResolvedValue({
        nodes: [{ id: 'node-no-status', role: 'worker' }]
      });

      render(<ClusterManagement />);

      await waitFor(() => {
        expect(screen.getByText('unknown')).toBeInTheDocument();
      });
    });

    test('displays N/A for missing resource values', async () => {
      clusterApi.listNodes.mockResolvedValue({
        nodes: [{ id: 'node-minimal', role: 'worker', status: 'active' }]
      });

      render(<ClusterManagement />);

      await waitFor(() => {
        const naValues = screen.getAllByText('N/A');
        expect(naValues.length).toBeGreaterThanOrEqual(4); // cpu, memory, storage, network
      });
    });
  });
});
