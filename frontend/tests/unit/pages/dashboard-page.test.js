// frontend/tests/unit/pages/dashboard-page.test.js
import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import Dashboard from '@/pages/dashboard';

// Mock the API functions
jest.mock('@/lib/api', () => ({
  clusterApi: {
    listNodes: jest.fn(),
  },
  databaseApi: {
    listDatabases: jest.fn(),
  },
  monitoringApi: {
    systemStatus: jest.fn(),
  },
  securityApi: {
    listAuditLogs: jest.fn(),
  },
  adminApi: {
    shutdownServer: jest.fn(),
  },
  authApi: {
    getCurrentUser: jest.fn(),
  },
  usersApi: {
    getUser: jest.fn(),
  },
}));

import { clusterApi, databaseApi, monitoringApi, securityApi, adminApi, authApi, usersApi } from '@/lib/api';

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

// Mock window.confirm and window.alert
beforeAll(() => {
  jest.spyOn(window, 'confirm').mockImplementation(() => false);
  jest.spyOn(window, 'alert').mockImplementation(() => {});
});

describe('Dashboard Page', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    jest.useFakeTimers();

    // Mock successful API responses
    clusterApi.listNodes.mockResolvedValue({
      nodes: [
        { id: 'node-1', role: 'leader', status: 'active' },
        { id: 'node-2', role: 'worker', status: 'active' },
      ]
    });

    databaseApi.listDatabases.mockResolvedValue({
      databases: [
        {
          id: 'db-1',
          databaseId: 'db-1',
          name: 'Test Database',
          vectorDimension: 128,
          indexType: 'HNSW'
        },
        {
          id: 'db-2',
          databaseId: 'db-2',
          name: 'Production DB',
          vectorDimension: 256,
          indexType: 'IVF'
        }
      ]
    });

    monitoringApi.systemStatus.mockResolvedValue({
      uptime: '5 days',
      cpu: '45%',
      memory: '2.1 GB',
      requests: '15,234'
    });

    securityApi.listAuditLogs.mockResolvedValue({
      logs: [
        { timestamp: '2026-01-19T10:00:00Z', message: 'User login successful' },
        { timestamp: '2026-01-19T09:30:00Z', action: 'Database created' },
      ]
    });

    authApi.getCurrentUser.mockReturnValue({ user_id: 'user-123' });
    usersApi.getUser.mockResolvedValue({ roles: ['user'] });
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  describe('Rendering', () => {
    test('renders dashboard title', async () => {
      render(<Dashboard />);

      expect(screen.getByText('JadeVectorDB Dashboard')).toBeInTheDocument();
    });

    test('renders refresh button', async () => {
      render(<Dashboard />);

      expect(screen.getByRole('button', { name: /refresh/i })).toBeInTheDocument();
    });

    test('shows loading state initially', async () => {
      render(<Dashboard />);

      // The loading text appears in empty state divs
      expect(screen.getAllByText(/loading/i).length).toBeGreaterThan(0);
    });
  });

  describe('Data Fetching', () => {
    test('fetches and displays cluster nodes', async () => {
      render(<Dashboard />);

      await waitFor(() => {
        expect(screen.getByText('node-1')).toBeInTheDocument();
        expect(screen.getByText('node-2')).toBeInTheDocument();
      });

      expect(clusterApi.listNodes).toHaveBeenCalled();
    });

    test('fetches and displays databases', async () => {
      render(<Dashboard />);

      await waitFor(() => {
        expect(screen.getByText('Test Database')).toBeInTheDocument();
        expect(screen.getByText('Production DB')).toBeInTheDocument();
      });

      expect(databaseApi.listDatabases).toHaveBeenCalledWith(5, 0);
    });

    test('fetches and displays system status', async () => {
      render(<Dashboard />);

      await waitFor(() => {
        expect(screen.getByText('5 days')).toBeInTheDocument();
        expect(screen.getByText('45%')).toBeInTheDocument();
        expect(screen.getByText('2.1 GB')).toBeInTheDocument();
      });

      expect(monitoringApi.systemStatus).toHaveBeenCalled();
    });

    test('fetches and displays audit logs', async () => {
      render(<Dashboard />);

      await waitFor(() => {
        expect(screen.getByText('User login successful')).toBeInTheDocument();
        expect(screen.getByText('Database created')).toBeInTheDocument();
      });

      expect(securityApi.listAuditLogs).toHaveBeenCalledWith(5, 0);
    });

    test('displays cluster status with node count', async () => {
      render(<Dashboard />);

      await waitFor(() => {
        expect(screen.getByText(/Cluster Status \(2 nodes\)/)).toBeInTheDocument();
      });
    });

    test('displays database count in header', async () => {
      render(<Dashboard />);

      await waitFor(() => {
        expect(screen.getByText(/Recent Databases \(2\)/)).toBeInTheDocument();
      });
    });
  });

  describe('Empty States', () => {
    test('shows empty state when no nodes found', async () => {
      clusterApi.listNodes.mockResolvedValue({ nodes: [] });

      render(<Dashboard />);

      await waitFor(() => {
        expect(screen.getByText('No cluster nodes found')).toBeInTheDocument();
      });
    });

    test('shows empty state when no databases found', async () => {
      databaseApi.listDatabases.mockResolvedValue({ databases: [] });

      render(<Dashboard />);

      await waitFor(() => {
        expect(screen.getByText('No databases found')).toBeInTheDocument();
      });
    });

    test('shows empty state when no logs found', async () => {
      securityApi.listAuditLogs.mockResolvedValue({ logs: [] });

      render(<Dashboard />);

      await waitFor(() => {
        expect(screen.getByText('No recent logs')).toBeInTheDocument();
      });
    });
  });

  describe('Error Handling', () => {
    test('handles cluster API error gracefully', async () => {
      clusterApi.listNodes.mockRejectedValue(new Error('Network error'));

      render(<Dashboard />);

      // Should still render the page without crashing
      await waitFor(() => {
        expect(screen.getByText('JadeVectorDB Dashboard')).toBeInTheDocument();
      });
    });

    test('handles database API error gracefully', async () => {
      databaseApi.listDatabases.mockRejectedValue(new Error('Network error'));

      render(<Dashboard />);

      await waitFor(() => {
        expect(screen.getByText('JadeVectorDB Dashboard')).toBeInTheDocument();
      });
    });

    test('handles all APIs failing gracefully', async () => {
      clusterApi.listNodes.mockRejectedValue(new Error('Network error'));
      databaseApi.listDatabases.mockRejectedValue(new Error('Network error'));
      monitoringApi.systemStatus.mockRejectedValue(new Error('Network error'));
      securityApi.listAuditLogs.mockRejectedValue(new Error('Network error'));

      render(<Dashboard />);

      // Should still render the page
      await waitFor(() => {
        expect(screen.getByText('JadeVectorDB Dashboard')).toBeInTheDocument();
      });
    });
  });

  describe('User Interactions', () => {
    test('refresh button triggers data fetch', async () => {
      render(<Dashboard />);

      await waitFor(() => {
        expect(screen.getByText('Test Database')).toBeInTheDocument();
      });

      // Clear mocks to track new calls
      jest.clearAllMocks();

      // Click refresh button
      const refreshButton = screen.getByRole('button', { name: /refresh/i });
      fireEvent.click(refreshButton);

      // Verify APIs were called again
      await waitFor(() => {
        expect(clusterApi.listNodes).toHaveBeenCalled();
        expect(databaseApi.listDatabases).toHaveBeenCalled();
      });
    });

    test('refresh button shows loading state while refreshing', async () => {
      render(<Dashboard />);

      await waitFor(() => {
        expect(screen.getByText('Test Database')).toBeInTheDocument();
      });

      // Click refresh - button text should change to "Refreshing..."
      const refreshButton = screen.getByRole('button', { name: /refresh/i });
      fireEvent.click(refreshButton);

      // The button should show refreshing state
      expect(screen.getByRole('button', { name: /refreshing/i })).toBeInTheDocument();
    });

    test('database links point to correct URLs', async () => {
      render(<Dashboard />);

      await waitFor(() => {
        expect(screen.getByText('Test Database')).toBeInTheDocument();
      });

      const dbLink = screen.getByRole('link', { name: 'Test Database' });
      expect(dbLink).toHaveAttribute('href', '/databases/db-1');
    });
  });

  describe('Shutdown Functionality', () => {
    test('shutdown button is visible', async () => {
      render(<Dashboard />);

      await waitFor(() => {
        // There's a test shutdown button that's always visible
        expect(screen.getByRole('button', { name: /shutdown server/i })).toBeInTheDocument();
      });
    });

    test('shutdown requires confirmation', async () => {
      window.confirm.mockReturnValueOnce(false);

      render(<Dashboard />);

      await waitFor(() => {
        expect(screen.getByText('Test Database')).toBeInTheDocument();
      });

      const shutdownButton = screen.getByRole('button', { name: /shutdown server/i });
      fireEvent.click(shutdownButton);

      // Confirm dialog should have been shown
      expect(window.confirm).toHaveBeenCalled();

      // API should NOT have been called since user cancelled
      expect(adminApi.shutdownServer).not.toHaveBeenCalled();
    });

    test('shutdown proceeds when confirmed', async () => {
      window.confirm.mockReturnValueOnce(true);
      adminApi.shutdownServer.mockResolvedValue({});

      render(<Dashboard />);

      await waitFor(() => {
        expect(screen.getByText('Test Database')).toBeInTheDocument();
      });

      const shutdownButton = screen.getByRole('button', { name: /shutdown server/i });
      fireEvent.click(shutdownButton);

      await waitFor(() => {
        expect(adminApi.shutdownServer).toHaveBeenCalled();
      });
    });
  });

  describe('Auto Refresh', () => {
    test('sets up auto refresh interval', async () => {
      render(<Dashboard />);

      await waitFor(() => {
        expect(screen.getByText('Test Database')).toBeInTheDocument();
      });

      // Initial calls
      expect(clusterApi.listNodes).toHaveBeenCalledTimes(1);

      // Advance timer by 30 seconds (auto-refresh interval)
      jest.advanceTimersByTime(30000);

      // Should have called APIs again
      await waitFor(() => {
        expect(clusterApi.listNodes).toHaveBeenCalledTimes(2);
      });
    });
  });

  describe('Admin Features', () => {
    test('fetches user roles on mount', async () => {
      render(<Dashboard />);

      await waitFor(() => {
        expect(authApi.getCurrentUser).toHaveBeenCalled();
        expect(usersApi.getUser).toHaveBeenCalledWith('user-123');
      });
    });

    test('shows admin shutdown button when user is admin', async () => {
      usersApi.getUser.mockResolvedValue({ roles: ['admin'] });

      render(<Dashboard />);

      await waitFor(() => {
        // Admin should see the actual shutdown button
        const shutdownButtons = screen.getAllByRole('button', { name: /shutdown server/i });
        expect(shutdownButtons.length).toBeGreaterThan(0);
      });
    });
  });

  describe('Node Status Display', () => {
    test('displays node roles correctly', async () => {
      render(<Dashboard />);

      await waitFor(() => {
        expect(screen.getByText('leader')).toBeInTheDocument();
        expect(screen.getByText('worker')).toBeInTheDocument();
      });
    });

    test('displays node status badges', async () => {
      render(<Dashboard />);

      await waitFor(() => {
        const activeBadges = screen.getAllByText('active');
        expect(activeBadges.length).toBe(2);
      });
    });
  });

  describe('System Status Metrics', () => {
    test('displays uptime metric', async () => {
      render(<Dashboard />);

      await waitFor(() => {
        expect(screen.getByText('5 days')).toBeInTheDocument();
        expect(screen.getByText('Uptime')).toBeInTheDocument();
      });
    });

    test('displays CPU usage metric', async () => {
      render(<Dashboard />);

      await waitFor(() => {
        expect(screen.getByText('45%')).toBeInTheDocument();
        expect(screen.getByText('CPU Usage')).toBeInTheDocument();
      });
    });

    test('displays memory usage metric', async () => {
      render(<Dashboard />);

      await waitFor(() => {
        expect(screen.getByText('2.1 GB')).toBeInTheDocument();
        expect(screen.getByText('Memory Usage')).toBeInTheDocument();
      });
    });

    test('displays total requests metric', async () => {
      render(<Dashboard />);

      await waitFor(() => {
        expect(screen.getByText('15,234')).toBeInTheDocument();
        expect(screen.getByText('Total Requests')).toBeInTheDocument();
      });
    });
  });
});
