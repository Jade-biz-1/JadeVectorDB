// frontend/tests/unit/pages/dashboard-page.test.js
import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import Dashboard from '@/pages/dashboard';

// Mock the API functions
jest.mock('@/lib/api', () => ({
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

import { databaseApi, monitoringApi, securityApi, adminApi, authApi, usersApi } from '@/lib/api';

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
      status: 'operational',
      uptime: '5 days',
      version: '1.0.0',
      performance: {
        database_count: 2,
        total_vectors: 100000
      },
      system: {
        cpu_usage_percent: 45,
        memory_usage_percent: 60,
        disk_usage_percent: 35
      }
    });

    securityApi.listAuditLogs.mockResolvedValue({
      events: [
        { timestamp: '2026-01-19T10:00:00Z', event_type: 'AUTH_SUCCESS', details: 'User login successful', user_id: 'user-1' },
        { timestamp: '2026-01-19T09:30:00Z', event_type: 'DATA_MODIFICATION', details: 'Database created', user_id: 'user-1' },
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
    test('fetches and displays databases', async () => {
      render(<Dashboard />);

      await waitFor(() => {
        expect(screen.getByText('Test Database')).toBeInTheDocument();
        expect(screen.getByText('Production DB')).toBeInTheDocument();
      });

      expect(databaseApi.listDatabases).toHaveBeenCalledWith(5, 0);
    });

    test('fetches and displays server info', async () => {
      render(<Dashboard />);

      await waitFor(() => {
        expect(screen.getByText('5 days')).toBeInTheDocument();
        expect(screen.getByText('1.0.0')).toBeInTheDocument();
      });

      expect(monitoringApi.systemStatus).toHaveBeenCalled();
    });

    test('fetches and displays audit logs', async () => {
      render(<Dashboard />);

      await waitFor(() => {
        expect(screen.getByText(/AUTH_SUCCESS/)).toBeInTheDocument();
        expect(screen.getByText(/DATA_MODIFICATION/)).toBeInTheDocument();
      });

      expect(securityApi.listAuditLogs).toHaveBeenCalledWith(5, 0);
    });

    test('displays server info card', async () => {
      render(<Dashboard />);

      await waitFor(() => {
        expect(screen.getByText('Server Info')).toBeInTheDocument();
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
    test('shows empty state when no databases found', async () => {
      databaseApi.listDatabases.mockResolvedValue({ databases: [] });

      render(<Dashboard />);

      await waitFor(() => {
        expect(screen.getByText('No databases found')).toBeInTheDocument();
      });
    });

    test('shows empty state when no logs found', async () => {
      securityApi.listAuditLogs.mockResolvedValue({ events: [] });

      render(<Dashboard />);

      await waitFor(() => {
        expect(screen.getByText('No recent logs')).toBeInTheDocument();
      });
    });

    test('shows server status unavailable when systemStatus is null', async () => {
      monitoringApi.systemStatus.mockResolvedValue(null);

      render(<Dashboard />);

      await waitFor(() => {
        expect(screen.getByText('Server status unavailable')).toBeInTheDocument();
      });
    });
  });

  describe('Error Handling', () => {
    test('handles database API error gracefully', async () => {
      databaseApi.listDatabases.mockRejectedValue(new Error('Network error'));

      render(<Dashboard />);

      await waitFor(() => {
        expect(screen.getByText('JadeVectorDB Dashboard')).toBeInTheDocument();
      });
    });

    test('handles all APIs failing gracefully', async () => {
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
        expect(databaseApi.listDatabases).toHaveBeenCalled();
        expect(monitoringApi.systemStatus).toHaveBeenCalled();
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
      expect(databaseApi.listDatabases).toHaveBeenCalledTimes(1);

      // Advance timer by 30 seconds (auto-refresh interval)
      jest.advanceTimersByTime(30000);

      // Should have called APIs again
      await waitFor(() => {
        expect(databaseApi.listDatabases).toHaveBeenCalledTimes(2);
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

  describe('System Resources Metrics', () => {
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
        expect(screen.getByText('60%')).toBeInTheDocument();
        expect(screen.getByText('Memory Usage')).toBeInTheDocument();
      });
    });

    test('displays total vectors metric', async () => {
      render(<Dashboard />);

      await waitFor(() => {
        expect(screen.getByText('100000')).toBeInTheDocument();
        expect(screen.getByText('Total Vectors')).toBeInTheDocument();
      });
    });

    test('displays disk usage metric', async () => {
      render(<Dashboard />);

      await waitFor(() => {
        expect(screen.getByText('35%')).toBeInTheDocument();
        expect(screen.getByText('Disk Usage')).toBeInTheDocument();
      });
    });
  });

  describe('Audit Log Display', () => {
    test('displays event_type and details in log entries', async () => {
      render(<Dashboard />);

      await waitFor(() => {
        expect(screen.getByText(/AUTH_SUCCESS.*User login successful/)).toBeInTheDocument();
        expect(screen.getByText(/DATA_MODIFICATION.*Database created/)).toBeInTheDocument();
      });
    });

    test('displays user_id in log time', async () => {
      render(<Dashboard />);

      await waitFor(() => {
        const matches = screen.getAllByText(/user-1/);
        expect(matches.length).toBeGreaterThan(0);
      });
    });
  });
});
