// frontend/tests/unit/pages/monitoring-page.test.js
import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import MonitoringDashboard from '@/pages/monitoring';

// Mock the API functions
jest.mock('@/lib/api', () => ({
  monitoringApi: {
    systemStatus: jest.fn(),
  },
  databaseApi: {
    listDatabases: jest.fn(),
  }
}));

import { monitoringApi, databaseApi } from '@/lib/api';

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

describe('Monitoring Dashboard Page', () => {
  const mockSystemStatus = {
    status: 'operational',
    checks: {
      database: 'ok',
      storage: 'ok',
      network: 'ok',
      memory: 'ok',
      cpu: 'ok'
    },
    metrics: {
      totalDatabases: 5,
      totalVectors: 150000,
      queriesPerSecond: 42,
      avgQueryLatencyMs: 15,
      storageUtilizationPercent: 35,
      uptime: '5 days'
    },
    timestamp: '2026-01-20T10:00:00Z'
  };

  const mockDatabases = [
    {
      databaseId: 'db-1',
      name: 'Production DB',
      status: 'online',
      stats: { vectorCount: 100000, indexCount: 3, storageSize: '2.5 GB' }
    },
    {
      databaseId: 'db-2',
      name: 'Test DB',
      status: 'online',
      stats: { vectorCount: 50000, indexCount: 2, storageSize: '1.2 GB' }
    }
  ];

  beforeEach(() => {
    jest.clearAllMocks();
    jest.useFakeTimers();

    // Mock successful API responses
    monitoringApi.systemStatus.mockResolvedValue(mockSystemStatus);
    databaseApi.listDatabases.mockResolvedValue({ databases: mockDatabases });
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  describe('Rendering', () => {
    test('renders page title', async () => {
      render(<MonitoringDashboard />);
      expect(screen.getByText('System Monitoring')).toBeInTheDocument();
    });

    test('renders refresh button', async () => {
      render(<MonitoringDashboard />);
      // Button text starts as "Refresh Status" then changes to "Refreshing..." during load
      await waitFor(() => {
        const button = screen.getByRole('button', { name: /refresh/i });
        expect(button).toBeInTheDocument();
      });
    });

    test('renders system overview section', async () => {
      render(<MonitoringDashboard />);
      expect(screen.getByText('System Overview')).toBeInTheDocument();
    });

    test('renders performance metrics section', async () => {
      render(<MonitoringDashboard />);
      expect(screen.getByText('Performance Metrics')).toBeInTheDocument();
    });

    test('renders health checks section', async () => {
      render(<MonitoringDashboard />);
      expect(screen.getByText('Health Checks')).toBeInTheDocument();
    });

    test('renders database status section', async () => {
      render(<MonitoringDashboard />);
      expect(screen.getByText('Database Status')).toBeInTheDocument();
    });
  });

  describe('System Status', () => {
    test('fetches and displays system status', async () => {
      render(<MonitoringDashboard />);

      await waitFor(() => {
        expect(screen.getByText('Operational')).toBeInTheDocument();
      });

      expect(monitoringApi.systemStatus).toHaveBeenCalled();
    });

    test('displays total databases metric', async () => {
      render(<MonitoringDashboard />);

      await waitFor(() => {
        expect(screen.getByText('5')).toBeInTheDocument();
        expect(screen.getByText('Total Databases')).toBeInTheDocument();
      });
    });

    test('displays total vectors metric', async () => {
      render(<MonitoringDashboard />);

      await waitFor(() => {
        expect(screen.getByText('150,000')).toBeInTheDocument();
        expect(screen.getByText('Total Vectors')).toBeInTheDocument();
      });
    });

    test('displays system uptime', async () => {
      render(<MonitoringDashboard />);

      await waitFor(() => {
        expect(screen.getByText('5 days')).toBeInTheDocument();
        expect(screen.getByText('System Uptime')).toBeInTheDocument();
      });
    });
  });

  describe('Health Checks', () => {
    test('displays health check items', async () => {
      render(<MonitoringDashboard />);

      await waitFor(() => {
        expect(screen.getByText('database')).toBeInTheDocument();
        expect(screen.getByText('storage')).toBeInTheDocument();
        expect(screen.getByText('network')).toBeInTheDocument();
        expect(screen.getByText('memory')).toBeInTheDocument();
        expect(screen.getByText('cpu')).toBeInTheDocument();
      });
    });

    test('displays health check status badges', async () => {
      render(<MonitoringDashboard />);

      await waitFor(() => {
        const okBadges = screen.getAllByText('ok');
        expect(okBadges.length).toBe(5);
      });
    });

    test('shows empty state when no health checks', async () => {
      monitoringApi.systemStatus.mockResolvedValue({
        status: 'operational',
        checks: {},
        timestamp: new Date().toISOString()
      });

      render(<MonitoringDashboard />);

      await waitFor(() => {
        expect(screen.getByText(/No health check data available/)).toBeInTheDocument();
      });
    });
  });

  describe('Performance Metrics', () => {
    test('displays queries per second', async () => {
      render(<MonitoringDashboard />);

      await waitFor(() => {
        expect(screen.getByText('Queries Per Second')).toBeInTheDocument();
        expect(screen.getByText('42')).toBeInTheDocument();
      });
    });

    test('displays average query time', async () => {
      render(<MonitoringDashboard />);

      await waitFor(() => {
        expect(screen.getByText('Avg Query Time (ms)')).toBeInTheDocument();
        expect(screen.getByText('15')).toBeInTheDocument();
      });
    });

    test('displays storage utilization', async () => {
      render(<MonitoringDashboard />);

      await waitFor(() => {
        expect(screen.getByText('Storage Utilization')).toBeInTheDocument();
        expect(screen.getByText('35%')).toBeInTheDocument();
      });
    });
  });

  describe('Database Status Table', () => {
    test('fetches and displays databases', async () => {
      render(<MonitoringDashboard />);

      await waitFor(() => {
        expect(screen.getByText('Production DB')).toBeInTheDocument();
        expect(screen.getByText('Test DB')).toBeInTheDocument();
      });

      expect(databaseApi.listDatabases).toHaveBeenCalled();
    });

    test('displays database table headers', async () => {
      render(<MonitoringDashboard />);

      await waitFor(() => {
        expect(screen.getByText('Production DB')).toBeInTheDocument();
      });

      expect(screen.getByText('Database Name')).toBeInTheDocument();
      // "Status" appears multiple times, so check within the table header context
      expect(screen.getByRole('columnheader', { name: /status/i })).toBeInTheDocument();
      expect(screen.getByRole('columnheader', { name: /vectors/i })).toBeInTheDocument();
      expect(screen.getByRole('columnheader', { name: /indexes/i })).toBeInTheDocument();
      expect(screen.getByRole('columnheader', { name: /storage/i })).toBeInTheDocument();
    });

    test('displays database vector counts', async () => {
      render(<MonitoringDashboard />);

      await waitFor(() => {
        expect(screen.getByText('100,000')).toBeInTheDocument();
        expect(screen.getByText('50,000')).toBeInTheDocument();
      });
    });

    test('shows empty state when no databases', async () => {
      databaseApi.listDatabases.mockResolvedValue({ databases: [] });

      render(<MonitoringDashboard />);

      await waitFor(() => {
        expect(screen.getByText(/No databases found/)).toBeInTheDocument();
      });
    });
  });

  describe('Refresh Functionality', () => {
    test('refresh button triggers data fetch', async () => {
      render(<MonitoringDashboard />);

      await waitFor(() => {
        expect(screen.getByText('Operational')).toBeInTheDocument();
      });

      jest.clearAllMocks();

      const refreshButton = screen.getByRole('button', { name: /refresh status/i });
      fireEvent.click(refreshButton);

      await waitFor(() => {
        expect(monitoringApi.systemStatus).toHaveBeenCalled();
        expect(databaseApi.listDatabases).toHaveBeenCalled();
      });
    });

    test('shows loading state during refresh', async () => {
      render(<MonitoringDashboard />);

      await waitFor(() => {
        expect(screen.getByText('Operational')).toBeInTheDocument();
      });

      // Make API delay
      monitoringApi.systemStatus.mockImplementation(() => new Promise(() => {}));

      const refreshButton = screen.getByRole('button', { name: /refresh status/i });
      fireEvent.click(refreshButton);

      expect(screen.getByRole('button', { name: /refreshing/i })).toBeInTheDocument();
    });

    test('auto-refreshes every 30 seconds', async () => {
      render(<MonitoringDashboard />);

      await waitFor(() => {
        expect(screen.getByText('Operational')).toBeInTheDocument();
      });

      expect(monitoringApi.systemStatus).toHaveBeenCalledTimes(1);

      // Advance timer by 30 seconds
      jest.advanceTimersByTime(30000);

      await waitFor(() => {
        expect(monitoringApi.systemStatus).toHaveBeenCalledTimes(2);
      });
    });

    test('displays last updated time', async () => {
      render(<MonitoringDashboard />);

      await waitFor(() => {
        expect(screen.getByText(/Last updated:/)).toBeInTheDocument();
      });
    });
  });

  describe('Error Handling', () => {
    test('handles system status API error', async () => {
      monitoringApi.systemStatus.mockRejectedValue(new Error('Network error'));

      render(<MonitoringDashboard />);

      await waitFor(() => {
        expect(screen.getByText('Unknown')).toBeInTheDocument();
      });
    });

    test('handles database API error', async () => {
      databaseApi.listDatabases.mockRejectedValue(new Error('Network error'));

      render(<MonitoringDashboard />);

      await waitFor(() => {
        expect(screen.getByText(/No databases found/)).toBeInTheDocument();
      });
    });

    test('still renders page when APIs fail', async () => {
      monitoringApi.systemStatus.mockRejectedValue(new Error('Network error'));
      databaseApi.listDatabases.mockRejectedValue(new Error('Network error'));

      render(<MonitoringDashboard />);

      expect(screen.getByText('System Monitoring')).toBeInTheDocument();
      expect(screen.getByText('System Overview')).toBeInTheDocument();
    });
  });

  describe('Recent Activity', () => {
    test('displays recent activity section', async () => {
      render(<MonitoringDashboard />);

      expect(screen.getByText('Recent Activity')).toBeInTheDocument();
    });

    test('shows activity items', async () => {
      render(<MonitoringDashboard />);

      expect(screen.getByText('System Started')).toBeInTheDocument();
      expect(screen.getByText('Database Activity')).toBeInTheDocument();
      expect(screen.getByText('Monitoring Active')).toBeInTheDocument();
    });
  });
});
