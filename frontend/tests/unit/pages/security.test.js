import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';

// Global mocks: next/head, next/router, next/link via jest.config.js
jest.mock('@/lib/api', () => require('./__mocks__/api'));
const { securityApi } = require('@/lib/api');

import SecurityMonitoring from '@/pages/security';

const mockLogs = [
  { id: 'log-1', timestamp: '2026-04-13T10:00:00Z', user: 'alice', event: 'login', status: 'success' },
  { id: 'log-2', timestamp: '2026-04-13T10:05:00Z', user: 'bob', event: 'database.delete', status: 'failure' },
];

describe('SecurityMonitoring page', () => {
  beforeEach(() => jest.clearAllMocks());

  it('renders the Security Monitoring heading', () => {
    securityApi.listAuditLogs.mockResolvedValue({ logs: [] });
    render(<SecurityMonitoring />);
    // Use heading role to avoid matching the (now suppressed) <title> tag
    expect(screen.getByRole('heading', { name: /security monitoring/i })).toBeInTheDocument();
  });

  it('renders the audit logs table headers', () => {
    securityApi.listAuditLogs.mockResolvedValue({ logs: [] });
    render(<SecurityMonitoring />);
    expect(screen.getByText(/audit logs/i)).toBeInTheDocument();
    expect(screen.getByText(/timestamp/i)).toBeInTheDocument();
    expect(screen.getByText(/event/i)).toBeInTheDocument();
    expect(screen.getByText(/status/i)).toBeInTheDocument();
  });

  it('shows loading state while fetching', () => {
    securityApi.listAuditLogs.mockImplementation(() => new Promise(() => {}));
    render(<SecurityMonitoring />);
    expect(screen.getByText(/loading/i)).toBeInTheDocument();
  });

  it('shows empty state when no audit logs exist', async () => {
    securityApi.listAuditLogs.mockResolvedValue({ logs: [] });
    render(<SecurityMonitoring />);
    await waitFor(() => {
      expect(screen.getByText(/no audit logs found/i)).toBeInTheDocument();
    });
  });

  it('renders audit log entries', async () => {
    securityApi.listAuditLogs.mockResolvedValue({ logs: mockLogs });
    render(<SecurityMonitoring />);
    await waitFor(() => {
      expect(screen.getByText('alice')).toBeInTheDocument();
      expect(screen.getByText('login')).toBeInTheDocument();
      expect(screen.getByText('bob')).toBeInTheDocument();
      expect(screen.getByText('database.delete')).toBeInTheDocument();
    });
  });

  it('shows success status with green style', async () => {
    securityApi.listAuditLogs.mockResolvedValue({ logs: [mockLogs[0]] });
    render(<SecurityMonitoring />);
    await waitFor(() => {
      const badge = screen.getByText('success');
      expect(badge).toHaveClass('bg-green-100');
      expect(badge).toHaveClass('text-green-800');
    });
  });

  it('shows failure status with red style', async () => {
    securityApi.listAuditLogs.mockResolvedValue({ logs: [mockLogs[1]] });
    render(<SecurityMonitoring />);
    await waitFor(() => {
      const badge = screen.getByText('failure');
      expect(badge).toHaveClass('bg-red-100');
      expect(badge).toHaveClass('text-red-800');
    });
  });

  it('handles API error gracefully (no crash)', async () => {
    const consoleSpy = jest.spyOn(console, 'error').mockImplementation(() => {});
    securityApi.listAuditLogs.mockRejectedValueOnce(new Error('Unauthorized'));
    render(<SecurityMonitoring />);
    await waitFor(() => {
      expect(consoleSpy).toHaveBeenCalled();
    });
    consoleSpy.mockRestore();
  });
});
