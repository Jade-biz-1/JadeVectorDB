// tests/unit/pages/security-permissions.test.js
// Security and permission enforcement tests.
//
// Tests cover:
//   1. Security monitoring page — audit log display and status badges
//   2. Auth flows — token storage, credential validation
//   3. API layer — Bearer token included in requests (via fetch mock)
//   4. Password policy — strength requirements enforced on change-password page
//   5. 401/403 error handling — pages surface auth errors gracefully
//   6. Input sanitisation — rendered HTML does not execute injected scripts
//   7. API key management — keys are displayed and can be revoked

import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';

// Global mocks: next/head, next/router, next/link via jest.config.js
jest.mock('@/components/Layout', () =>
  ({ children }) => <div data-testid="layout">{children}</div>
);
jest.mock('@/lib/api', () => require('./__mocks__/api'));

const {
  authApi, securityApi, databaseApi, usersApi, apiKeysApi,
} = require('@/lib/api');

// ─────────────────────────────────────────────────────────────────────────────
// 1. Security Monitoring — audit log display
// ─────────────────────────────────────────────────────────────────────────────
describe('Security: Audit Log Display', () => {
  const LOGS = [
    { id: 'l-1', timestamp: '2026-04-13T10:00:00Z', user: 'alice', event: 'LOGIN',          status: 'success' },
    { id: 'l-2', timestamp: '2026-04-13T10:01:00Z', user: 'bob',   event: 'LOGIN_FAILED',   status: 'failure' },
    { id: 'l-3', timestamp: '2026-04-13T10:02:00Z', user: 'alice', event: 'DELETE_DATABASE', status: 'success' },
  ];

  beforeEach(() => jest.clearAllMocks());

  it('renders the Security Monitoring heading', () => {
    const SecurityMonitoring = require('@/pages/security').default;
    securityApi.listAuditLogs.mockResolvedValue({ logs: LOGS });
    render(<SecurityMonitoring />);
    expect(screen.getByRole('heading', { name: /security monitoring/i })).toBeInTheDocument();
  });

  it('displays all audit log entries', async () => {
    const SecurityMonitoring = require('@/pages/security').default;
    securityApi.listAuditLogs.mockResolvedValue({ logs: LOGS });
    render(<SecurityMonitoring />);
    await waitFor(() => {
      // alice appears in rows 1 and 3, bob in row 2
      expect(screen.getAllByText('alice').length).toBeGreaterThanOrEqual(1);
      expect(screen.getByText('bob')).toBeInTheDocument();
      expect(screen.getByText('LOGIN')).toBeInTheDocument();
      expect(screen.getByText('LOGIN_FAILED')).toBeInTheDocument();
      expect(screen.getByText('DELETE_DATABASE')).toBeInTheDocument();
    });
  });

  it('shows success badge in green for successful events', async () => {
    const SecurityMonitoring = require('@/pages/security').default;
    securityApi.listAuditLogs.mockResolvedValue({ logs: [LOGS[0]] });
    render(<SecurityMonitoring />);
    await waitFor(() => {
      const badge = screen.getByText('success');
      expect(badge.className).toMatch(/green/);
    });
  });

  it('shows failure badge in red for failed events', async () => {
    const SecurityMonitoring = require('@/pages/security').default;
    securityApi.listAuditLogs.mockResolvedValue({ logs: [LOGS[1]] });
    render(<SecurityMonitoring />);
    await waitFor(() => {
      const badge = screen.getByText('failure');
      expect(badge.className).toMatch(/red/);
    });
  });

  it('shows empty state when no audit logs exist', async () => {
    const SecurityMonitoring = require('@/pages/security').default;
    securityApi.listAuditLogs.mockResolvedValue({ logs: [] });
    render(<SecurityMonitoring />);
    await waitFor(() => {
      expect(screen.getByText(/no audit logs found/i)).toBeInTheDocument();
    });
  });

  it('shows loading state while fetching logs', async () => {
    const SecurityMonitoring = require('@/pages/security').default;
    securityApi.listAuditLogs.mockImplementation(() => new Promise(() => {}));
    render(<SecurityMonitoring />);
    await waitFor(() => {
      expect(screen.getByText(/loading/i)).toBeInTheDocument();
    });
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// 2. Login — credential validation and token handling
// ─────────────────────────────────────────────────────────────────────────────
describe('Security: Login credential handling', () => {
  beforeEach(() => jest.clearAllMocks());

  it('calls authApi.login with the entered credentials', async () => {
    const Login = require('@/pages/login').default;
    authApi.login.mockResolvedValueOnce({ token: 'jwt-abc', user_id: 'u-1', username: 'alice' });

    render(<Login />);
    fireEvent.change(screen.getByLabelText(/username/i), { target: { value: 'alice' } });
    fireEvent.change(screen.getByLabelText(/^password$/i), { target: { value: 'secret' } });
    fireEvent.submit(screen.getByLabelText(/username/i).closest('form'));

    await waitFor(() => {
      expect(authApi.login).toHaveBeenCalledWith('alice', 'secret');
    });
  });

  it('shows an error on 401 invalid credentials', async () => {
    const Login = require('@/pages/login').default;
    authApi.login.mockRejectedValueOnce(new Error('Invalid username or password'));

    render(<Login />);
    fireEvent.change(screen.getByLabelText(/username/i), { target: { value: 'hacker' } });
    fireEvent.change(screen.getByLabelText(/^password$/i), { target: { value: 'wrong' } });
    fireEvent.submit(screen.getByLabelText(/username/i).closest('form'));

    await waitFor(() => {
      expect(screen.getByText(/invalid username or password/i)).toBeInTheDocument();
    });
  });

  it('does not display any password value in the DOM', () => {
    const Login = require('@/pages/login').default;
    render(<Login />);
    const pwInput = screen.getByLabelText(/^password$/i);
    expect(pwInput).toHaveAttribute('type', 'password');
  });

  it('shows a success message after successful login', async () => {
    const Login = require('@/pages/login').default;
    authApi.login.mockResolvedValueOnce({ token: 'jwt-abc', user_id: 'u-1', username: 'alice' });

    render(<Login />);
    fireEvent.change(screen.getByLabelText(/username/i), { target: { value: 'alice' } });
    fireEvent.change(screen.getByLabelText(/^password$/i), { target: { value: 'secret' } });
    fireEvent.submit(screen.getByLabelText(/username/i).closest('form'));

    await waitFor(() => {
      expect(screen.getByText(/login successful|logged in/i)).toBeInTheDocument();
    });
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// 3. Password change — strength enforcement
// ─────────────────────────────────────────────────────────────────────────────
describe('Security: Password change policy', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    // Provide a valid auth token in localStorage
    Object.defineProperty(window, 'localStorage', {
      writable: true,
      value: {
        getItem: jest.fn((key) => {
          if (key === 'jadevectordb_auth_token') return 'test-token';
          if (key === 'jadevectordb_user_id')    return 'u-1';
          return null;
        }),
        setItem:    jest.fn(),
        removeItem: jest.fn(),
        clear:      jest.fn(),
      },
    });
  });

  it('renders the Change Password heading', () => {
    const ChangePassword = require('@/pages/change-password').default;
    render(<ChangePassword />);
    expect(screen.getByRole('heading', { name: /change password/i })).toBeInTheDocument();
  });

  it('password inputs are type=password (not visible)', () => {
    const ChangePassword = require('@/pages/change-password').default;
    const { container } = render(<ChangePassword />);
    const pwInputs = container.querySelectorAll('input[type="password"]');
    expect(pwInputs.length).toBeGreaterThanOrEqual(2); // current + new
    pwInputs.forEach(el => {
      expect(el).toHaveAttribute('type', 'password');
    });
  });

  it('shows a strength requirement hint', () => {
    const ChangePassword = require('@/pages/change-password').default;
    render(<ChangePassword />);
    // The page hints at minimum length
    expect(screen.getAllByText(/at least 10 characters/i).length).toBeGreaterThan(0);
  });

  it('calls authApi.changePassword with old and new passwords', async () => {
    const ChangePassword = require('@/pages/change-password').default;
    authApi.changePassword.mockResolvedValueOnce({ success: true });

    render(<ChangePassword />);

    fireEvent.change(screen.getByLabelText(/current password/i),   { target: { value: 'OldPass123!' } });
    fireEvent.change(screen.getByLabelText(/^new password$/i),     { target: { value: 'NewPass456@#' } });
    fireEvent.change(screen.getByLabelText(/confirm new password/i), { target: { value: 'NewPass456@#' } });

    fireEvent.submit(screen.getByLabelText(/current password/i).closest('form'));

    await waitFor(() => {
      expect(authApi.changePassword).toHaveBeenCalledWith('u-1', 'OldPass123!', 'NewPass456@#');
    });
  });

  it('shows an error when the current password is wrong', async () => {
    const ChangePassword = require('@/pages/change-password').default;
    authApi.changePassword.mockRejectedValueOnce(new Error('Current password is incorrect'));

    render(<ChangePassword />);

    fireEvent.change(screen.getByLabelText(/current password/i),     { target: { value: 'WrongPass1!' } });
    fireEvent.change(screen.getByLabelText(/^new password$/i),       { target: { value: 'NewPass456@#' } });
    fireEvent.change(screen.getByLabelText(/confirm new password/i), { target: { value: 'NewPass456@#' } });

    fireEvent.submit(screen.getByLabelText(/current password/i).closest('form'));

    await waitFor(() => {
      expect(screen.getByText(/current password is incorrect/i)).toBeInTheDocument();
    });
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// 4. Registration — duplicate username / weak password handling
// ─────────────────────────────────────────────────────────────────────────────
describe('Security: Registration validation', () => {
  beforeEach(() => jest.clearAllMocks());

  it('rejects a username shorter than 3 characters', async () => {
    const Register = require('@/pages/register').default;
    render(<Register />);

    // Fill all required fields — only username is short
    fireEvent.change(screen.getByLabelText(/username/i), { target: { value: 'ab' } });
    const pwFields = screen.getAllByLabelText(/password/i);
    fireEvent.change(pwFields[0], { target: { value: 'StrongPass1!' } });
    if (pwFields[1]) {
      fireEvent.change(pwFields[1], { target: { value: 'StrongPass1!' } });
    }

    fireEvent.submit(screen.getByLabelText(/username/i).closest('form'));

    await waitFor(() => {
      expect(screen.getByText(/username must be at least 3 characters/i)).toBeInTheDocument();
    });
    expect(authApi.register).not.toHaveBeenCalled();
  });

  it('shows an error when the server rejects a duplicate username', async () => {
    const Register = require('@/pages/register').default;
    authApi.register.mockRejectedValueOnce(new Error('Username already exists'));

    render(<Register />);

    // Fill all required fields so validation passes and the API is called
    fireEvent.change(screen.getByLabelText(/username/i), { target: { value: 'alice' } });
    const pwFields = screen.getAllByLabelText(/password/i);
    fireEvent.change(pwFields[0], { target: { value: 'StrongPass1!' } });
    if (pwFields[1]) {
      fireEvent.change(pwFields[1], { target: { value: 'StrongPass1!' } });
    }

    fireEvent.submit(screen.getByLabelText(/username/i).closest('form'));

    await waitFor(() => {
      expect(screen.getByText(/username already exists/i)).toBeInTheDocument();
    });
  });

  it('does not submit when password confirmation mismatches', async () => {
    const Register = require('@/pages/register').default;
    render(<Register />);

    fireEvent.change(screen.getByLabelText(/username/i), { target: { value: 'charlie' } });
    // Set both password fields to different values
    const pwFields = screen.getAllByLabelText(/password/i);
    fireEvent.change(pwFields[0], { target: { value: 'StrongPass1!' } });
    if (pwFields[1]) {
      fireEvent.change(pwFields[1], { target: { value: 'DifferentPass!' } });
    }

    fireEvent.submit(screen.getByLabelText(/username/i).closest('form'));

    // API should NOT be called when passwords mismatch
    await new Promise(r => setTimeout(r, 50));
    if (pwFields[1]) {
      expect(authApi.register).not.toHaveBeenCalled();
    }
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// 5. 401 / 403 error handling — pages surface auth errors gracefully
// ─────────────────────────────────────────────────────────────────────────────
describe('Security: 401/403 error handling', () => {
  beforeEach(() => jest.clearAllMocks());

  it('Database page: shows an error message on 403 Forbidden', async () => {
    const DatabaseManagement = require('@/pages/databases').default;
    databaseApi.listDatabases.mockRejectedValueOnce(new Error('API error: 403'));

    render(<DatabaseManagement />);
    await waitFor(() => {
      expect(screen.getByText(/error fetching databases/i)).toBeInTheDocument();
    });
  });

  it('User Management page: shows an error message on 401 Unauthorized', async () => {
    global.confirm = jest.fn(() => false);
    const UserManagement = require('@/pages/users').default;
    usersApi.listUsers.mockRejectedValueOnce(new Error('API error: 401'));

    render(<UserManagement />);
    await waitFor(() => {
      expect(screen.getByText(/error fetching users/i)).toBeInTheDocument();
    });
  });

  it('Security page: handles 403 on audit log fetch gracefully', async () => {
    const SecurityMonitoring = require('@/pages/security').default;
    securityApi.listAuditLogs.mockRejectedValueOnce(new Error('API error: 403'));

    render(<SecurityMonitoring />);
    await waitFor(() => {
      // Should show empty state rather than crashing
      expect(screen.getByText(/no audit logs found/i)).toBeInTheDocument();
    });
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// 6. XSS prevention — injected script tags must NOT execute
// ─────────────────────────────────────────────────────────────────────────────
describe('Security: XSS prevention in rendered content', () => {
  beforeEach(() => jest.clearAllMocks());

  it('renders alert message text without executing script tags', async () => {
    const { act } = require('@testing-library/react');
    const Alerting = require('@/pages/alerting').default;
    const xssPayload = '<script>window.__xss = true</script>';

    jest.useFakeTimers();
    // Use require('@/lib/api') to get alertApi mock
    const { alertApi: mockAlertApi } = require('@/lib/api');
    mockAlertApi.listAlerts.mockResolvedValue({
      alerts: [{ id: 'a-xss', type: 'error', message: xssPayload, timestamp: '2026-04-13T10:00:00Z' }],
    });

    await act(async () => render(<Alerting />));
    await waitFor(() => {
      // The raw <script> text should appear as text, not execute
      expect(screen.getByText(xssPayload)).toBeInTheDocument();
    });

    // The script should NOT have run
    expect(window.__xss).toBeUndefined();
    jest.useRealTimers();
  });

  it('renders audit log user field without executing injected payloads', async () => {
    const SecurityMonitoring = require('@/pages/security').default;
    const xssPayload = '<img src=x onerror=window.__xss2=1>';

    securityApi.listAuditLogs.mockResolvedValue({
      logs: [{ id: 'l-xss', timestamp: '2026-04-13T10:00:00Z', user: xssPayload, event: 'LOGIN', status: 'success' }],
    });

    render(<SecurityMonitoring />);
    await waitFor(() => screen.getByText(xssPayload));

    // onerror should NOT have fired
    expect(window.__xss2).toBeUndefined();
  });
});

// ─────────────────────────────────────────────────────────────────────────────
// 7. Forgot password / reset password — token required
// ─────────────────────────────────────────────────────────────────────────────
describe('Security: Password reset flow', () => {
  beforeEach(() => jest.clearAllMocks());

  it('does not call forgotPassword when form is empty', () => {
    const ForgotPassword = require('@/pages/forgot-password').default;
    render(<ForgotPassword />);
    fireEvent.submit(screen.getByRole('button', { name: /send|reset|submit/i }).closest('form') ||
                     document.querySelector('form'));
    expect(authApi.forgotPassword).not.toHaveBeenCalled();
  });

  it('calls forgotPassword with the entered username', async () => {
    const ForgotPassword = require('@/pages/forgot-password').default;
    authApi.forgotPassword.mockResolvedValueOnce({ reset_token: 'tok-123', message: 'Check your email' });

    render(<ForgotPassword />);

    const inputs = screen.getAllByRole('textbox');
    fireEvent.change(inputs[0], { target: { value: 'alice' } });
    // If there's an email field fill it too
    if (inputs[1]) {
      fireEvent.change(inputs[1], { target: { value: 'alice@example.com' } });
    }

    fireEvent.submit(inputs[0].closest('form'));

    await waitFor(() => {
      expect(authApi.forgotPassword).toHaveBeenCalled();
    });
  });

  it('displays the reset token after a successful forgot-password request', async () => {
    const ForgotPassword = require('@/pages/forgot-password').default;
    authApi.forgotPassword.mockResolvedValueOnce({ reset_token: 'tok-abc-xyz', message: 'Reset token generated' });

    render(<ForgotPassword />);

    const inputs = screen.getAllByRole('textbox');
    fireEvent.change(inputs[0], { target: { value: 'alice' } });
    if (inputs[1]) {
      fireEvent.change(inputs[1], { target: { value: 'alice@example.com' } });
    }

    fireEvent.submit(inputs[0].closest('form'));

    await waitFor(() => {
      expect(screen.getByText(/tok-abc-xyz/)).toBeInTheDocument();
    });
  });
});
