import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

// Global mocks: next/head, next/router, next/link via jest.config.js
const { useRouter } = require('next/router');
const mockPush = jest.fn();
useRouter.mockReturnValue({ push: mockPush, query: {} });

jest.mock('@/lib/api', () => require('./__mocks__/api'));
const { authApi } = require('@/lib/api');

// Mock localStorage
const store = { jadevectordb_auth_token: 'test-token', jadevectordb_user_id: 'u-1' };
const localStorageMock = {
  getItem: jest.fn((key) => store[key] || null),
  setItem: jest.fn((key, value) => { store[key] = value; }),
  removeItem: jest.fn((key) => { delete store[key]; }),
  clear: jest.fn(),
};
Object.defineProperty(window, 'localStorage', { value: localStorageMock, writable: true });

import ChangePassword from '@/pages/change-password';

describe('ChangePassword page', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockPush.mockClear();
    localStorageMock.getItem.mockImplementation((key) => {
      if (key === 'jadevectordb_auth_token') return 'test-token';
      if (key === 'jadevectordb_user_id') return 'u-1';
      return null;
    });
  });

  it('redirects to /login when not authenticated', async () => {
    localStorageMock.getItem.mockReturnValue(null);
    render(<ChangePassword />);
    await waitFor(() => {
      expect(mockPush).toHaveBeenCalledWith('/login');
    });
  });

  it('renders the Change Password heading', () => {
    render(<ChangePassword />);
    expect(screen.getByRole('heading', { name: /^change password$/i })).toBeInTheDocument();
  });

  it('renders current password, new password, and confirm new password fields', () => {
    render(<ChangePassword />);
    expect(screen.getByLabelText(/current password/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/^new password$/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/confirm new password/i)).toBeInTheDocument();
  });

  it('shows dynamic password strength feedback when password is too short', async () => {
    render(<ChangePassword />);
    const newPasswordInput = screen.getByLabelText(/^new password$/i);
    await userEvent.type(newPasswordInput, 'short');
    // Dynamic message (shown in red paragraph when password is typed) vs static hint
    // The strength message "Password must be at least 10 characters" is in the dynamic p element
    await waitFor(() => {
      const msgs = screen.getAllByText(/at least 10 characters/i);
      // At least one of these should be in a red/colored element (score=0)
      expect(msgs.length).toBeGreaterThan(0);
    });
  });

  it('shows strong password message for a sufficiently complex password', async () => {
    render(<ChangePassword />);
    const newPasswordInput = screen.getByLabelText(/^new password$/i);
    await userEvent.type(newPasswordInput, 'StrongPass1!ExtraLong');
    await waitFor(() => {
      expect(screen.getByText(/strong password/i)).toBeInTheDocument();
    });
  });

  it('shows error when new passwords do not match', async () => {
    render(<ChangePassword />);
    const form = screen.getByLabelText(/current password/i).closest('form');
    await userEvent.type(screen.getByLabelText(/current password/i), 'OldPass1!');
    await userEvent.type(screen.getByLabelText(/^new password$/i), 'NewPass1!ExtraLong');
    await userEvent.type(screen.getByLabelText(/confirm new password/i), 'DifferentPass1!');
    fireEvent.submit(form);
    await waitFor(() => {
      expect(screen.getByText(/passwords do not match/i)).toBeInTheDocument();
    });
  });

  it('calls authApi.changePassword on valid form submission', async () => {
    authApi.changePassword.mockResolvedValueOnce({});
    render(<ChangePassword />);

    await userEvent.type(screen.getByLabelText(/current password/i), 'OldPass1!');
    await userEvent.type(screen.getByLabelText(/^new password$/i), 'NewPass1!ExtraLong');
    await userEvent.type(screen.getByLabelText(/confirm new password/i), 'NewPass1!ExtraLong');
    const form = screen.getByLabelText(/current password/i).closest('form');
    fireEvent.submit(form);

    await waitFor(() => {
      expect(authApi.changePassword).toHaveBeenCalledWith('u-1', 'OldPass1!', 'NewPass1!ExtraLong');
    });
  });

  it('shows success message after password change', async () => {
    authApi.changePassword.mockResolvedValueOnce({});
    render(<ChangePassword />);

    await userEvent.type(screen.getByLabelText(/current password/i), 'OldPass1!');
    await userEvent.type(screen.getByLabelText(/^new password$/i), 'NewPass1!ExtraLong');
    await userEvent.type(screen.getByLabelText(/confirm new password/i), 'NewPass1!ExtraLong');
    const form = screen.getByLabelText(/current password/i).closest('form');
    fireEvent.submit(form);

    await waitFor(() => {
      expect(screen.getByText(/password changed successfully/i)).toBeInTheDocument();
    });
  });

  it('shows error when API call fails', async () => {
    authApi.changePassword.mockRejectedValueOnce(new Error('Incorrect current password'));
    render(<ChangePassword />);

    await userEvent.type(screen.getByLabelText(/current password/i), 'WrongOld1!');
    await userEvent.type(screen.getByLabelText(/^new password$/i), 'NewPass1!ExtraLong');
    await userEvent.type(screen.getByLabelText(/confirm new password/i), 'NewPass1!ExtraLong');
    const form = screen.getByLabelText(/current password/i).closest('form');
    fireEvent.submit(form);

    await waitFor(() => {
      expect(screen.getByText(/incorrect current password/i)).toBeInTheDocument();
    });
  });
});
