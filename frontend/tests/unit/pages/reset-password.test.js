import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

// Global mocks: next/head, next/router, next/link via jest.config.js
const { useRouter } = require('next/router');
const mockPush = jest.fn();
useRouter.mockReturnValue({ push: mockPush, query: {} });

jest.mock('@/lib/api', () => require('./__mocks__/api'));
const { authApi } = require('@/lib/api');

import ResetPassword from '@/pages/reset-password';

describe('ResetPassword page', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockPush.mockClear();
  });

  it('renders the Reset Password heading', () => {
    render(<ResetPassword />);
    expect(screen.getByRole('heading', { name: /reset password/i })).toBeInTheDocument();
  });

  it('renders all required fields', () => {
    render(<ResetPassword />);
    expect(screen.getByLabelText(/user id/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/reset token/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/^new password/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/confirm.*password/i)).toBeInTheDocument();
  });

  it('shows error when fields are empty', async () => {
    render(<ResetPassword />);
    const form = screen.getByRole('button', { name: /reset password/i }).closest('form');
    fireEvent.submit(form);
    await waitFor(() => {
      expect(screen.getByText(/please fill in all fields/i)).toBeInTheDocument();
    });
  });

  it('shows error when passwords do not match', async () => {
    render(<ResetPassword />);
    await userEvent.type(screen.getByLabelText(/user id/i), 'u-1');
    await userEvent.type(screen.getByLabelText(/reset token/i), 'tok-abc');
    await userEvent.type(screen.getByLabelText(/^new password/i), 'NewPass1!');
    await userEvent.type(screen.getByLabelText(/confirm.*password/i), 'Different1!');
    const form = screen.getByRole('button', { name: /reset password/i }).closest('form');
    fireEvent.submit(form);
    await waitFor(() => {
      expect(screen.getByText(/passwords do not match/i)).toBeInTheDocument();
    });
  });

  it('calls authApi.resetPassword on valid submission', async () => {
    authApi.resetPassword.mockResolvedValueOnce({});
    render(<ResetPassword />);

    await userEvent.type(screen.getByLabelText(/user id/i), 'u-1');
    await userEvent.type(screen.getByLabelText(/reset token/i), 'tok-abc');
    await userEvent.type(screen.getByLabelText(/^new password/i), 'NewPass12345!');
    await userEvent.type(screen.getByLabelText(/confirm.*password/i), 'NewPass12345!');
    fireEvent.click(screen.getByRole('button', { name: /reset password/i }));

    await waitFor(() => {
      expect(authApi.resetPassword).toHaveBeenCalledWith('u-1', 'tok-abc', 'NewPass12345!');
    });
  });

  it('shows success message after reset', async () => {
    authApi.resetPassword.mockResolvedValueOnce({});
    render(<ResetPassword />);

    await userEvent.type(screen.getByLabelText(/user id/i), 'u-1');
    await userEvent.type(screen.getByLabelText(/reset token/i), 'tok-abc');
    await userEvent.type(screen.getByLabelText(/^new password/i), 'NewPass12345!');
    await userEvent.type(screen.getByLabelText(/confirm.*password/i), 'NewPass12345!');
    fireEvent.click(screen.getByRole('button', { name: /reset password/i }));

    await waitFor(() => {
      expect(screen.getByText(/password reset successfully/i)).toBeInTheDocument();
    });
  });

  it('shows error when API call fails', async () => {
    authApi.resetPassword.mockRejectedValueOnce(new Error('Invalid reset token'));
    render(<ResetPassword />);

    await userEvent.type(screen.getByLabelText(/user id/i), 'u-1');
    await userEvent.type(screen.getByLabelText(/reset token/i), 'bad-tok');
    await userEvent.type(screen.getByLabelText(/^new password/i), 'NewPass12345!');
    await userEvent.type(screen.getByLabelText(/confirm.*password/i), 'NewPass12345!');
    fireEvent.click(screen.getByRole('button', { name: /reset password/i }));

    await waitFor(() => {
      expect(screen.getByText(/invalid reset token/i)).toBeInTheDocument();
    });
  });
});
