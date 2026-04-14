import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

// Global mocks: next/head, next/router, next/link via jest.config.js
jest.mock('@/lib/api', () => require('./__mocks__/api'));
const { authApi } = require('@/lib/api');

import ForgotPassword from '@/pages/forgot-password';

describe('ForgotPassword page', () => {
  beforeEach(() => jest.clearAllMocks());

  it('renders the Forgot Password heading', () => {
    render(<ForgotPassword />);
    expect(screen.getByRole('heading', { name: /forgot password/i })).toBeInTheDocument();
  });

  it('renders username and email fields', () => {
    render(<ForgotPassword />);
    expect(screen.getByLabelText(/username/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/email/i)).toBeInTheDocument();
  });

  it('renders links to login and reset-password pages', () => {
    render(<ForgotPassword />);
    expect(screen.getByRole('link', { name: /sign in/i })).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /already have a reset token/i })).toBeInTheDocument();
  });

  it('shows error when both fields are empty', async () => {
    render(<ForgotPassword />);
    const form = screen.getByRole('button', { name: /send reset instructions/i }).closest('form');
    fireEvent.submit(form);
    await waitFor(() => {
      expect(screen.getByText(/please enter either your username or email/i)).toBeInTheDocument();
    });
  });

  it('calls authApi.forgotPassword when username is provided', async () => {
    authApi.forgotPassword.mockResolvedValueOnce({});
    render(<ForgotPassword />);

    await userEvent.type(screen.getByLabelText(/username/i), 'alice');
    fireEvent.click(screen.getByRole('button', { name: /send reset instructions/i }));

    await waitFor(() => {
      expect(authApi.forgotPassword).toHaveBeenCalledWith('alice', '');
    });
  });

  it('calls authApi.forgotPassword when email is provided', async () => {
    authApi.forgotPassword.mockResolvedValueOnce({});
    render(<ForgotPassword />);

    await userEvent.type(screen.getByLabelText(/email/i), 'alice@example.com');
    fireEvent.click(screen.getByRole('button', { name: /send reset instructions/i }));

    await waitFor(() => {
      expect(authApi.forgotPassword).toHaveBeenCalledWith('', 'alice@example.com');
    });
  });

  it('shows success message after request', async () => {
    authApi.forgotPassword.mockResolvedValueOnce({});
    render(<ForgotPassword />);

    await userEvent.type(screen.getByLabelText(/username/i), 'alice');
    fireEvent.click(screen.getByRole('button', { name: /send reset instructions/i }));

    await waitFor(() => {
      expect(screen.getByText(/password reset instructions/i)).toBeInTheDocument();
    });
  });

  it('displays reset token in success message when returned', async () => {
    authApi.forgotPassword.mockResolvedValueOnce({ reset_token: 'tok-abc', user_id: 'u-1' });
    render(<ForgotPassword />);

    await userEvent.type(screen.getByLabelText(/username/i), 'alice');
    fireEvent.click(screen.getByRole('button', { name: /send reset instructions/i }));

    await waitFor(() => {
      expect(screen.getByText(/tok-abc/)).toBeInTheDocument();
    });
  });

  it('shows error when API call fails', async () => {
    authApi.forgotPassword.mockRejectedValueOnce(new Error('User not found'));
    render(<ForgotPassword />);

    await userEvent.type(screen.getByLabelText(/username/i), 'ghost');
    fireEvent.click(screen.getByRole('button', { name: /send reset instructions/i }));

    await waitFor(() => {
      expect(screen.getByText(/user not found/i)).toBeInTheDocument();
    });
  });

  it('disables button while loading', async () => {
    authApi.forgotPassword.mockImplementation(() => new Promise(() => {}));
    render(<ForgotPassword />);

    await userEvent.type(screen.getByLabelText(/username/i), 'alice');
    fireEvent.click(screen.getByRole('button', { name: /send reset instructions/i }));

    await waitFor(() => {
      expect(screen.getByRole('button', { name: /processing/i })).toBeDisabled();
    });
  });
});
