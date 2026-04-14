import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

// Global mocks via jest.config.js: next/head, next/router, next/link
const { useRouter } = require('next/router');
const mockPush = jest.fn();
useRouter.mockReturnValue({ push: mockPush, query: {} });

jest.mock('@/lib/api', () => require('./__mocks__/api'));
const { authApi } = require('@/lib/api');

import Login from '@/pages/login';

describe('Login page', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockPush.mockClear();
  });

  it('renders the login heading', () => {
    render(<Login />);
    expect(screen.getByRole('heading', { name: /^login$/i })).toBeInTheDocument();
  });

  it('renders username and password fields', () => {
    render(<Login />);
    expect(screen.getByLabelText(/username/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/password/i)).toBeInTheDocument();
  });

  it('renders forgot password and register links', () => {
    render(<Login />);
    expect(screen.getByText(/forgot your password/i)).toBeInTheDocument();
    expect(screen.getByText(/sign up/i)).toBeInTheDocument();
  });

  it('shows error when only username is provided (no password)', async () => {
    // HTML required prevents empty submission; fill username only and submit the form directly
    render(<Login />);
    const form = screen.getByRole('button', { name: /sign in/i }).closest('form');
    await userEvent.type(screen.getByLabelText(/username/i), 'admin');
    fireEvent.submit(form);
    await waitFor(() => {
      expect(screen.getByText(/please enter both username and password/i)).toBeInTheDocument();
    });
  });

  it('calls authApi.login with correct credentials on submit', async () => {
    authApi.login.mockResolvedValueOnce({ username: 'admin' });
    render(<Login />);

    await userEvent.type(screen.getByLabelText(/username/i), 'admin');
    await userEvent.type(screen.getByLabelText(/password/i), 'secret');
    fireEvent.click(screen.getByRole('button', { name: /sign in/i }));

    await waitFor(() => {
      expect(authApi.login).toHaveBeenCalledWith('admin', 'secret');
    });
  });

  it('shows success message on successful login', async () => {
    authApi.login.mockResolvedValueOnce({ username: 'admin' });
    render(<Login />);

    await userEvent.type(screen.getByLabelText(/username/i), 'admin');
    await userEvent.type(screen.getByLabelText(/password/i), 'secret');
    fireEvent.click(screen.getByRole('button', { name: /sign in/i }));

    await waitFor(() => {
      expect(screen.getByText(/successfully logged in/i)).toBeInTheDocument();
    });
  });

  it('shows must-change-password message when flag is set', async () => {
    authApi.login.mockResolvedValueOnce({ must_change_password: true });
    render(<Login />);

    await userEvent.type(screen.getByLabelText(/username/i), 'admin');
    await userEvent.type(screen.getByLabelText(/password/i), 'secret');
    fireEvent.click(screen.getByRole('button', { name: /sign in/i }));

    await waitFor(() => {
      expect(screen.getByText(/must change your password/i)).toBeInTheDocument();
    });
  });

  it('shows error message when login fails', async () => {
    authApi.login.mockRejectedValueOnce(new Error('Invalid credentials'));
    render(<Login />);

    await userEvent.type(screen.getByLabelText(/username/i), 'admin');
    await userEvent.type(screen.getByLabelText(/password/i), 'wrongpass');
    fireEvent.click(screen.getByRole('button', { name: /sign in/i }));

    await waitFor(() => {
      expect(screen.getByText(/invalid credentials/i)).toBeInTheDocument();
    });
  });

  it('disables the submit button while loading', async () => {
    authApi.login.mockImplementation(() => new Promise(() => {}));
    render(<Login />);

    await userEvent.type(screen.getByLabelText(/username/i), 'admin');
    await userEvent.type(screen.getByLabelText(/password/i), 'secret');
    fireEvent.click(screen.getByRole('button', { name: /sign in/i }));

    await waitFor(() => {
      expect(screen.getByRole('button', { name: /signing in/i })).toBeDisabled();
    });
  });
});
