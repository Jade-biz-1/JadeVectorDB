import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

// Global mocks: next/head, next/router, next/link via jest.config.js
const { useRouter } = require('next/router');
const mockPush = jest.fn();
useRouter.mockReturnValue({ push: mockPush, query: {} });

jest.mock('@/lib/api', () => require('./__mocks__/api'));
const { authApi } = require('@/lib/api');

import Home from '@/pages/index';

describe('Home (landing/auth) page', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockPush.mockClear();
  });

  it('renders the JadeVectorDB brand heading', () => {
    render(<Home />);
    expect(screen.getAllByText(/JadeVectorDB/i).length).toBeGreaterThan(0);
  });

  it('renders login form fields by default', () => {
    render(<Home />);
    expect(screen.getByPlaceholderText(/username/i)).toBeInTheDocument();
    expect(screen.getByPlaceholderText(/password/i)).toBeInTheDocument();
  });

  it('can switch to register mode showing email field', async () => {
    render(<Home />);
    // Find the switch button (between login/register modes)
    const registerToggle = screen.getByRole('button', { name: /register|sign up|create account/i });
    fireEvent.click(registerToggle);
    await waitFor(() => {
      expect(screen.getByPlaceholderText(/email/i)).toBeInTheDocument();
    });
  });

  it('calls authApi.login on login submit', async () => {
    authApi.login.mockResolvedValueOnce({ username: 'admin' });
    render(<Home />);

    await userEvent.type(screen.getByPlaceholderText(/username/i), 'admin');
    await userEvent.type(screen.getByPlaceholderText(/password/i), 'secret');
    fireEvent.click(screen.getByRole('button', { name: /sign in|login/i }));

    await waitFor(() => {
      expect(authApi.login).toHaveBeenCalledWith('admin', 'secret');
    });
  });

  it('redirects to /databases after successful login', async () => {
    authApi.login.mockResolvedValueOnce({ username: 'admin' });
    render(<Home />);

    await userEvent.type(screen.getByPlaceholderText(/username/i), 'admin');
    await userEvent.type(screen.getByPlaceholderText(/password/i), 'secret');
    fireEvent.click(screen.getByRole('button', { name: /sign in|login/i }));

    await waitFor(() => {
      expect(mockPush).toHaveBeenCalledWith('/databases');
    });
  });

  it('shows error message on login failure', async () => {
    authApi.login.mockRejectedValueOnce(new Error('Invalid credentials'));
    render(<Home />);

    await userEvent.type(screen.getByPlaceholderText(/username/i), 'admin');
    await userEvent.type(screen.getByPlaceholderText(/password/i), 'wrong');
    fireEvent.click(screen.getByRole('button', { name: /sign in|login/i }));

    await waitFor(() => {
      expect(screen.getByText(/invalid credentials/i)).toBeInTheDocument();
    });
  });
});
